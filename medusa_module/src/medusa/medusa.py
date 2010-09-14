from portaudio import *
import numpy as np


# Select the correct name for the shared library, dependent on platform
if platform.system() == "Windows":
    libname = get_python_lib() + "\\medusa\\medusa.dll"
else:
    libname = find_library("medusa")

if libname == None:
    raise RuntimeError("Unable to locate library `medusa`")


# Instantiate FFI reference to libmedusa
cmedusa = ctypes.CDLL(libname)


# struct ContigArrayHandle [in `medusa.h`]
class ContigArrayHandle (ctypes.Structure):
    """
    Used in the `void *userData` field of the `callback_play_ndarray` function in `libmedusa`.

    Definition from `medusa.h`:

    typedef struct ContigArrayHandle {
        PyObject *x;      // Refers to the (assumed-contiguous) ndarray
        int chan_i;       // Iterates along dimension 0, must be < PyArray_DIM(x, 0)
        int samp_i;       // Iterates along dimension 1, must be < PyArray_DIM(x, 1)
        double samp_freq; // Sampling frequency in Hertz
        double scale;     // Scaling factor for each sample, should be in the interval [0, 1]
    } ContigArrayHandle;
    """
    _fields_ = (("x",         py_object), # Ends up being cast as `(PyArrayObject *)`
                ("chan_i",    c_int),
                ("samp_i",    c_int),
                ("samp_freq", c_double),
                ("scale",     c_double),
                ("loop",      c_int))


# struct ToneData [in 'medusa.h']
class ToneData (ctypes.Structure):
    """
    Used in the `void *userData` field of the `callback_play_tone` function in `libmedusa`.

    Definition from `medusa.h`:

    typedef struct ToneData {
        int channels;     // Number of channels total
        int chan_out;     // Output channel
        double tone_freq; // Frequency of the tone that will be generated
        double samp_freq; // Sampling frequency
    } ToneData;
    """
    _fields_ = (("t",         c_uint),
                ("channels",  c_int),
                ("chan_out",  c_int),
                ("tone_freq", c_double),
                ("samp_freq", c_double),
                ("scale",     c_double))


class Device:
    input_device_index = None
    output_device_index = None

    def __init__(self, in_index, out_index):
        self.input_device_index = in_index
        self.output_device_index = out_index


    def create_tone(self, tone_freq, samp_freq=44100.0, scale=1.0, channels=1, chan_out=1, samp_format=paFloat32):
        # Index of `chan_out` is 1-based as passed, but translated to a 0-based index in the `ToneStream` constructor
        s = ToneStream(self, channels, chan_out, tone_freq, samp_freq, scale, samp_format)
        s.open()
        return s


    def open_array(self, arr, samp_freq=44100.0, scale=1.0, loop=False, samp_format=paFloat32):
        s = ArrayStream(self, arr, samp_freq, scale, loop, samp_format)
        s.open()
        return s


class Stream:
    # device : PaDevice
    device = None

    # Our handle on the stream
    stream_p = None

    # `ctypes.Structure` to be pointed to in `void *userData` arg of the Portaudio callback
    user_data = None

    # samp_format : ctypes.c_ulong
    samp_format = None

    # samp_freq : ctypes.c_double
    samp_freq = None

    # : `PaStreamParameters(ctypes.Structure)`
    in_param = None
    out_param = None

    # False: Should be a `PaStreamCallback` function in `libmedusa` via `cmedusa`
    # True: Should be a `c_int` that corresponds to an enum in `medusa.h`
    callback = None

    def open(self):
        #raise RuntimeError("This instance method requires subclass implementation")
        if self.in_param == None:
            in_param = None
        else:
            in_param = byref(self.in_param)

        if self.out_param == None:
            out_param = None
        else:
            out_param = byref(self.out_param)

        if self.callback == None:
            raise RuntimeError("No PaStreamCallback defined (self.callback == None)")

        self.stream_p = cmedusa.open_stream(self.stream_p, in_param, out_param, py_object(self), byref(self.user_data), c_int(self.callback))

    def start(self):
        err = pa.Pa_StartStream(self.stream_p)
        ERROR_CHECK(err)
        return err

    def stop(self):
        err = pa.Pa_StopStream(self.stream_p)
        ERROR_CHECK(err)

    def time(self):
        t = pa.Pa_GetStreamTime(self.stream_p)
        if t:
            return pa.Pa_GetStreamTime(self.stream_p)
        else:
            raise RuntimeError("An Portaudio error occurred in `self.time()`: return value of `Pa_GetStreamTime` == 0")

    def play(self):
        self.open()
        self.start()

    def pause(self):
        self.stop()

    def is_playing(self):
        err = pa.Pa_IsStreamStopped(self.stream_p)
        ERROR_CHECK(err)
        print err
        return bool(err)


class ArrayStream(Stream):
    # Callback-specific attributes
    callback = 0
    arr = None

    def __init__(self, device, arr, samp_freq, scale, loop=False, samp_format=paFloat32):
        # Manually convert Python boolean to C-friendly "true" or "false" ints
        if loop:
            loop = c_int(1)
        else:
            loop = c_int(0)

        # `callback_ndarray` currently requires arrays with two dimensions
        if len(arr.shape) == 1:
            n = arr.shape[0]
            arr = arr.reshape((n,1))

        self.arr = np.ascontiguousarray(arr)

        self.user_data = ContigArrayHandle(py_object(self.arr), 0, 0, samp_freq, scale, loop)
        self.stream_p = c_void_p()
        self.device = device
        self.samp_format = samp_format
        self.samp_freq = samp_freq

        channels = arr.shape[1]

        self.out_param = PaStreamParameters(c_int(device.output_device_index), c_int(channels), samp_format, 1.0, None)

    def play(self):
        if not self.is_paused():
            print "was paused"
            self.user_data.chan_i = c_int(0)
            self.user_data.samp_i = c_int(0)
            self.open()  # Reopen stream, just in case
        self.start()


class ToneStream (Stream):
    callback = 1

    def __init__(self, device, channels, chan_out, tone_freq, samp_freq, scale, samp_format=paFloat32):
        # Ensure that our desired output channels is in the 1-indexed range of available channels
        assert chan_out <= channels

        chan_out -= 1  # Since actual channel indices are 0-based, not 1-based

        self.user_data = ToneData(0, channels, chan_out, tone_freq, samp_freq, scale)
        self.stream_p = c_void_p()
        self.device = device
        self.samp_format = samp_format
        self.samp_freq = samp_freq

        # Include a fix to have what is currently `1.0` be the default low latency value of the stream we've indexed
        self.out_param = PaStreamParameters(c_int(device.output_device_index), c_int(channels), samp_format, 1.0, None)


def generateHostApiInfo():
    HostApiInfoPointer = POINTER(PaHostApiInfo)
    api_count = pa.Pa_GetHostApiCount()
    for i in xrange(api_count):
        p = ctypes.cast(pa.Pa_GetHostApiInfo(i), HostApiInfoPointer)
        hai = p[0]
        yield hai


def generateDeviceInfo():
    DeviceInfoPointer = POINTER(PaDeviceInfo)
    device_count = pa.Pa_GetDeviceCount()

    ERROR_CHECK(device_count)

    if device_count == 0:
        raise RuntimeError("No devices found")

    for i in xrange(device_count):
        p = ctypes.cast(pa.Pa_GetDeviceInfo(i), DeviceInfoPointer)
        di = p[0]
        yield di


def printAvailableDevices(host_api=None, verbose=False):
    # If necessary, wrap `host_api` in a list so it is iterable
    if isinstance(host_api, str):
        host_api = [host_api]

    if host_api == None:
        # No constraints
        devices = list(generateDeviceInfo())
    else:
        # Remap user-friendly aliases to integer enum values
        host_api = [HostApiTypeAliases[x] for x in host_api]

        # Filter output of `generateDeviceInfo()`
        devices = [di for di in generateDeviceInfo() if (di.hostApi in host_api)]

    if len(devices) == 0:
        print "No devices found for given hostApi(s):", ",".join([HostApiTypeAliases[x] for x in host_api])
        return None

    if verbose:
        for i,di in enumerate(devices):
            print "index:", i
            print "structVersion", di.structVersion
            print "name", di.name
            print "hostApi:", PaHostApiTypeId.from_int[di.hostApi]
            print "maxInputChannels:", di.maxInputChannels
            print "maxOutputChannels:", di.maxOutputChannels
            print "defaultLowInputLatency", di.defaultLowInputLatency
            print "defaultLowOutputLatency", di.defaultLowOutputLatency
            print "defaultHighInputLatency", di.defaultHighInputLatency
            print "defaultHighOutputLatency", di.defaultHighOutputLatency
            print "defaultSampleRate", di.defaultSampleRate
            print ""
    else:
        for i,di in enumerate(devices):
            print "index:", i
            print "name", di.name
            print "hostApi:", PaHostApiTypeId.from_int[di.hostApi]
            print "maxInputChannels:", di.maxInputChannels
            print "maxOutputChannels:", di.maxOutputChannels
            print "defaultSampleRate", di.defaultSampleRate
            print ""


def open_device(out_device_index=None, in_device_index=None):
    if out_device_index == None:
        out_device_index = pa.Pa_GetDefaultOutputDevice()

    d = Device(in_device_index, out_device_index)
    return d


def open_default_device():
    out_di = pa.Pa_GetDefaultOutputDevice()
    in_di = pa.Pa_GetDefaultInputDevice()

    d = Device(in_di, out_di)
    return d


def start_streams(streams, open_streams=False, normalize=False):
    if open_streams:
        [s.open() for s in streams]

    if normalize:
        scale_factor = 1./(len(streams))
        for i,x in enumerate(streams):
            if isinstance(x, ArrayStream):
                streams[i].cah.scale = scale_factor
            elif isinstance(x, ToneStream):
                streams[i].td.scale = scale_factor

    num_streams = len(streams)
    STREAM_P_ARRAY_TYPE = c_void_p * num_streams  # custom-length type
    stream_p_array = STREAM_P_ARRAY_TYPE(*[s.stream_p for s in streams])
    cmedusa.start_streams(stream_p_array, c_int(num_streams))


def init():
    """
    Attempts to initialize Portaudio.
    """
    err = pa.Pa_Initialize()
    ERROR_CHECK(err)
    return True


def terminate():
    """
    Attempts to terminate Portaudio.
    """
    err = pa.Pa_Terminate()
    ERROR_CHECK(err)
    return True