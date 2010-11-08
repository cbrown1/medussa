from portaudio import *
import sndfile
import numpy as np
from time import sleep


# Select the correct name for the shared library, dependent on platform
if platform.system() == "Windows":
    libname = get_python_lib() + "\\medussa.dll"
elif platform.system() == "Linux":
    libname = "/usr/local/lib/libmedussa.so"
else:
    libname = find_library("medussa")

if libname == None:
    raise RuntimeError("Unable to locate library `medussa`")


# Instantiate FFI reference to libmedussa
cmedussa = ctypes.CDLL(libname)


class Device:
    in_index = None
    in_device_info = None
    in_name = None
    in_hostapi = None

    out_index = None
    out_device_info = None
    out_name = None
    out_hostapi = None

    def __init__(self, in_index=None, out_index=None):
        """
        Note that, because we have overridden `__setattr__`, an index
        assignment in general will automatically update the user-friendly
        instance attributes which should always be determined
        """
        if in_index != None:
            self.in_index = in_index
        if out_index != None:
            self.out_index = out_index

    def __setattr__(self, name, val):
        """
        We override the `__setattr__` method for the device class so that, by
        just assigning a valid `PaDeviceIndex` to `in_index` or `out_index`,
        the user-friendly convenience attributes that are wholly dependent on
        this index will be updated automatically.
        """
        if name == "in_index":
            # Argument validation
            if not isinstance(val, int):
                raise RuntimeError("Device index must be a positive integer")
            if not (val < pa.Pa_GetDeviceCount()):
                raise RuntimeError("Device index out of range")

            # Get the `DeviceInfo` for this index
            ptr = ctypes.cast(pa.Pa_GetDeviceInfo(val), DeviceInfoPointer) # get pointer to DeviceInfo
            di = ptr[0] # dereference pointer, using a local variable for convenient access in this function

            self.in_device_info = di
            self.in_name = di.name
            self.in_hostapi = PaHostApiTypeId.from_int[di.hostApi] # user-friendly hostapi

            # Do actual requested attribute assignment.
            self.__dict__[name] = val
        elif name == "out_index":
            # Argument validation
            if not isinstance(val, int):
                raise RuntimeError("Device index must be a positive integer")
            if not (val < pa.Pa_GetDeviceCount()):
                raise RuntimeError("Device index out of range")

            # Get the `DeviceInfo` for this index
            ptr = ctypes.cast(pa.Pa_GetDeviceInfo(val), DeviceInfoPointer) # get pointer to DeviceInfo
            di = ptr[0] # dereference pointer, using a local variable for convenient access in this function

            self.out_device_info = di
            self.out_name = di.name
            self.out_hostapi = PaHostApiTypeId.from_int[di.hostApi] # user-friendly hostapi

            # Do actual requested attribute assignment.
            self.__dict__[name] = val
        else:
            # Any other attribute assignment is business as usual, for now.
            self.__dict__[name] = val

    def create_tone(self, tone_freq, fs):
        s = ToneStream(self, fs, np.array([1.0, 1.0]), tone_freq)
        return s

    def open_array(self, arr, fs):
        s = ArrayStream(self, fs, None, arr)
        return s

    def open_file(self, finpath):
        s = SndfileStream(self, None, finpath)
        return s

class Stream:
    # device : PaDevice
    device = None

    # Attributes for `Pa_OpenStream()`
    stream_ptr = None
    in_param = None
    out_param = None
    fs = None
    callback = None

    # Mixing matrix for computing output
    mix_mat = None
    pa_fpb = 0 # `paFramesPerBufferUnspecified' == 0

    def __setattr__(self, name, val):
        if name == "fs":
            # enforce fs as floating point (add nonnegative check?)
            self.__dict__[name] = float(val)
        elif name == "mix_mat" or name == "arr":
            # enforce array contiguity
            self.__dict__[name] = np.ascontiguousarray(val)
        else:
            self.__dict__[name] = val

    def open(self):
        if self.callback == None:
            raise RuntimeError("No PaStreamCallback defined (self.callback == None)")

        spin_ptr = None
        spout_ptr = None

        if self.in_param != None:
            spin_ptr = StreamParametersPointer(self.in_param)

        if self.out_param != None:
            spout_ptr = StreamParametersPointer(self.out_param)
            self.spout_ptr = ctypes.addressof(self.out_param)

        self.stream_ptr = cmedussa.open_stream(py_object(self), spin_ptr, spout_ptr, self.callback_ptr)

    def start(self):
        err = pa.Pa_StartStream(self.stream_ptr)
        ERROR_CHECK(err)
        return err

    def stop(self):
        err = pa.Pa_StopStream(self.stream_ptr)
        ERROR_CHECK(err)

    def pa_time(self):
        t = pa.Pa_GetStreamTime(self.stream_ptr)
        if t:
            return t.value
        else:
            raise RuntimeError("Error indicated by `Pa_GetStreamTime()` -> 0")

    def play(self):
        if (self.stream_ptr == None):
            self.open()
            self.start()
        elif self.is_playing():
            self.pause()
            err = pa.Pa_CloseStream(self.stream_ptr)
            ERROR_CHECK(err)
            self.open()
            self.start()
        else:
            self.open()
            self.start()

    def pause(self):
        self.stop()

    def is_playing(self):
        err = pa.Pa_IsStreamActive(self.stream_ptr)
        ERROR_CHECK(err)
        return bool(err)


class ToneStream(Stream):
    tone_freq = None
    t = None

    def __init__(self, device, fs, mix_mat, tone_freq):
        # Initialize `Stream` attributes
        # OLD: self.callback_ptr = ctypes.cast(ctypes.pointer(cmedussa.callback_tone), c_void_p)
        self.callback = cmedussa.callback_tone
        self.callback_ptr = cmedussa.callback_tone
        self.device = device
        self.mix_mat = mix_mat
        self.stream_p = 0
        self.fs = fs

        # Initialize this class' attributes
        self.tone_freq = tone_freq
        self.t = 0

        # Find a smart way to determine this value,
        # which has to be hardcoded into the callback
        self.pa_fpb = 1024

        #self.out_param = PaStreamParameters(devindex, channel_count, sample_format, sugg_lat, hostapispecstrminfo)
        self.out_param = PaStreamParameters(self.device.out_index,
                                            self.mix_mat.shape[0], # number of rows is output dimension
                                            paFloat32,
                                            self.device.out_device_info.defaultLowInputLatency,
                                            None)


class FiniteStream(Stream):
    loop = None
    pa_fpb = 1024  # This lets us avoid `malloc` in the callback
    cursor = 0

    frames = None # Total length of the signal in frames
    duration = None # Total length of the signal in milliseconds

    def time(self, pos=None, posunit="ms"):
        """
        If `pos` is `None`, returns the current cursor position in milliseconds.
        Otherwise, sets the cursor position to `pos`, as deterimined by the argument to `posunit`.

        `posunit` may be of value:
            "ms": assume `pos` is of type `float`
            "sec": `assume `pos` is of type float`
            "frames": assume `pos` is of type `int`
        """
        if pos == None:
            return self.cursor / 44100.0 * 1000.0
        elif posunit == "ms":
            newcursor = int(pos / 1000.0 * 44100.0)
            if not (newcursor < self.frames):
                raise RuntimeError("New cursor position %d exceeds signal frame count %d." % (newcursor, self.frames))
            self.cursor = int(pos / 1000.0 * 44100.0)
        elif posunit == "sec":
            newcursor = int(pos * 44100.0)
            if not (newcursor < self.frames):
                raise RuntimeError("New cursor position %d exceeds signal frame count %d." % (newcursor, self.frames))
            self.cursor = int(pos * 44100.0)
        elif posunit == "frames":
            assert isinstance(pos, int)
            if not (pos < self.frames):
                raise RuntimeError("New cursor position %d exceeds signal frame count %d." % (newcursor, self.frames))
            self.cursor = pos
        else:
            raise RuntimeError("Bad argument to `posunit`")


class ArrayStream(FiniteStream):
    arr = None

    def __init__(self, device, fs, mix_mat, arr, loop=False):
        if len(arr.shape) == 1:
            arr = arr.reshape(arr.size, 1)

        # Initialize `Stream` attributes
        self.callback = cmedussa.callback_ndarray
        self.callback_ptr = cmedussa.callback_ndarray
        self.device = device

        if mix_mat == None:
            self.mix_mat = np.eye(arr.shape[1])
        else:
            self.mix_mat = mix_mat
        #self.stream_ptr = 0
        self.fs = fs

        # Initialize `FiniteStream` attributes
        self.loop = loop

        # Initialize this class' attributes
        self.arr = arr

        # Set length data
        self.frames = self.arr.shape[0]
        self.duration = self.frames / float(self.fs) * 1000

        #if self.arr.shape[1] <= self.device.out_device_info.maxOutputChannels:
        #    output_channels = self.arr.shape[1]
        #else:
        #    output_channels = self.device.out_device_info.maxOutputChannels

        output_channels = self.device.out_device_info.maxOutputChannels

        self.mix_mat = np.resize(self.mix_mat, (output_channels, self.mix_mat.shape[1]))
        if output_channels > self.mix_mat.shape[1]:
            self.mix_mat
            self.mix_mat[self.mix_mat.shape[1]:,:] *= 0.0  # zero out extra rows which, by default, are just repeated in memory

        # print "DEBUG: output_channels == %d" % (output_channels,)

        self.out_param = PaStreamParameters(self.device.out_index,
                                            output_channels, # number of rows is output dimension
                                            paFloat32,
                                            self.device.out_device_info.defaultLowInputLatency,
                                            None)


class SndfileStream(FiniteStream):
    fin = None
    finfo = None
    finpath = None

    def __init__(self, device, mix_mat, finpath, loop=False):
        # Initialize `Stream` attributes
        self.callback = cmedussa.callback_sndfile_read
        self.callback_ptr = cmedussa.callback_sndfile_read
        self.device = device

        # Initialize `FiniteStream` attributes
        self.loop = loop

        # Initialize this class' attributes
        self.finpath = finpath
        self.finfo = sndfile.SF_INFO(0,0,0,0,0,0)
        self.fin = sndfile.csndfile.sf_open(finpath, sndfile.SFM_READ, byref(self.finfo))

        print self.fin, type(self.fin)
        print self.finfo.frames
        print self.finfo.samplerate
        print self.finfo.channels

        # set sampling frequency
        self.fs = self.finfo.samplerate

        # set actual device output channels
        output_channels = self.device.out_device_info.maxOutputChannels

        # set signal length
        self.frames = self.finfo.frames
        self.duration = self.finfo.frames / float(self.finfo.samplerate) * 1000.0

        if mix_mat == None:
            self.mix_mat = np.eye(self.finfo.channels)
        else:
            self.mix_mat = mix_mat
        self.mix_mat = np.resize(self.mix_mat, (output_channels, self.mix_mat.shape[1]))
        if output_channels > self.mix_mat.shape[1]:
            self.mix_mat
            self.mix_mat[self.mix_mat.shape[1]:,:] *= 0.0  # zero out extra rows which, by default, are just repeated in memory

        # print "DEBUG: output_channels == %d" % (output_channels,)

        self.out_param = PaStreamParameters(self.device.out_index,
                                            output_channels, # number of rows is output dimension
                                            paFloat32,
                                            self.device.out_device_info.defaultLowInputLatency,
                                            None)

    def __del__(self):
        sndfile.csndfile.sf_close(c_void_p(self.fin))

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
    """
    This differs from `open_device()` (i.e. with no arguments) only in
    that a default output device is also opened.
    """
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
    cmedussa.start_streams(stream_p_array, c_int(num_streams))


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


def playarr(arr, fs, channel=1):
    """
    Plays an array on the default device with blocking, Matlab-style.
    """
    d = open_default_device()
    s = ArrayStream(d, arr, fs)
    s.open()
    s.play()
    while s.is_playing():
        sleep(.01)

def playfile(filename, channel=1):
    """
    Plays a soundfile on the default device with blocking, Matlab-style.

    Use with care! Long soundfiles will cause the interpreter to lock for a
    correspondingly long time!
    """
    d = open_default_device()
    s = SndfileStream(d, filename)
    s.open()
    s.play()
    while s.is_playing():
        sleep(.01)
