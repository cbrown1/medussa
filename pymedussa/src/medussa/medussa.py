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
            self.__dict__[name] = value
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
        elif name == "mix_mat":
            # enforce array contiguity
            self.__dict__[name] = np.ascontiguousarray(val)
        else:
            self.__dict__[name] = val

    def open(self):
        if self.callback == None:
            raise RuntimeError("No PaStreamCallback defined (self.callback == None)")

        self.stream_p = cmedussa.open_stream(py_object(self))

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

    def __init__(self, tone_freq, mix_mat):
        self.tone_freq = tone_freq
        self.mix_mat = mix_mat

        # Find a smart way to determine this value,
        # which has to be hardcoded into the callback
        self.pa_fpb = 1000



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
