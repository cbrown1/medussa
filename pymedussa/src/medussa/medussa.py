# -*- coding: utf-8 -*-
from portaudio import *
import sndfile
import numpy as np
from time import sleep
import atexit
import rkit
from os.path import exists

# Select the correct name for the shared library, dependent on platform
if platform.system() == "Windows":
    libname = get_python_lib() + "\\medussa\\medussa.dll"
    if not exists(libname):
        raise RuntimeError("Unable to locate library: " + libname)
elif platform.system() == "Linux":
    libname = "/usr/local/lib/libmedussa.so"
else:
    libname = find_library("medussa")
    if libname == None:
        raise RuntimeError("Unable to locate library `medussa`")


# Instantiate FFI reference to libmedussa
cmedussa = ctypes.CDLL(libname)


@atexit.register
def medussa_exit():
    pa.Pa_Terminate()


class StreamUserData(ctypes.Structure):
    """
    struct stream_user_data {
        void *parent;

        void *device;

        PaStream *stream;
        PaStreamParameters *in_param;
        PaStreamParameters *out_param;
        double fs;
        PaStreamCallback *callback;

        double *mix_mat;
        double *mute_mat;
        int pa_fpb;
    };
    """
    _fields_ = (("parent",    c_void_p),
                ("device",    py_object),
                ("stream",    c_void_p),
                ("in_param",  c_void_p),
                ("out_param", c_void_p),
                ("fs",        c_double),
#                ("callback",  c_void_p),
                ("mix_mat",   POINTER(c_double)),
                ("mix_mat_0", c_int),
                ("mix_mat_1", c_int),
                ("mute_mat",  POINTER(c_double)),
                ("mute_mat_0", c_int),
                ("mute_mat_1", c_int),
                ("pa_fpb",    c_int))


class FiniteUserData(ctypes.Structure):
    """
    struct finite_user_data {
        void *parent;

        unsigned int loop;
        unsigned int cursor;
        int frames;
        double duration;
    };
    """
    _fields_ = (("parent",   c_void_p),
                ("loop",     c_int),
                ("cursor",   c_uint),
                ("frames",   c_uint),
                ("duration", c_double))


class ArrayUserData(ctypes.Structure):
    """
    struct array_user_data {
        void *parent;
        PyObject *self;

        double *ndarr;
    };
    """
    _fields_ = (("parent", c_void_p),
                ("self",   py_object),
                ("ndarr",  POINTER(c_double)),
                ("ndarr_0", c_int),
                ("ndarr_1", c_int))


class Device:
    """
    Audio device object.
    """
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
        """
        Returns a stream object representing a pure tone.

        Parameters
        ----------
        tone_freq : int
            The frequency, in Hz, of the tone.
        fs : int
            The sampling frequency.

        Returns
        -------
        s : Stream object
            The stream object.
        """
        s = ToneStream(self, fs, None, tone_freq)
        return s

    def create_white(self, fs):
        """
        Returns a stream object representing Gaussian/white noise.

        Parameters
        ----------
        fs : int
            The sampling frequency.

        Returns
        -------
        s : Stream object
            The stream object.
        """
        s = WhiteStream(self, fs, None)
        return s

    def open_array(self, arr, fs):
        """
        Returns a stream object representing an ndarray.

        Parameters
        ----------
        arr : array
            The array of audio data.
        fs : int
            The sampling frequency.

        Returns
        -------
        s : Stream object
            The stream object.
        """
        s = ArrayStream(self, fs, None, arr)
        return s

    def open_file(self, finpath):
        """
        Returns a stream object representing a soundfile on disk.

        Parameters
        ----------
        finpath : string
            The path to the sound file.

        Returns
        -------
        s : Stream object
            The stream object.
        """
        s = SndfileStream(self, None, finpath)
        return s


class Stream(object):
    """
    Maximally-generic stream class.
    """
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
    mute_mat = None
    pa_fpb = 0 # `paFramesPerBufferUnspecified' == 0

    # structs for the callbacks
    stream_user_data = StreamUserData()


    def __setattr__(self, name, val):
        if name == "fs":
            # enforce fs as floating point (add nonnegative check?)
            self.__dict__[name] = float(val)
        elif name == "mix_mat" or name == "arr" or name == "mute_mat":
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
        """
        Stops playback of the stream (Playback cursor is reset to zero).
        """
        err = pa.Pa_StopStream(self.stream_ptr)
        ERROR_CHECK(err)

    def pa_time(self):
        """
        Returns the portaudio time.
        """
        t = pa.Pa_GetStreamTime(self.stream_ptr)
        if t:
            return t.value
        else:
            raise RuntimeError("Error indicated by `Pa_GetStreamTime()` -> 0")

    def play(self):
        """
        Starts playback of the stream.
        """
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
        """
        Pauses playback of the stream (Playback cursor is not reset).
        """
        self.stop()

    def is_playing(self):
        """
        Boolean indicating whether the stream is currently playing.
        """
        err = pa.Pa_IsStreamActive(self.stream_ptr)
        ERROR_CHECK(err)
        return bool(err)

    def mute(self):
        """
        Mutes the stream. Mix matrix is unaffected.
        """
        # simply swaps the mix matrix with a zero matrix of same shape, or back
        self.mix_mat, self.mute_mat = self.mute_mat, self.mix_mat

    def __del__(self):
        #pa.Pa_StopStream(self.stream_ptr)
        pa.Pa_CloseStream(self.stream_ptr)


class ToneStream(Stream):
    """
    Stream object representing a pure tone.
    """
    tone_freq = None
    t = None

    def __init__(self, device, fs, mix_mat, tone_freq):
        # Initialize `Stream` attributes
        # OLD: self.callback_ptr = ctypes.cast(ctypes.pointer(cmedussa.callback_tone), c_void_p)
        self.callback = cmedussa.callback_tone
        self.callback_ptr = cmedussa.callback_tone
        self.device = device

        if mix_mat == None:
            output_channels = self.device.out_device_info.maxOutputChannels
            self.mix_mat = np.ones((output_channels,1))
        else:
            self.mix_mat = mix_mat

        self.mute_mat = self.mix_mat * 0.0
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


class WhiteStream(Stream):
    """
    Stream object representing white noise.
    """
    rk_state = None
    rk_state_ptr = None

    def __init__(self, device, fs, mix_mat):
        # Initialize `Stream` attributes
        # OLD: self.callback_ptr = ctypes.cast(ctypes.pointer(cmedussa.callback_tone), c_void_p)
        self.callback = cmedussa.callback_white
        self.callback_ptr = cmedussa.callback_white
        self.device = device

        if mix_mat == None:
            output_channels = self.device.out_device_info.maxOutputChannels
            self.mix_mat = np.ones((output_channels,1))
        else:
            self.mix_mat = mix_mat

        self.mute_mat = self.mix_mat * 0.0
        self.stream_p = 0
        self.fs = fs

        # Initialize this class' attributes
        self.rk_state = rkit.Rk_state()
        self.rk_state_ptr = ctypes.addressof(self.rk_state)
        cmedussa.rk_randomseed(byref(self.rk_state))

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
    """
    Generic stream object used to derive sndfilestream and arraystream objects.
    """
    loop = None
    pa_fpb = 1024  # This lets us avoid `malloc` in the callback
    cursor = 0

    frames = None # Total length of the signal in frames
    duration = None # Total length of the signal in milliseconds

    finite_user_data = FiniteUserData()

    def time(self, pos=None, posunit="ms"):
        """
        Gets or sets the current cursor position.
        If `pos` is `None`, returns the current cursor position in ms.
        Otherwise, sets the cursor position to `pos`, as deterimined by
        the argument to `posunit`.

        Parameters
        ----------
        pos : numeric
            The position to set the cursor to.
        posunit : string
            The units of pos. May be of value:
            "ms": assume `pos` is of type `float` [default]
            "sec": `assume `pos` is of type float`
            "frames": assume `pos` is of type `int`

        Returns
        -------
        pos : numeric
            The current position of the cursor. This value is returned if
            no input `pos` is specified.
        """
        if pos == None:
            return self.cursor / self.fs * 1000.0
        elif posunit == "ms":
            newcursor = int(pos / 1000.0 * self.fs)
            if not (newcursor < self.frames):
                raise RuntimeError("New cursor position %d exceeds signal frame count %d." % (newcursor, self.frames))
            self.cursor = int(pos / 1000.0 * self.fs)
        elif posunit == "sec":
            newcursor = int(pos * self.fs)
            if not (newcursor < self.frames):
                raise RuntimeError("New cursor position %d exceeds signal frame count %d." % (newcursor, self.frames))
            self.cursor = int(pos * self.fs)
        elif posunit == "frames":
            assert isinstance(pos, int)
            if not (pos < self.frames):
                raise RuntimeError("New cursor position %d exceeds signal frame count %d." % (newcursor, self.frames))
            self.cursor = pos
        else:
            raise RuntimeError("Bad argument to `posunit`")


class ArrayStream(FiniteStream):
    """
    Stream object representing a NumPy array.
    """
    arr = None
    array_user_data = ArrayUserData()

    def __setattr__(self, name, val):
        if name == "stream":
            self.stream_user_data.stream = val
            self.__dict__[name] = val
        elif name == "in_param":
            self.stream_user_data.in_param = ctypes.cast(ctypes.pointer(val), ctypes.c_void_p)
            self.__dict__[name] = val
        elif name == "out_param":
            self.stream_user_data.out_param = ctypes.cast(ctypes.pointer(val), ctypes.c_void_p)
            self.__dict__[name] = val
        elif name == "fs":
            self.stream_user_data.fs = float(val)
            self.__dict__[name] = float(val)
#        elif name == "callback":
#            self.stream_user_data.callback = int(val)
#            self.__dict__[name] = val
        elif name == "mix_mat":
            self.__dict__[name] = np.ascontiguousarray(val)
            self.stream_user_data.mix_mat = self.mix_mat.ctypes.data_as(POINTER(c_double))
            self.stream_user_data.mix_mat_0 = self.mix_mat.shape[0]
            self.stream_user_data.mix_mat_1 = self.mix_mat.shape[1]
        elif name == "mute_mat":
            self.__dict__[name] = np.ascontiguousarray(val)
            self.stream_user_data.mute_mat = self.mute_mat.ctypes.data_as(POINTER(c_double))
            self.stream_user_data.mute_mat_0 = self.mute_mat.shape[0]
            self.stream_user_data.mute_mat_1 = self.mute_mat.shape[1]
        elif name == "pa_fpb":
            self.stream_user_data.pa_fpb = val
            self.__dict__[name] = val
        elif name == "loop":
            self.finite_user_data.loop = val
            self.__dict__[name] = val
        elif name == "cursor":
            self.finite_user_data.cursor = val
            self.__dict__[name] = val
        elif name == "frames":
            self.finite_user_data.frames = val
            self.__dict__[name] = val
        elif name == "duration":
            self.finite_user_data.duration = val
            self.__dict__[name] = val
        elif name == "arr":
            self.__dict__[name] = np.ascontiguousarray(val)
            self.array_user_data.ndarr = self.arr.ctypes.data_as(POINTER(c_double))
            self.array_user_data.ndarr_0 = val.shape[0]
            self.array_user_data.ndarr_1 = val.shape[1]
        else:
            self.__dict__[name] = val

    # We need only override names that are modified by a given callback
    def __getattribute__(self, name):
        if name == "cursor":
            return self.finite_user_data.cursor
        else:
            return object.__getattribute__(self, name)

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
        self.mute_mat = self.mix_mat * 0.0

        # print "DEBUG: output_channels == %d" % (output_channels,)

        self.out_param = PaStreamParameters(self.device.out_index,
                                            output_channels, # number of rows is output dimension
                                            paFloat32,
                                            self.device.out_device_info.defaultLowInputLatency,
                                            None)
        self.array_user_data.parent = ctypes.cast(ctypes.pointer(self.finite_user_data), ctypes.c_void_p)
        self.finite_user_data.parent = ctypes.cast(ctypes.pointer(self.stream_user_data), ctypes.c_void_p)
        self.user_data = ctypes.addressof(self.array_user_data)



class SndfileStream(FiniteStream):
    """
    Stream object representing a sound file on disk.
    """
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

        #print "DEBUG:", self.fin, type(self.fin)
        #print "DEBUG:", self.finfo.frames
        #print "DEBUG:", self.finfo.samplerate
        #print "DEBUG:", self.finfo.channels

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

        self.mute_mat = self.mix_mat * 0.0

        self.out_param = PaStreamParameters(self.device.out_index,
                                            output_channels, # number of rows is output dimension
                                            paFloat32,
                                            self.device.out_device_info.defaultLowInputLatency,
                                            None)

    def __del__(self):
        #pa.Pa_StopStream(self.stream_ptr)
        pa.Pa_CloseStream(self.stream_ptr)
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


def getAvailableDevices(host_api=None, verbose=False):
    '''
    Returns a list containing information on the available audio devices.

    Parameters
    ----------
    host_api : string
        Filters the list of devices to include only the specified host_api.
    verbose : Bool
        Include more information.

    Returns
    -------
    devices : list
        The list of devices.
    '''
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
        return None
    else:
        return devices


def printAvailableDevices(host_api=None, verbose=False):
    '''
    Displays information on the available audio devices.

    Parameters
    ----------
    host_api : string
        Filters the list of devices to include only the specified host_api.
    verbose : Bool
        Print more information.

    Returns
    -------
    None
    '''
    devices = getAvailableDevices(host_api, verbose)

    if len(devices) == 0:
        print "No devices found for given hostApi(s):", ",".join([HostApiTypeAliases[x] for x in host_api])
        return None

    if verbose:
        for i,di in enumerate(devices):
            print "index:", i
            print " structVersion", di.structVersion
            print " name", di.name
            print " hostApi:", PaHostApiTypeId.from_int[di.hostApi]
            print " maxInputChannels:", di.maxInputChannels
            print " maxOutputChannels:", di.maxOutputChannels
            print " defaultLowInputLatency", di.defaultLowInputLatency
            print " defaultLowOutputLatency", di.defaultLowOutputLatency
            print " defaultHighInputLatency", di.defaultHighInputLatency
            print " defaultHighOutputLatency", di.defaultHighOutputLatency
            print " defaultSampleRate", di.defaultSampleRate
            print ""
    else:
        for i,di in enumerate(devices):
            print "index:", i
            print " name", di.name
            print " hostApi:", PaHostApiTypeId.from_int[di.hostApi]
            print " maxInputChannels:", di.maxInputChannels
            print " maxOutputChannels:", di.maxOutputChannels
            print " defaultSampleRate", di.defaultSampleRate
            print ""


def open_device(out_device_index=None, in_device_index=None):
    '''
    Opens the specified input and output devices.
    Use None for default devices.

    Parameters
    ----------
    out_device_index : int
        Index to the desired output device.
    in_device_index : int
        Index to the desired input device.

    Returns
    -------
    d : Device object
        Object representing the specified devices.
    '''
    if out_device_index == None:
        out_device_index = pa.Pa_GetDefaultOutputDevice()

    d = Device(in_device_index, out_device_index)
    return d


def open_default_device():
    """
    This differs from `open_device()` (with no arguments) only in
    that a default output device is also opened.
    """
    out_di = pa.Pa_GetDefaultOutputDevice()
    in_di = pa.Pa_GetDefaultInputDevice()

    d = Device(in_di, out_di)
    return d


def start_streams(streams, open_streams=False, normalize=False):
    """
    Tries to start playback of specified streams as synchronously as possible.

    Parameters
    ----------
    streams : list
        List of stream objects.

    Returns
    -------
    None
    """
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

    Parameters
    ----------
    arr : ndarray
        The array to play. Each column is treated as a channel.
    fs : int
        The sampling frequency.

    Returns
    -------
    None
    """
    d = open_default_device()
    s = d.open_array(arr, fs)
    s.play()
    while s.is_playing():
        sleep(.01)


def playfile(filename):
    """
    Plays a soundfile on the default device with blocking, Matlab-style.

    Use with care! Long soundfiles will cause the interpreter to lock for a
    correspondingly long time!

    Parameters
    ----------
    filename : str
        The path to the file to play.

    Returns
    -------
    None
    """
    d = open_default_device()
    s = d.open_file(filename)
    s.play()
    while s.is_playing():
        sleep(.01)


def readfile(finpath):
    """
    Read any libsndfile-compatible sound file into an ndarray.

    Parameters
    ----------
    filename : str

    Returns
    -------
    (arr, fs) : (ndarray, float)
    """
    finfo = sndfile.SF_INFO(0,0,0,0,0,0)
    fin = sndfile.csndfile.sf_open(finpath, sndfile.SFM_READ, byref(finfo))

    fs = finfo.samplerate
    BUFFTYPE = ctypes.c_double * (finfo.frames * finfo.channels)
    buff = BUFFTYPE()
    frames_read = sndfile.csndfile.sf_readf_double(fin, byref(buff), finfo.frames)
    #print "frames_read", frames_read

    err = sndfile.csndfile.sf_close(c_void_p(fin))
    #print "err", err

    #print finfo.frames, finfo.channels
    arr = np.ascontiguousarray(np.zeros((finfo.frames, finfo.channels)))

    for i in xrange(finfo.frames):
        for j in xrange(finfo.channels):
            arr[i][j] = buff[i*finfo.channels + j]

    return (arr, float(fs))
