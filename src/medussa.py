# -*- coding: utf-8 -*-
from portaudio import *
import sndfile
import numpy as np
from time import sleep
import atexit
import rkit
from os.path import exists, join
import inspect
from pink import Pink_noise_t
import platform

pyver = "%s.%s" % (platform.python_version_tuple()[0], platform.python_version_tuple()[1])


# Select the correct name for the shared library, dependent on platform
if platform.system() == "Windows":
    libname = get_python_lib() + "\\medussa\\medussa.dll"
    if not exists(libname):
        raise RuntimeError("Unable to locate library: " + libname)
elif platform.system() == "Linux":
    libname = '/usr/local/lib/python%s/site-packages/medussa/libmedussa.so' % pyver
    #libname = join(get_python_lib(), 'medussa', 'libmedussa.so')
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

class SndfileUserData(ctypes.Structure):
    """
    struct sndfile_user_data {
        void *parent;
        PyObject *self;

        SNDFILE *fin;
        char    *finpath;
        SF_INFO *finfo;
    };

    """
    _fields_ = (("parent",  c_void_p),
                ("self",    py_object),
                ("fin",     c_void_p),
                ("finpath", c_char_p),
                ("finfo",   POINTER(sndfile.SF_INFO)))

class ToneUserData(ctypes.Structure):
    """
    struct tone_user_data {
        void *parent;
        PyObject *self;

        unsigned int t;
        double tone_freq;
    };
    """
    _fields_ = (("parent",    c_void_p),
                ("self",      py_object),
                ("t",         c_uint),
                ("tone_freq", c_double))


class WhiteUserData(ctypes.Structure):
    """
    struct white_user_data {
        void *parent;
        PyObject *self;

        rk_state *rks;
    };
    """
    _fields_ = (("parent", c_void_p),
                ("self",   py_object),
                ("rks",    c_void_p))


class PinkUserData(ctypes.Structure):
    """
    struct pink_user_data {
        void *parent;
        PyObject *self;

        pink_noise_t *pn;
    };
    """
    _fields_ = (("parent", c_void_p),
                ("self",   py_object),
                ("pn",     c_void_p))


class Device(object):
    """
    Medussa object representing an audio device.

    Contains methods to create various streams, and information about the
    hardware device it represents.
    """
    _in_index = None
    in_device_info = None
    in_name = None
    in_hostapi = None

    out_index = None
    out_device_info = None
    out_name = None
    out_hostapi = None

    out_channels = None

    @property
    def in_index(self):
        return self._in_index

    @in_index.setter
    def in_index(self, val):
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
        self._in_index = val

    @in_index.deleter
    def in_index(self):
        del self._in_index

    @property
    def out_index(self):
        return self._out_index

    @out_index.setter
    def out_index(self, value):
        # Argument validation
        if not isinstance(value, int):
            raise RuntimeError("Device index must be a positive integer")
        if not (value < pa.Pa_GetDeviceCount()):
            raise RuntimeError("Device index out of range")

        # Get the `DeviceInfo` for this index
        ptr = ctypes.cast(pa.Pa_GetDeviceInfo(value), DeviceInfoPointer) # get pointer to DeviceInfo

        di = ptr[0] # dereference pointer, using a local variable for convenient access in this function

        self.out_device_info = di
        self.out_name = di.name
        self.out_hostapi = PaHostApiTypeId.from_int[di.hostApi] # user-friendly hostapi

        # Do actual requested attribute assignment.
        self._out_index = value

    @out_index.deleter
    def out_index(self):
        del self._out_index

    def __init__(self, in_index=None, out_index=None, out_channels=None):
        if in_index != None:
            self.in_index = in_index
        if out_index != None:
            self.out_index = out_index
        if out_channels != None:
            self.out_channels = out_channels


    def create_tone(self, tone_freq, fs=None):
        """
        Returns a Medussa stream object representing a pure tone.

        Parameters
        ----------
        tone_freq : int
            The frequency, in Hz, of the tone.
        fs : int
            The sampling frequency. Don't specify for the device's default.

        Returns
        -------
        s : Stream object
            The stream object.
        """
        if fs is None:
            fs = self.out_device_info.defaultSampleRate
        s = ToneStream(self, fs, None, tone_freq)
        return s

    def create_white(self, fs=None):
        """
        Returns a Medussa stream object representing Gaussian/white noise.

        Parameters
        ----------
        fs : int
            The sampling frequency. Don't specify for the device's default.

        Returns
        -------
        s : Stream object
            The stream object.
        """
        if fs is None:
            fs = self.out_device_info.defaultSampleRate
        s = WhiteStream(self, fs, None)
        return s

    def create_pink(self, fs=None):
        """
        Returns a Medussa stream object representing pink noise.

        Parameters
        ----------
        fs : int
            The sampling frequency. Don't specify for the device's default.

        Returns
        -------
        s : Stream object
            The stream object.
        """
        if fs is None:
            fs = self.out_device_info.defaultSampleRate
        s = PinkStream(self, fs, None)
        return s

    def open_array(self, arr, fs):
        """
        Returns a Medussa stream object representing an ndarray.

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

    def open_file(self, file_name):
        """
        Returns a Medussa stream object representing a soundfile on disk.

        Parameters
        ----------
        file_name : string
            The path to the sound file.

        Returns
        -------
        s : Stream object
            The stream object.
        """
        s = SndfileStream(self, None, file_name)
        return s


class Stream(object):
    """
    Generic stream class.
    """
    # device : PaDevice
    device = None

    # Attributes for `Pa_OpenStream()`
    stream_ptr = None
    in_param = None
    out_param = None
    fs = None

    # Mixing matrix for computing output
    _mix_mat = None
    _mute_mat = None
#    pa_fpb = 0 # `paFramesPerBufferUnspecified' == 0

    # structs for the callbacks
    stream_user_data = None

    @property
    def stream(self):
        return self.stream_user_data.stream

    @stream.setter
    def stream(self, val):
        self.stream_user_data.stream = val

    @stream.deleter
    def stream(self):
        del self.stream_user_data.stream

    @property
    def in_param(self):
        return self.stream_user_data.in_param

    @in_param.setter
    def in_param(self, val):
        self.stream_user_data.in_param = ctypes.cast(ctypes.pointer(val),
                                                     ctypes.c_void_p)

    @in_param.deleter
    def in_param(self):
        del self.stream_user_data.in_param

    @property
    def out_param(self):
        return self._out_param

    @out_param.setter
    def out_param(self, val):
        self.stream_user_data.out_param = ctypes.addressof(val)
        self._out_param = val

    @out_param.deleter
    def out_param(self):
        del self._out_param

    @property
    def fs(self):
        return self.stream_user_data.fs

    @fs.setter
    def fs(self, val):
        self.stream_user_data.fs = val

    @property
    def mix_mat(self):
        return self._mix_mat

    @mix_mat.setter
    def mix_mat(self, val):
        self._mix_mat = np.ascontiguousarray(val)
        self.stream_user_data.mix_mat = self.mix_mat.ctypes.data_as(POINTER(c_double))
        self.stream_user_data.mix_mat_0 = self.mix_mat.shape[0]
        self.stream_user_data.mix_mat_1 = self.mix_mat.shape[1]

    @mix_mat.deleter
    def mix_mat(self):
        del self._mix_mat

    @property
    def mute_mat(self):
        return self._mute_mat

    @mute_mat.setter
    def mute_mat(self, val):
        self._mute_mat = np.ascontiguousarray(val)
        self.stream_user_data.mute_mat = self.mute_mat.ctypes.data_as(POINTER(c_double))
        self.stream_user_data.mute_mat_0 = self.mute_mat.shape[0]
        self.stream_user_data.mute_mat_1 = self.mute_mat.shape[1]

    @mute_mat.deleter
    def mute_mat(self):
        del self._mute_mat

    @property
    def pa_fpb(self):
        return self.stream_user_data.pa_fpb

    @pa_fpb.setter
    def pa_fpb(self, val):
        self.stream_user_data.pa_fpb = val

    @pa_fpb.deleter
    def pa_fpb(self):
        del self.stream_user_data.pa_fpb



    def open(self):
        self.stream_ptr = cmedussa.open_stream(py_object(self),
                                               self.stream_user_data.in_param,
                                               self.stream_user_data.out_param,
                                               self.callback_ptr)

    def start(self):
        err = pa.Pa_StartStream(self.stream_ptr)
        ERROR_CHECK(err)
        return err

    def stop(self):
        """
        Stops playback of the stream (Playback cursor is reset to zero).
        """
        if pa.Pa_IsStreamStopped(self.stream_ptr):
            return
        else:
            err = pa.Pa_StopStream(self.stream_ptr)
            ERROR_CHECK(err)

    def pa_time(self):
        """
        Returns the portaudio time, which is in seconds.
        """
        t = pa.Pa_GetStreamTime(self.stream_ptr)
        if t:
            return t
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
        if pa.Pa_IsStreamStopped(self.stream_ptr):
            return
        else:
            err = pa.Pa_StopStream(self.stream_ptr)
            ERROR_CHECK(err)

    def is_playing(self):
        """
        Boolean indicating whether the stream is currently playing.
        """
        err = pa.Pa_IsStreamActive(self.stream_ptr)
        ERROR_CHECK(err)
        return bool(err)

    def mute(self):
        """
        Mutes or unmutes the stream.

        Mix matrix is unaffected. Playback will continue while stream is muted.
        """
        # simply swaps the mix matrix with a zero matrix of same shape, or back
        self.mix_mat, self.mute_mat = self.mute_mat, self.mix_mat

    def __del__(self):
        pa.Pa_CloseStream(self.stream_ptr)


class ToneStream(Stream):
    """
    Stream object representing a pure tone.

    Methods
    -------
    play
        Starts playback of the stream.
    stop
        Stops playback of the stream (Playback cursor is reset to zero).
    pause
        Pauses playback of the stream (Playback cursor is not reset).
    mute
        Mutes or unmutes the stream.
        Mix matrix is unaffected. Playback will continue while stream is muted.
    is_playing
        Boolean indicating whether the stream is currently playing.

    Properties
    ----------
    tone_freq
        Frequency, in Hz, of the tone.
    fs
        Sampling frequency, in Hz.
    mix_mat
        A NumPy array that acts as a mixer. The number of columns corresponds
        to the number of input channels (in the case of a tonestream, 1),
        and the number of rows corresponds to the number of output channels,
        which is hardware specific, and can be get/set with dev.out_channels.
        The values of the mix_mat are are floats between 0. and 1., and are
        used to specify the playback level of each `input` channel on each
        `output` channel.
    """
    tone_freq = None
    t = None
    tone_user_data = None

    @property
    def tone_freq(self):
        return self.tone_user_data.tone_freq

    @tone_freq.setter
    def tone_freq(self, val):
        self.tone_user_data.tone_freq = val

    @property
    def t(self):
        return self.tone_user_data.t

    @t.setter
    def t(self, val):
        self.tone_user_data.t = val


    def __init__(self, device, fs, mix_mat, tone_freq):
        # Initialize `Stream` attributes
        self.callback_ptr = cmedussa.callback_tone
        self.device = device

        self.stream_user_data = StreamUserData()
        self.tone_user_data = ToneUserData()

        if mix_mat == None:
            if self.device.out_channels == None:
                out_channels = self.device.out_device_info.maxOutputChannels
            else:
                out_channels = self.device.out_channels
            self.mix_mat = np.ones((out_channels,1))
        else:
            self.mix_mat = mix_mat

        self.mute_mat = self.mix_mat * 0.0
        self.fs = fs

        # Initialize this class' attributes
        self.tone_freq = tone_freq
        self.t = 0

        # Find a smart way to determine this value,
        # which has to be hardcoded into the callback
        self.pa_fpb = 1024

        self.out_param = PaStreamParameters(self.device.out_index,
                                            self.mix_mat.shape[0],
                                            paFloat32,
                                            self.device.out_device_info.defaultLowInputLatency,
                                            None)
        self.tone_user_data.parent = ctypes.addressof(self.stream_user_data)
        self.user_data = ctypes.addressof(self.tone_user_data)

class WhiteStream(Stream):
    """
    Stream object representing Gaussian white noise, which has a flat spectrum.

    Methods
    -------
    play
        Starts playback of the stream.
    stop
        Stops playback of the stream (Playback cursor is reset to zero).
    pause
        Pauses playback of the stream (Playback cursor is not reset).
    mute
        Mutes or unmutes the stream.
        Mix matrix is unaffected. Playback will continue while stream is muted.
    is_playing
        Boolean indicating whether the stream is currently playing.

    Properties
    ----------
    fs
        Sampling frequency, in Hz.
    mix_mat
        A NumPy array that acts as a mixer. The number of columns corresponds
        to the number of input channels (in the case of a whitestream, 1),
        and the number of rows corresponds to the number of output channels,
        which is hardware specific, and can be get/set with dev.out_channels.
        The values of the mix_mat are are floats between 0. and 1., and are
        used to specify the playback level of each `input` channel on each
        `output` channel.
    """
    rk_state = None
    white_user_data = None

    @property
    def rk_state(self):
        return self._rk_state

    @rk_state.setter
    def rk_state(self, val):
        self._rk_state = val
        self.white_user_data.rks = ctypes.addressof(self._rk_state)

    @rk_state.deleter
    def rk_state(self):
        del self._rk_state

    def __init__(self, device, fs, mix_mat):
        self.callback_ptr = cmedussa.callback_white
        self.device = device

        self.stream_user_data = StreamUserData()
        self.white_user_data = WhiteUserData()

        if mix_mat == None:
            if self.device.out_channels == None:
                out_channels = self.device.out_device_info.maxOutputChannels
            else:
                out_channels = self.device.out_channels
            self.mix_mat = np.ones((out_channels,1))
        else:
            self.mix_mat = mix_mat

        self.mute_mat = self.mix_mat * 0.0
        self.fs = fs

        # Initialize this class' attributes
        self.rk_state = rkit.Rk_state()
        cmedussa.rk_randomseed(byref(self.rk_state))

        # Find a smart way to determine this value,
        # which has to be hardcoded into the callback
        self.pa_fpb = 1024

        self.out_param = PaStreamParameters(self.device.out_index,
                                            self.mix_mat.shape[0],
                                            paFloat32,
                                            self.device.out_device_info.defaultLowInputLatency,
                                            None)

        self.white_user_data.parent = ctypes.addressof(self.stream_user_data)
        self.user_data = ctypes.addressof(self.white_user_data)


class PinkStream(Stream):
    """
    Stream object representing pink noise, which has equal energy per octave.

    Methods
    -------
    play
        Starts playback of the stream.
    stop
        Stops playback of the stream (Playback cursor is reset to zero).
    pause
        Pauses playback of the stream (Playback cursor is not reset).
    mute
        Mutes or unmutes the stream.
        Mix matrix is unaffected. Playback will continue while stream is muted.
    is_playing
        Boolean indicating whether the stream is currently playing.

    Properties
    ----------
    fs
        Sampling frequency, in Hz.
    mix_mat
        A NumPy array that acts as a mixer. The number of columns corresponds
        to the number of input channels (in the case of a pinkstream, 1),
        and the number of rows corresponds to the number of output channels,
        which is hardware specific, and can be get/set with dev.out_channels.
        The values of the mix_mat are are floats between 0. and 1., and are
        used to specify the playback level of each `input` channel on each
        `output` channel.
    """
    pink_user_data = None

    def __init__(self, device, fs, mix_mat):
        self.callback_ptr = cmedussa.callback_pink
        self.device = device

        self.stream_user_data = StreamUserData()
        self.pink_user_data = PinkUserData()

        if mix_mat == None:
            if self.device.out_channels == None:
                out_channels = self.device.out_device_info.maxOutputChannels
            else:
                out_channels = self.device.out_channels
            self.mix_mat = np.ones((out_channels,1))
        else:
            self.mix_mat = mix_mat

        self.mute_mat = self.mix_mat * 0.0
        self.fs = fs

        self.pn = Pink_noise_t()
        self.pink_user_data.pn = ctypes.addressof(self.pn)
        cmedussa.initialize_pink_noise(self.pink_user_data.pn, 4)

        self.pa_fpb = 1024

        self.out_param = PaStreamParameters(self.device.out_index,
                                            self.mix_mat.shape[0],
                                            paFloat32,
                                            self.device.out_device_info.defaultLowInputLatency,
                                            None)

        self.pink_user_data.parent = ctypes.addressof(self.stream_user_data)
        self.user_data = ctypes.addressof(self.pink_user_data)


class FiniteStream(Stream):
    """
    Generic stream object used to derive sndfilestream and arraystream objects.
    """
    loop = None
    pa_fpb = 1024  # This lets us avoid `malloc` in the callback
    cursor = 0

    frames = None # Total length of the signal in frames
    duration = None # Total length of the signal in milliseconds

    finite_user_data = None

    @property
    def loop(self):
        return self.finite_user_data.loop

    @loop.setter
    def loop(self, val):
        self.finite_user_data.loop = val

    @property
    def cursor(self):
        return self.finite_user_data.cursor

    @cursor.setter
    def cursor(self, val):
        self.finite_user_data.cursor = val

    @property
    def frames(self):
        return self.finite_user_data.frames

    @frames.setter
    def frames(self, val):
        self.finite_user_data.frames = val

    @property
    def duration(self):
        return self.finite_user_data.duration

    @duration.setter
    def duration(self, val):
        self.finite_user_data.duration = val

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

    def stop(self):
        super(FiniteStream, self).stop()
        self.cursor = 0

class ArrayStream(FiniteStream):
    """
    Stream object representing a NumPy array.

    You can use medussa.read_file to load soundfiles into NumPy arrays.
    """
    _arr = None
    array_user_data = None

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, val):
        self._arr = np.ascontiguousarray(val)
        self.array_user_data.ndarr = self._arr.ctypes.data_as(POINTER(c_double))
        self.array_user_data.ndarr_0 = val.shape[0]
        self.array_user_data.ndarr_1 = val.shape[1]

    @arr.deleter
    def arr(self):
        del self._arr

    def __init__(self, device, fs, mix_mat, arr, loop=False):
        if len(arr.shape) == 1:
            arr = arr.reshape(arr.size, 1)

        self.stream_user_data = StreamUserData()
        self.finite_user_data = FiniteUserData()
        self.array_user_data = ArrayUserData()

        # Initialize `Stream` attributes
        self.callback_ptr = cmedussa.callback_ndarray
        self.device = device

        if mix_mat == None:
            self.mix_mat = np.eye(arr.shape[1])
        else:
            self.mix_mat = mix_mat

        self.fs = fs

        # Initialize `FiniteStream` attributes
        self.loop = loop

        # Initialize this class' attributes
        self.arr = arr

        # Set length data
        self.frames = self.arr.shape[0]
        self.duration = self.frames / float(self.fs) * 1000

        out_channels = self.device.out_channels

        self.mix_mat = np.resize(self.mix_mat,
                                 (out_channels, self.mix_mat.shape[1]))

        if out_channels > self.mix_mat.shape[1]:
            # zero out extra rows which, by default, are just repeated in memory
            self.mix_mat[self.mix_mat.shape[1]:,:] *= 0.0

        self.mute_mat = self.mix_mat * 0.0

        self.out_param = PaStreamParameters(self.device.out_index,
                                            out_channels,
                                            paFloat32,
                                            self.device.out_device_info.defaultLowInputLatency,
                                            None)

        self.array_user_data.parent = ctypes.addressof(self.finite_user_data)
        self.finite_user_data.parent = ctypes.addressof(self.stream_user_data)
        self.user_data = ctypes.addressof(self.array_user_data)


class SndfileStream(FiniteStream):
    """
    Stream object representing a sound file on disk.

    The audio data are not loaded into memory, but rather are streamed from
    disk.
    """
    fin = None
    finpath = None
    finfo = None
    sndfile_user_data = None

    @property
    def finpath(self):
        return self._finpath

    @finpath.setter
    def finpath(self, val):
        # Only permit assignment to `finpath` attribute if we are in `__init__`
        if inspect.stack()[1][3] == "__init__":
            self.sndfile_user_data.finpath = c_char_p(val)
            self._finpath = val
        else:
            raise RuntimeError("`%s` attribute is immutable." % (name))

    @finpath.deleter
    def finpath(self):
        del self._finpath

    @property
    def fin(self):
        return self._fin

    @fin.setter
    def fin(self, val):
        # Only permit assignment to `fin` attribute if we are in `__init__`
        if inspect.stack()[1][3] == "__init__":
            self.sndfile_user_data.fin = val
            self._fin = val
        else:
            raise RuntimeError("`%s` attribute is immutable." % (name))

    @fin.deleter
    def fin(self):
        del self._fin

    @property
    def finfo(self):
        return self._finfo

    @finfo.setter
    def finfo(self, val):
        # Only permit assignment to `finfo` attribute if we are in `__init__`
        if inspect.stack()[1][3] == "__init__":
            self._finfo = val
            self.sndfile_user_data.finfo = ctypes.cast(ctypes.pointer(val), POINTER(sndfile.SF_INFO))
#            self.sndfile_user_data.finfo = ctypes.addressof(self.finfo)
        else:
            raise RuntimeError("`%s` attribute is immutable." % (name))

    def __init__(self, device, mix_mat, finpath, loop=False):
        # Initialize `Stream` attributes
        self.callback_ptr = cmedussa.callback_sndfile_read
        self.device = device

        self.stream_user_data = StreamUserData()
        self.finite_user_data = FiniteUserData()
        self.sndfile_user_data = SndfileUserData()

        # Initialize `FiniteStream` attributes
        self.loop = loop

        self.cursor = 0

        # Initialize this class' attributes
        self.finpath = finpath
        self.finfo = sndfile.SF_INFO(0,0,0,0,0,0)
        self.fin = sndfile.csndfile.sf_open(finpath,
                                            sndfile.SFM_READ,
                                            byref(self.finfo))

        # set sampling frequency
        self.fs = self.finfo.samplerate

        # set actual device output channels
        if self.device.out_channels == None:
            out_channels = self.device.out_device_info.maxOutputChannels
        else:
            out_channels = self.device.out_channels

        # set signal length
        self.frames = self.finfo.frames
        self.duration = self.finfo.frames / float(self.finfo.samplerate) * 1000.0

        if mix_mat == None:
            self.mix_mat = np.eye(self.finfo.channels)
        else:
            self.mix_mat = mix_mat
        self.mix_mat = np.resize(self.mix_mat,
                                 (out_channels, self.mix_mat.shape[1]))

        if out_channels > self.mix_mat.shape[1]:
            # zero out extra rows which, by default, are just repeated in memory
            self.mix_mat[self.mix_mat.shape[1]:,:] *= 0.0

        self.mute_mat = self.mix_mat * 0.0

        self.out_param = PaStreamParameters(self.device.out_index,
                                            out_channels,
                                            paFloat32,
                                            self.device.out_device_info.defaultLowInputLatency,
                                            None)
        self.sndfile_user_data.parent = ctypes.addressof(self.finite_user_data)
        self.finite_user_data.parent = ctypes.addressof(self.stream_user_data)
        self.user_data = ctypes.addressof(self.sndfile_user_data)

    def __del__(self):
        #pa.Pa_CloseStream(self.stream_ptr)
        sndfile.csndfile.sf_close(c_void_p(self.fin))

def generate_hostapi_info():
    HostApiInfoPointer = POINTER(PaHostApiInfo)
    api_count = pa.Pa_GetHostApiCount()
    for i in xrange(api_count):
        p = ctypes.cast(pa.Pa_GetHostApiInfo(i), HostApiInfoPointer)
        hai = p[0]
        yield hai


def generate_device_info():
    DeviceInfoPointer = POINTER(PaDeviceInfo)
    device_count = pa.Pa_GetDeviceCount()

    ERROR_CHECK(device_count)

    if device_count == 0:
        raise RuntimeError("No devices found")

    for i in xrange(device_count):
        p = ctypes.cast(pa.Pa_GetDeviceInfo(i), DeviceInfoPointer)
        di = p[0]
        yield di


def getAvailableDevices(hostapi=None, verbose=False):
    """
    Returns a list containing information on the available audio devices.

    Parameters
    ----------
    hostapi : string
        Filters the list of devices to include only the specified hostapi.
    verbose : Bool
        Include more information.

    Returns
    -------
    devices : list
        The list of devices.
    """
    # If necessary, wrap `hostapi` in a list so it is iterable
    if isinstance(hostapi, str):
        hostapi = [hostapi]

    if hostapi == None:
        # No constraints
        devices = list(generate_device_info())
    else:
        # Remap user-friendly aliases to integer enum values
        hostapi = [HostApiTypeAliases[x] for x in hostapi]

        # Filter output of generate_device_info()`
        devices = [di for di in generate_device_info() if (di.hostApi in hostapi)]

    if len(devices) == 0:
        return None
    else:
        return devices


def print_available_devices(hostapi=None, verbose=False):
    """
    Displays information on the available audio devices.

    Parameters
    ----------
    hostapi : string
        Filters the list of devices to include only the specified hostapi.
    verbose : Bool
        Print more information.

    Returns
    -------
    None
    """
    devices = getAvailableDevices(hostapi, verbose)

    if len(devices) == 0:
        print("No devices found for given hostApi(s): %s" % ",".join([HostApiTypeAliases[x] for x in hostapi]))
        return None

    if verbose:
        for i,di in enumerate(devices):
            print("index: %s" %  i)
            print(" structVersion: %s" % di.structVersion)
            print(" name: %s" % di.name)
            print(" hostApi: %s" % PaHostApiTypeId.from_int[di.hostApi])
            print(" maxInputChannels: %s" % di.maxInputChannels)
            print(" maxOutputChannels: %s" % di.maxOutputChannels)
            print(" defaultLowInputLatency: %s" % di.defaultLowInputLatency)
            print(" defaultLowOutputLatency: %s" % di.defaultLowOutputLatency)
            print(" defaultHighInputLatency: %s" % di.defaultHighInputLatency)
            print(" defaultHighOutputLatency: %s" % di.defaultHighOutputLatency)
            print(" defaultSampleRate: %s" % di.defaultSampleRate)
            print("")
    else:
        for i,di in enumerate(devices):
            print("index: %s" % i)
            print(" name: %s" % di.name)
            print(" hostApi: %s" % PaHostApiTypeId.from_int[di.hostApi])
            print(" maxInputChannels: %s" % di.maxInputChannels)
            print(" maxOutputChannels: %s" % di.maxOutputChannels)
            print(" defaultSampleRate: %s" % di.defaultSampleRate)
            print("")


def open_device(out_device_index=None, in_device_index=None, out_channels=2):
    """
    Opens the specified input and output devices.
    If no input device is specified, none will be used. If no output device
    is specified, the default device will be used.

    Parameters
    ----------
    out_device_index : int
        Index to the desired output device.
    in_device_index : int
        Index to the desired input device.
    out_channels : int
        The number of output channels to use. PortAudio is not always correct
        in reporting this number, and can sometimes return values like 128.
        This is often not a problem, but because of the way mix_mat works, it
        is important for Medussa to have the correct value. Thus, you have 3
        options (you can always change it later by modifying the property
        dev.out_channels):

         - Don't specify out_channels. Medussa will set it to 2
         - Specify `None`. Medussa will use the PortAudio value
         - Specify a number. Medussa will use that number

    Returns
    -------
    d : Device object
        Object representing the specified devices.
    """
    if out_device_index == None:
        out_device_index = pa.Pa_GetDefaultOutputDevice()

    d = Device(in_device_index, out_device_index, out_channels)
    return d


def open_default_device(out_channels=2):
    """
    Opens the default input and output devices.

    Parameters
    ----------
    out_channels : int
        The number of output channels to use. PortAudio is not always correct
        in reporting this number, and can sometimes return values like 128.
        This is often not a problem, but because of the way mix_mat works, it
        is important for Medussa to have the correct value. Thus, Medussa
        sets out_channels to 2 by default. If your device has more
        channels, set the value here or in dev.out_channels.

    Returns
    -------
    d : Device object
        Object representing the specified devices.

    """
    out_di = pa.Pa_GetDefaultOutputDevice()
    in_di = pa.Pa_GetDefaultInputDevice()

    d = Device(in_di, out_di, out_channels)
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


def play_arr(arr, fs, channel=1):
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


def play_file(file_name):
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
    s = d.open_file(file_name)
    s.play()
    while s.is_playing():
        sleep(.01)


def read_file(file_name):
    """
    Reads a sound file with any libsndfile-compatible format into an ndarray.

    Parameters
    ----------
    filename : str

    Returns
    -------
    (arr, fs) : (ndarray, float)
    """
    finfo = sndfile.SF_INFO(0,0,0,0,0,0)
    fin = sndfile.csndfile.sf_open(file_name, sndfile.SFM_READ, byref(finfo))

    fs = finfo.samplerate

    BUFFTYPE = ctypes.c_double * (finfo.frames * finfo.channels)
    buff = BUFFTYPE()

    frames_read = cmedussa.readfile_helper(fin, byref(buff), finfo.frames)

    err = sndfile.csndfile.sf_close(c_void_p(fin))

    arr = np.ascontiguousarray(np.zeros((finfo.frames, finfo.channels)))

    for i in xrange(finfo.frames):
        for j in xrange(finfo.channels):
            arr[i][j] = buff[i*finfo.channels + j]

    return (arr, float(fs))


def write_file(file_name, arr, fs,
              format=(sndfile.formats.SF_FORMAT_WAV[0] | sndfile.formats.SF_FORMAT_PCM_16[0]),
              frames=None):
    """
    Writes an ndarray to a sound file with any libsndfile-compatible format.

    Parameters
    ----------
    file_name : str
        The name of the file to write to.
    arr : ndarray
        The array of data to write.
    fs : int
        The sampling frequency.
    format : int
        TODO: add description, and some examples
    frames : int
        The number of frames to write.

    Returns
    -------
    frames_written : int
        The number of frames that were written to the file.
    """
    if frames == None:
        frames = arr.shape[0]

    finfo = sndfile.SF_INFO(0,0,0,0,0,0)
    finfo.samplerate = int(fs)
    finfo.channels = arr.shape[1]
    finfo.format = c_int(format)

    arr = np.ascontiguousarray(arr)
    _arr = arr.ctypes.data_as(POINTER(c_double))

    frames_written = cmedussa.writefile_helper(file_name,
                                               byref(finfo),
                                               _arr,
                                               format,
                                               frames)

    return frames_written


def write_wav(file_name, arr, fs, bits='s16', frames=None):
    """
    Convenience function to write a wavefile.

    Parameters
    ----------
    file_name : str
        The name of the file to write to.
    arr : ndarray
        The array of data to write.
    fs : int
        The sampling frequency.
    bits : int
        The bit depth. For wavefiles, libsndfile can handle 8, 16, 24, or 32 bits.
        You can also use a string to specify either signed 16-, 24- or 32-bit
        integers ('s16', 's24', 's32'), or unsigned 8-bit integers ('u8').
    frames : int
        The number of frames to write.

    Returns
    -------
    frames_written : int
        The number of frames that were written to the file.
    """
    majformat = sndfile.formats.SF_FORMAT_WAV[0]

    subformat = {8: sndfile.formats.SF_FORMAT_PCM_U8[0],
                 16: sndfile.formats.SF_FORMAT_PCM_16[0],
                 24: sndfile.formats.SF_FORMAT_PCM_24[0],
                 32: sndfile.formats.SF_FORMAT_PCM_32[0],
                 's16': sndfile.formats.SF_FORMAT_PCM_16[0],
                 's24': sndfile.formats.SF_FORMAT_PCM_24[0],
                 's32': sndfile.formats.SF_FORMAT_PCM_32[0],
                 'u8': sndfile.formats.SF_FORMAT_PCM_U8[0]}

    endformat = majformat | subformat[bits]

    return write_file(file_name, arr, fs, format=endformat, frames=frames)


def write_flac(file_name, arr, fs, bits='s16', frames=None):
    """
    Convenience function to write a FLAC audio file.

    Parameters
    ----------
    file_name : str
        The name of the file to write to.
    arr : ndarray
        The array of data to write.
    fs : int
        The sampling frequency.
    bits : int
        The bit depth. For flac files, libsndfile can handle 8, 16, or 24 bits.
        You can also use a string to specify signed 8- 16-, or 24-bit
        integers ('s8', 's16', 's24').
    frames : int
        The number of frames to write.

    Returns
    -------
    frames_written : int
        The number of frames that were written to the file.
    """
    majformat = sndfile.formats.SF_FORMAT_FLAC[0]

    subformat = {8: sndfile.formats.SF_FORMAT_PCM_S8[0],
                 16: sndfile.formats.SF_FORMAT_PCM_16[0],
                 24: sndfile.formats.SF_FORMAT_PCM_24[0],
                 's8': sndfile.formats.SF_FORMAT_PCM_S8[0],
                 's16': sndfile.formats.SF_FORMAT_PCM_16[0],
                 's24': sndfile.formats.SF_FORMAT_PCM_24[0]}

    endformat = majformat | subformat[bits]

    return write_file(file_name, arr, fs, format=endformat, frames=frames)


def write_ogg(file_name, arr, fs, frames=None):
    """
    Convenience function to write an Ogg Vorbis audio file.

    Parameters
    ----------
    file_name : str
        The name of the file to write to.
    arr : ndarray
        The array of data to write.
    fs : int
        The sampling frequency.
    frames : int
        The number of frames to write.

    Returns
    -------
    frames_written : int
        The number of frames that were written to the file.

    Notes
    -----
    Bit depth is not specified with the Vorbis format, but rather is variable.
  """
    majformat = sndfile.formats.SF_FORMAT_OGG[0]

    subformat = sndfile.formats.SF_FORMAT_VORBIS[0]

    endformat = majformat | subformat

    return write_file(file_name, arr, fs, format=endformat, frames=frames)
