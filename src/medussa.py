# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import atexit
import inspect
import weakref
import platform

pymaj = platform.python_version_tuple()[0]
pymin = platform.python_version_tuple()[1]
pyver = "%s.%s" % (pymaj, pymin)
if pymaj == "2":
    from portaudio import *
    from sndfile import SF_INFO, csndfile, SFM_READ, sf_formats
    from pink import Pink_noise_t
    from rkit import Rk_state
else:
    from .portaudio import *
    from .sndfile import SF_INFO, csndfile, SFM_READ, sf_formats
    from .pink import Pink_noise_t
    from .rkit import Rk_state
    xrange = range


if platform.system() == "Windows":
    libname = os.path.join(get_python_lib(), 'medussa', 'medussa.dll')
    if not os.path.exists(libname):
        raise RuntimeError("Unable to locate library: " + libname)
elif platform.system() == "Linux":
    libname = os.path.join(get_python_lib(), 'medussa', 'libmedussa.so')
    if not os.path.exists(libname):
        raise RuntimeError("Unable to locate library: " + libname)
else:
    libname = find_library("medussa")
    if libname == None:
        raise RuntimeError("Unable to locate library `medussa`")


# Instantiate FFI reference to libmedussa
cmedussa = ctypes.CDLL(libname)

device_instances = lambda: list(Device.instances())
stream_instances = lambda: list(Stream.instances())

@atexit.register
def medussa_exit():
    pa.Pa_Terminate()


###################
## Data Structs

STREAM_COMMAND_SET_MATRICES = c_int(0)
STREAM_COMMAND_FREE_MATRICES = c_int(1)
STREAM_COMMAND_SET_IS_MUTED = c_int(2)

class StreamCommand(ctypes.Structure):
    """
    struct stream_command{
        int command;
        void *data_ptr0;
        void *data_ptr1;
        double data_double;
        unsigned int data_uint;
    };
    """
    _fields_ = (("command",     c_int),
                ("data_ptr0",   c_void_p),
                ("data_ptr1",   c_void_p),
                ("data_double", c_double),
                ("data_uint",   c_uint))
    
        
class StreamUserData(ctypes.Structure):
    """
    struct stream_user_data {
        void *parent;

        PyObject *device;
        
        PaStream *stream;
        PaStreamParameters *in_param;
        PaStreamParameters *out_param;
        double fs;

        stream_command_queues *command_queues;
        
        int is_muted;
        medussa_dmatrix *mix_mat;
        medussa_dmatrix *mute_mat;

        int pa_fpb;
    };
    """
    _fields_ = (("parent",    c_void_p),
                ("device",    py_object),
                ("stream",    c_void_p),
                ("in_param",  c_void_p),
                ("out_param", c_void_p),
                ("fs",        c_double),
                ("command_queues", c_void_p),
                ("is_muted",  c_int),
                ("mix_mat",   c_void_p),
                ("mute_mat",  c_void_p),
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
        char    *file_name;
        SF_INFO *finfo;
    };

    """
    _fields_ = (("parent",  c_void_p),
                ("self",    py_object),
                ("fin",     c_void_p),
                ("file_name", c_char_p),
                ("finfo",   POINTER(SF_INFO)))

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


###################
## Object Classes

class Device(object):
    """
    Medussa object representing an audio device.

    Contains methods to create various streams, and information about the
    hardware device it represents.

    Methods
    -------
    create_pink
        Creates a stream representing pink noise.
    create_white
        Creates a stream representing white noise.
    create_tone
        Creates a stream representing a pure tone.
    open_array
        Creates a stream representing a NumPy array.
    open_file
        Creates a stream representing a sound file on disk.

    Properties
    ----------
    out_channels
        The number of output channels to use. PortAudio is not always correct
        in reporting this number, and can sometimes return spurious values like
        128. In other contexts, this is often not a problem. But because of the
        way mix_mat works, it is important for this value to not be too large.
        Thus, it is set to 2 by default. You can always change it later by
        modifying the property device.out_channels.
    out_name
        The name of the output device, as reported by Port Audio.
    out_hostapi
        The host API of the output device, as reported by Port Audio.
    out_index
        The index of the output device, as reported by Port Audio.

    Notes
    -----
    There is currently no support for recording. We plan to add this feature
    in a future release.

    """

    _instances = set()

    # [Culled from] http://effbot.org/pyfaq/how-do-i-get-a-list-of-all-instances-of-a-given-class.htm
    # [License] http://effbot.org/zone/copyright.htm
    @classmethod
    def instances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead

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

    @property
    def child_streams(self):
        """
        Returns a generator that yields each Stream instance that depends on
        this Device instance.
        """
        return (s for s in Stream.instances() if s.device == self)

    @property
    def out_channels(self):
        return self._out_channels

    @out_channels.setter
    def out_channels(self, val):
        for s in self.child_streams:
            if s.mix_mat.shape[0] < val:
                m = s.mix_mat.shape[0]
                n = s.mix_mat.shape[1]
                x = np.zeros((val,n))
                x[:m,:n] = s.mix_mat
                s.mix_mat = x
            else:
                s.mix_mat = s.mix_mat[:val]
        self._out_channels = val

    def __init__(self, in_index=None, out_index=None, out_channels=None):
        if in_index != None:
            self.in_index = in_index
        if out_index != None:
            self.out_index = out_index
        if out_channels != None:
            self.out_channels = out_channels

        self._instances.add(weakref.ref(self))


    def create_tone(self, tone_freq, fs=None):
        """
        Returns a stream object representing a pure tone.

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
        Returns a stream object representing Gaussian/white noise.

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
        Returns a stream object representing pink noise.

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

    def open_file(self, file_name):
        """
        Returns a stream object representing a soundfile on disk.

        Parameters
        ----------
        file_name : string
            The path to the sound file.

        Returns
        -------
        s : Stream object
            The stream object.

        """
        s = SoundfileStream(self, None, file_name)
        return s


# Given a proposed mix_mat and the number of source and output channels
# return a valid (correctly shaped) mix_mat as follows:
# if mix_mat is None: allocate a correctly shaped matrix of zeros except
# for the diagonal, which is set to 1s.
# if mix_mat is valid, return a copy of mix_mat conformed to the correct 
# shape with any added elements set to zero.
def _util_allocate_or_conform_mix_mat( mix_mat, source_channels, out_channels ):
    
    shape = (out_channels, source_channels)

    if mix_mat == None:
        mix_mat = np.zeros(shape)
        for i in range( 0, min(shape) ):
            mix_mat[i,i] = 1.0
    else:
        if mix_mat.shape != shape:
            mix_mat = np.copy(mix_mat)
            mixmat.resize( shape ) # fills missing entries with zeros

    return mix_mat


class Stream(object):
    """
    Generic stream class.
    """
    _instances = set()

    @classmethod
    def instances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead

    @property # (read-only)
    def fs(self):
        return self._stream_user_data.fs

    @fs.setter
    def fs(self, val):
        raise AttributeError( "can't set attribute (stream.fs is read only)" )

    # mix_mat is a public property. __mix_mat is the underlying attribute storage
    @property
    def mix_mat(self):
        return self.__mix_mat

    @mix_mat.setter
    def mix_mat(self, val):
        if hasattr(self,'__mix_mat'):
            # if we already have a __mix_mat (i.e. any time after construction)
            # then conform the new mix_mat to the correct shape
            val = _util_allocate_or_conform_mix_mat( val, self.__mix_mat.shape[1], self.__mix_mat.shape[0] )      
        
        self.__mix_mat = np.ascontiguousarray(val)

        # allocate new C mix_mat and mute_mat matrices and send to PA callback
        # mix_mat is a copy of __mix_mat's data. mute_mat has the same shape but is zeroed.

        cmd = StreamCommand()
        cmd.command = STREAM_COMMAND_SET_MATRICES
        cmd.data_ptr0 = cmedussa.alloc_medussa_dmatrix( self.mix_mat.shape[0], self.mix_mat.shape[1], self.mix_mat.ctypes.data_as(POINTER(c_double)) )
        cmd.data_ptr1 = cmedussa.alloc_medussa_dmatrix( self.mix_mat.shape[0], self.mix_mat.shape[1], 0 )        
        self._post_command_to_pa_callback( cmd )
                                                        
    @mix_mat.deleter
    def mix_mat(self):
        del self.__mix_mat
    

    # _pa_fpb is queried from C side cmedussa.open_stream. this is brittle. FIXME (note that marking this as __pa_fpb breaks for some reason)
    @property
    def _pa_fpb(self):
        return self._stream_user_data.pa_fpb

    @_pa_fpb.setter
    def _pa_fpb(self, val):
        self._stream_user_data.pa_fpb = val

    def open(self):
        self._stream_ptr = cmedussa.open_stream(py_object(self),
                                               self._stream_user_data.in_param,
                                               self._stream_user_data.out_param,
                                               self._callback_ptr)

    def start(self):
        err = pa.Pa_StartStream(self._stream_ptr)
        ERROR_CHECK(err)
        return err

    def stop(self):
        """
        Stops playback of the stream (Playback cursor is reset to zero).
        """
        if pa.Pa_IsStreamStopped(self._stream_ptr):
            return
        else:
            err = pa.Pa_StopStream(self._stream_ptr)
            ERROR_CHECK(err)

    def play(self):
        """
        Starts playback of the stream.
        """
        if (self._stream_ptr == None):
            self.open()
        if not self.is_playing:
            self.start()

    def pause(self):
        """
        Pauses playback of the stream (Playback cursor is not reset).
        """
        if pa.Pa_IsStreamStopped(self._stream_ptr):
            return
        else:
            err = pa.Pa_StopStream(self._stream_ptr)
            ERROR_CHECK(err)

    @property
    def is_playing(self):
        """
        Boolean indicating whether the stream is currently playing.
        """
        if self._stream_ptr:
            err = pa.Pa_IsStreamActive(self._stream_ptr)
            ERROR_CHECK(err)
            return bool(err)
        else:
            return False;
    
    @is_playing.setter
    def is_playing(self, val):
        if val:
            self.play()
        else:
            self.pause()

    def pa_time(self):
        """
        Returns the current portaudio time, which is in seconds.
        """
        t = pa.Pa_GetStreamTime(self._stream_ptr)
        if t:
            return t
        else:
            raise RuntimeError("Error indicated by `Pa_GetStreamTime()` -> 0")

    @property
    def is_muted(self):
        return self._is_muted

    @is_muted.setter
    def is_muted(self, val):
        self.mute(bool(val))

    def mute(self, val=None): #FIXME I think we should default val to True
        """
        Mutes or unmutes the stream.

        Mix matrix is unaffected. Playback will continue while stream is muted.
        """
        if val is None:
            return self.is_muted
        else:
            if bool(val) != self._is_muted:
                self._is_muted = bool(val)
                                                        
                cmd = StreamCommand()
                cmd.command = STREAM_COMMAND_SET_IS_MUTED
                cmd.data_uint = int(self._is_muted)
                self._post_command_to_pa_callback( cmd )
                
        return self.is_muted

    def unmute(self):
        return self.mute(False)

    def __init__(self):
        self._stream_user_data = StreamUserData()
        self._stream_user_data.command_queues = cmedussa.alloc_stream_command_queues();
        self._stream_user_data.is_muted = 0;
        self._stream_user_data.mix_mat = 0;
        self._stream_user_data.mute_mat = 0;

        self._stream_ptr = None

        self._is_muted = False
        
    def _init2(self, device, fs, callback_ptr, callback_command_exec_ptr, callback_user_data, mix_mat, source_channels):
        
        self._device = device
        self._stream_user_data.fs = fs
        self._callback_ptr = callback_ptr
        self._callback_command_exec_ptr = callback_command_exec_ptr
        self._callback_user_data = ctypes.addressof(callback_user_data)

        if device.out_channels == None:
            out_channels = device.out_device_info.maxOutputChannels
        else:
            out_channels = device.out_channels
    
        self.mix_mat = _util_allocate_or_conform_mix_mat( mix_mat, source_channels, out_channels )

        self._out_param = PaStreamParameters(self._device.out_index,
                                            out_channels,
                                            paFloat32,
                                            self._device.out_device_info.defaultLowOutputLatency,
                                            None)
        self._stream_user_data.out_param = ctypes.addressof(self._out_param)
        
        # Find a smart way to determine this value,
        # which has to be hardcoded into the callback
        self._pa_fpb = 1024

        self._instances.add(weakref.ref(self))

    def _close_stream_and_flush_commands(self):
        # derived classes must call this at the beginning of their __del__()
        # it ensures that the pa stream is closed and all commands have been finalised
        # before any teardown happens.
        
        if self._stream_ptr:
            pa.Pa_CloseStream(self._stream_ptr)
            self._stream_ptr = None

        # Ensure that the callback end of the command queue is empty,
        # then process any results to free data.

        # (the callback isn't running, so we can execute its commands here, to flush the queue)
        cmedussa.execute_commands_in_pa_callback( self._stream_user_data.command_queues, self._callback_command_exec_ptr, self._callback_user_data ); 
        cmedussa.process_results_from_pa_callback( self._stream_user_data.command_queues );
        
    def __del__(self):
        assert( self._stream_ptr == None ) # derived class didn't call _close_stream_and_flush_commands?
        
        cmedussa.free_stream_command_queues( self._stream_user_data.command_queues );

        cmedussa.free_medussa_dmatrix( self._stream_user_data.mix_mat );
        cmedussa.free_medussa_dmatrix( self._stream_user_data.mute_mat );

        self._stream_user_data.out_param = None
        del self._out_param
        
    def _post_command_to_pa_callback( self, cmd ):
        # This function is more elaborate than simply posting the command to the queue
        # because we want to make sure any results from earlier commands get processed.
        # Also, when the callback isn't running, we want to execute the commands immediately
        # to prevent the queue from filling with commands.
                                                        
        # first make sure the queues are as empty as possible
        if self.is_playing:
            cmedussa.process_results_from_pa_callback( self._stream_user_data.command_queues )
        else:
            cmedussa.process_results_from_pa_callback( self._stream_user_data.command_queues )
            # (the callback isn't running, so we can execute its commands here, to flush the queue)
            cmedussa.execute_commands_in_pa_callback( self._stream_user_data.command_queues, self._callback_command_exec_ptr, self._callback_user_data )
            cmedussa.process_results_from_pa_callback( self._stream_user_data.command_queues )
 
        # post our message to the callback
        if cmedussa.post_command_to_pa_callback( self._stream_user_data.command_queues, ctypes.addressof(cmd) ) != 1:

            # if the command queue is full, we wait for it to have space
            if self.is_playing:
                while cmedussa.post_command_to_pa_callback( self._stream_user_data.command_queues, ctypes.addressof(cmd) ) != 1:                           
                    time.sleep(.01) 
                    cmedussa.process_results_from_pa_callback( self._stream_user_data.command_queues )
            else:
                assert( False ) # we should never get here. the queue should not be full if the stream isn't running

        # if the callback isn't running, execute the command immediately
        if not self.is_playing:
            # (the callback isn't running, so we can execute its commands here, to flush the queue)
            cmedussa.execute_commands_in_pa_callback( self._stream_user_data.command_queues, self._callback_command_exec_ptr, self._callback_user_data )
            cmedussa.process_results_from_pa_callback( self._stream_user_data.command_queues )


class ToneStream(Stream):
    """
    Medussa stream object representing a pure tone.

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
        to the number of source channels (in the case of a tonestream, 1),
        and the number of rows corresponds to the number of device output
        channels, which is accessible with dev.out_channels. The values of
        the mix_mat are floats between 0. and 1., and are used to specify the
        playback level of each source channel on each output channel. A default
        mix_mat will have ones along the diagonal, and zeros everywhere else
        (source channel 1 to output device channel 1, source 2 to ouput 2,
        etc). A mix_mat of all ones would route all source channels to all
        device output channels.

    """
    _instances = set()

    @classmethod
    def instances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead

    @property
    def tone_freq(self):
        return self._tone_user_data.tone_freq

    @tone_freq.setter
    def tone_freq(self, val):
        self._tone_user_data.tone_freq = val

    def __init__(self, device, fs, mix_mat, tone_freq):
        super(ToneStream, self).__init__()

        self._tone_user_data = ToneUserData()
        self._tone_user_data.parent = ctypes.addressof(self._stream_user_data)
        self._tone_user_data.t = 0

        super(ToneStream, self)._init2( device, fs, cmedussa.callback_tone, \
                                        cmedussa.execute_tone_user_data_command, \
                                        self._tone_user_data, mix_mat, 1 )

        self.tone_freq = tone_freq
        
        self._instances.add(weakref.ref(self))

    def __del__(self):
        super(ToneStream, self)._close_stream_and_flush_commands()
        super(ToneStream, self).__del__()


class WhiteStream(Stream):
    """
    Medussa stream object representing Gaussian white noise, which has a flat spectrum.

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
        to the number of source channels (in the case of a tonestream, 1),
        and the number of rows corresponds to the number of device output
        channels, which is accessible with dev.out_channels. The values of
        the mix_mat are floats between 0. and 1., and are used to specify the
        playback level of each source channel on each output channel. A default
        mix_mat will have ones along the diagonal, and zeros everywhere else
        (source channel 1 to output device channel 1, source 2 to ouput 2,
        etc). A mix_mat of all ones would route all source channels to all
        device output channels.

    """
    _instances = set()

    @classmethod
    def instances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead

    def __init__(self, device, fs, mix_mat):
        super(WhiteStream, self).__init__()
        
        self._white_user_data = WhiteUserData()
        self._white_user_data.parent = ctypes.addressof(self._stream_user_data)
        
        self._rk_state = Rk_state()
        cmedussa.rk_randomseed(byref(self._rk_state))
        self._white_user_data.rks = ctypes.addressof(self._rk_state)
            
        super(WhiteStream, self)._init2( device, fs, cmedussa.callback_white, \
                                         cmedussa.execute_white_user_data_command, \
                                         self._white_user_data, mix_mat, 1 )

        self._instances.add(weakref.ref(self))

    def __del__(self):
        super(WhiteStream, self)._close_stream_and_flush_commands()
        super(WhiteStream, self).__del__()

        
class PinkStream(Stream):
    """
    Medussa stream object representing pink noise, which has equal energy per octave.

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
        to the number of source channels (in the case of a tonestream, 1),
        and the number of rows corresponds to the number of device output
        channels, which is accessible with dev.out_channels. The values of
        the mix_mat are floats between 0. and 1., and are used to specify the
        playback level of each source channel on each output channel. A default
        mix_mat will have ones along the diagonal, and zeros everywhere else
        (source channel 1 to output device channel 1, source 2 to ouput 2,
        etc). A mix_mat of all ones would route all source channels to all
        device output channels.

    """
    _instances = set()

    @classmethod
    def instances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead

    def __init__(self, device, fs, mix_mat):
        super(PinkStream, self).__init__()
        
        self._pink_user_data = PinkUserData()
        self._pink_user_data.parent = ctypes.addressof(self._stream_user_data)
        
        self._pn = Pink_noise_t()
        self._pink_user_data.pn = ctypes.addressof(self._pn)
        cmedussa.initialize_pink_noise(self._pink_user_data.pn, 24)

        super(PinkStream, self)._init2( device, fs, cmedussa.callback_pink, \
                                        cmedussa.execute_pink_user_data_command, \
                                        self._pink_user_data, mix_mat, 1 )

        self._instances.add(weakref.ref(self))

    def __del__(self):
        super(PinkStream, self)._close_stream_and_flush_commands()
        super(PinkStream, self).__del__()


class FiniteStream(Stream):
    """
    Generic stream class used to derive SoundfileStream and ArrayStream classes.
    """
    _instances = set()

    @classmethod
    def instances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead

    @property
    def is_looping(self):
        return bool(self._finite_user_data.loop)

    @is_looping.setter
    def is_looping(self, val):
        self._finite_user_data.loop = val

    def loop(self, state=None):
        if state is not None:
            self.is_looping = state
        return self.is_looping

    @property
    def cursor(self):
        return self._finite_user_data.cursor

    @cursor.setter
    def cursor(self, val):
        self._finite_user_data.cursor = val

    @property
    def frames(self):
        return self._finite_user_data.frames

    @frames.setter
    def frames(self, val):
        raise AttributeError( "can't set attribute (stream.frames is read only)" )

    @property
    def duration(self):
        return self._finite_user_data.duration

    @duration.setter
    def duration(self, val):
        raise AttributeError( "can't set attribute (stream.duration is read only)" )

    def stop(self):
        """
        Stops playback of the stream (Playback cursor is reset to zero).
        """
        try:
            super(FiniteStream, self).stop()
        except:
            self.cursor = 0
            raise

        self.cursor = 0
        
    def play(self):
        """
        Starts playback of the stream.
        """
        if (self._stream_ptr == None):
            self.open()
        if not self.is_playing:
            if self.cursor == 0 and not pa.Pa_IsStreamStopped(self._stream_ptr):
                pa.Pa_StopStream(self._stream_ptr)
                self.start()
            else:
                self.start()
        
    def time(self, pos=None, units="ms"):
        """
        Gets or sets the current playback cursor position.

        Parameters
        ----------
        pos : numeric
            The cursor cursor position. If `pos` is `None`, the function
            returns the current position. Otherwise, the current cursor
            position will be set to `pos`.
        units : string
            The units of pos. May be of value:
            "ms": assume `pos` is of type `float` [default]
            "sec": `assume `pos` is of type float`
            "frames": assume `pos` is of type `int`

        Returns
        -------
        pos : numeric
            The current position of the cursor. This value is returned only
            if no input `pos` is specified.

        """
        if pos == None:
            if units == "ms":
                return self.cursor / self.fs * 1000.0
            elif units == "sec":
                return self.cursor / self.fs
            elif units == "frames":
                return self.cursor
        elif units == "ms":
            newcursor = int(pos / 1000.0 * self.fs)
            if not (newcursor < self.frames):
                raise RuntimeError("New cursor position %d exceeds signal frame count %d." % (newcursor, self.frames))
            self.cursor = int(pos / 1000.0 * self.fs)
        elif units == "sec":
            newcursor = int(pos * self.fs)
            if not (newcursor < self.frames):
                raise RuntimeError("New cursor position %d exceeds signal frame count %d." % (newcursor, self.frames))
            self.cursor = int(pos * self.fs)
        elif units == "frames":
            assert isinstance(pos, int)
            if not (pos < self.frames):
                raise RuntimeError("New cursor position %d exceeds signal frame count %d." % (newcursor, self.frames))
            self.cursor = pos
        else:
            raise RuntimeError("Bad argument to `units`")

    def __init__(self):
        super(FiniteStream, self).__init__()
        self._finite_user_data = FiniteUserData()
        self._finite_user_data.parent = ctypes.addressof(self._stream_user_data)

        self._finite_user_data.cursor = 0
        self._finite_user_data.loop = 0
        
    def _init2(self, device, fs, callback_ptr, callback_command_exec_ptr, callback_user_data, mix_mat, source_channels, frames ):
        self._finite_user_data.frames = frames
        self._finite_user_data.duration = frames / float(fs)

        super(FiniteStream, self)._init2( device, fs, callback_ptr, \
                                          callback_command_exec_ptr, \
                                          callback_user_data, mix_mat, \
                                          source_channels )
        
        self._instances.add(weakref.ref(self))

    def __del__(self):
        super(FiniteStream, self).__del__()


class ArrayStream(FiniteStream):
    """
    Medussa stream object representing a NumPy array.

    You can use medussa.read_file to load soundfiles into NumPy arrays.

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
    time
        Gets or sets the current playback cursor position.

    Properties
    ----------
    fs
        Sampling frequency, in Hz.
    mix_mat
        A NumPy array that acts as a mixer. The number of columns corresponds
        to the number of source channels (in the case of a tonestream, 1),
        and the number of rows corresponds to the number of device output
        channels, which is accessible with dev.out_channels. The values of
        the mix_mat are floats between 0. and 1., and are used to specify the
        playback level of each source channel on each output channel. A default
        mix_mat will have ones along the diagonal, and zeros everywhere else
        (source channel 1 to output device channel 1, source 2 to ouput 2,
        etc). A mix_mat of all ones would route all source channels to all
        device output channels.
    is_looping
        Gets or sets whether the stream will continue playing from the
        beginning when it reaches the end.
    frames
        The number of frames (samples per source channel). (read-only)
    duration
        The duration in seconds. (read-only)
    arr
        The array of audio data. (read-only)

    """
    _instances = set()

    @classmethod
    def instances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead

    @property
    def arr(self):
        return self.__arr

    @arr.setter
    def arr(self, val):
        raise AttributeError( "can't set attribute (stream.arr is read only)" )
        
    def __set_arr(self, val):
        if not val.dtype == np.dtype('double'):
            raise TypeError('Array must have `double` dtype.')
        self.__arr = np.ascontiguousarray(val)
        self._array_user_data.ndarr = self.__arr.ctypes.data_as(POINTER(c_double))
        self._array_user_data.ndarr_0 = val.shape[0]
        self._array_user_data.ndarr_1 = val.shape[1]

    def __init__(self, device, fs, mix_mat, arr, is_looping=False):
        super(ArrayStream, self).__init__()

        self._array_user_data = ArrayUserData()
        self._array_user_data.parent = ctypes.addressof(self._finite_user_data)
        
        if len(arr.shape) == 1:
            arr = arr.reshape(arr.size, 1)

        self.__set_arr( arr )

        channels = arr.shape[1] 
        frames = arr.shape[0]  
        super(ArrayStream, self)._init2( device, fs, cmedussa.callback_ndarray, \
                                         cmedussa.execute_array_user_data_command, \
                                         self._array_user_data, mix_mat, channels, frames )

        # Initialize `FiniteStream` attributes
        self.is_looping = is_looping
        
        self._instances.add(weakref.ref(self))

    def __del__(self):
        super(ArrayStream, self)._close_stream_and_flush_commands()
        super(ArrayStream, self).__del__()
        
        
class SoundfileStream(FiniteStream):
    """
    Medussa stream object representing a sound file on disk.

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
    time
        Gets or sets the current playback cursor position.

    Properties
    ----------
    fs
        Sampling frequency, in Hz.
    mix_mat
        A NumPy array that acts as a mixer. The number of columns corresponds
        to the number of source channels (in the case of a tonestream, 1),
        and the number of rows corresponds to the number of device output
        channels, which is accessible with dev.out_channels. The values of
        the mix_mat are floats between 0. and 1., and are used to specify the
        playback level of each source channel on each output channel. A default
        mix_mat will have ones along the diagonal, and zeros everywhere else
        (source channel 1 to output device channel 1, source 2 to ouput 2,
        etc). A mix_mat of all ones would route all source channels to all
        device output channels.
    is_looping
        Gets or sets whether the stream will continue playing from the
        beginning when it reaches the end.
    frames
        The number of frames (samples per source channel). (read-only)
    duration
        The duration in seconds. (read-only)
    file_name
        The path to the sound file. (read-only)

    Notes
    -----
    The audio data are not loaded into memory, but rather are streamed from
    disk during playback.

    """
    _instances = set()

    @classmethod
    def instances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead


    @property
    def file_name(self):
        if pymaj == '3':
            return self.__file_name.decode('utf-8')
        else:
            return self.__file_name

    @file_name.setter
    def file_name(self, val):
        raise AttributeError( "can't set attribute (stream.file_name is read only)" )

    def __set_file_name(self, val): 
        if pymaj == '3':
            self.__file_name = bytes(val, 'utf-8')
            self._sndfile_user_data.file_name = c_char_p(self.__file_name)
        else:
            self.__file_name = val
            self._sndfile_user_data.file_name = c_char_p(self.__file_name)

    def __init__(self, device, mix_mat, file_name, is_looping=False):
        super(SoundfileStream, self).__init__()
        
        self._sndfile_user_data = SndfileUserData()
        self._sndfile_user_data.parent = ctypes.addressof(self._finite_user_data)
        
        # Initialize this class' attributes

        if not os.path.isfile(file_name):
            raise IOError('File not found: %s' % file_name)
    
        self.__set_file_name( file_name )
        
        self._finfo = SF_INFO(0,0,0,0,0,0)
        self._sndfile_user_data.finfo = ctypes.cast(ctypes.pointer(self._finfo),
                                                       POINTER(SF_INFO))
        # self.sndfile_user_data.finfo = ctypes.addressof(self._finfo)

        if pymaj == '3':
            self._fin = csndfile.sf_open(bytes(file_name, 'utf-8'),
                                        SFM_READ,
                                        byref(self._finfo))
        else:
            self._fin = csndfile.sf_open(file_name,
                                        SFM_READ,
                                        byref(self._finfo))
        self._sndfile_user_data.fin = self._fin

        fs = self._finfo.samplerate
        frames = self._finfo.frames
        super(SoundfileStream, self)._init2( device, fs, cmedussa.callback_sndfile_read, \
                                             cmedussa.execute_sndfile_read_user_data_command, \
                                             self._sndfile_user_data, mix_mat, self._finfo.channels, frames )

        # Initialize `FiniteStream` attributes
        self.is_looping = is_looping

        self._instances.add(weakref.ref(self))

    def __del__(self):
        super(SoundfileStream, self)._close_stream_and_flush_commands()
        csndfile.sf_close(c_void_p(self._fin))
        super(SoundfileStream, self).__del__()
        

###################
## General Methods

def get_default_output_device_index():
    """
    Returns the index to the system default audio output device.

    Parameters
    ----------
    None.

    Returns
    -------
    device_ind : int
        The index to the default output device.

    """
    devices = [(i,x) for (i,x) in enumerate(get_available_devices()) if x.name == 'default']
    if devices == []:
        return pa.Pa_GetDefaultOutputDevice()
    else:
        i,d = devices[0]
        if d.maxOutputChannels > 0:
            return i
        else:
            return pa.Pa_GetDefaultOutputDevice()


def get_default_input_device_index():
    """
    Returns the index to the system default audio input device.

    Parameters
    ----------
    None.

    Returns
    -------
    device_ind : int
        The index to the default input device.

	Notes
	-----
	Input (recording) has not bee implemented yet.

	"""
    devices = [(i,x) for (i,x) in enumerate(get_available_devices()) if x.name == 'default']
    if devices == []:
        return pa.Pa_GetDefaultInputDevice()
    else:
        i,d = devices[0]
        if d.maxInputChannels > 0:
            return i
        else:
            return pa.Pa_GetDefaultInputDevice()


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


def get_available_devices(hostapi=None, verbose=False):
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
    devices = get_available_devices(hostapi, verbose)

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

    If no output device is specified, the default device will be used. If no
    input device is specified, none will be used.


    Parameters
    ----------
    out_device_index : int
        Index to the desired output device.
    in_device_index : int
        Index to the desired input device.
    out_channels : int
        The number of output channels to use. PortAudio is not always correct
        in reporting this number, and can sometimes return spurious values like
        128. In other contexts, this is often not a problem. But because of the
        way mix_mat works, it is important for this value to not be too large.
        Thus, you have 3 options (you can always change it later by modifying
        the property dev.out_channels):

         - Don't specify out_channels. Medussa will set it to 2
         - Specify `None`. Medussa will use the PortAudio value
         - Specify a number. Medussa will use that number

    Returns
    -------
    d : Device object
        Object representing the specified devices.

	Notes
	-----
	Input (recording) has not bee implemented yet.

	"""
    if out_device_index == None:
        out_device_index = get_default_output_device_index()

    d = Device(in_device_index, out_device_index, out_channels)
    return d


def open_default_device(out_channels=2):
    """
    Opens the default input and output devices.

    Parameters
    ----------
    out_channels : int
        The number of output channels to use. PortAudio is not always correct
        in reporting this number, and can sometimes return spurious values like
        128. In other contexts, this is often not a problem. But because of the
        way mix_mat works, it is important for this value to not be too large.
        Thus, you have 3 options (you can always change it later by modifying
        the property dev.out_channels):

         - Don't specify out_channels. Medussa will set it to 2
         - Specify `None`. Medussa will use the PortAudio value
         - Specify a number. Medussa will use that number

    Returns
    -------
    d : Device object
        Object representing the specified devices.

    Notes
	-----
	Input (recording) has not bee implemented yet.

	"""
    out_di = get_default_output_device_index()
    in_di = get_default_input_device_index()

    d = Device(in_di, out_di, out_channels)
    return d


def start_streams(*args):
    """
    Tries to start playback of specified streams as synchronously as possible.

    Parameters
    ----------
    streams : tuple
        Tuple of stream objects.

    Returns
    -------
    None

    """
    [s.open() for s in args]

    num_streams = len(args)
    STREAM_P_ARRAY_TYPE = c_void_p * num_streams  # custom-length type
    stream_p_array = STREAM_P_ARRAY_TYPE(*[s._stream_ptr for s in args])
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


def play_array(arr, fs, output_device_id=None, volume=1.):
    """
    Plays a NumPy array with blocking, Matlab-style (synchronous playback).

    Parameters
    ----------
    arr : ndarray
        The array to play. Each column is treated as a channel.
    fs : int
        The sampling frequency.
    output_device_id : int
        The id of the output device to play from. [Ommit for system default]
    volume : scalar
        Volume during playback. 0. <= 1. [Default = 1]

    Returns
    -------
    None

    """
    d = open_device(output_device_id)
    s = d.open_array(arr, fs)
    s.mix_mat *= float(volume)
    s.play()
    while s.is_playing:
        time.sleep(.01)


def play_file(file_name, output_device_id=None, volume=1., duration=0, max_duration=10):
    """
    Plays a soundfile with blocking (synchronous playback).

    Parameters
    ----------
    file_name : str
        The path to the file to play.
    output_device_id : int
        The id of the output device to play from. [Ommit for system default]
    volume : scalar
        Volume during playback. 0. <= 1. [Default = 1]
    duration : scalar
        The amount of the file to play, in seconds. Useful if you want to
        play the first few seconds of a file. `max_duration` is ignored if
        this property is set. [Default is the entire file]
    max_duration : scalar
        The maximum file duration to play, in seconds. Because this function
        is blocking, this is a sort of sanity check in case you accidentally
        pass a file that is exceedingly long. The check is not performed if
		either this is set to 0, or if `duration` is set. [Default = 10 sec]

    Returns
    -------
    None

    Notes
    -----
    Use with care! Long soundfiles will cause the interpreter to lock for a
    correspondingly long time!

    """
    d = open_device(output_device_id)
    s = d.open_file(file_name)
    if max_duration > 0 and s.duration > max_duration:
        raise RuntimeError("The duration of the soundfile is longer than max_duration.\nTo play this file, set max_duration > %0.2f (or use 0 to bypass this check)." % (s.duration))
    if duration is None:
        duration = s.duration
    else:
        duration = duration
        max_duration = 0
    s.mix_mat *= float(volume)
    s.play()
    while s.is_playing:
        pass
        #if s.time() > duration:
        #    s.stop()


def read_file(file_name):
    """
    Reads a sound file with any libsndfile-compatible format into an ndarray.

    Parameters
    ----------
    file_name : str
        The path to the sound file. Can be relative or absolute.

    Returns
    -------
    (arr, fs) : (ndarray, float)
        A 2-element tuple containing the audio data as a NumPy array, and
        the sample rate as a float.

    Notes
    -----
    The file IO functions in Medussa are intended to be extremely light
    wrappers to libsndfile, and not a full python implementation of its API.
    For that, you want scikits.audiolab.

    """
    if not os.path.isfile(file_name):
        raise IOError('File not found: %s' % file_name)

    finfo = SF_INFO(0,0,0,0,0,0)
    if pymaj == '3':
        fin = csndfile.sf_open(bytes(file_name, 'utf-8'), SFM_READ, byref(finfo))
    else:
        fin = csndfile.sf_open(file_name, SFM_READ, byref(finfo))

    fs = finfo.samplerate

    BUFFTYPE = ctypes.c_double * (finfo.frames * finfo.channels)
    buff = BUFFTYPE()

    frames_read = cmedussa.readfile_helper(fin, byref(buff), finfo.frames)

    err = csndfile.sf_close(c_void_p(fin))

    arr = np.ascontiguousarray(np.zeros((finfo.frames, finfo.channels)))

    for i in xrange(finfo.frames):
        for j in xrange(finfo.channels):
            arr[i][j] = buff[i*finfo.channels + j]

    return (arr, float(fs))


def write_file(file_name, arr, fs,
              fmt=(sf_formats.SF_CONTAINER_WAV | sf_formats.SF_ENCODING_PCM_16),
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
    fmt : int
        A bitwise-or combination of an SF_CONTAINER format and an SF_ENCODING format.
        See http://www.mega-nerd.com/libsndfile/ for a relatively complete list of
        which encoding formats can be used with which container formats.
        Here are a few examples:
            # a wav file with 16-bit signed integers (standard wav format):
            fmt = sf_formats.SF_CONTAINER_WAV | sf_formats.SF_ENCODING_PCM_16
            # a flac file with 24-bit integers
            fmt = sf_formats.SF_CONTAINER_FLAC | sf_formats.SF_ENCODING_PCM_24
    frames : int
        The number of frames to write.

    Returns
    -------
    frames_written : int
        The number of frames that were written to the file.

    Notes
    -----
    Existing files will be over-written!

    The file IO functions in Medussa are intended to be extremely light
    wrappers to libsndfile, and not a full python implementation of its API.
    For that, you want http://pypi.python.org/pypi/scikits.audiolab/

    """
    if not arr.dtype == np.dtype('double'):
        raise TypeError('Array must have `double` dtype.')

    if frames == None:
        frames = arr.shape[0]

    finfo = SF_INFO(0,0,0,0,0,0)
    finfo.samplerate = int(fs)
    if arr.ndim == 1:
        finfo.channels = 1
    elif arr.ndim == 2:
        finfo.channels = arr.shape[1]
    else:
        raise TypeError('Array dimension must == 1 or 2')
    finfo.format = c_int(format)

    arr = np.ascontiguousarray(arr)
    _arr = arr.ctypes.data_as(POINTER(c_double))

    if pymaj == '3':
        frames_written = cmedussa.writefile_helper(bytes(file_name, 'utf-8'),
                                                   byref(finfo),
                                                   _arr,
                                                   format,
                                                   frames)
    else:
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

    Notes
    -----
    Existing files will be over-written!

    The file IO functions in Medussa are intended to be extremely light
    wrappers to libsndfile, and not a full python implementation of its API.
    For that, you want scikits.audiolab.

    """
    majformat = sf_formats.SF_CONTAINER_WAV

    subformat = {8: sf_formats.SF_ENCODING_PCM_U8,
                 16: sf_formats.SF_ENCODING_PCM_16,
                 24: sf_formats.SF_ENCODING_PCM_24,
                 32: sf_formats.SF_ENCODING_PCM_32,
                 's16': sf_formats.SF_ENCODING_PCM_16,
                 's24': sf_formats.SF_ENCODING_PCM_24,
                 's32': sf_formats.SF_ENCODING_PCM_32,
                 'u8': sf_formats.SF_ENCODING_PCM_U8}

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

    Notes
    -----
    Existing files will be over-written!

    The file IO functions in Medussa are intended to be extremely light
    wrappers to libsndfile, and not a full python implementation of its API.
    For that, you want scikits.audiolab.

    """
    majformat = sf_formats.SF_CONTAINER_FLAC

    subformat = {8: sf_formats.SF_ENCODING_PCM_S8,
                 16: sf_formats.SF_ENCODING_PCM_16,
                 24: sf_formats.SF_ENCODING_PCM_24,
                 's8': sf_formats.SF_ENCODING_PCM_S8,
                 's16': sf_formats.SF_ENCODING_PCM_16,
                 's24': sf_formats.SF_ENCODING_PCM_24}

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
    Existing files will be over-written!

    Bit depth is not specified with the Vorbis format, but rather is variable.

    The file IO functions in Medussa are intended to be extremely light
    wrappers to libsndfile, and not a full python implementation of its API.
    For that, you want scikits.audiolab.

    """
    majformat = sf_formats.SF_CONTAINER_OGG

    subformat = sf_formats.SF_ENCODING_VORBIS

    endformat = majformat | subformat

    return write_file(file_name, arr, fs, format=endformat, frames=frames)
