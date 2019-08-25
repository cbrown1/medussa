# -*- coding: utf-8 -*-

# Copyright (c) 2010-2019 Christopher Brown
#
# This file is part of Medussa.
#
# Medussa is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Medussa is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Medussa.  If not, see <http://www.gnu.org/licenses/>.
#
# Comments and/or additions are welcome. Send e-mail to: cbrown1@pitt.edu.
#

import atexit
import datetime
import numpy as np
import os
import platform
import time
import weakref
import ctypes

from .portaudio import *
from .sndfile import SF_INFO, SF_INFO_p, csndfile, SFM_READ, sf_formats, SNDFILE, SNDFILE_p
from .pink import Pink_noise_t
from .rkit import Rk_state
from ctypes import c_void_p, byref, cast

#Some portaudio calls may return "paNoDevice" on error (e.g., no default device
#is available). In port audio, this value is #define'd to ((PaDeviceIndex)-1).
PA_NODEVICE = -1

pymaj = platform.python_version_tuple()[0]
if pymaj == "3":
    xrange = range


def __abi_suffix():
    if "2" == pymaj:
        return ".pyd" if platform.system() == "Windows" else ".so"
    import sysconfig
    return sysconfig.get_config_var('EXT_SUFFIX')

libname_base = 'libmedussa{}'.format(__abi_suffix())
# distutils behavior across platforms is fairly portable, no need to overcomplicate this
libname = os.path.join(os.path.dirname(os.path.abspath(__file__)), libname_base)
if not os.path.exists(libname):
    raise RuntimeError("Unable to locate library: " + libname)


def _to_cstr(s):
    return s if pymaj == "2" else bytes(s, "utf-8")


def _from_cstr(b):
    return b if pymaj == "2" else b.decode("utf-8")

# Instantiate FFI reference to libmedussa
cmedussa = ctypes.CDLL(libname)

device_instances = lambda: list(Device.instances())
stream_instances = lambda: list(Stream.instances())

@atexit.register
def medussa_exit():
    pa.Pa_Terminate()

c_double_p = POINTER(c_double)


class medussa_dmatrix (ctypes.Structure):
    pass
medussa_dmatrix_p = POINTER(medussa_dmatrix)


class medussa_stream_command_queues (ctypes.Structure):
    pass
medussa_stream_command_queues_p = POINTER(medussa_stream_command_queues)


class FileStream (ctypes.Structure):
    pass
FileStream_p = POINTER(FileStream)

###################
## Data Structs

STREAM_COMMAND_SET_MATRICES = c_int(0)
STREAM_COMMAND_FREE_MATRICES = c_int(1)
STREAM_COMMAND_SET_IS_MUTED = c_int(2)
FINITE_STREAM_COMMAND_SET_CURSOR = c_int(3)

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
StreamCommandPointer = POINTER(StreamCommand)


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
        medussa_dmatrix *fade_inc_mat;
        medussa_dmatrix *target_mix_mat;
        int mix_mat_fade_countdown_frames;
        
        int pa_fpb;
    };
    """
    _fields_ = (("parent",    c_void_p),
                ("device",    py_object),
                ("stream",    StreamPointer),
                ("in_param",  StreamParametersPointer),
                ("out_param", StreamParametersPointer),
                ("fs",        c_double),
                ("command_queues", medussa_stream_command_queues_p),
                ("is_muted",  c_int),
                ("mix_mat",   medussa_dmatrix_p),
                ("mute_mat",  medussa_dmatrix_p),
                ("fade_inc_mat",  medussa_dmatrix_p),
                ("target_mix_mat",  medussa_dmatrix_p),
                ("mix_mat_fade_countdown_frames",  c_int),
                ("pa_fpb",    c_int))
StreamUserDataPointer = POINTER(StreamUserData)


class FiniteUserData(ctypes.Structure):
    """
    struct finite_user_data {
        void *parent;

        unsigned int loop;
        unsigned int cursor;
        int frames;
        double duration;
        medussa_dmatrix *temp_mat;
    };
    """
    _fields_ = (("parent",   c_void_p),
                ("loop",     c_int),
                ("cursor",   c_uint),
                ("frames",   c_uint),
                ("duration", c_double),
                ("temp_mat", medussa_dmatrix_p))


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
                ("ndarr",  c_double_p),
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
                ("fin",     SNDFILE_p),
                ("file_name", c_char_p),
                ("finfo",   SF_INFO_p),
                ("file_stream", FileStream_p))

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

_cm = cmedussa
_cm.readfile_helper.argtypes = [SNDFILE_p, c_double_p, c_int]
_cm.writefile_helper.argtypes = [c_char_p, SF_INFO_p, c_double_p, c_int, c_int]
_cm.alloc_medussa_dmatrix.restype = medussa_dmatrix_p
_cm.free_medussa_dmatrix.argtypes = [medussa_dmatrix_p]
_cm.alloc_stream_command_queues.restype = medussa_stream_command_queues_p
_cm.free_stream_command_queues.argtypes = [medussa_stream_command_queues_p]
_cm.post_command_to_pa_callback.argtypes = [medussa_stream_command_queues_p, StreamCommandPointer]
_cm.process_results_from_pa_callback.argtypes = [medussa_stream_command_queues_p]
_cm.execute_commands_in_pa_callback.argtypes = [medussa_stream_command_queues_p, c_void_p, c_void_p]
_cm.open_stream.restype = StreamPointer
_cm.open_stream.argtypes = [py_object, StreamParametersPointer, StreamParametersPointer, c_void_p]
_cm.start_streams.argtypes = [POINTER(StreamPointer), c_int]
_cm.allocate_file_stream.restype = FileStream_p
_cm.allocate_file_stream.argtypes = [SNDFILE_p, SF_INFO_p, c_int, c_int]
_cm.free_file_stream.argtypes = [FileStream_p]
_cm.initialize_pink_noise.argtypes = [c_void_p, c_int]
_cm.rk_randomseed.argtypes = [POINTER(Rk_state)]

# The following don't need prototypes, they are passed as pointer values only (not called)
# _cm.callback_tone
# _cm.execute_tone_user_data_command
# _cm.callback_white
# _cm.execute_white_user_data_command
# _cm.callback_pink
# _cm.execute_pink_user_data_command
# _cm.callback_ndarray
# _cm.execute_array_user_data_command
# _cm.callback_sndfile_read
# _cm.execute_sndfile_read_user_data_command

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
        ptr = pa.Pa_GetDeviceInfo(val) # get pointer to DeviceInfo
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
        ptr = pa.Pa_GetDeviceInfo(value) # get pointer to DeviceInfo

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
        self._out_channels = val

    def __init__(self, in_index=None, out_index=None, out_channels=None):
        if in_index != None:
            if (in_index < 0):
                raise ValueError("input device must be >= 0: found {ind}".format(ind=in_index))

            self.in_index = in_index

        if out_index != None:
            if (out_index < 0):
                raise ValueError("output device must be >= 0: found {ind}".format(ind=out_index))

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
def _util_allocate_or_conform_mix_mat( mix_mat, out_channels, source_channels ):
    
    shape = (out_channels, source_channels)

    if isinstance(mix_mat,np.ndarray):
        if mix_mat.shape != shape:
            mix_mat = np.copy(mix_mat)
            mix_mat.resize( shape ) # fills missing entries with zeros
    else:
        mix_mat = np.zeros(shape)
        for i in range( 0, min(shape) ):
            mix_mat[i,i] = 1.0

#    if mix_mat == None:
#        mix_mat = np.zeros(shape)
#        for i in range( 0, min(shape) ):
#            mix_mat[i,i] = 1.0
#    else:
#        if mix_mat.shape != shape:
#            mix_mat = np.copy(mix_mat)
#            mix_mat.resize( shape ) # fills missing entries with zeros

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

    # fade duration in seconds
    @property
    def mix_mat_fade_duration(self):
        return self.__mix_mat_fade_duration

    @mix_mat_fade_duration.setter
    def mix_mat_fade_duration(self, val):
        self.__mix_mat_fade_duration = val

    # mix_mat is a public property. __mix_mat is the underlying attribute storage
    @property
    def mix_mat(self):
        return self.__mix_mat

    @mix_mat.setter
    def mix_mat(self, val):
        self.fade_mix_mat_to( val, self.__mix_mat_fade_duration )
       
    @mix_mat.deleter
    def mix_mat(self):
        del self.__mix_mat

    # usually setting mix_mat fades over s.mix_mat_fade_duration seconds
    # you can call fade_mix_mat_to to explicitly specify a fade time in seconds
    # a fade_duration of 0 disables fading.
    # note that irrespective of the fade duration, s.mix_mat reflects
    # the target value immediately.
    def fade_mix_mat_to(self, val, fade_duration ):

        self.fade_mix_mat_start_time = datetime.datetime.now()
        self.fade_mix_mat_fade_duration = fade_duration

        if hasattr(self,'_Stream__mix_mat'): # must use mangled name here http://bugs.python.org/issue8264
            # if we already have a __mix_mat (i.e. any time after construction)
            # then conform the new mix_mat to the correct shape
            val = _util_allocate_or_conform_mix_mat( val, self.__mix_mat.shape[0], self.__mix_mat.shape[1] )
        
        self.__mix_mat = np.ascontiguousarray(val)

        # allocate new C mix_mat and mute_mat matrices and send to PA callback
        # mix_mat is a copy of __mix_mat's data.
        
        cmd = StreamCommand()
        cmd.command = STREAM_COMMAND_SET_MATRICES
        cmd.data_ptr0 = ctypes.cast(
                cmedussa.alloc_medussa_dmatrix( self.mix_mat.shape[0], self.mix_mat.shape[1], self.mix_mat.ctypes.data_as(c_double_p)),
                c_void_p)
        cmd.data_ptr1 = 0
        cmd.data_uint = int(fade_duration * self.fs)
        self._post_command_to_pa_callback( cmd )
                                                        
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
        if not self._stream_ptr:
            raise RuntimeError("Failed to open stream.")
        # Save address of stream as python int to ease recovering the pointer value on the C side
        self._stream_ptr_addr = ctypes.addressof(self._stream_ptr[0])
    
    def start(self):
        """
        Starts playback of the stream.
        """
        if not pa.Pa_IsStreamStopped(self._stream_ptr): # needed since some callbacks call paComplete
            pa.Pa_StopStream(self._stream_ptr)          # so streams can be inactive but not stopped
            
        err = pa.Pa_StartStream(self._stream_ptr)
        PA_ERROR_CHECK(err)
        return err

    def stop(self):
        """
        Stops playback of the stream.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        if pa.Pa_IsStreamStopped(self._stream_ptr):
            return
        else:
            err = pa.Pa_StopStream(self._stream_ptr)
            PA_ERROR_CHECK(err)

    def play(self):
        """
        Starts playback of the stream.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        if (self._stream_ptr == None):
            self.open()
        if not self.is_playing:
            self.start()

    def pause(self):
        """
        Pauses playback of the stream (Playback cursor is not reset).
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        if pa.Pa_IsStreamStopped(self._stream_ptr):
            return
        else:
            err = pa.Pa_StopStream(self._stream_ptr)
            PA_ERROR_CHECK(err)

    @property
    def is_playing(self):
        """
        Boolean indicating whether the stream is currently playing.
        """
        if self._stream_ptr:
            err = pa.Pa_IsStreamActive(self._stream_ptr)
            PA_ERROR_CHECK(err)
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
    def is_fading(self):
        """
        Returns whether the stream is currently fading from one mix_mat value
        to another.
        """

        if (self.fade_mix_mat_start_time == None or
                self.fade_mix_mat_fade_duration == None):
            return False

        td = datetime.timedelta(seconds=self.fade_mix_mat_fade_duration)
        return self.fade_mix_mat_start_time + td >= datetime.datetime.now()

    @property
    def is_muted(self):
        return self._is_muted

    @is_muted.setter
    def is_muted(self, val):
        self.mute(bool(val))

    def mute(self, val=None): #FIXME I think we should default val to True
        """
        Mutes or unmutes the stream.

        Parameters
        ----------
        val : boolean
            True to mute, false to unmute. Do not pass a val to get current 
            mute state.
            
        Returns
        -------
        val : boolean
            True is muted, otherwise false. Val will be returned if no input 
            argument is specified.

        Notes
        -----
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

    # Notes on error handling during construction:
    # __init__ and _init2 should catch exceptions and clean up any already allocated  
    # resources before re-raising the exception. Subclasess that catch
    # an exception during their __init__ should do the same (clean up their own resources),
    # and also call their superclass's _free_init_resources() method before
    # re-raising the exception.
    # the main point is: any resources allocated during __init__ should be
    # deterministically deallocated if __init__ raises an exception.
    def __init__(self):
        self._stream_user_data = StreamUserData()
        
        self._stream_user_data.command_queues = None

        self._stream_user_data.mix_mat = None;
        self._stream_user_data.mute_mat = None;
        self._stream_user_data.fade_inc_mat = None;
        self._stream_user_data.target_mix_mat = None;

        self._stream_ptr = None
        self._stream_ptr_addr = 0

        try:
            self._stream_user_data.command_queues = cmedussa.alloc_stream_command_queues();
            if not self._stream_user_data.command_queues:
                raise MemoryError
            
            self._stream_user_data.is_muted = 0;
            self._stream_user_data.mix_mat_fade_countdown_frames = 0;
            self._is_muted = False

            self.fade_mix_mat_start_time = None
            self.fade_mix_mat_fade_duration = None
           
        except:
            self.__free_command_queues()
            raise
        
    def _init2(self, device, fs, callback_ptr, callback_command_exec_ptr, callback_user_data, mix_mat, source_channels):
        try:
            self._device = device
            self._stream_user_data.fs = fs
            self._callback_ptr = callback_ptr
            self._callback_command_exec_ptr = callback_command_exec_ptr
            # XXX This is read as int in C and should not be represented by c_void_p
            self._callback_user_data = ctypes.addressof(callback_user_data)

            if device.out_channels == None:
                self._out_channels = device.out_device_info.maxOutputChannels
            else:
                self._out_channels = device.out_channels
     
            # mute_mat and fade_inc_mat are only allocated once.
            # mix_mat and target_mix_mat get allocated/deallocated as necessary
            self._stream_user_data.mix_mat = cmedussa.alloc_medussa_dmatrix( self._out_channels, source_channels, 0 )
            self._stream_user_data.mute_mat = cmedussa.alloc_medussa_dmatrix( self._out_channels, source_channels, 0 )
            self._stream_user_data.fade_inc_mat = cmedussa.alloc_medussa_dmatrix( self._out_channels, source_channels, 0 )
            self._stream_user_data.target_mix_mat = cmedussa.alloc_medussa_dmatrix( self._out_channels, source_channels, 0 )
            
            self.mix_mat_fade_duration = 0.005
            # initial mix_mat is installed without fading
            self.fade_mix_mat_to( _util_allocate_or_conform_mix_mat( mix_mat, self._out_channels, source_channels ), 0 )

            self._out_param = PaStreamParameters(self._device.out_index,
                                                 self._out_channels,
                                                 paFloat32,
                                                 self._device.out_device_info.defaultLowOutputLatency,
                                                 None)
            self._stream_user_data.out_param = ctypes.pointer(self._out_param)

            # Find a smart way to determine this value,
            # which has to be hardcoded into the callback
            self._pa_fpb = 1024

            self._instances.add(weakref.ref(self))

            self.fade_mix_mat_start_time = None
            self.fade_mix_mat_fade_duration = self.mix_mat_fade_duration

        except:
            self._free_init_resources()
            raise
    
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

    def __free_command_queues(self):
        if self._stream_user_data.command_queues:
            cmedussa.free_stream_command_queues( self._stream_user_data.command_queues )
            self._stream_user_data.command_queues = None
        
    def __free_dmatrices(self):
        # free_medussa_dmatrix can handle null ptrs so we don't check for them here
        # but reset to None to avoid double-free
        cmedussa.free_medussa_dmatrix( self._stream_user_data.mix_mat )
        self._stream_user_data.mix_mat = None
        cmedussa.free_medussa_dmatrix( self._stream_user_data.mute_mat )
        self._stream_user_data.mute_mat = None;
        cmedussa.free_medussa_dmatrix( self._stream_user_data.fade_inc_mat )
        self._stream_user_data.fade_inc_mat = None
        cmedussa.free_medussa_dmatrix( self._stream_user_data.target_mix_mat )
        self._stream_user_data.target_mix_mat = None

    def _free_init_resources(self): # derived classes should call this if they catch an error in __init__
        self.__free_command_queues()
        self.__free_dmatrices()
        
    def __del__(self):
        assert( self._stream_ptr == None ) # derived class didn't call _close_stream_and_flush_commands?
        
        self.__free_command_queues()
        self.__free_dmatrices()
        
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
        if cmedussa.post_command_to_pa_callback( self._stream_user_data.command_queues, byref(cmd) ) != 1:

            # if the command queue is full, we wait for it to have space
            if self.is_playing:
                while cmedussa.post_command_to_pa_callback( self._stream_user_data.command_queues, byref(cmd) ) != 1:
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
    mute
        Mutes or unmutes the stream. Mix matrix is unaffected. Playback will
        continue while stream is muted.
    pause
        Pauses playback of the stream (for this type of stream, same as 
        stop).
    play
        Starts playback of the stream from current cursor position. If the 
        current position is the end, the cursor position is reset to zero.
    stop
        Stops playback of the stream (for this type of stream, same as 
        pause).

    Properties
    ----------
    fs : float (read-only)
        The sampling frequency, in Hz.
    is_muted : boolean
        Whether the stream is currently muted (has no affect on mix_mat).
    is_playing : boolean
        Whether the stream is currently playing.
    mix_mat : 2-d NumPy array
        Acts as a mixer. The number of columns corresponds to the number of 
        source channels (in the case of a stereo sound file, 2), and the 
        number of rows corresponds to the number of device output channels, 
        which is accessible with dev.out_channels. The values of the mix_mat 
        are floats between 0. and 1., and are used to specify the playback 
        level of each source channel on each outputchannel. A default mix_mat 
        will have ones along the diagonal, and zeros everywhere else (source 
        channel 1 routed to output device channel 1, source 2 to ouput 2, 
        etc). A mix_mat of all ones would route all source channels to all 
        device output channels. 
        
        Use mix_mat to change overall level (ie., volume) in addition to 
        individual channel levels. To change the overall level (of all 
        channels, leaving the relative channel levels unchanged), you can do 
        something like stream.mix_mat *= .5.
    mix_mat_fade_duration : float
        When mix_mat is changed, the changes in level are faded linearly from 
        those in the old to those in the new mix_mat. This property sets the 
        duration of the fade. Set this to zero for no fade. 
    tone_freq : float
        The frequency, in Hz, of the tone.

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
        try:
            self._tone_user_data = ToneUserData()
            self._tone_user_data.parent = cast(byref(self._stream_user_data), c_void_p)
            self._tone_user_data.t = 0

            super(ToneStream, self)._init2( device, fs, cmedussa.callback_tone, \
                                            cmedussa.execute_tone_user_data_command, \
                                            self._tone_user_data, mix_mat, 1 )

            self.tone_freq = tone_freq
            
            self._instances.add(weakref.ref(self))
        except:
            super(ToneStream, self)._free_init_resources()
            raise

    def __del__(self):
        super(ToneStream, self)._close_stream_and_flush_commands()
        super(ToneStream, self).__del__()


class WhiteStream(Stream):
    """
    Medussa stream object representing Gaussian white noise, which has a flat
    spectrum.

    Methods
    -------
    mute
        Mutes or unmutes the stream. Mix matrix is unaffected. Playback will
        continue while stream is muted.
    pause
        Pauses playback of the stream (for this type of stream, same as 
        stop).
    play
        Starts playback of the stream from current cursor position. If the 
        current position is the end, the cursor position is reset to zero.
    stop
        Stops playback of the stream (for this type of stream, same as 
        pause).

    Properties
    ----------
    fs : float (read-only)
        The sampling frequency, in Hz.
    is_muted : boolean
        Whether the stream is currently muted (has no affect on mix_mat).
    is_playing : boolean
        Whether the stream is currently playing.
    mix_mat : 2-d NumPy array
        Acts as a mixer. The number of columns corresponds to the number of 
        source channels (in the case of a stereo sound file, 2), and the 
        number of rows corresponds to the number of device output channels, 
        which is accessible with dev.out_channels. The values of the mix_mat 
        are floats between 0. and 1., and are used to specify the playback 
        level of each source channel on each outputchannel. A default mix_mat 
        will have ones along the diagonal, and zeros everywhere else (source 
        channel 1 routed to output device channel 1, source 2 to ouput 2, 
        etc). A mix_mat of all ones would route all source channels to all 
        device output channels. 
        
        Use mix_mat to change overall level (ie., volume) in addition to 
        individual channel levels. To change the overall level (of all 
        channels, leaving the relative channel levels unchanged), you can do 
        something like stream.mix_mat *= .5.
    mix_mat_fade_duration : float
        When mix_mat is changed, the changes in level are faded linearly from 
        those in the old to those in the new mix_mat. This property sets the 
        duration of the fade. Set this to zero for no fade. 

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
        try:
            self._white_user_data = WhiteUserData()
            self._white_user_data.parent = cast(byref(self._stream_user_data), c_void_p)

            self._rk_state = Rk_state()
            cmedussa.rk_randomseed(byref(self._rk_state))
            self._white_user_data.rks = cast(byref(self._rk_state), c_void_p)

            super(WhiteStream, self)._init2( device, fs, cmedussa.callback_white, \
                                             cmedussa.execute_white_user_data_command, \
                                             self._white_user_data, mix_mat, 1 )

            self._instances.add(weakref.ref(self))
        except:
            super(WhiteStream, self)._free_init_resources()
            raise

    def __del__(self):
        super(WhiteStream, self)._close_stream_and_flush_commands()
        super(WhiteStream, self).__del__()

        
class PinkStream(Stream):
    """
    Medussa stream object representing pink noise, which has equal energy per
    octave.
    
    Methods
    -------
    mute
        Mutes or unmutes the stream. Mix matrix is unaffected. Playback will
        continue while stream is muted.
    pause
        Pauses playback of the stream (for this type of stream, same as 
        stop).
    play
        Starts playback of the stream from current cursor position. If the 
        current position is the end, the cursor position is reset to zero.
    stop
        Stops playback of the stream (for this type of stream, same as 
        pause).

    Properties
    ----------
    fs : float (read-only)
        The sampling frequency, in Hz.
    is_muted : boolean
        Whether the stream is currently muted (has no affect on mix_mat).
    is_playing : boolean
        Whether the stream is currently playing.
    mix_mat : 2-d NumPy array
        Acts as a mixer. The number of columns corresponds to the number of 
        source channels (in the case of a stereo sound file, 2), and the 
        number of rows corresponds to the number of device output channels, 
        which is accessible with dev.out_channels. The values of the mix_mat 
        are floats between 0. and 1., and are used to specify the playback 
        level of each source channel on each outputchannel. A default mix_mat 
        will have ones along the diagonal, and zeros everywhere else (source 
        channel 1 routed to output device channel 1, source 2 to ouput 2, 
        etc). A mix_mat of all ones would route all source channels to all 
        device output channels. 
        
        Use mix_mat to change overall level (ie., volume) in addition to 
        individual channel levels. To change the overall level (of all 
        channels, leaving the relative channel levels unchanged), you can do 
        something like stream.mix_mat *= .5.
    mix_mat_fade_duration : float
        When mix_mat is changed, the changes in level are faded linearly from 
        those in the old to those in the new mix_mat. This property sets the 
        duration of the fade. Set this to zero for no fade. 

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
        try:
            self._pink_user_data = PinkUserData()
            self._pink_user_data.parent = cast(byref(self._stream_user_data), c_void_p)

            self._pn = Pink_noise_t()
            self._pink_user_data.pn = cast(byref(self._pn), c_void_p)
            cmedussa.initialize_pink_noise(byref(self._pn), 24)

            super(PinkStream, self)._init2( device, fs, cmedussa.callback_pink, \
                                            cmedussa.execute_pink_user_data_command, \
                                            self._pink_user_data, mix_mat, 1 )

            self._instances.add(weakref.ref(self))
        except:
            super(PinkStream, self)._free_init_resources()
            raise
        
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
        raise AttributeError( "can't set attribute (stream.cursor is read only). use stream.time() instead" )

    @property
    def cursor_is_at_end(self):
        return (self._finite_user_data.cursor >= self._finite_user_data.frames)

    @cursor_is_at_end.setter
    def cursor_is_at_end(self, val):
        raise AttributeError( "can't set attribute (stream.cursor_is_at_end is read only)" )
    
    def request_seek( self, positionFrames ):
        """
        Update playback cursor asynchronously. If the stream is running
        the cursor attribute will reflect the change after the PortAudio
        callback next executes.
        """
        cmd = StreamCommand()
        cmd.command = FINITE_STREAM_COMMAND_SET_CURSOR
        cmd.data_uint = c_uint(positionFrames)
        self._post_command_to_pa_callback( cmd )

    def _reset_cursor_when_inactive( self ): # internal method. only safe when stream is not active
        assert( pa.Pa_IsStreamActive(self._stream_ptr) == 0 )
        self._finite_user_data.cursor = 0

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
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        try:
            super(FiniteStream, self).stop()
        except:
            self._reset_cursor_when_inactive()
            raise

        self._reset_cursor_when_inactive()
        
    def play(self):
        """
        Starts playback of the stream.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Notes
        -----
        If playback cursor is exactly at the end of the stream (you can check 
        the boolean property stream.cursor_is_at_end to see if it is), it is 
        reset to zero.
        
        """
        if (self._stream_ptr == None):
            self.open()
        if not self.is_playing:
            if self.cursor_is_at_end: # if cursor is at end, play from start. otherwise continue from current pos
                self._reset_cursor_when_inactive()
            self.start()
            
    def time(self, pos=None, units="sec"):
        """
        Gets or sets the current playback cursor position.

        Parameters
        ----------
        pos : numeric
            The cursor cursor position. If `pos` is `None`, the function
            returns the current position. Otherwise the current cursor
            position is updated to `pos` asynchronously via request_seek()
        units : string
            The units of pos. May be of value:
            "ms": assume pos is of type float
            "sec": assume pos is of type float [default]
            "frames": assume pos is of type int

        Returns
        -------
        pos : numeric
            The current position of the cursor. This value is returned only
            if input argument `pos` is unspecified or None.

        """
        if pos == None:
            if units == "ms":
                return round(self.cursor / self.fs * 1000.0, 3)
            elif units == "sec":
                return round(self.cursor / self.fs, 4)
            elif units == "frames":
                return self.cursor
        elif units == "ms":
            newcursor = int(pos / 1000.0 * self.fs)
        elif units == "sec":
            newcursor = int(pos * self.fs)
        elif units == "frames":
            assert isinstance(pos, int)
            newcursor = pos
        else:
            raise RuntimeError("Bad argument to `units`")

        if newcursor > self.frames:
            # we allow newcursor to be equal to self.frames. this signifies that it is at the end
            raise RuntimeError("New cursor position %d exceeds signal frame count %d." % (newcursor, self.frames))
        self.request_seek( newcursor )

    def __init__(self):
        super(FiniteStream, self).__init__()
        try:
            self._finite_user_data = FiniteUserData()
            self._finite_user_data.parent = cast(byref(self._stream_user_data), c_void_p)
            self._finite_user_data.temp_mat = None
            self._finite_user_data.cursor = 0
            self._finite_user_data.loop = 0
        except:
            super(FiniteStream, self)._free_init_resources()
            raise
        
    def _init2(self, device, fs, callback_ptr, callback_command_exec_ptr, callback_user_data, mix_mat, source_channels, frames, is_looping):

        self._finite_user_data.frames = frames
        self._finite_user_data.duration = frames / float(fs)

        super(FiniteStream, self)._init2( device, fs, callback_ptr, \
                                          callback_command_exec_ptr, \
                                          callback_user_data, mix_mat, \
                                          source_channels )
        try:
            self._finite_user_data.temp_mat = cmedussa.alloc_medussa_dmatrix( self._out_channels, 1, 0 )

            self._finite_user_data.loop = is_looping
            
            self._instances.add(weakref.ref(self))
        except:
            self._free_init_resources()
            raise

    def _free_init_resources(self):
        if self._finite_user_data.temp_mat:
            cmedussa.free_medussa_dmatrix( self._finite_user_data.temp_mat )
            self._finite_user_data.temp_mat = None
        super(FiniteStream, self)._free_init_resources()

    def __del__(self):
        cmedussa.free_medussa_dmatrix( self._finite_user_data.temp_mat )
        self._finite_user_data.temp_mat = None
        super(FiniteStream, self).__del__()


class ArrayStream(FiniteStream):
    """
    Medussa stream object representing a NumPy array.

    You can use medussa.read_file to load sound files into NumPy arrays.

    Methods
    -------
    loop
        Gets or sets whether the stream will loop (continue playing from
        the beginning when it reaches the end).
    mute
        Mutes or unmutes the stream. Mix matrix is unaffected. Playback will
        continue while stream is muted.
    pause
        Pauses playback of the stream (Playback cursor is not reset).
    play
        Starts playback of the stream from current cursor position. If the 
        current position is the end, the cursor position is reset to zero.
    stop
        Stops playback of the stream (Playback cursor is reset to zero).
    time
        Gets or sets the current playback cursor position, default units = ms

    Properties
    ----------
    cursor : long (read-only)
        The current cursor position, in samples (or more precisely, frames). 
    cursor_is_at_end : boolean
        Whether the cursor position is exactly at the end of the stream.
    duration : float (read-only)
        The stream duration in seconds.
    file_name : str (read-only)
        The path to the sound file.
    frames : long (read-only)
        The number of samples per source channel.
    fs : float (read-only)
        The sampling frequency, in Hz.
    is_looping : boolean
        Whether the stream will continue playing from the beginning when it 
        reaches the end.
    is_muted : boolean
        Whether the stream is currently muted (has no affect on mix_mat).
    is_playing : boolean
        Whether the stream is currently playing (setting to False is like 
        calling pause; it does not reset the stream cursor).
    mix_mat : 2-d NumPy array
        Acts as a mixer. The number of columns corresponds to the number of 
        source channels (in the case of a stereo sound file, 2), and the 
        number of rows corresponds to the number of device output channels, 
        which is accessible with dev.out_channels. The values of the mix_mat 
        are floats between 0. and 1., and are used to specify the playback 
        level of each source channel on each outputchannel. A default mix_mat 
        will have ones along the diagonal, and zeros everywhere else (source 
        channel 1 routed to output device channel 1, source 2 to ouput 2, 
        etc). A mix_mat of all ones would route all source channels to all 
        device output channels. 
        
        Use mix_mat to change overall level (ie., volume) in addition to 
        individual channel levels. To change the overall level (of all 
        channels, leaving the relative channel levels unchanged), you can do 
        something like stream.mix_mat *= .5.
    mix_mat_fade_duration : float
        When mix_mat is changed, the changes in level are faded linearly from 
        those in the old to those in the new mix_mat. This property sets the 
        duration of the fade. Set this to zero for no fade. 

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
        self._array_user_data.ndarr = self.__arr.ctypes.data_as(c_double_p)
        self._array_user_data.ndarr_0 = val.shape[0]
        self._array_user_data.ndarr_1 = val.shape[1]

    def __init__(self, device, fs, mix_mat, arr, is_looping=False):
        super(ArrayStream, self).__init__()
        try:
            self._array_user_data = ArrayUserData()
            self._array_user_data.parent = cast(byref(self._finite_user_data), c_void_p)

            if len(arr.shape) == 1:
                arr = arr.reshape(arr.size, 1)

            self.__set_arr( arr )

            channels = arr.shape[1] 
            frames = arr.shape[0]  
            super(ArrayStream, self)._init2( device, fs, cmedussa.callback_ndarray, \
                                             cmedussa.execute_array_user_data_command, \
                                             self._array_user_data, mix_mat, channels, frames, is_looping )
            
            self._instances.add(weakref.ref(self))
        except:
            super(ArrayStream, self)._free_init_resources()
            raise

    def __del__(self):
        super(ArrayStream, self)._close_stream_and_flush_commands()
        super(ArrayStream, self).__del__()
        
        
class SoundfileStream(FiniteStream):
    """
    Medussa stream object representing a sound file on disk.

    Methods
    -------
    loop
        Gets or sets whether the stream will loop (continue playing from
        the beginning when it reaches the end).
    mute
        Mutes or unmutes the stream. Mix matrix is unaffected. Playback will
        continue while stream is muted.
    pause
        Pauses playback of the stream (Playback cursor is not reset).
    play
        Starts playback of the stream from current cursor position. If the 
        current position is the end, the cursor position is reset to zero.
    stop
        Stops playback of the stream (Playback cursor is reset to zero).
    time
        Gets or sets the current playback cursor position, default units = ms

    Properties
    ----------
    cursor : long (read-only)
        The current cursor position, in samples (or more precisely, frames). 
    cursor_is_at_end : boolean
        Whether the cursor position is exactly at the end of the stream.
    duration : float (read-only)
        The stream duration in seconds.
    file_name : str (read-only)
        The path to the sound file.
    frames : long (read-only)
        The number of samples per source channel.
    fs : float (read-only)
        The sampling frequency, in Hz.
    is_looping : boolean
        Whether the stream will continue playing from the beginning when it 
        reaches the end.
    is_muted : boolean
        Whether the stream is currently muted (has no affect on mix_mat).
    is_playing : boolean
        Whether the stream is currently playing (setting to False is like 
        calling pause; it does not reset the stream cursor).
    mix_mat : 2-d NumPy array
        Acts as a mixer. The number of columns corresponds to the number of 
        source channels (in the case of a stereo sound file, 2), and the 
        number of rows corresponds to the number of device output channels, 
        which is accessible with dev.out_channels. The values of the mix_mat 
        are floats between 0. and 1., and are used to specify the playback 
        level of each source channel on each outputchannel. A default mix_mat 
        will have ones along the diagonal, and zeros everywhere else (source 
        channel 1 routed to output device channel 1, source 2 to ouput 2, 
        etc). A mix_mat of all ones would route all source channels to all 
        device output channels. 
        
        Use mix_mat to change overall level (ie., volume) in addition to 
        individual channel levels. To change the overall level (of all 
        channels, leaving the relative channel levels unchanged), you can do 
        something like stream.mix_mat *= .5.
    mix_mat_fade_duration : float
        When mix_mat is changed, the changes in level are faded linearly from 
        those in the old to those in the new mix_mat. This property sets the 
        duration of the fade. Set this to zero for no fade. 

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
        return _from_cstr(self.__file_name)

    @file_name.setter
    def file_name(self, val):
        raise AttributeError( "can't set attribute (stream.file_name is read only)" )

    def __set_file_name(self, val):
        self.__file_name = _to_cstr(val)
        self._sndfile_user_data.file_name = c_char_p(self.__file_name)

    def __init__(self, device, mix_mat, file_name, is_looping=False):
        super(SoundfileStream, self).__init__()
        try:
            self._sndfile_user_data = SndfileUserData()
            self._sndfile_user_data.file_stream = None
            self._fin = None
        except:
            super(SoundfileStream, self)._free_init_resources()
            raise
            
        try:
            self._sndfile_user_data.parent = cast(byref(self._finite_user_data), c_void_p)

            # Initialize this class' attributes

            if not os.path.isfile(file_name):
                raise IOError('File not found: %s' % file_name)
        
            self.__set_file_name( file_name )
            
            self._finfo = SF_INFO(0,0,0,0,0,0)
            self._sndfile_user_data.finfo = ctypes.pointer(self._finfo)

            self._fin = csndfile.sf_open(_to_cstr(file_name),
                                        SFM_READ,
                                        byref(self._finfo))

            if not self._fin:
                raise RuntimeError("Error opening soundfile: %s" % csndfile.sf_strerror( self._fin ))

            self._sndfile_user_data.fin = self._fin

            buffer_size_frames = 32 * 1024
            buffer_queue_duration_seconds = 5 # 5 seconds read ahead
            buffer_queue_duration_frames = self._finfo.samplerate * buffer_queue_duration_seconds
            buffer_count = max( 5, int(buffer_queue_duration_frames / buffer_size_frames) + 1 )

            self._sndfile_user_data.file_stream = cmedussa.allocate_file_stream(self._fin, self._sndfile_user_data.finfo, buffer_count, buffer_size_frames)

            if not self._sndfile_user_data.file_stream:
                raise RuntimeError("Error allocating async. file stream")
            
            fs = self._finfo.samplerate
            frames = self._finfo.frames
            super(SoundfileStream, self)._init2( device, fs, cmedussa.callback_sndfile_read, \
                                                 cmedussa.execute_sndfile_read_user_data_command, \
                                                 self._sndfile_user_data, mix_mat, self._finfo.channels, frames, is_looping )

            self._instances.add(weakref.ref(self))
            
        except:
            self._free_init_resources()
            raise

    def _free_init_resources( self ):
        self._free_sndfile_and_file_stream()
        super(SoundfileStream, self)._free_init_resources()

    def _free_sndfile_and_file_stream( self ):
        if self._fin:
            csndfile.sf_close(self._fin)
            self._fin = None
            
        if self._sndfile_user_data.file_stream:
            cmedussa.free_file_stream(self._sndfile_user_data.file_stream)
            self._sndfile_user_data.file_stream = None
            
    def __del__(self):
        super(SoundfileStream, self)._close_stream_and_flush_commands()
        self._free_sndfile_and_file_stream();
        super(SoundfileStream, self).__del__()
        

###################
## General Methods

def get_default_output_device_index():
    """
    Returns the index to the system default audio output device.

    Parameters
    ----------
    None

    Returns
    -------
    device_ind : int
        The index to the default output device.

    """
    devices = [(i,x) for (i,x) in enumerate(get_available_devices()) if x.name == 'default']
    if devices == []:
        output_device = pa.Pa_GetDefaultOutputDevice()
        return output_device if output_device != PA_NODEVICE else None
    else:
        i,d = devices[0]
        if d.maxOutputChannels > 0:
            return i
        else:
            output_device = pa.Pa_GetDefaultOutputDevice()
            return output_device if output_device != PA_NODEVICE else None


def get_default_input_device_index():
    """
    Returns the index to the system default audio input device.

    Parameters
    ----------
    None

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
        input_device = pa.Pa_GetDefaultInputDevice()
        return input_device if input_device != PA_NODEVICE else None
    else:
        i,d = devices[0]
        if d.maxInputChannels > 0:
            return i
        else:
            input_device = pa.Pa_GetDefaultInputDevice()
            return input_device if input_device != PA_NODEVICE else None


def generate_hostapi_info():
    api_count = pa.Pa_GetHostApiCount()
    for i in xrange(api_count):
        p = pa.Pa_GetHostApiInfo(i)
        hai = p[0]
        yield hai


def generate_device_info():
    device_count = pa.Pa_GetDeviceCount()

    PA_ERROR_CHECK(device_count)

    if device_count == 0:
        raise RuntimeError("No devices found")

    for i in xrange(device_count):
        p = pa.Pa_GetDeviceInfo(i)
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
    PA_ERROR_CHECK(err)
    return True


def terminate():
    """
    Attempts to terminate Portaudio.
    """
    err = pa.Pa_Terminate()
    PA_ERROR_CHECK(err)
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


def play_file(file_name, output_device_id=None, volume=1., duration=10):
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
        play the first few seconds of a file. The default value is 10 s, 
        which is intended to be a sort of sanity check in case you 
        accidentally pass a file that is exceedingly long (since it is a 
        blocking function). To play an entire file regardless of how long it 
        is, set `duration` to 0. 

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
    if duration == 0:
        duration = s.duration
    s.mix_mat *= float(volume)
    s.play()
    while s.is_playing:
        if s.time() > duration:
            s.stop()


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
    fin = csndfile.sf_open(_to_cstr(file_name), SFM_READ, byref(finfo))

    if not fin:
        raise RuntimeError("Error opening soundfile: %s" % csndfile.sf_strerror( self._fin ))
        
    fs = finfo.samplerate

    BUFFTYPE = ctypes.c_double * (finfo.frames * finfo.channels)
    buff = BUFFTYPE()

    frames_read = cmedussa.readfile_helper(fin, buff, finfo.frames)

    err = csndfile.sf_close(fin)

    arr = np.ascontiguousarray(np.zeros((finfo.frames, finfo.channels)))

    for i in xrange(finfo.frames):
        for j in xrange(finfo.channels):
            arr[i][j] = buff[i*finfo.channels + j]

    #Samples with 1 channel need to be reshaped, so that they can be directly
    #passed to, for example, play_array.
    if (len(arr.shape) == 2 and arr.shape[1] == 1):
        arr = arr[:,0]

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
    Existing files will be over-written without warning!

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
    finfo.format = c_int(fmt)

    arr = np.ascontiguousarray(arr)
    _arr = arr.ctypes.data_as(c_double_p)

    frames_written = cmedussa.writefile_helper(_to_cstr(file_name),
                                               byref(finfo),
                                               _arr,
                                               fmt,
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
    bits : int or str
        The bit depth. For wavefiles, libsndfile can handle 8, 16, 24, or 32 
        bits. You can also use a string to specify either signed 16-, 24- or 
        32-bit integers ('s16', 's24', 's32'), or unsigned 8-bit integers 
        ('u8').
    frames : int
        The number of frames to write.

    Returns
    -------
    frames_written : int
        The number of frames that were written to the file.

    Notes
    -----
    Existing files will be over-written without warning!

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

    return write_file(file_name, arr, fs, fmt=endformat, frames=frames)


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
    bits : int or str
        The bit depth. For flac files, libsndfile can handle 8, 16, or 24 
        bits. You can also use a string to specify signed 8- 16-, or 24-bit
        integers ('s8', 's16', 's24').
    frames : int
        The number of frames to write.

    Returns
    -------
    frames_written : int
        The number of frames that were written to the file.

    Notes
    -----
    Existing files will be over-written without warning!

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

    return write_file(file_name, arr, fs, fmt=endformat, frames=frames)


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
    Existing files will be over-written without warning!

    Bit depth is not specified with the Vorbis format, but rather is 
    variable.

    The file IO functions in Medussa are intended to be extremely light
    wrappers to libsndfile, and not a full python implementation of its API.
    For that, you want scikits.audiolab.

    """
    majformat = sf_formats.SF_CONTAINER_OGG

    subformat = sf_formats.SF_ENCODING_VORBIS

    endformat = majformat | subformat

    return write_file(file_name, arr, fs, fmt=endformat, frames=frames)
