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

import platform
import ctypes
from ctypes.util import find_library
from ctypes import c_int, c_uint, c_long, c_ulong, c_float, c_double, c_char_p, c_void_p, py_object, byref, POINTER
import os
import sys
from os import path as _p

# Select the correct name for the shared library, dependent on platform
if platform.system() == "Windows":
    libpath = _p.join(_p.dirname(_p.abspath(__file__)), "lib", "portaudio.dll")
else:
    libpath = "portaudio"
libname = find_library(libpath)
if libname == None:
    raise RuntimeError("Unable to locate library: `{}`".format(libpath))

# Load the shared library
# In linux, if this doesn't work try:
#su -c "echo '/usr/local/lib' >> /etc/ld.so.conf"
pa = ctypes.CDLL(libname)

# Alias `typedef int PaError`
PaError = c_int


# Alias `typedef int PaDeviceIndex`
PaDeviceIndex = c_int


# Alias `typedef int PaHostApiIndex`
PaHostApiIndex = c_int


# Alias `typedef double PaTime`
PaTime = c_double


# Aliases for `#define` macros in portaudio.h
# ###########################################

# Special values for `PaDeviceIndex`
paNoDevice = c_int(-1)  # ((PaDeviceIndex)-1)
paUseHostApiSpecificDeviceSpecification = c_int(-2)  # ((PaDeviceIndex)-2)

# A type used to specify one or more sample formats.
# Each value indicates a possible format for sound data passed to and from
# the stream callback, Pa_ReadStream and Pa_WriteStream.
# http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#4582d93c2c2e60e12be3d74c5fe00b96
paFloat32        = c_ulong(0x00000001)  # ((PaSampleFormat) 0x00000001)
paInt32          = c_ulong(0x00000002)  # ((PaSampleFormat) 0x00000002)
paInt24          = c_ulong(0x00000004)  # ((PaSampleFormat) 0x00000004)
paInt16          = c_ulong(0x00000008)  # ((PaSampleFormat) 0x00000008)
paInt8           = c_ulong(0x00000010)  # ((PaSampleFormat) 0x00000010)
paUInt8          = c_ulong(0x00000020)  # ((PaSampleFormat) 0x00000020)
paCustomFormat   = c_ulong(0x00010000)  # ((PaSampleFormat) 0x00010000)
paNonInterleaved = c_ulong(0x80000000)  # ((PaSampleFormat) 0x80000000)

# Special comparison constant for return value of `Pa_IsFormatSupported`
paFormatIsSupported = c_int(0)  # (0)

# Can be passed as the framesPerBuffer parameter to Pa_OpenStream() or
# Pa_OpenDefaultStream() to indicate that the stream callback will accept
# buffers of any size.
paFramesPerBufferUnspecified = c_int(0)  # (0)

# Flags used to control the behavior of a stream.
# They are passed as parameters to Pa_OpenStream or Pa_OpenDefaultStream.
# Multiple flags may be ORed together.
paNoFlag                                = c_ulong(0)           # ((PaStreamFlags) 0)
paClipOff                               = c_ulong(0x00000001)  # ((PaStreamFlags) 0x00000001)
paDitherOff                             = c_ulong(0x00000002)  # ((PaStreamFlags) 0x00000002)
paNeverDropInput                        = c_ulong(0x00000004)  # ((PaStreamFlags) 0x00000004)
paPrimeOutputBuffersUsingStreamCallback = c_ulong(0x000000080) # ((PaStreamFlags) 0x00000008)
paPlatformSpecificFlags                 = c_ulong(0xFFFF0000)  # ((PaStreamFlags)0xFFFF0000)

# Flag bit constants for the statusFlags to PaStreamCallback.
paInputUnderflow  = c_ulong(0x00000001)  # ((PaStreamCallbackFlags) 0x00000001)
paInputOverflow   = c_ulong(0x00000002)  # ((PaStreamCallbackFlags) 0x00000002)
paOutputUnderflow = c_ulong(0x00000004)  # ((PaStreamCallbackFlags) 0x00000004)
paOutputOverflow  = c_ulong(0x00000008)  # ((PaStreamCallbackFlags) 0x00000008)
paPrimingOutput   = c_ulong(0x00000010)  # ((PaStreamCallbackFlags) 0x00000010)


# Models `typedef enum PaErrorCode { ... } PaErrorCode`
class PaErrorCode (c_int):
    paNoError                  = c_int(0)
    paNotInitialized           = c_int(-10000)
    paUnanticipatedHostError   = c_int(-9999)
    paInvalidChannelCount      = c_int(-9998)
    paInvalidSampleRate        = c_int(-9997)
    paInvalidDevice            = c_int(-9996)
    paInvalidFlag              = c_int(-9995)
    paSampleFormatNotSupported = c_int(-9994)
    paBadIODeviceCombination   = c_int(-9993)
    paInsufficientMemory       = c_int(-9992)
    paBufferTooBig             = c_int(-9991)
    paBufferTooSmall           = c_int(-9990)
    paNullCallback             = c_int(-9989)
    paBadStreamPtr             = c_int(-9988)
    paTimedOut                 = c_int(-9987)
    paInternalError            = c_int(-9986)
    paDeviceUnavailable        = c_int(-9985)
    paIncompatibleHostApiSpecificStreamInfo = c_int(-9984)
    paStreamIsStopped                  = c_int(-9983)
    paStreamIsNotStopped               = c_int(-9982)
    paInputOverflowed                  = c_int(-9981)
    paOutputUnderflowed                = c_int(-9980)
    paHostApiNotFound                  = c_int(-9979)
    paInvalidHostApi                   = c_int(-9978)
    paCanNotReadFromACallbackStream    = c_int(-9977)
    paCanNotWriteToACallbackStream     = c_int(-9976)
    paCanNotReadFromAnOutputOnlyStream = c_int(-9975)
    paCanNotWriteToAnInputOnlyStream   = c_int(-9974)
    paIncompatibleStreamHostApi        = c_int(-9973)
    paBadBufferPtr                     = c_int(-9972)


# Models `typedef enum PaHostApiTypeId { ... } PaHostApiTypeId`
class PaHostApiTypeId:
    """
    Unchanging unique identifiers for each supported host API. This type is
    used in the PaHostApiInfo structure.

    The values are guaranteed to be unique and to never change, thus allowing
    code to be written that conditionally uses host API specific extensions.
    """
    paInDevelopment   = c_int(0)
    paDirectSound     = c_int(1)
    paMME             = c_int(2)
    paASIO            = c_int(3)
    paSoundManager    = c_int(4)
    paCoreAudio       = c_int(5) # Note that there is no enum => 6 in the Portaudio api
    paOSS             = c_int(7)
    paALSA            = c_int(8)
    paAL              = c_int(9)
    paBeOS            = c_int(10)
    paWDMKS           = c_int(11)
    paJACK            = c_int(12)
    paWASAPI          = c_int(13)
    paAudioScienceHPI = c_int(14)
    from_int = {0:  "paInDevelopment",
                1:  "paDirectSound",
                2:  "paMME",
                3:  "paASIO",
                4:  "paSoundManager",
                5:  "paCoreAudio",
                7:  "paOSS",
                8:  "paALSA",
                9:  "paAL",
                10: "paBeOS",
                11: "paWDMKS",
                12: "paJACK",
                13: "paWASAPI",
                14: "paAudioScienceHPI"}


# Sane names for user references via `str`
# First we define it as a tuple, then we add its converse and cast as `dict`
HostApiTypeAliases = ((0, "indevelopment"),
                      (1, "dsound"),
                      (2,  "mme"),
                      (3,  "asio"),
                      (4,  "soundmanager"),
                      (5,  "coreaudio"),
                      (7,  "oss"),
                      (8,  "alsa"),
                      (9,  "al"),
                      (10, "beos"),
                      (11, "wdmks"),
                      (12, "jack"),
                      (13, "wasapi"),
                      (14, "audiosciencehpi"))
HostApiTypeAliases = dict(HostApiTypeAliases + tuple((x,i) for (i,x) in HostApiTypeAliases))


# struct PaHostApiInfo
class PaHostApiInfo (ctypes.Structure):
    """
    struct PaHostApiInfo
        int             structVersion
        PaHostApiTypeId type
        const char *    name
        int             deviceCount
        PaDeviceIndex   defaultInputDevice
        PaDeviceIndex   defaultOutputDevice
    """
    _fields_ = (("structVersion",       c_int),
                ("type",                c_int), # PaHostApiTypeId
                ("name",                c_char_p),
                ("deviceCount",         c_int),
                ("defaultInputDevice",  c_int), # PaDeviceIndex
                ("defaultOutputDevice", c_int)) # PaDeviceIndex
HostApiInfoPointer = POINTER(PaHostApiInfo)


# struct PaHostErrorInfo
class PaHostErrorInfo (ctypes.Structure):
    """
    struct PaHostErrorInfo
        PaHostApiTypeId hostApiType
        long            errorCode
        const char *    errorText
    """
    _fields_ = (("hostApiType", c_int), # PaHostApiTypeId
                ("errorCode",   c_long),
                ("errorText",   c_char_p))
HostErrorInfoPointer = POINTER(PaHostErrorInfo)


# struct PaDeviceInfo
class PaDeviceInfo (ctypes.Structure):
    """
    struct PaDeviceInfo
        int            structVersion
        const char *   name
        PaHostApiIndex hostApi
        int            maxInputChannels
        int            maxOutputChannels
        PaTime         defaultLowInputLatency
        PaTime         defaultLowOutputLatency
        PaTime         defaultHighInputLatency
        PaTime         defaultHighOutputLatency
        double         defaultSampleRate
    """
    _fields_ = (("structVersion",     c_int),
                ("name",              c_char_p),
                ("hostApi",           c_int), # PaHostApiIndex
                ("maxInputChannels",  c_int),
                ("maxOutputChannels", c_int),
                ("defaultLowInputLatency",   c_double), # PaTime
                ("defaultLowOutputLatency",  c_double), # PaTime
                ("defaultHighInputLatency",  c_double), # PaTime
                ("defaultHighOutputLatency", c_double), # PaTime
                ("defaultSampleRate",        c_double))
# Type used for getting `DeviceInfo` structs
DeviceInfoPointer = POINTER(PaDeviceInfo)

# struct PaStreamParameters
class PaStreamParameters (ctypes.Structure):
    """
    struct PaStreamParameters
        PaDeviceIndex  device
        int            channelCount
        PaSampleFormat sampleFormat
        PaTime         suggestedLatency
        void *         hostApiSpecificStreamInfo
    """
    _fields_ = (("device",                    c_int), # PaDeviceIndex
                ("channelCount",              c_int),
                ("sampleFormat",              c_ulong),  # PaSampleFormat
                ("suggestedLatency",          c_double), # PaTime
                ("hostApiSpecificStreamInfo", c_void_p))
StreamParametersPointer = POINTER(PaStreamParameters)


# struct PaStreamCallbackTimeInfo
class PaStreamCallbackTimeInfo (ctypes.Structure):
    """
    struct PaStreamCallbackTimeInfo
        PaTime inputBufferAdcTime
        PaTime currentTime
        PaTime outputBufferDacTime
    """
    _fields_ = (("inputBufferAdcTime",  c_double), # PaTime
                ("currentTime",         c_double), # PaTime
                ("outputBufferDacTime", c_double)) # PaTime
StreamCallbackTimeInfoPointer = POINTER(PaStreamCallbackTimeInfo)


# struct PaStreamInfo
class PaStreamInfo (ctypes.Structure):
    """
    struct PaStreamInfo
        int    structVersion
        PaTime inputLatency
        PaTime outputLatency
        double sampleRate
    """
    _fields_ = (("structVersion", c_int),
                ("inputLatency",  c_double), # PaTime
                ("outputLatency", c_double), # PaTime
                ("sampleRate",    c_double))
StreamInfoPointer = POINTER(PaStreamInfo)


# Opaque struct for enforcing API type validation only.
class PaStream (ctypes.Structure):
    pass
StreamPointer = POINTER(PaStream)


def PA_ERROR_CHECK(err):
    """
    If `err` is an actual Portaudio error (< 0), raises a RuntimeError whose message is the Portaudio error text.
    """
    if err < 0:
        raise RuntimeError("PaError(%d): %s" % (err, pa.Pa_GetErrorText(err)))

# Explicitly mark pointer arguments for 64-bit compatibility
pa.Pa_GetStreamTime.restype  = c_double  # c_double ~ PaTime
pa.Pa_GetStreamTime.argtypes = [StreamPointer]
pa.Pa_GetVersionText.restype = c_char_p
pa.Pa_GetDeviceInfo.restype  = DeviceInfoPointer
pa.Pa_GetHostApiInfo.restype = HostApiInfoPointer
pa.Pa_GetErrorText.restype = c_char_p
pa.Pa_GetLastHostErrorInfo.restype = HostErrorInfoPointer
pa.Pa_GetStreamInfo.restype = StreamInfoPointer
pa.Pa_GetStreamInfo.argtypes = [StreamPointer]
pa.Pa_CloseStream.argtypes = [StreamPointer]
pa.Pa_StartStream.argtypes = [StreamPointer]
pa.Pa_StopStream.argtypes = [StreamPointer]
pa.Pa_AbortStream.argtypes = [StreamPointer]
pa.Pa_IsStreamStopped.argtypes = [StreamPointer]
pa.Pa_IsStreamActive.argtypes = [StreamPointer]
