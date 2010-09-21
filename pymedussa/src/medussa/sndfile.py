import platform
from ctypes.util import find_library
from distutils.sysconfig import get_python_lib

from ctypes import *


# Select the correct name for the shared library, dependent on platform
if platform.system() == "Windows":
    libname = get_python_lib() + "\\libsndfile-1.dll"
else:
    libname = find_library("sndfile") # untested

if libname == None:
    raise RuntimeError("Unable to locate library `libsndfile`")

csndfile = CDLL(libname)

class SF_INFO (Structure):
    """
    See: http://www.mega-nerd.com/libsndfile/api.html#open

    typedef struct {
        sf_count_t  frames ;
        int         samplerate ;
        int         channels ;
        int         format ;
        int         sections ;
        int         seekable ;
    } SF_INFO ;
    """
    _fields_ = (("frames",     c_uint),
                ("format",     c_int),
                ("samplerate", c_int),
                ("channels",   c_int),
                ("sections",   c_int),
                ("seekable",   c_int))

# sndfile.h `#define` macros
SFM_READ  = c_int(0x10)
SFM_WRITE = c_int(0x20)
SFM_RDWR  = c_int(0x30)