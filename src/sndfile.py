# -*- coding: utf-8 -*-

import platform
pyver = platform.python_version_tuple()[0]
if pyver == "2":
    from sndfile_formats import formats
else:
    from .sndfile_formats import formats


from ctypes.util import find_library
from distutils.sysconfig import get_python_lib
from os.path import exists
from ctypes import *


# Select the correct name for the shared library, dependent on platform
if platform.system() == "Windows":
    libname = get_python_lib() + "\\medussa\\libsndfile-1.dll"
    if not exists(libname):
	    raise RuntimeError("Unable to locate library: " + libname)
else:
    libname = find_library("sndfile")
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

# set argument and return types for relevant `libsndfile' functions
#csndfile.sf_seek.restype =
csndfile.sf_seek.argtypes = [c_void_p, c_uint, c_int]

