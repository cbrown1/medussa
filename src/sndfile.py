# -*- coding: utf-8 -*-

# Copyright (c) 2010-2012 Christopher Brown
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

from .sndfile_formats import sndfile_formats

sf_formats = sndfile_formats()

import os
import platform
from ctypes.util import find_library
from distutils.sysconfig import get_python_lib
from ctypes import *

# Select the correct name for the shared library, dependent on platform
if platform.system() == "Windows":
    LIBSNDFILE = 'libsndfile-1.dll'
    libsearchpath = [
        os.path.join(get_python_lib(), "medussa", "dlls", LIBSNDFILE),
        os.path.join(get_python_lib(), "medussa", LIBSNDFILE),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), LIBSNDFILE),
		os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dlls', LIBSNDFILE),
        os.path.join(os.environ["ProgramFiles"], "Mega-Nerd", "libsndfile","bin",LIBSNDFILE),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'medussa', LIBSNDFILE)
		]
    libname = ""
    foundlib = False
    for libpath in libsearchpath:
        libname = libpath
        if os.path.exists(libname):
            foundlib = True
            break
    if not foundlib:
        raise RuntimeError("Unable to locate library: libsndfile")
else:
    libname = find_library("sndfile")
    if libname == None:
        raise RuntimeError("Unable to locate library `libsndfile`")

# Load the shared library
# In linux, if this doesn't work try:
#su -c "echo '/usr/local/lib' >> /etc/ld.so.conf"
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

    _fields_ = (("frames",     c_longlong),
                ("samplerate", c_int),
                ("channels",   c_int),
                ("format",     c_int),
                ("sections",   c_int),
                ("seekable",   c_int))

# sndfile.h `#define` macros
SFM_READ  = c_int(0x10)
SFM_WRITE = c_int(0x20)
SFM_RDWR  = c_int(0x30)

SFC_GET_LIB_VERSION = c_int(0x1000)

def get_libsndfile_version():
	s = create_string_buffer(128)
	csndfile.sf_command (None, SFC_GET_LIB_VERSION, s, len(s))
	return string_at(s)

# set argument and return types for relevant `libsndfile' functions
#csndfile.sf_seek.restype =
csndfile.sf_seek.argtypes = [c_void_p, c_uint, c_int]

csndfile.sf_strerror.restype   = c_char_p
