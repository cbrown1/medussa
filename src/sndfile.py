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

from .sndfile_formats import sndfile_formats

sf_formats = sndfile_formats()

import os
import platform
import sys
from ctypes.util import find_library
from ctypes import *
from os import path as _p

# Select the correct name for the shared library, dependent on platform
if platform.system() == "Windows":
    libpath = _p.join(_p.dirname(_p.abspath(__file__)), 'lib', 'sndfile.dll')
else:
    libpath = "sndfile"
libname = find_library(libpath)
if libname == None:
    raise RuntimeError("Unable to locate library `{}`".format(libpath))

# Load the shared library
# In linux, if this doesn't work try:
#su -c "echo '/usr/local/lib' >> /etc/ld.so.conf"
csndfile = CDLL(libname)

sf_count_t = c_longlong

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

    _fields_ = (("frames",     sf_count_t),
                ("samplerate", c_int),
                ("channels",   c_int),
                ("format",     c_int),
                ("sections",   c_int),
                ("seekable",   c_int))
SF_INFO_p = POINTER(SF_INFO)

# sndfile.h `#define` macros
SFM_READ  = c_int(0x10)
SFM_WRITE = c_int(0x20)
SFM_RDWR  = c_int(0x30)

SFC_GET_LIB_VERSION = c_int(0x1000)

class SNDFILE (Structure):
    pass
SNDFILE_p = POINTER(SNDFILE)

# XXX 64-bit platforms except ILP64 require pointer returns to be marked as such,
# otherwise truncation occurs on casting to c_int which is the default restype.
# This in most cases leads to the segfault on first use of such int-disguised-pointer.
csndfile.sf_seek.restype = sf_count_t
csndfile.sf_seek.argtypes = [SNDFILE_p, sf_count_t, c_int]
csndfile.sf_strerror.restype = c_char_p
csndfile.sf_strerror.argtypes = [SNDFILE_p]
csndfile.sf_open.restype = SNDFILE_p
csndfile.sf_close.argtypes = [SNDFILE_p]
csndfile.sf_command.argtypes = [SNDFILE_p, c_int, c_void_p, c_int]


def get_libsndfile_version():
	s = create_string_buffer(128)
	csndfile.sf_command (None, SFC_GET_LIB_VERSION, s, len(s))
	return string_at(s)

