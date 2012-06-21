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

from .medussa import (play_array, play_file, read_file, 
                    write_file, write_wav, write_flac, write_ogg, 
                    Stream, ToneStream, PinkStream, WhiteStream, 
                    FiniteStream, ArrayStream, SoundfileStream, 
                    get_default_output_device_index, 
                    get_default_input_device_index, 
                    Device, open_device, open_default_device, 
                    generate_hostapi_info, generate_device_info, 
                    get_available_devices, print_available_devices, 
                    device_instances, stream_instances, start_streams, 
                    cmedussa, init, terminate, 
                    sf_formats, 
                    )
from .docs import (package_name, version, url, author, author_email,
                    long_help, short_description, long_description)
from .portaudio import pa, PA_ERROR_CHECK
from .sndfile import get_libsndfile_version
	
__doc__ = "%s\n\n%s" % (long_description, long_help)
__package_name__ = package_name
__version__ = version
__description__ = short_description
__author__ = author
__author_email__ = author_email
__url__ = url
__pa_version__ = "%s, Release %d" % (pa.Pa_GetVersionText(), pa.Pa_GetVersion())
__libsndfile_version__ = get_libsndfile_version()
del(package_name, version, url, author, author_email, long_help, short_description, long_description)

err = pa.Pa_Initialize()
try:
    PA_ERROR_CHECK(err)
except RuntimeError as e:
    raise ImportError(e.message)
