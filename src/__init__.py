# -*- coding: utf-8 -*-

import platform
pyver = platform.python_version_tuple()[0]
if pyver == "2":
    from medussa import (play_array, play_file, cmedussa,
                        Device, Stream, FiniteStream,
                        ToneStream, WhiteStream, PinkStream,
                        ArrayStream, SndfileStream, generate_hostapi_info,
                        generate_device_info, print_available_devices,
                        start_streams, open_device, open_default_device, init,
                        sf_formats, terminate, read_file,
                        write_file, write_wav, write_flac, write_ogg,
                        device_instances, stream_instances,
                        get_default_output_device_index,
                        get_default_input_device_index)
    from docs import (package_name, version, url, author, author_email,
                        members, short_description, long_description)
    from portaudio import pa, ERROR_CHECK
else:
    from .medussa import (play_array, play_file, cmedussa,
                        Device, Stream, FiniteStream,
                        ToneStream, WhiteStream, PinkStream,
                        ArrayStream, SndfileStream, generate_hostapi_info,
                        generate_device_info, print_available_devices,
                        start_streams, open_device, open_default_device, init,
                        sf_formats, terminate, read_file,
                        write_file, write_wav, write_flac, write_ogg,
                        device_instances, stream_instances,
                        get_default_output_device_index,
                        get_default_input_device_index)
    from .docs import (package_name, version, url, author, author_email,
                        members, short_description, long_description)
    from .portaudio import pa, ERROR_CHECK

__doc__ = "%s\n\n%s" % (long_description, members)
__package_name__ = package_name
__version__ = version
__description__ = short_description
__author__ = author
__author_email__ = author_email
__url__ = url
__pa_version__ = "%s, Release %d" % (pa.Pa_GetVersionText(), pa.Pa_GetVersion())

err = pa.Pa_Initialize()
try:
    ERROR_CHECK(err)
except RuntimeError as e:
    raise ImportError(e.message)
