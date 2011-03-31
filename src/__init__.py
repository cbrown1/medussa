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
                        terminate, read_file,
                        write_file, write_wav, write_flac, write_ogg,
                        device_instances, stream_instances)
    from portaudio import pa, ERROR_CHECK
else:
    from .medussa import (play_array, play_file, cmedussa,
                        Device, Stream, FiniteStream,
                        ToneStream, WhiteStream, PinkStream,
                        ArrayStream, SndfileStream, generate_hostapi_info,
                        generate_device_info, print_available_devices,
                        start_streams, open_device, open_default_device, init,
                        terminate, read_file,
                        write_file, write_wav, write_flac, write_ogg,
                        device_instances, stream_instances)
    from .portaudio import pa, ERROR_CHECK

import docs
__doc__ = "%s%s" % (docs.long_description, docs.members)
__package_name__ = docs.package_name
__version__ = docs.version
__author__ = docs.author
__author_email__ = docs.author_email
__url__ = docs.url
__pa_version__ = "%s, Release %d" % (pa.Pa_GetVersionText(), pa.Pa_GetVersion())

err = pa.Pa_Initialize()
try:
    ERROR_CHECK(err)
except RuntimeError as e:
    raise ImportError(e.message)
