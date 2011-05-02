# -*- coding: utf-8 -*-

import platform
pymaj = platform.python_version_tuple()[0]
pymin = platform.python_version_tuple()[1]
pyver = "%s.%s" % (pymaj, pymin)
if pymaj == "2":
    from medussa import (play_array, play_file, read_file, 
                        write_file, write_wav, write_flac, write_ogg, 
                        Stream, ToneStream, PinkStream, WhiteStream, 
						FiniteStream, ArrayStream, SndfileStream, 
						get_default_output_device_index, 
						get_default_input_device_index, 
						Device, open_device, open_default_device, 
                        generate_hostapi_info, generate_device_info, 
						get_available_devices, print_available_devices, 
                        device_instances, stream_instances, start_streams, 
						cmedussa, init, terminate, 
                        sf_formats, 
                        )
    from docs import (package_name, version, url, author, author_email,
                        long_help, short_description, long_description)
    from portaudio import pa, ERROR_CHECK
else:
    from .medussa import (play_array, play_file, read_file, 
                        write_file, write_wav, write_flac, write_ogg, 
                        Stream, ToneStream, PinkStream, WhiteStream, 
						FiniteStream, ArrayStream, SndfileStream, 
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
    from .portaudio import pa, ERROR_CHECK

__doc__ = "%s\n\n%s" % (long_description, long_help)
__package_name__ = package_name
__version__ = version
__description__ = short_description
__author__ = author
__author_email__ = author_email
__url__ = url
__pa_version__ = "%s, Release %d" % (pa.Pa_GetVersionText(), pa.Pa_GetVersion())
del(package_name, version, url, author, author_email, long_help, short_description, long_description)

err = pa.Pa_Initialize()
try:
    ERROR_CHECK(err)
except RuntimeError as e:
    raise ImportError(e.message)
