# -*- coding: utf-8 -*-

"""\
 Medussa is a cross-platform, high-performance, high-level audio library
 based on Port Audio and libsndfile. You can play NumPy arrays, stream
 sound files from disk, or create pure tones or 'on-line' white or pink
 noise. There are high-level functions like playarr (similar to matlab's
 wavplay). You can also access specific host api's or devices, create
 streams as needed, and control them all asynchronously. Or for the most
 control, you can access the port audio library directly. Sweet!
 """

#from medussa import playarr, playfile, cmedussa, Device, ContigArrayHandle, ArrayStream, ToneData, ToneStream, SndfileStream, SndfileData, generateHostApiInfo, generateDeviceInfo, printAvailableDevices, start_streams, open_device, open_default_device, init, terminate
from medussa import (play_arr, play_file, cmedussa, Device, ToneStream, 
                    ArrayStream, SndfileStream, generate_hostapi_info, 
                    generate_device_info, print_available_devices, 
                    start_streams, open_device, open_default_device, init, 
                    terminate, read_file, write_file, write_wav)
from portaudio import pa, ERROR_CHECK

__version__ = "1.0"
__pa_version__ = "%s, Release %d" % (pa.Pa_GetVersionText(), pa.Pa_GetVersion())

err = pa.Pa_Initialize()
try:
    ERROR_CHECK(err)
except RuntimeError as e:
    raise ImportError(e.message)
