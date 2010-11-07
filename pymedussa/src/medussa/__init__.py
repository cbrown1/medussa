# -*- coding: utf-8 -*-
#from medussa import playarr, playfile, cmedussa, Device, ContigArrayHandle, ArrayStream, ToneData, ToneStream, SndfileStream, SndfileData, generateHostApiInfo, generateDeviceInfo, printAvailableDevices, start_streams, open_device, open_default_device, init, terminate
from medussa import playarr, playfile, cmedussa, Device, ToneStream, ArrayStream, SndfileStream, generateHostApiInfo, generateDeviceInfo, printAvailableDevices, start_streams, open_device, open_default_device, init, terminate
from portaudio import pa, ERROR_CHECK

__version__ = "1.0"
__pa_version__ = "Release %d, %s" % (pa.Pa_GetVersion(), pa.Pa_GetVersionText())

err = pa.Pa_Initialize()
try:
    ERROR_CHECK(err)
except RuntimeError as e:
    raise ImportError(e.message)