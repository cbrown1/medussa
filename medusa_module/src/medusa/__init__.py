from medusa import cmedusa, Device, ContigArrayHandle, ArrayStream, ToneData, ToneStream, generateHostApiInfo, generateDeviceInfo, printAvailableDevices, start_streams, open_device, open_default_device, init, terminate
from portaudio import pa, ERROR_CHECK

err = pa.Pa_Initialize()
try:
    ERROR_CHECK(err)
except RuntimeError as e:
    raise ImportError(e.message)