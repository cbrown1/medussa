from medusa import cmedusa, Device, ContigArrayHandle, ArrayStream, ToneData, ToneStream, generateHostApiInfo, generateDeviceInfo, printAvailableDevices, start_streams, open_device, open_default_device
from portaudio import pa

pa.Pa_Initialize()