# -*- coding: utf-8 -*-

"""\
 Medussa is a cross-platform, high-performance, high-level audio library
 based on Port Audio and libsndfile. You can play NumPy arrays, stream
 sound files from disk, or create pure tones or 'on-line' white or pink
 noise. There are high-level functions like playarr (similar to matlab's
 wavplay). You can also access specific host api's or devices, create
 streams as needed, and control them all asynchronously. Or for the most
 control, you can access the port audio library directly. Sweet!

 Methods
 -------
 play_arr
    Plays a NumPy array with blocking, Matlab-style.
 play_file
    Plays a sound file on disk with blocking, Matlab-style.
open_device
    Returns an object representing a sound device.
open_default_device
    Returns an object representing the default sound device.
print_available_devices
    Prints a list of available sound devices, with info.
read_file
    Reads a soundfile from disk into a NumPy array.
write_file
    Writes a NumPy array to a soundfile on disk.
write_wav
    Writes a NumPy array to a wave file on disk.
write_flac
    Writes a NumPy array to a flac file on disk.
write_ogg
    Writes a NumPy array to a ogg vorbis file on disk.

Properties
----------
__version__
    The Medussa library version.
__pa_version__
    The Port Audio library version.

"""

from medussa import (play_arr, play_file, cmedussa, Device, ToneStream,
                     ArrayStream, SndfileStream, generate_hostapi_info,
                     generate_device_info, print_available_devices,
                     start_streams, open_device, open_default_device, init,
                     terminate, read_file,
                     write_file, write_wav, write_flac, write_ogg)
from portaudio import pa, ERROR_CHECK

__version__ = "1.0"
__pa_version__ = "%s, Release %d" % (pa.Pa_GetVersionText(), pa.Pa_GetVersion())

err = pa.Pa_Initialize()
try:
    ERROR_CHECK(err)
except RuntimeError as e:
    raise ImportError(e.message)
