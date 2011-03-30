# -*- coding: utf-8 -*-

package_name = "medussa"
version = "1.0"
author = "Christopher A. Brown, Joseph K. Ranweiler"
author_email = "c-b /at/ asu.edu"
maintainer = 'Christopher Brown'
maintainer_email = 'c-b /at/ asu.edu'
url = "http://www.medussa.us"
keywords = 'sound audio wavplay'
license = 'GPL'
platforms = 'Win32, Linux'
short_description = "Medussa: A cross-platform high-level audio library for Python"
long_description = """\
 Medussa is a cross-platform high-level audio library for Python
 based on Port Audio and libsndfile. You can play NumPy arrays, stream
 sound files from disk, or create pure tones or 'on-line' white or pink
 noise. There are high-level functions like playarr (similar to matlab's
 wavplay). You can also access specific host api's or devices, create
 streams as needed, and control them all asynchronously. Or for the most
 control, you can access the port audio library directly. Sweet!

 """

members = """\
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
start_streams
    Tries to start playback of specified streams as synchronously as possible.
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
