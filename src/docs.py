# -*- coding: utf-8 -*-

# Copyright (c) 2010-2019 Christopher Brown
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

package_name = "medussa"
version = "2.0.2"
author = "Christopher A. Brown, Joseph K. Ranweiler, Ross Bencina, Ryan W. Moore, Kutay B. Sezginel, Andrzej Ciarkowski"
author_email = "cbrown1@pitt.edu"
maintainer = "Christopher Brown"
maintainer_email = "cbrown1@pitt.edu"
url = "https://github.com/cbrown1/medussa"
keywords = "sound audio signal wavplay"
license = "GNU General Public License v3 or later"
platforms = "Win32, Linux"
short_description = "Medussa: A cross-platform high-level audio library for Python"
long_description = """\
Medussa is an easy to use high-level cross-platform audio library for Python
based on Port Audio and libsndfile. You can play NumPy arrays, stream sound
files of various formats from disk, or create pure tones or 'on-line' white
or pink noise. There are high-level functions like play_array (similar to
matlab's old wavplay function but with more features). Or you can access 
specific host api's or devices, create streams as needed, and control them 
all asynchronously. Or for the most control, you can access the port audio 
library directly. Sweet!
"""

long_help = """\
 Methods
 -------
 play_array
    Plays a NumPy array with blocking, Matlab-style.
 play_file
    Plays a sound file on disk with blocking, Matlab-style.
 open_device
    Returns an object representing a audio device.
 open_default_device
    Returns an object representing the default audio device.
 print_available_devices
    Prints a list of available audio devices, with info.
 start_streams
    Tries to start playback of specified streams as synchronously as possible.
 read_file
    Reads a soundfile from disk into a NumPy array.
 write_file
    Writes a NumPy array to a soundfile on disk in a specified format.
 write_wav
    Writes a NumPy array to a soundfile on disk in wave format.
 write_flac
    Writes a NumPy array to a soundfile on disk in flac format (lossless).
 write_ogg
    Writes a NumPy array to a soundfile on disk in ogg vorbis format (lossy).

 Properties
 ----------
 __version__
    The Medussa library version.
 __pa_version__
    The Port Audio library version.
 __libsndfile_version__
    The libsndfile library version.

 Notes
 -----
 We have had some trouble with Port Audio on linux with respect to audio
 devices. There are two potential problems we are aware of.

 The first one has to do with the default audio device. We had been using
 the function Pa_GetDefaultOutputDevice when needed, but we experience
 mixer trouble on several machines we tested. We finally realized that that
 function was always returning the device whose index is 0, which is not
 always the system default device. So Medussa looks for a device whose name
 is 'default' and uses that, otherwise, use 0. The bottom line is that if you
 have trouble playing streams on the default device, look at what others are
 available, and try some. You might have better results with another one.

 The second problem is with the number of output channels, which is not
 always reported accurately by Port Audio. This seems to occur most
 frequently with the default device, although we have not confirmed this.
 On several machines we have tested, the maximum number of output channels
 is reported to be 32, or sometimes 128, when the actual value is 2. Thus,
 because we cannot trust this value, we have decided to set the device
 property out_channels to 2 (always), but we have made it settable, so
 if your device actually has more than 2 channels, you can set it. But
 again, you will have to do this manually.

 As of this release, there is no support for recording. We plan to add this
 in a future release.

"""
