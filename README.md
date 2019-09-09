[![Build Status](https://travis-ci.org/cbrown1/medussa.svg?branch=master)](https://travis-ci.org/cbrown1/medussa)

# Medussa: A cross-platform high-level audio library for Python

## About

Medussa is an easy to use high-level cross-platform audio library for Python
based on [Port Audio](http://www.portaudio.com/) and [libsndfile](http://www.mega-nerd.com/libsndfile/). You can play [NumPy](https://numpy.org/) arrays, stream sound
files of various formats from disk, or create pure tones or 'on-line' white
or pink noise. There are high-level functions like play_array (similar to
matlab's old [wavplay](https://www.mathworks.com/matlabcentral/fileexchange/71798-wavplay/) function but with more features). Or you can access specific host api's or devices, create
streams as needed, and control them all asynchronously. Or for the most
control, you can access the port audio library directly. Sweet!

Medussa runs on Linux, Windows, and OS X platforms, and Python 2 and 3. 

- Current version: 2.0.1
- [License](COPYING.md)
- [Website](https://github.com/cbrown1/medussa)
- [Version History](NEWS.md)
- [Authors](AUTHORS.md)
- [Installation instructions](INSTALL.md)
- [Usage examples](USAGE.md)


## Copyright & License

*Copyright (c) 2010-2019 Christopher A. Brown*

Medussa is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Medussa is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

