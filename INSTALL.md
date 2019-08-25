# Medussa: A cross-platform high-level audio library for Python

## Build & Installation Instructions

### On Linux

#### Build Requirements

You will need the devel packages for python and numpy (>= 1.3). Header files for portaudio and libsndfile are included. 

### Runtime requirements

- [Python](http://www.python.org/). Any version >= 2.6. 
- [NumPy](http://numpy.scipy.org/). Version 1.3 or later should work.
- [Portaudio](http://www.portaudio.com/). Tested with v19, November 21, 2011, SVN rev 1788 (later versions should be fine).
- [Libsndfile](http://www.mega-nerd.com/libsndfile/). Tested with version 1.0.25, July 13 2011 (later versions should be fine).

#### Building and installing

(This is assuming that you have a compatible Python and NumPy installed)

From a command prompt, go the the Medussa directory and type:
 
```bash
python setup.py build && sudo python setup.py install
```


----

### On Windows
 
#### Building and installing

This Assumes that you have a compatible Python and NumPy (see above) installed, and you're using MSVC 10. You don't need libsndfile or Portaudio installed, as those dlls are included with Medussa.

From a command prompt, go the the Medussa directory and type:
 
```bash
python setup.py build
python setup.py install
```


#### Building the Medussa DLL

(this doesn't actually install anything, it only builds the lib):

- Open the Visual Studio 10 project at: \medussa\lib\build\win\msvc10\medusa.vcxproj
- Select the Solution Configuration according to the Python version you're using (eg., Release Py2.7).
- Choose Build Solution from the Build menu (or hit F7).
