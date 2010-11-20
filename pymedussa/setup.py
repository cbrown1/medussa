# -*- coding: utf-8 -*-
from distutils.core import setup
import platform

medussa_package = ['medussa']
medussa_package_dir = 'src/medussa'
medussa_requires = ['numpy (>=1.2)',]
medussa_package_data = ['*.py']
medussa_data_files = []

if platform.system() == "Windows":
	medussa_package_data.append('../../lib/medussa.dll')
	medussa_package_data.append('../../lib/portaudio_x86.dll')
	medussa_package_data.append('../../lib/libsndfile-1.dll')
else:
	medussa_data_files.append('lib/libmedussa.so')
	medussa_requires.append('portaudio (>=19.0)')
	medussa_requires.append('libsndfile (>=1.0)')

setup(name='medussa',
	version='1.0',
	description='Medussa: A cross-platform high-level audio library',
	author='Christopher A. Brown, Joseph K. Ranweiler',
	author_email='c-b /at/ asu.edu',
	maintainer = 'Christopher Brown',
	maintainer_email = 'c-b /at/ asu.edu',
	url='http://www.medussa.us',
	packages = medussa_package,
	requires = medussa_requires,
	package_dir={'medussa': medussa_package_dir},
	package_data={'medussa': medussa_package_data},
	data_files=[('lib',medussa_data_files)],
	keywords = 'sound audio wavplay',
	license = 'GPL',
	platforms = 'Win32, Linux',
	long_description="""\
 Medussa is a cross-platform, high-performance, high-level audio library
 based on Port Audio and libsndfile. You can play NumPy arrays, stream
 sound files from disk, or create pure tones, or 'on-line' white or pink
 noise. There are high-level functions like the playarr function (like
 matlab's wavplay) function. Medussa also allows you to access specific
 host api's or devices, create streams as needed, and control them all
 asynchronously. Or, for the most control, you can access the port audio
 library directly. Sweet!
 """,
	classifiers=[
		"License :: OSI Approved :: GNU General Public License (GPL)",
		"Programming Language :: Python",
		"Operating System :: Microsoft :: Windows",
		"Operating System :: POSIX",
		"Operating System :: MacOS :: MacOS X",
		"Programming Language :: Python",
		"Programming Language :: Python :: 2.5",
		"Programming Language :: Python :: 2.6",
		"Programming Language :: Python :: 2.7",
		"Environment :: Console",
		"Development Status :: 4 - Beta",
		"Intended Audience :: Developers",
		"Intended Audience :: Science/Research",
		"Topic :: Multimedia :: Sound/Audio",
		"Topic :: Scientific/Engineering",
	],
)
