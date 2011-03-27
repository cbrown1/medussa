# -*- coding: utf-8 -*-
from distutils.core import setup
import platform
from distutils.sysconfig import get_python_lib
from os.path import join

pyver = "%s.%s" % (platform.python_version_tuple()[0], platform.python_version_tuple()[1])

medussa_package = ['medussa']
medussa_package_dir = 'src'
medussa_package_data = ['*.py']
medussa_data_files = []
medussa_data_files_path = ''
medussa_requires = ['numpy (>=1.2)',]

if platform.system() == "Windows":
	medussa_data_files.append('lib/build/win/py%s/medussa.dll' % pyver)
	medussa_data_files.append('lib/build/win/portaudio_x86.dll')
	medussa_data_files.append('lib/build/win/libsndfile-1.dll')
	medussa_data_files_path = join(get_python_lib(prefix=''), 'medussa')
else:
	medussa_data_files.append('lib/build/linux/py%s/libmedussa.so' % pyver)
	medussa_data_files_path = join(get_python_lib(prefix='/usr/local'), 'medussa')

#TODO: Pull version, author, etc., from imported medussa
setup(name='medussa',
	version='1.0',
	description='Medussa: A cross-platform high-level audio library for Python',
	author='Christopher A. Brown, Joseph K. Ranweiler',
	author_email='c-b /at/ asu.edu',
	maintainer = 'Christopher Brown',
	maintainer_email = 'c-b /at/ asu.edu',
	url='http://www.medussa.us',
	packages = medussa_package,
	requires = medussa_requires,
	package_dir={'medussa': medussa_package_dir},
	package_data={'medussa': medussa_package_data},
	data_files=[(medussa_data_files_path, medussa_data_files)],
	keywords = 'sound audio wavplay',
	license = 'GPL',
	platforms = 'Win32, Linux',
	long_description="""\
 Medussa is a cross-platform high-level audio library for Python
 based on Port Audio and libsndfile. You can play NumPy arrays, stream
 sound files from disk, or create pure tones or 'on-line' white or pink
 noise. There are high-level functions like playarr (similar to matlab's
 wavplay). You can also access specific host api's or devices, create
 streams as needed, and control them all asynchronously. Or for the most
 control, you can access the port audio library directly. Sweet!
 """,
	classifiers=[
		"License :: OSI Approved :: GNU General Public License (GPL)",
		"Programming Language :: Python",
		"Operating System :: Microsoft :: Windows",
		"Operating System :: POSIX",
		"Operating System :: MacOS :: MacOS X",
		"Programming Language :: Python",
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
