# -*- coding: utf-8 -*-

# Copyright (c) 2010-2012 Christopher Brown
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

from setuptools import setup, Extension
from distutils.sysconfig import get_python_lib
import glob
import os
import platform
import distutils
from distutils.command.build_ext import build_ext
import sys

pymaj = platform.python_version_tuple()[0]
pymin = platform.python_version_tuple()[1]
pyver = "%s.%s" % (pymaj, pymin)

sys.path.insert(0,os.path.abspath(r'./src'))
docs =  __import__('docs', fromlist=['package_name', 'version', 'url', 
                    'author', 'author_email', 'long_help', 
                    'short_description', 'long_description', 'maintainer', 
                    'maintain_email', 'keywords', 'platforms'])
del sys.path[0]

medussa_package = [docs.package_name]
medussa_package_dir = 'src'
medussa_package_data = ['*.py']
medussa_data_files = []
medussa_data_files_path = ''
medussa_install_requires = ['numpy >=1.3']
medussa_requires = ['numpy (>=1.3)',]
medussa_setup_requires = ['numpy >=1.3']

library_dirs = []
libraries = ['portaudio', 'sndfile']

if platform.system() == "Windows":
    medussa_data_files.append('lib/build/win/py%s/medussa.dll' % pyver)
    medussa_data_files.append('lib/build/win/portaudio_x86.dll')
    medussa_data_files.append('lib/build/win/libsndfile-1.dll')
    medussa_data_files_path = os.path.join(get_python_lib(prefix=''), 'medussa')
    library_dirs.append('./lib/lib')

    libraries.append('advapi32')
else:
    medussa_data_files_path = os.path.join(get_python_lib(), 'medussa')

class build_medussa_c_extension(build_ext):
    """CURRENTLY BROKEN: setuptools will automatically download and build
    numpy, but then the numpy get_include call will fail. Be sure to install
    numpy first.
    
    This seems unnecessarily complex, but: We want to add
    numpy.get_include() as an include directory for the libmedussa extension.
    But, numpy might not be installed at the time that the Extension object is
    constructed. Instead, we delay adding numpy's include directory until right
    before we build the Extension. By the time the Extension is built,
    setuptools will have ensured that numpy is installed (via setup_requires).
    This code will then be able to import numpy and add its include directory
    to the include directory search paths."""

    def run(self):
        if (self.include_dirs is not None):
            try:
                import numpy as np
                self.include_dirs.append(np.get_include())
            except ImportError as e:
                raise distutils.errors.DistutilsSetupError(
                    "Error occured finding numpy include directory. " +\
                    "Be sure to install numpy first. "+\
                    "This should not occur: " + str(e))

        #Continue with the build as normal.
        build_ext.run(self)


cmedussa = Extension('.'.join([docs.package_name, 'libmedussa']), 
    include_dirs=['lib', os.path.join('lib', 'include')],
    libraries=libraries,
    library_dirs=library_dirs,
    sources=glob.glob(os.path.join('lib', 'src', '*.c')))

setup(name=docs.package_name,
    version=docs.version,
    description=docs.short_description,
    author=docs.author,
    author_email=docs.author_email,
    maintainer = docs.maintainer,
    maintainer_email = docs.maintainer_email,
    url=docs.url,
    packages = medussa_package,
    cmdclass=dict(build_ext=build_medussa_c_extension),
    install_requires = medussa_install_requires,
    setup_requires = medussa_setup_requires,
    requires = medussa_requires,
    package_dir={docs.package_name: medussa_package_dir},
    package_data={docs.package_name: medussa_package_data},
    data_files=[(medussa_data_files_path, medussa_data_files)],
    keywords = docs.keywords,
    license = docs.license,
    platforms = docs.platforms,
    long_description = docs.long_description,
    ext_modules = [cmedussa],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        #"Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        #"Programming Language :: Python :: 3.1",
        "Programming Language :: Python :: 3.2",
        "Environment :: Console",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
    ],
)
