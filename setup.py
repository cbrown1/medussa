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

from setuptools import setup, Extension
from distutils.sysconfig import get_python_lib
import glob
import os
import platform
import distutils
from distutils.command.build_ext import build_ext
import numpy
import sys
import sysconfig

pymaj = platform.python_version_tuple()[0]
pymin = platform.python_version_tuple()[1]
pyver = "{}.{}".format(pymaj, pymin)

sys.path.insert(0,os.path.abspath(r'./src'))
docs =  __import__('docs', fromlist=['package_name', 'version', 'url',
                    'author', 'author_email', 'long_help',
                    'short_description', 'long_description', 'maintainer',
                    'maintain_email', 'keywords', 'platforms'])
del sys.path[0]

medussa_package = [docs.package_name]
medussa_package_dir = 'src'
medussa_package_data = ['*.py']
medussa_data_files = ['symbols.lst']
medussa_data_files_path = 'medussa'
medussa_install_requires = ['numpy >=1.3']
medussa_requires = ['numpy (>=1.3)',]
medussa_setup_requires = ['numpy >=1.3']

library_dirs = []
libraries = ['portaudio', 'sndfile']

_DEBUG = False
if "--debug" in sys.argv:
    _DEBUG = True
    sys.argv.remove("--debug")

if platform.system() == "Windows":
    # install_deps.windows.sh puts correct arch libs in there
    library_dirs.append('src/lib')
    libraries.append('advapi32')
    medussa_package_data.append('lib/*.dll')
    # TODO support debug builds
    extra_compile_args = []
else:
    medussa_data_files_path = os.path.join(get_python_lib(), 'medussa')
    extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
    extra_compile_args += ["-Wall", "-Wextra"]
    if _DEBUG:
        extra_compile_args += ["-ggdb3", "-O0", "-UNDEBUG"]
    else:
        extra_compile_args += ["-DNDEBUG", "-O3"]

def get_exported_symbols():
    return [l.strip() for l in open('symbols.lst')]


cmedussa = Extension('.'.join([docs.package_name, 'libmedussa']),
    include_dirs=[numpy.get_include(), 'lib', os.path.join('lib', 'include')],
    libraries=libraries,
    library_dirs=library_dirs,
    language="c++",
    export_symbols=get_exported_symbols(),
    extra_compile_args=extra_compile_args,
    sources=glob.glob(os.path.join('lib', 'src', '*.c')) +
            glob.glob(os.path.join('lib', 'src', '*.cpp'))
    )

setup(name=docs.package_name,
    version=docs.version,
    description=docs.short_description,
    author=docs.author,
    author_email=docs.author_email,
    maintainer = docs.maintainer,
    maintainer_email = docs.maintainer_email,
    url=docs.url,
    packages = medussa_package,
    include_package_data=True,
    install_requires = medussa_install_requires,
    setup_requires = medussa_setup_requires,
    requires = medussa_requires,
    eager_resources = ['setup.lst'],
    package_dir={docs.package_name: medussa_package_dir},
    package_data={docs.package_name: medussa_package_data},
    data_files=[(medussa_data_files_path, medussa_data_files)],
    keywords = docs.keywords,
    license = docs.license,
    platforms = docs.platforms,
    long_description = docs.long_description,
    ext_modules = [cmedussa],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        # "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Environment :: Console",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Mixers",
        "Topic :: Multimedia :: Sound/Audio :: Players",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Natural Language :: English",
        ],
)
