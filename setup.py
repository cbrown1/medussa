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

from distutils.core import setup
from distutils.sysconfig import get_python_lib
import os
import platform
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
medussa_requires = ['numpy (>=1.2)',]

if platform.system() == "Windows":
    medussa_data_files.append('lib/build/win/py%s/medussa.dll' % pyver)
    medussa_data_files.append('lib/build/win/portaudio_x86.dll')
    medussa_data_files.append('lib/build/win/libsndfile-1.dll')
    medussa_data_files_path = os.path.join(get_python_lib(prefix=''), 'medussa')
else:
    medussa_data_files.append('lib/build/linux/py%s/libmedussa.so' % pyver)
    medussa_data_files_path = os.path.join(get_python_lib(), 'medussa')

setup(name=docs.package_name,
    version=docs.version,
    description=docs.short_description,
    author=docs.author,
    author_email=docs.author_email,
    maintainer = docs.maintainer,
    maintainer_email = docs.maintainer_email,
    url=docs.url,
    packages = medussa_package,
    requires = medussa_requires,
    package_dir={docs.package_name: medussa_package_dir},
    package_data={docs.package_name: medussa_package_data},
    data_files=[(medussa_data_files_path, medussa_data_files)],
    keywords = docs.keywords,
    license = docs.license,
    platforms = docs.platforms,
    long_description = docs.long_description,
    classifiers=[
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        #"Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.1",
        "Programming Language :: Python :: 3.2",
        "Environment :: Console",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
    ],
)
