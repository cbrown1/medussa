# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.sysconfig import get_python_lib
import os
import platform
import sys

pymaj = platform.python_version_tuple()[0]
pymin = platform.python_version_tuple()[1]
pyver = "%s.%s" % (pymaj, pymin)

sys.path.append('src')
if pymaj == "2":
    from docs import (package_name, version, url, author, author_email,
                        long_help, short_description, long_description, 
						maintainer, maintainer_email, keywords, platforms, 
						)
else:
    from .docs import (package_name, version, url, author, author_email,
                        long_help, short_description, long_description,
						maintainer, maintainer_email, keywords, platforms, 
						)

medussa_package = [package_name]
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
    medussa_data_files_path = os.path.join(get_python_lib(prefix='/usr/local'), 'medussa')

setup(name=package_name,
    version=version,
    description=short_description,
    author=author,
    author_email=author_email,
    maintainer = maintainer,
    maintainer_email = maintainer_email,
    url=url,
    packages = medussa_package,
    requires = medussa_requires,
    package_dir={package_name: medussa_package_dir},
    package_data={package_name: medussa_package_data},
    data_files=[(medussa_data_files_path, medussa_data_files)],
    keywords = keywords,
    license = license,
    platforms = platforms,
    long_description = long_description,
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
