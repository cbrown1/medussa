#!/bin/bash

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

PYVER=$1
if [ -z "$PYVER" ]; then
  PYMAJ=`python -c "import platform; print(platform.python_version_tuple()[0])"`
  PYMIN=`python -c "import platform; print(platform.python_version_tuple()[1])"`
  PYVER="${PYMAJ}.${PYMIN}"
  echo "Building libmedussa.so for default python version (${PYVER})"
fi

make clean

pythonIncludeDirectory=/usr/include/python${PYVER}/
if [ ! -d "$pythonIncludeDirectory" ]; then
    pythonIncludeDirectory=/usr/local/include/python${PYVER}/
    if [ ! -d "$pythonIncludeDirectory" ]; then
        pythonIncludeDirectory=/usr/local/include/python${PYVER}m/
        if [ ! -d "$pythonIncludeDirectory" ]; then
		    pythonIncludeDirectory=/opt/python${PYVER}/include/python${PYVER}/
			if [ ! -d "$pythonIncludeDirectory" ]; then
		        pythonIncludeDirectory=/opt/python${PYVER}/include/python${PYVER}m/
			    if [ ! -d "$pythonIncludeDirectory" ]; then
			        echo "Can't find python include folder!"
			        exit 1
                fi
			fi
        fi
    fi
fi

numpyIncludeDirectory=$(python${PYVER} -c "from numpy import distutils; print(distutils.misc_util.get_numpy_include_dirs()[0])")
if [ ! -d "$numpyIncludeDirectory" ]; then
    numpyIncludeDirectory=/usr/lib/python${PYVER}/site-packages/numpy/core/include/
    if [ ! -d "$numpyIncludeDirectory" ]; then
        numpyIncludeDirectory=/usr/local/lib/python${PYVER}/site-packages/numpy/core/include/
        if [ ! -d "$numpyIncludeDirectory" ]; then
		    numpyIncludeDirectory=/opt/python${PYVER}/lib/python${PYVER}/site-packages/numpy/core/include/
		    if [ ! -d "$numpyIncludeDirectory" ]; then
		        echo "Can't find numpy include folder!"
		        exit 1
		    fi
        fi
    fi
fi

sharedLibIncludeDirectory=../../include

make FLAGS="-I$sharedLibIncludeDirectory -I$pythonIncludeDirectory -I$numpyIncludeDirectory"
if [ $? -eq 0 ] ; then
  mkdir ./py${PYVER}
  cp libmedussa.so ./py${PYVER}
  exit 0
else
  echo "Building libmedussa.so failed!"
  exit 1
fi
