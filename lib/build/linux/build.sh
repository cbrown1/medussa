#!/bin/bash
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
		    pythonIncludeDirectory=/opt/python${PYVER}/include/
			if [ ! -d "$pythonIncludeDirectory" ]; then
			    echo "Can't find python include folder!"
			    exit 1
			fi
        fi
    fi
fi

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

make FLAGS="-I$pythonIncludeDirectory -I$numpyIncludeDirectory"
if [ $? -eq 0 ] ; then
  mkdir ./py${PYVER}
  cp libmedussa.so ./py${PYVER}
  exit 0
else
  echo "Building libmedussa.so failed!"
  exit 1
fi
