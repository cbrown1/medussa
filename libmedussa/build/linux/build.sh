#!/bin/bash
PYVER=$1
if [ -z "$PYVER" ]; then
  PYMAJ=`python -c "import platform; print(platform.python_version_tuple()[0])"`
  PYMIN=`python -c "import platform; print(platform.python_version_tuple()[1])"`
  PYVER="${PYMAJ}.${PYMIN}"
fi

SITEPACKAGES=`python${PYVER} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"`

make clean
make PYVER=${PYVER} FLAGS="-I/usr/include/python${PYVER} -I${SITEPACKAGES}/numpy/core/include/"
if [ $? -eq 0 ] ; then
  mkdir ../../../pymedussa/lib/py${PYVER}
  cp libmedussa.so ../../../pymedussa/lib/py${PYVER}
else
  echo "Nope!"
fi
