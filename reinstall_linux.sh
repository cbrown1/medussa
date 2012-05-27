#!/bin/bash

set -e

pyver=$1
if [ "$pyver" == "" ]; then
	echo "Usage: reinstall_linux.sh pyver # where pyver ~= 2.7";
	exit
fi

# Build lib
cd ./lib/build/linux
if [ -d py$pyver ]; then
    rm -r py$pyver
fi
./build.sh $pyver
cd ../../..

installdir=$(python${pyver} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

if [ -d ${installdir}/medussa ]; then
    sudo rm -r ${installdir}/medussa
fi
if [ -e ${installdir}/medussa-*.egg-info ]; then
    sudo rm ${installdir}/medussa-*.egg-info
fi
if [ -d build ]; then
    sudo rm -r build
fi
python${pyver} setup.py build
sudo python${pyver} setup.py install

