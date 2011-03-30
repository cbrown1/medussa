#!/bin/bash

set -e

pyver=$1
if [ "$pyver" == "" ]; then
	echo "Usage: reinstall_linux.sh pyver # where pyver ~= 2.7";
	exit
fi

# Build lib
cd ./lib/build/linux
./build.sh $pyver
cd ../../..

installdir=$(python${pyver} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(prefix='/usr/local'))")

sudo rm -r ${installdir}/medussa
sudo rm ${installdir}/medussa-*.egg-info
sudo python${pyver} setup.py install --prefix='/usr/local'
