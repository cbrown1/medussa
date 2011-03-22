#!/bin/bash

pyver=$1
if [ "$pyver" == "" ]; then
	echo "Usage: reinstall_linux.sh pyver # where pyver ~= 2.7";
	exit
fi

sudo rm /usr/local/lib/libmedussa.so
sudo rm -r /usr/local/lib/python${pyver}/dist-packages/medussa

sudo python${pyver} setup.py install
