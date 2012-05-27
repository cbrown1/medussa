#!/bin/bash

set -e

release=1
arch="i386"; #"all"; #"i386"; #'amd64'
dist="el6"

pyver=$1
pyvermaj=${pyver:0:1}
pyvermin=${pyver:2:3}

pydist="py${pyvermaj}${pyvermin}-opt"

if [ "$pyver" == "" ]; then
	echo "Usage: package_linux_rpm.sh pyver # where pyver ~= 2.7";
	exit
fi

pybin="python${pyver}";

# Get metadata
ver=$(${pybin} setup.py --version);
package=$(${pybin} setup.py --name);
maintainer=$(${pybin} setup.py --maintainer);
maintaineremail=$(${pybin} setup.py --maintainer-email);

# Build lib
cd ./lib/build/linux
if [ -d py$pyver ]; then
    rm -r py$pyver
fi
./build.sh $pyver
cd ../../..

if [ -d build ]; then
    sudo rm -r build
fi
python${pyver} setup.py build
sudo python${pyver} setup.py bdist_rpm --fix-python --binary-only --force-arch=${arch} --no-autoreq --distribution-name=${dist} --release=${release} --packager="${maintainer} <${maintaineremail}>" 

mv -f dist/${package}-${ver}-${release}.${arch}.rpm dist/${package}-${ver}-${release}.${pydist}.${dist}.${arch}.rpm

