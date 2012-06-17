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
sdescription=$(${pybin} setup.py --description);
ldescription=$(${pybin} setup.py --long-description);
long_description="${sdescription}\n\n${ldescription}\n\nThis is a custom build of Medussa to /opt/python${pyver}\n";

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
sudo python${pyver} setup.py bdist_rpm --fix-python --binary-only --force-arch=${arch} --no-autoreq --distribution-name=${dist} --release=${release} --vendor="${maintainer} <${maintaineremail}>"

mv -f dist/${package}-${ver}-${release}.${arch}.rpm dist/${package}-${ver}-${release}.${pydist}.${dist}.${arch}.rpm

