#!/bin/bash

set -e

# deb-specific variables. Set as needed
architecture="i386"; #"all"; #"i386"; #'amd64'
dependencies="libportaudio2, libsndfile1, python-numpy (>= 1.2)" # Use '' for none. delimit with a comma.
section='python';  # Most python packages seem to go here

pyver=$1
if [ "$pyver" == "" ]; then
	echo "Usage: package_linux.sh pyver # where pyver ~= 2.7";
	exit
fi

# Build lib
cd ./lib/build/linux
./build.sh $pyver
cd ../../..

pybin="python${pyver}";

# Get metadata
ver=$(${pybin} setup.py --version);
package=$(${pybin} setup.py --name);
required="${pybin} (= ${pyver}), ${dependencies}"
#req=$(${pybin} setup.py --requires);    # Look for python-specific requires
#if [ -n "$req" ]; then                  # prepend with 'python-' and add to list
#	required=$(echo "${required}, python-${req}" | sed -n '1h;2,$H;${g;s/\n/, python-/g;p}');
#fi
author=$(${pybin} setup.py --author);
authoremail=$(${pybin} setup.py --author-email);
maintainer=$(${pybin} setup.py --maintainer);
maintaineremail=$(${pybin} setup.py --maintainer-email);
sdescription=$(${pybin} setup.py --description);      # Build deb description from py short desc
ldescription=$(${pybin} setup.py --long-description); # and py long desc
url=$(${pybin} setup.py --url);
description=$(echo -e "${ldescription}"); # Add url to end of desc if present
if [ "$url" != "UNKNOWN" ]; then
    description=$(echo -e "${description}\n .\n ${url}");
fi

base="${section}-${package}_${ver}_py${pyver}_${architecture}"
debname="${base}.deb"
rpmname="${base}.rpm"

# Clean
#rm -r build
#rm ./docs/*~
${pybin} setup.py clean

# Build
${pybin} setup.py build

##
# Create rpm [broken]
#TODO: Add the custom description
#${pybin} setup.py bdist_rpm --fix-python --force-arch=${architecture} --group="python" --binary-only --dist-dir="./dist/${rpmname}"


####
## Create deb

# Fake install
${pybin} setup.py install --root dist/deb
cd dist

# Get installed size
fss=$(du -s deb)
set -- $fss
fs=$1

cd deb

# Create Deb control file
mkdir -p DEBIAN
control="./DEBIAN/control"
touch "${control}" 2> /dev/null
if [ -e "${control}" ] ; then
  echo "Package: python-${package}" > "${control}"
  echo "Version: ${ver}" >> "${control}"
  if [ "$url" != "UNKNOWN" ]; then
    echo "Homepage: ${url}" >> "${control}"
  fi
  echo "Maintainer: ${maintainer} <${maintaineremail}>" >> "${control}"
  echo "Installed-Size: ${fs}" >> "${control}"
  echo "Section: ${section}" >> "${control}"
  echo "Architecture: ${architecture}" >> "${control}"
  echo "Depends: ${required}" >> "${control}"
  echo "Description: ${description}" >> "${control}"
fi

# Create md5sum file
md5sum `find . -type f | grep -v '^[.]/DEBIAN/'  | sed -e 's/.\///'` >DEBIAN/md5sums

cd ..

## Copy documentation
#mkdir -p deb/usr/share/doc/$package
#cp ../docs/* deb/usr/share/doc/$package

dpkg-deb -b deb ${debname}

rm -r deb
cd ..

# Clean
#rm -r build
#rm -r ${package}.egg-info

# Make source & egg distributions
#python setup.py sdist --formats zip,gztar
#python setup.py bdist_egg

# Upload
#ftp -n $hostname <<EOF
#quote USER $username
#quote PASS $password
#prompt
#passive
#binary
#lcd ./dist
#cd /public_html/packages
#put ${deb}
#quit
#EOF
