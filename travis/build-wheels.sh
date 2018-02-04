#!/bin/bash
set -e -x

# Install system packages required by our library
wget http://www.portaudio.com/archives/pa_stable_v19_20140130.tgz && tar -xzvf pa_stable_v19_20140130.tgz && cd portaudio && ./configure && make && make install && cd ..
wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.27.tar.gz && tar -xzvf libsndfile-1.0.27.tar.gz && cd libsndfile-1.0.27 && ./configure && make && make install && cd ..

## Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [ "$PYBIN" != "/opt/python/cp33-cp33m/bin" ]; then
        "${PYBIN}/pip" install -U --only-binary=numpy numpy
        "${PYBIN}/pip" wheel /io/ -w /wheelhouse
    fi
done

find /wheelhouse -name 'numpy*' -exec rm {} \;

mkdir /wheelhouse/manylinux

for whl in /wheelhouse/*.whl; do
    # Bundle external shared libraries into the wheels
    auditwheel repair "$whl" -w /wheelhouse/manylinux
done

"${PYBIN}/pip" install twine
"${PYBIN}/twine" upload -u $PYPI_USER -p $PYPI_PASS /wheelhouse/manylinux/*.whl

## Install packages and test
#for PYBIN in /opt/python/*/bin; do
#    if [ "$PYBIN" != "/opt/python/cp33-cp33m/bin" ]; then
#        "${PYBIN}/pip" install medussa --no-index -f wheelhouse
#    fi
##    (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
#done

#git config --global user.email "cbrown1@pitt.edu"
#git config --global user.name "cbrown1"
