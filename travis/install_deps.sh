#! /usr/bin/env bash
set -eux
set -o pipefail

if [[ "${TRAVIS_OS_NAME:-}" != osx ]]
then
    # XXX cibuildwheel docs state we can use yum to install deps in manylinux as it's CentOS
    yum -y install \
        libsndfile-devel \
        portaudio-devel
fi

pip install -U --only-binary=numpy numpy
