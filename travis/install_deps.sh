#! /usr/bin/env bash
set -eux
set -o pipefail

case "${TRAVIS_OS_NAME:-}" in
osx)
    # XXX watch out for travis "homebrew" addon failing the update
    # TODO when brew install starts failing, remove HOMEBREW_NO_AUTO_UPDATE=1,
    # this will slow down the build but it will work
    HOMEBREW_NO_AUTO_UPDATE=1 brew install \
        libsndfile \
        portaudio
    ;;
# TRAVIS_OS_NAME is missing in linux-docker
*)
    # XXX cibuildwheel docs state we can use yum to install deps in manylinux as it's CentOS
    yum -y install \
        libsndfile-devel \
        portaudio-devel
    ;;
esac

pip install -U --only-binary=numpy numpy
