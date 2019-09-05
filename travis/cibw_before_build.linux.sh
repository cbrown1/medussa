#! /usr/bin/env bash
set -eux
set -o pipefail

# XXX cibuildwheel docs state we can use yum to install deps in manylinux as it's CentOS
yum -y install \
    libsndfile-devel \
    portaudio-devel
