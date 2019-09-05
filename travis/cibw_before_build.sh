#! /usr/bin/env bash
# Top-level script started by cibuildwheel to prepare build of python
# wheel for each build configuration
set -eux
set -o pipefail

# Some platform-independent deps first
# TODO cosider using requirements.txt for this?
python -m pip install --upgrade --only-binary=numpy numpy

# Linux is running in container, doesn't preserve env vars
OS_SCRIPT="travis/cibw_before_build.${TRAVIS_OS_NAME:-linux}.sh"
[[ ! -x "${OS_SCRIPT}" ]] || "./${OS_SCRIPT}"
