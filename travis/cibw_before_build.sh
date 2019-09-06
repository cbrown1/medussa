#! /usr/bin/env bash
# Top-level script started by cibuildwheel to prepare build of python
# wheel for each build configuration
set -eux
set -o pipefail

python -m pip install -r requirements.txt

OS_SCRIPT="travis/cibw_before_build.${TRAVIS_OS_NAME}.sh"
[[ ! -x "${OS_SCRIPT}" ]] || "./${OS_SCRIPT}"
