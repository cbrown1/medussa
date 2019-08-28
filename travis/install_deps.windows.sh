#! /usr/bin/env bash
set -eux
set -o pipefail

python -m pip install --upgrade --only-binary=numpy numpy

DLL2LIB="${PWD}/travis/dll2lib.cmd"

function install_dll {
    local bits="$1"
    local url="$2"
    local rename_as="$3"
    local dll_name="${url##*/}"
    local tmp_dir="temp.win${bits}"
    local stem
    stem=$( basename "${rename_as}" )

    mkdir -p "${tmp_dir}"
    pushd "${tmp_dir}"
    [[ -f "${dll_name}" ]] || wget "${url}"
    [[ -f "${dll_name}" ]] || { >&2 echo "failed to download ${tmp_dir}/${dll_name}" && return 1; }

    cp "${dll_name}" "${stem}.dll"
    dll_name="${stem}.dll"
    local lib_name="${stem}.lib"
    [[ -f "${lib_name}" ]] || "${DLL2LIB}" "${bits}" "${dll_name}"
    [[ -f "${lib_name}" ]] || { >&2 echo "failed to generate ${tmp_dir}/${lib_name}" && return 1; }
    popd
    mkdir -p "$( dirname "${rename_as}" )"
    cp "${tmp_dir}/${dll_name}" "${rename_as}.dll"
    cp "${tmp_dir}/${lib_name}" "${rename_as}.lib"
}

bits=$( python -c 'import struct; print(struct.calcsize("P") * 8)' )
lib_dir="src/lib"
mkdir -p "${lib_dir}"
install_dll "${bits}" \
    "https://github.com/bastibe/libsndfile-binaries/raw/master/libsndfile${bits}bit.dll" \
    "${lib_dir}/sndfile"
install_dll "${bits}" \
    "https://github.com/spatialaudio/portaudio-binaries/raw/master/libportaudio${bits}bit.dll" \
    "${lib_dir}/portaudio"
