#! /usr/bin/env bash
set -eux
set -o pipefail

function safe_download {
    local url="$1"
    local arch_name="${url##*/}"
    [[ -f "tmp/download/${arch_name}" ]] || {
        mkdir -p tmp/download/unverified
        >&2 pushd tmp/download/unverified
        wget "${url}"
        [[ -f "${arch_name}" ]] || {
            >&2 echo "failed to download ${arch_name}"
            return 1
        }
        >&2 popd
        local url_no_scheme="${url#*://}"
        local exp_sum
        read -r exp_sum < "deps/${url_no_scheme}.md5"
        exp_sum="${exp_sum//$'\r'/}"
        local down_sum
        down_sum=$( md5sum < "tmp/download/unverified/${arch_name}" | cut -d' ' -f1 )
        [[ "${exp_sum}" == "${down_sum}" ]] || {
            rm "tmp/download/unverified/${arch_name}"
            >&2 echo "${url_no_scheme} md5 checksum mismatch, aborting"
            return 1
        }
        mv "tmp/download/unverified/${arch_name}" tmp/download
    }
    echo "tmp/download/${arch_name}"
}

function extract_dll {
    local arch_path="$1"
    local bits="$2"
    local dll_path
    dll_path=$( tar -tf "${arch_path}" | grep "${bits}bit.dll" )
    [[ -f "tmp/${dll_path}" ]] || {
        tar -x -C tmp -f "${arch_path}" "${dll_path}"
        [[ -f "tmp/${dll_path}" ]] || {
            >&2 echo "failed to extract ${dll_path}"
            return 1
        }
    }
    echo "tmp/${dll_path}"
}

DLL2LIB="${PWD}/travis/dll2lib_vcrun.cmd"

function install_dll {
    local bits="$1"
    local url="$2"
    local rename_as="$3"
    local arch_path
    arch_path=$( safe_download "${url}" )
    local dll_path
    dll_path=$( extract_dll "${arch_path}" "${bits}" )
    local tmp_dir="tmp/win${bits}"
    mkdir -p "${tmp_dir}"
    local stem
    stem=$( basename "${rename_as}" )
    cp "${dll_path}" "${tmp_dir}/${stem}.dll"
    >&2 pushd "${tmp_dir}"
    local dll_name="${stem}.dll"
    local lib_name="${stem}.lib"
    [[ -f "${lib_name}" ]] || "${DLL2LIB}" "${bits}" "${dll_name}"
    [[ -f "${lib_name}" ]] || {
        >&2 echo "failed to generate ${tmp_dir}/${lib_name}"
        return 1
    }
    >&2 popd
    mkdir -p "$( dirname "${rename_as}" )"
    cp "${tmp_dir}/${dll_name}" "${rename_as}.dll"
    cp "${tmp_dir}/${lib_name}" "${rename_as}.lib"
}

bits=$( python -c 'import struct; print(struct.calcsize("P") * 8)' )
lib_dir="src/lib"
mkdir -p "${lib_dir}"
install_dll "${bits}" \
    "https://github.com/bastibe/libsndfile-binaries/archive/1.0.27.tar.gz" \
    "${lib_dir}/sndfile"
install_dll "${bits}" \
    "https://github.com/spatialaudio/portaudio-binaries/archive/portaudio-19.6.0.tar.gz" \
    "${lib_dir}/portaudio"
