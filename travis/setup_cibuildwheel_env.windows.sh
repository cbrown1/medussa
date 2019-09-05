
function _setup_cibuildwheel_env {
    local ver=( ${1//./ } )
    local suffix
    [[ "${2:-}" == 32 ]] || suffix=-x64
    local x86_opt
    [[ "${2:-}" != 32 ]] || x86_opt=--x86
    local py_dir="Python${ver[0]}${ver[1]}${suffix}"
    [[ 2 != "${ver[0]}" ]] || {
        choco install --yes vcredist2008
        choco install --yes --ignore-dependencies vcpython27
    }
    choco install "python${ver[0]}" ${x86_opt} \
        "--version=${1}" \
        --yes \
        --side-by-side \
        --override \
        --installarguments "'/quiet  InstallAllUsers=1 TargetDir=c:\\${py_dir}'"
    export PATH="/c/${py_dir}:/c/${py_dir}/Scripts:$PATH"
    local platform
    [[ "${2:-}" == 32 ]] && platform=win32 || platform=win_amd64
    export CIBW_PLATFORM=windows
    export CIBW_BUILD="cp${ver[0]}${ver[1]}-${platform}"
}

_setup_cibuildwheel_env "$@"
