
function _setup_python {
    local ver=( ${1//./ } )
    local suffix
    [[ "${2:-}" == 32 ]] || suffix=-x64
    local x86_opt
    [[ "${2:-}" != 32 ]] || x86_opt=--x86
    local py_dir="Python${ver[0]}${ver[1]}${suffix}"
    choco install "python${ver[0]}" ${x86_opt} \
        "--version=${1}" \
        --yes \
        --side-by-side \
        --override \
        --installarguments "'/quiet  InstallAllUsers=1 TargetDir=c:\\${py_dir}'"
    export PATH="/c/${py_dir}:/c/${py_dir}/Scripts:$PATH"
}

_setup_python "$@"
