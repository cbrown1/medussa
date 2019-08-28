:: We only need this to have paths to dumpbin & lib in PATH
call "c:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
bash travis/install_deps.windows.sh
