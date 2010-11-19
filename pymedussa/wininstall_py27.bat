REM uninstalling
msiexec /uninstall dist\medussa-1.0.win32-py2.7.msi /qn

REM Building new installer
build_win.bat 2.7

REM Installing
msiexec /i dist\medussa-1.0.win32-py2.7.msi /quiet
