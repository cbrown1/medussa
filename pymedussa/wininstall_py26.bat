REM uninstalling
msiexec /uninstall dist\medussa-1.0.win32-py2.6.msi /qn

REM Building new installer
build_win.bat 2.6

REM Installing
msiexec /i dist\medussa-1.0.win32-py2.6.msi /quiet
