REM uninstalling
msiexec /uninstall dist\medussa-1.0.win32-py2.6.msi /qn

REM Building new installer
python setup.py bdist_msi

REM Installing
msiexec /i dist\medussa-1.0.win32-py2.6.msi /quiet
