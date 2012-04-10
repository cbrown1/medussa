@echo off
set ver=%1

if %ver%.==. goto NoArg

set maj=%ver:~0,-2%
set min=%ver:~2%

REM uninstalling
msiexec /uninstall dist\medussa-1.0.win32-py%ver%.msi /qn
REM (ignore uninstall errors)
echo Uninstalling DONE.

REM Building new installer
call package_win.bat %ver%
if ERRORLEVEL 1 goto BuildingInstallerError
echo Building new installer DONE.

REM Installing
msiexec /i dist\medussa-1.0.win32-py%ver%.msi /quiet
if ERRORLEVEL 1 goto InstallError
echo Installing DONE.

goto End1

:BuildingInstallerError
echo Error building
goto End1

:InstallError
echo Error installing
goto End1

:NoArg
echo Pass Python version number as major.minor. Example: reinstall_win.bat 2.7
goto End1

:End1
