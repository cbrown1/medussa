set ver=%1

if %ver%.==. goto NoArg

set maj=%ver:~0,-2%
set min=%ver:~2%


REM uninstalling
msiexec /uninstall dist\medussa-1.0.win32-py%ver%.msi /qn

REM Building new installer
package_win.bat %ver%

REM Installing
msiexec /i dist\medussa-1.0.win32-py%ver%.msi /quiet

:NoArg
echo Pass Python version number as major.minor. Example: reinstall_win.bat 2.7
goto End1
:End1
