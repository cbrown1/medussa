@echo off
set ver=%1

if %ver%.==. goto NoArg

set maj=%ver:~0,-2%
set min=%ver:~2%

REM Build installer
c:\python%maj%%min%\python setup.py bdist_msi --plat-name="win32" --target-version="%ver%"
goto End1

:NoArg
echo Pass Python version number as major.minor. Example: package_win.bat 2.7
goto End1
:End1
