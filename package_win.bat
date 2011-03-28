@echo off
set ver=%1

if %ver%.==. goto NoArg

set maj=%ver:~0,-2%
set min=%ver:~2%

rd /s /q build

REM Build lib
call "C:\Program Files\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" x86
cd lib\build\win\msvc10
msbuild medussa.sln /property:Configuration="Release Py%ver%"
cd ..
mkdir py%ver%
copy "msvc10\Release Py%ver%\medussa.dll" py%ver% 
cd ..\..\..

REM Build installer
c:\python%maj%%min%\python setup.py bdist_msi --plat-name="win32" --target-version="%ver%"
goto End1

:NoArg
echo Pass Python version number as major.minor. Example: package_win.bat 2.7
goto End1
:End1
