@echo off
set ver=%1

if %ver%.==. goto NoArg

set maj=%ver:~0,-2%
set min=%ver:~2%

rd /s /q build

REM Build lib
call "C:\Program Files\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" x86
if ERRORLEVEL 1 call "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" x86
cd lib\build\win\msvc10
msbuild medussa.sln /property:Configuration="Release Py%ver%"
if ERRORLEVEL 1 goto BuildErrorLib
cd ..
mkdir py%ver%
copy "msvc10\Release Py%ver%\medussa.dll" py%ver% 
cd ..\..\..

REM Build installer
c:\python%maj%%min%\python setup.py bdist_msi --plat-name="win32" --target-version="%ver%"
if ERRORLEVEL 1 goto BuildErrorInstaller
goto End1

:BuildErrorLib
echo Error building medussa.dll
cd ..\..\..\..
goto End1
:BuildErrorInstaller
echo Error building Installer
goto End1
:NoArg
echo Pass Python version number as major.minor. Example: package_win.bat 2.7
goto End1
:End1
