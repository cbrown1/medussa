# Build lib
if (len(sys.argv) > 1) and (sys.argv[1] in ['build', 'install']):
    if platform.system() == "Windows":
        if os.environ['MedussaVcvarsOk']!="1":
            subprocess.call ([r"C:\Program Files\Microsoft Visual Studio 10.0\VC\vcvarsall.bat", "x86"])
            if os.environ['ERRORLEVEL']!="1":
                os.putenv('MedussaVcvarsOk', 1)
            else:
                raise OSError( "Error setting up MSVS build environment" )
        os.chdir(r'lib\build\win\msvc10')
        subprocess.call (["msbuild medussa.sln", "/property:Configuration=\"Release Py\"" + pyver])
        if os.environ['ERRORLEVEL']=="1":
            raise OSError( "Error building lib" )
        else:
            print("msbuild OK")
        os.chdir('..')

        os.mkdir("py" + pyver)
        shutil.copy(r"msvc10\Release Py" + pyver + r"\medussa.dll", "py" + pyver )
        os.chdir(r'..\..\..')
    else:
        os.chdir(r'./lib/build/linux')
        if os.path.isdir('py'+pyver):
            shutil.rmtree('py'+pyver)
        subprocess.call (["./build.sh", pyver])
        os.chdir('../../..')


