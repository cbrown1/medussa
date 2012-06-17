build_lib = True
for i in range(len(sys.argv)):
    if sys.argv[i] == "--no-lib-built":
        print "Don't build lib"
        build_lib = False
        sys.argv.remove("--no-lib-built")

# Build lib
if build_lib:
    if platform.system() == "Windows":
        #if (not os.environ.has_key('MedussaVcvarsOk')) or (os.environ['MedussaVcvarsOk']!="1"):
        #    print "No MedussaVcvarsOk"
        ret = subprocess.call ("\"C:/Program Files/Microsoft Visual Studio 10.0/VC/vcvarsall.bat\" x86")
        if (ret != "1"):
            print "MSVS environmet set"
        else:
            raise OSError( "Error setting up MSVS build environment" )
        os.chdir(r'lib\build\win\msvc10')
        ret = subprocess.call ("msbuild medussa.sln /property:Configuration=\"Release Py\"" + pyver)
        if (ret!="1"):
            print("msbuild OK")
        else:
            raise OSError( "Error building lib" )
        os.chdir('..')

        if os.path.isdir('py'+pyver):
            shutil.rmtree('py'+pyver)
        os.mkdir("py" + pyver)
        shutil.copy(r"msvc10\Release Py" + pyver + r"\medussa.dll", "py" + pyver )
        os.chdir(r'..\..\..')
    else:
        os.chdir(r'./lib/build/linux')
        if os.path.isdir('py'+pyver):
            shutil.rmtree('py'+pyver)
        subprocess.call (["./build.sh", pyver])
        os.chdir('../../..')

