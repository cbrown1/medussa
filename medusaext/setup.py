from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc, get_python_lib
import numpy
import os
import sys
sys.path.append('src')

inc_dirs = ['.', 'include',
            get_python_inc(),
            numpy.get_include()]
lib_dirs = [get_python_lib(), 'C:\\dev\\medusaext\\lib']
libs = []

if os.name == 'posix':
    inc_dirs.append('/usr/local/include')
    lib_dirs.append('/usr/local/lib')
    libs.append('portaudio')
elif os.name == 'nt':
    libs.append('portaudio_x86')
    files = [('Lib/site-packages', 'portaudio_x86.dll')]

ext_src = ['src/callbacks.c', 'src/medusaext.c']

ext = Extension('_medusaext',
                include_dirs = inc_dirs,
                libraries = libs,
                library_dirs = lib_dirs,
                sources = ext_src)

setup (name = 'medusaext',
       version = '1.0',
       description = 'Wrapper for PortAudio.',
       #py_modules=['medusaext'],
       ext_modules = [ext])
