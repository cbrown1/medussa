from distutils.core import setup
import platform

if platform.system() == "Windows":
    medussa_package_data = ['../../lib/medussa.dll',
                            '../../lib/portaudio_x86.dll',
                            '../../lib/libsndfile-1.dll']
else:
    medussa_package_data = []


setup(name='medussa',
      version='1.0',
      description='Medussa Audio Library',
      author='Joseph K. Ranweiler',
      author_email='jranweil@asu.edu',
      url='',
      packages=['medussa'],
      package_dir={'medussa': 'src/medussa'},
      package_data={'medussa': medussa_package_data}
     )
