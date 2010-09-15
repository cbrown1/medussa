from distutils.core import setup
import platform

if platform.system() == "Windows":
    medusa_package_data = ['medusa.dll', 'portaudio_x86.dll']
else:
    medusa_package_data = []


setup(name='medusa',
      version='1.0',
      description='Medusa Audio Library',
      author='Joseph K. Ranweiler',
      author_email='jranweil@asu.edu',
      url='',
      packages=['medusa'],
      package_dir={'medusa': 'lib'},
      package_data={'medusa': medusa_package_data}
     )
