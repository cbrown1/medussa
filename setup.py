from distutils.core import setup

setup(name='medusa',
      version='1.0',
      description='Medusa Audio Library',
      author='Joseph K. Ranweiler',
      author_email='jranweil@asu.edu',
      url='',
      packages=['medusa'],
      package_dir={'medusa': 'src/medusa'},
      package_data={'medusa': ['*.dll']}
     )
