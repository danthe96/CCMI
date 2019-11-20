from setuptools import setup

required = []

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='ccmi',
      version='1.0.0',
      description='Classifier based Conditional Mutual Information Estimation',
      author='Sudipto Mukherjee',
      author_email='sudipm@uw.edu',
      url='https://github.com/sudiptodip15/CCMI',
      packages=['CIT'],
      install_requires=required)
