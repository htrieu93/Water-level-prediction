from setuptools import find_packages, setup

# read the contents of README file
from os import path
from io import open  # for Python 2 and 3 compatibility

this_directory = path.abspath(path.dirname(__file__))

# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='src',
    author='Hieu Trieu',
    author_email='htrieu93@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)