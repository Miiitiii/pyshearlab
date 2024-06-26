"""Setup script for huest.

Installation command::

    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages
import os

root_path = os.path.dirname(__file__)
requires = open(os.path.join(root_path, 'requirements.txt')).readlines()

setup(
    name='pyshearlab',

    version='0.0.2',

    description='Shearlets in python',

    url='https://github.com/Miiitiii/pyshearlab',

    author='Stefan Loock',
    Editor='Mahdi Firouzbakht',

    license='GPL',

    packages=find_packages(exclude=['*test*']),
    package_dir={'pyshearlab': 'pyshearlab'},
    
    install_requires=[requires]
)
