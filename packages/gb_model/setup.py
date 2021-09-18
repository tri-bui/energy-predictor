#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
from pathlib import Path
from setuptools import find_packages, setup


# Meta-data.
NAME = 'gb_model'
DESCRIPTION = 'LightGBM regression model.'
URL = 'https://github.com/tri-bui/gep_gbm'
EMAIL = 'buitri91@gmail.com'
AUTHOR = 'Tri Bui'
REQUIRES_PYTHON = '>=3.7'


# Version
CURRENT_PATH = Path(__file__).resolve().parent
PACKAGE_PATH = CURRENT_PATH / NAME
with open(PACKAGE_PATH / 'VERSION') as v:
    VERSION = v.read().strip()


# Long description
try:
    with io.open(CURRENT_PATH / 'README.md', encoding='utf-8') as f:
        LONG_DESCRIPTION = '\n' + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION


# Requirements
def list_reqs(filename=(CURRENT_PATH / 'requirements.txt')):
    with open(filename) as req:
        return req.read().splitlines()


# Setup
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={'gb_model': ['VERSION']},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
