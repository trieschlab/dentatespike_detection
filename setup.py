#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    'pandas >= 0.18.0',
    'pathlib',
    'ripple_detection',
    'scipy',
    'numpy']

TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='dentatespike_detection',
    version='0.1',
    license='MIT',
    description=('Tools to detect dentate spikes'),
    author='Tristan Stoeber',
    author_email='tristan.stoeber@posteo.net',
    url='https://github.com/trieschlab/dentatespike_detection',
    packages=find_packages(),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
