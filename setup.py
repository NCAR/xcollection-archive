#!/usr/bin/env python

"""The setup script."""

from os.path import exists

from setuptools import find_packages, setup

import versioneer

readme = open("README.rst").read() if exists("README.rst") else ""


setup(
    name="xcollection",
    description="xcollection",
    long_description=readme,
    maintainer="Matt Long",
    maintainer_email="mclong@ucar.edu",
    url="https://github.com/NCAR/xcollection",
    packages=find_packages(),
    package_dir={"xcollection": "xcollection"},
    include_package_data=True,
    install_requires=[],
    license="Apache 2.0",
    zip_safe=False,
    keywords="xcollection",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
