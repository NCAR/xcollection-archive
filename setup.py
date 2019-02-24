#!/usr/bin/env python

"""The setup script."""

from os.path import exists

from setuptools import find_packages, setup

import versioneer

if exists("requirements.txt"):
    with open("requirements.txt") as f:
        install_requires = f.read().strip().split("\n")
else:
    install_requires = ["intake", "xarray", "pyyaml", "tqdm", "intake-xarray", "dask"]

if exists("README.rst"):
    with open("README.rst") as f:
        long_description = f.read()
else:
    long_description = ""

setup(
    name="xcollection",
    description="xcollection",
    long_description=long_description,
    maintainer="Matt Long",
    maintainer_email="mclong@ucar.edu",
    url="https://github.com/NCAR/xcollection",
    packages=find_packages(),
    package_dir={"xcollection": "xcollection"},
    include_package_data=True,
    install_requires=install_requires,
    license="Apache 2.0",
    zip_safe=False,
    keywords="xcollection",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
