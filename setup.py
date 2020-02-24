#!/usr/bin/python3
# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import sys
import setuptools

if sys.version_info < (3, 6):
    sys.exit("Python versions less than 3.6 are not supported")

VERSION = "0.1.0"

SCRIPTS = (
    "tensorshow tensorstat tensormon serve-imagenet-dir serve-imagenet-shards".split()
)

setuptools.setup(
    author="Thomas Breuel",
    author_email="tmbdev+removeme@gmail.com",
    description="Distributed preprocessing for deep learning.",
    install_requires="webdataset pyzmq msgpack torch".split(),
    keywords="object store, client, deep learning",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    name="tensorcom",
    packages=["tensorcom"],
    python_requires=">=3.6",
    scripts=SCRIPTS,
    url="http://github.com/tmbdev/tensorcom",
    version=VERSION,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
