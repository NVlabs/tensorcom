#!/usr/bin/python3
# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

from __future__ import print_function

import glob
import sys
from distutils.core import setup  # , Extension, Command

scripts = "tensorshow tensorstat tensormon serve-imagenet-dir serve-imagenet-shards".split()

setup(
    name='tensorcom',
    version='v0.0',
    author="Thomas Breuel",
    description="Distributed preprocessing for deep learning.",
    packages=["tensorcom"],
    scripts=scripts,
)
