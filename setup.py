# SPDX-License-Identifier: Apache-2.0

import os
import sys
import onnx_tensorrt
from setuptools import setup, find_packages

def no_publish():
    blacklist = ['register']
    for cmd in blacklist:
        if cmd in sys.argv:
            raise RuntimeError("Command \"{}\" blacklisted".format(cmd))


REQUIRED_PACKAGES = [
    "pycuda",
    "numpy",
    "onnx"
]

def main():
    no_publish()
    setup(
        name="onnx_tensorrt",
        version=onnx_tensorrt.__version__,
        description="ONNX-TensorRT - TensorRT backend for running ONNX models",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        url="https://github.com/onnx/onnx-tensorrt",
        author="NVIDIA",
        author_email="svc_tensorrt@nvidia.com",
        classifiers=[
            'Intended Audience :: Developers',
            'Programming Language :: Python :: 3',
        ],
        install_requires=REQUIRED_PACKAGES,
        packages=find_packages(),
        zip_safe=True,
    )

if __name__ == '__main__':
    main()
