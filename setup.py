# SPDX-License-Identifier: Apache-2.0

import re
import sys
from setuptools import setup, find_packages

def no_publish():
    blacklist = ['register']
    for cmd in blacklist:
        if cmd in sys.argv:
            raise RuntimeError("Command \"{}\" blacklisted".format(cmd))

def read_version(filepath = 'onnx_tensorrt/version.py'):
    try:
        with open(filepath) as f:
            version_text = f.read()
        version, = re.findall('__version__ = "(.*)"', version_text)
        return version
    except Exception as error:
        raise RuntimeError(f"Failed to read version from {filepath}: {error}") from error

REQUIRED_PACKAGES = [
    "pycuda",
    "numpy",
    "onnx"
]

def main():
    no_publish()
    setup(
        name="onnx_tensorrt",
        version=read_version('onnx_tensorrt/version.py'),
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
