# SPDX-License-Identifier: Apache-2.0
from importlib.metadata import version as _v, PackageNotFoundError  # Py3.8+
from . import backend

try:
    __version__ = _v("onnx_tensorrt")
except PackageNotFoundError:
    __version__ = "0+unknown"
