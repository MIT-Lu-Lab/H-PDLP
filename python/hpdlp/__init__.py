# Copyright 2025 Haihao Lu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""H-PDLP: Python bindings for the GPU-accelerated first-order LP solver."""

import os as _os
import platform as _platform

# Windows only: register CUDA bin for dependent DLL loading.
if _platform.system() == "Windows":  # pragma: no cover
    _cuda_path = _os.environ.get("CUDA_PATH")
    if _cuda_path:
        _bin_path = _os.path.join(_cuda_path, "bin")
        if _os.path.isdir(_bin_path):
            _cuda_dll_directory = _os.add_dll_directory(_bin_path)

from .model import Model, read
from . import PDLP

# versioning
from importlib.metadata import version as _version, PackageNotFoundError as _PackageNotFoundError
# get version from the installed package metadata (set by CMake project())
try:
    __version__ = _version("hpdlp")
except _PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["Model", "PDLP", "read", "__version__"]
