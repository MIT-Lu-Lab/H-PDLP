# Build Options

See [Installation](../README.md#installation) for requirements and the default build commands.

The default GPU architecture selection is `all-major`; override it with `-DCMAKE_CUDA_ARCHITECTURES=...` if needed. The SpMV backend is selected automatically at compile time: `cusparseSpMVOp` for cuSPARSE ≥ 12.8.1 (the build always enables cuSPARSE experimental APIs), and `cusparseSpMV` otherwise.

Optional CMake flags (for example, `-DHPDLP_BUILD_TESTS=ON`):

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `-DHPDLP_BUILD_STATIC_LIB` | `ON` | Build static library `hpdlp_core` |
| `-DHPDLP_BUILD_SHARED_LIB` | `ON` | Build shared library |
| `-DHPDLP_BUILD_CLI` | `ON` | Build CLI executable |
| `-DHPDLP_BUILD_PYTHON` | `OFF` | Build Python bindings (requires Python development files and pybind11) |
| `-DHPDLP_BUILD_TESTS` | `OFF` | Build the test suite (run with `ctest`) |

The CLI, Python bindings, and tests require the static library.

For the Python package, `python -m pip install .` configures the required build options automatically; see the [Python installation guide](../python/README.md#installation).
