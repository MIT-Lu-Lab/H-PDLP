# Tests

From the repository root, with the [CUDA build requirements](../README.md#installation), build and run the C/CUDA suite in Release:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHPDLP_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The `cpu`, `gpu`, and `integration` labels select model/API checks, CUDA algorithm checks, and complete solves. For example, `ctest --test-dir build -L cpu --output-on-failure` runs the CPU subset without a GPU device, using that CUDA build environment. GPU tests report a skip when no usable device is available. Run the full suite on a CUDA GPU for release verification.

The Python package has its own pytest suite; see [Python tests](../python/README.md#tests).

The [Build workflow](../.github/workflows/build.yml) compiles the native targets and Python bindings with CUDA 12.4 and 13.3. It does not run the solver tests on a GPU.
