# H-PDLP: A GPU-Accelerated First-Order LP Solver

[![Build](https://github.com/MIT-Lu-Lab/H-PDLP/actions/workflows/build.yml/badge.svg)](https://github.com/MIT-Lu-Lab/H-PDLP/actions/workflows/build.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**H-PDLP** is a first-order linear programming solver for NVIDIA GPUs. It supports primal and dual infeasibility detection.

- A generalized PDHG method solves a homogeneous self-dual (HSD) embedding that encodes optimal solutions and infeasibility certificates.
- Box cones handle two-sided constraints and variable bounds without explicit bound slacks.
- Each iteration uses sparse matrix-vector products, vector operations, and reductions, with no matrix factorization.

[Quick Start](#quick-start) · [CLI](docs/CLI.md) · [Python](python/README.md) · [C API](docs/C_API.md)

H-PDLP solves linear programs of the form

```math
\begin{aligned}
\min_{x} \quad & c^\top x + c_0 \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & \ell_v \le x \le u_v.
\end{aligned}
```

Maximization is supported through MPS `OBJSENSE` and the C/Python APIs. For MPS inputs with integer variables, H-PDLP solves the **LP relaxation**; integrality is not enforced.

## Installation

Linux and WSL2 are the primary build environments. Native Windows builds have not been validated.

- **GPU:** an NVIDIA GPU supported by the selected CUDA toolkit.
- **CUDA Toolkit:** 12.4+ with NVCC, cuBLAS, cuSPARSE, and cuRAND.
- **Build tools:** CMake 3.24+, Git, and a C99 / C++17 compiler compatible with the toolkit.

CMake fetches [PSLP](https://github.com/dance858/PSLP) (v0.0.11) for presolve and uses ZLIB to read `.mps.gz` files. The first build needs network access; ZLIB is also downloaded if not installed.

```bash
git clone https://github.com/MIT-Lu-Lab/H-PDLP.git
cd H-PDLP
cmake -S . -B build
cmake --build build -j
```

On Linux, this creates `build/hpdlp`, `build/libhpdlp.so` (shared), and `build/libhpdlp_core.a` (static). The default build type is Release. See [Build Options](docs/BUILD.md) for GPU architecture selection, library targets, and SpMV backends.

## Quick Start

Solve the LP relaxation of the bundled [MIPLIB instance `2club200v15p5scn`](https://miplib.zib.de/instance_details_2club200v15p5scn.html) from the repository root:

```bash
mkdir -p result
./build/hpdlp examples/2club200v15p5scn.mps.gz result/
```

At default tolerances, expect `OPTIMAL` and a primal objective of approximately **-121.2**. Results are written to `result/2club200v15p5scn_summary.txt`. See [example details](examples/README.md) for the data source and interpretation.

`OPTIMAL` means the configured feasibility and optimality tolerances are satisfied. `TIME_LIMIT` and `ITERATION_LIMIT` do not guarantee feasibility or optimality. Infeasibility statuses may return rays instead of solutions. Check the [termination status](docs/C_API.md#results-and-termination-status) before using returned vectors.

## Interfaces

### Command Line

```bash
./build/hpdlp --time_limit 60 --eps_opt 1e-6 --eps_feas 1e-6 examples/2club200v15p5scn.mps.gz result/
```

Both `.mps` and `.mps.gz` inputs are supported. The output directory must already exist.

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--time_limit` | `3600.0` | Iteration CPU-time limit in seconds. |
| `--eps_opt` | `1e-4` | Relative optimality tolerance. |
| `--eps_feas` | `1e-4` | Relative feasibility tolerance. |
| `--quiet` | Off | Disable verbose logging. |

The time limit excludes presolve and initialization; it is not a wall-clock limit. Run `./build/hpdlp --help` for all options. The [CLI Guide](docs/CLI.md) explains parameters, preprocessing, output files, and diagnostic logs.

### Python

With Python 3.10+ and the CUDA build requirements above, install from the repository root:

```bash
python -m pip install .
```

The install command also installs NumPy and SciPy. To solve the bundled instance:

```python
import hpdlp

model = hpdlp.read("examples/2club200v15p5scn.mps.gz")
model.setParams(TimeLimit=60, OptimalityTol=1e-6, FeasibilityTol=1e-6)
model.optimize()
print(model.StatusName)
if model.Status == hpdlp.PDLP.OPTIMAL:
    print(model.ObjVal, model.X)
```

See the [Python API Guide](python/README.md) for building models with NumPy/SciPy, installation options, parameters, warm starts, and certificates.

### C

The C API accepts CSR, CSC, COO, and dense matrices. See the [C API Guide](docs/C_API.md) for an example, compilation instructions, parameters, and warm starts. Public headers: [`hpdlp.h`](include/hpdlp.h) and [`hpdlp_types.h`](include/hpdlp_types.h).

## Algorithm Overview

H-PDLP applies restarted reflected Halpern iteration to a generalized primal-dual hybrid gradient (PDHG) operator for the HSD embedding. The primal and dual box-cone subproblems reduce to one-dimensional root finding and are solved by a warm-started safeguarded Newton method.

The implementation uses PSLP presolve and geometric-mean, Ruiz, Pock-Chambolle, and bound/objective scaling. See the [CLI Guide](docs/CLI.md#solver-options) for defaults and controls.

## Tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHPDLP_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

See the [test guide](test/README.md) for CPU/GPU subsets and [Python tests](python/README.md#tests) for pytest instructions. CI builds everything and runs the CPU-only C tests and the non-GPU Python tests.

## Acknowledgements

H-PDLP uses adaptive restart and primal-weight update techniques from [cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx).

## Citation

Citation information will be added here.

## Support

Use [GitHub Issues](https://github.com/MIT-Lu-Lab/H-PDLP/issues) for bug reports and feature requests. Include the H-PDLP commit or version, OS, GPU, CUDA version, a reproducing command or example, and relevant logs.

## License

H-PDLP is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details. The bundled MIPLIB instance is third-party data; its source is documented in [examples/README.md](examples/README.md).
