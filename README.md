# HoPDLP: A GPU-Accelerated First-Order LP Solver

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Version](https://img.shields.io/badge/version-v0.0.1-blue.svg)

**HoPDLP** is a GPU-accelerated first-order solver for linear programming with support for primal and dual infeasibility detection. It combines a generalized primal-dual hybrid gradient (PDHG) operator for a homogeneous self-dual (HSD) embedding with restarted reflected Halpern iteration. The embedding encodes primal-dual optimal solutions and infeasibility certificates in a single conic system.

HoPDLP represents two-sided constraint and variable bounds directly through box cones, without introducing explicit bound slacks. The box-cone projections reduce to one-dimensional root-finding problems, so each iteration uses sparse matrix-vector products, vector operations, and reductions, with no matrix factorization. The implementation also adapts practical techniques from [cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx), including adaptive restarting and primal-weight updates.

HoPDLP solves linear programs of the form
```math
\begin{aligned}
\min_{x} \quad & c^\top x + c_0 \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & \ell_v \le x \le u_v.
\end{aligned}
```

Maximization is supported through MPS `OBJSENSE` or `set_objective_sense` in the C API.

## Installation

### Requirements

* **GPU:** NVIDIA GPU with CUDA 12.4+ (cuBLAS, cuSPARSE, cuRAND).
* **Build Tools:** CMake (≥ 3.24), a C99 / C++17 compiler, NVCC.

CMake uses an installed **ZLIB** for `.mps.gz` decompression, or downloads it if unavailable. It fetches **[PSLP](https://github.com/dance858/PSLP)** (v0.0.11) for presolve.

### Build from Source
Clone the repository and compile the project using CMake.
```bash
git clone https://github.com/LucasBoTang/HoPDLP.git
cd HoPDLP
cmake -S . -B build
cmake --build build -j
```
On Linux, this creates `./build/hopdlp`, `./build/libhopdlp.so` (shared), and `./build/libhopdlp_core.a` (static). The default build type is Release.

### Quick Start

Download an instance from MIPLIB and solve its LP relaxation:

```bash
mkdir -p test
wget -P test/ https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz
./build/hopdlp test/2club200v15p5scn.mps.gz test/
```

The solver prints iteration statistics and writes results to `test/`. Check `test/2club200v15p5scn_summary.txt` for the termination status, objective values, and residuals. `OPTIMAL` means the solution satisfies the configured feasibility and optimality tolerances.

### Build Options

The default GPU architecture selection is `all-major`; override it with `-DCMAKE_CUDA_ARCHITECTURES=...` if needed. The SpMV backend is selected automatically at compile time: `cusparseSpMVOp` for cuSPARSE ≥ 12.8.1 with experimental APIs enabled (the default build), and `cusparseSpMV` otherwise.

Optional CMake flags (for example, `-DHOPDLP_BUILD_TESTS=ON`):

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `-DHOPDLP_BUILD_STATIC_LIB` | `ON` | Build static library `hopdlp_core` |
| `-DHOPDLP_BUILD_SHARED_LIB` | `ON` | Build shared library |
| `-DHOPDLP_BUILD_CLI` | `ON` | Build CLI executable |
| `-DHOPDLP_BUILD_TESTS` | `OFF` | Build the test suite (run with `ctest`) |

The CLI and tests require the static library.

### Tests

Build and run the C/CUDA suite in Release:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHOPDLP_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The `cpu`, `gpu`, and `integration` labels select model/API checks, CUDA
algorithm checks, and complete solves. For example,
`ctest --test-dir build -L cpu --output-on-failure` runs the CPU subset
without a GPU device, using the CUDA build environment above. GPU tests
report a skip when no usable device is available. Run the full suite on a
CUDA GPU for release verification.

The Python package has its own pytest suite; see [Python tests](python/README.md#tests).

### Python Package

Build and install the Python interface from this repository (Python 3.10+,
NumPy, SciPy, and the CUDA build requirements above):

```bash
python -m pip install .
```

For a specific GPU architecture, use, for example,
`python -m pip install . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=89`.
The CUDA compiler must support the selected architecture. The Python build
enables `HOPDLP_BUILD_PYTHON` and links the static solver into a pybind11 module.

## Usage & Interfaces

### Command-line Interface

After building the project, the `./build/hopdlp` binary can be invoked from the command line as follows:

```bash
./build/hopdlp [OPTIONS] <mps_file> <output_directory>
```

#### Arguments

- `<mps_file>`: The path to the input linear programming problem. Both plain (`.mps`) and gzipped (`.mps.gz`) files are supported.
- `<output_directory>`: An existing directory where the output files will be saved.

For MPS inputs with integer variables, HoPDLP solves the LP relaxation; integrality is not enforced.

For example, set a time limit and tighter feasibility and optimality tolerances:

```bash
./build/hopdlp --time_limit 60 --eps_opt 1e-6 --eps_feas 1e-6 \
    test/2club200v15p5scn.mps.gz test/
```

Use `./build/hopdlp --help` to list the available options.

#### Solver Options

**Termination**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--time_limit` | `double` | Iteration time limit in CPU seconds. | `3600.0` |
| `--iter_limit` | `int` | Iteration limit. | `2147483647` |
| `--eps_opt` | `double` | Relative optimality tolerance. | `1e-4` |
| `--eps_feas` | `double` | Relative feasibility tolerance. | `1e-4` |
| `--eps_farkas_homogeneous` | `double` | Residual-to-margin tolerance for homogeneous infeasibility certificates. | `1e-9` |
| `--eps_farkas_displacement` | `double` | Residual-to-margin tolerance for displacement infeasibility certificates. | `1e-10` |
| `--eval_freq` | `int` | Termination-check frequency (iterations), at least 3. | `200` |
| `--opt_norm` | `string` | Norm for optimality criteria: `l2` or `linf`. | `l2` |

**General**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `-h`, `--help` | `flag` | Display the help message. | N/A |
| `-v`, `--verbose` | `flag` | Enable verbose logging. | `true` |
| `-q`, `--quiet` | `flag` | Disable verbose logging. | `false` |
| `-d`, `--diagnose` | `flag` | Add HSD scale, step-size, primal-weight and block-norm columns to the iteration log. | `false` |

**Preprocessing / Scaling**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--no_presolve` | `flag` | Disable presolve (PSLP). | `false` |
| `--l_inf_ruiz_iter` | `int` | Iterations for L-∞ Ruiz rescaling. | `10` |
| `--no_pock_chambolle` | `flag` | Disable Pock-Chambolle rescaling. | `false` |
| `--pock_chambolle_alpha` | `float` | Pock-Chambolle α. | `1.0` |
| `--no_bound_obj_rescaling` | `flag` | Disable bound / objective rescaling. | `false` |

**Solver algorithm**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--step_size_factor` | `float` | Step-size scaling factor in (0, 1). | `0.998` |
| `--no_active_set_step_size_boost` | `flag` | Disable active-set step-size boosting. | `false` |
| `--singular_value_max_iter` | `int` | Max power iterations for σ_max(A). | `5000` |
| `--singular_value_tol` | `float` | Power-iteration tolerance. | `1e-4` |
| `--cone_newton_max_iter` | `int` | Max Newton iterations for cone projection. | `100` |
| `--cone_newton_tol` | `float` | Newton tolerance for cone projection. | `1e-6` |

#### Output Files
The solver writes a summary and any available primal/dual vectors to `<output_directory>`. For an input `INSTANCE.mps.gz`, the filenames are:
```
<output_directory>/
├── INSTANCE_summary.txt          # Statistics, timings, and termination status
├── INSTANCE_primal_solution.txt  # Candidate solution or primal ray, when available
└── INSTANCE_dual_solution.txt    # Candidate dual solution or dual ray, when available
```

The summary reports termination status, timing, iteration count, objective values, and residuals. Vector files are written when available; check the termination status to distinguish solution candidates from infeasibility rays.

`TIME_LIMIT` and `ITERATION_LIMIT` do not guarantee feasibility or optimality. Presolve may detect infeasibility without returning a ray. See the [termination-status reference](docs/C_API.md#results-and-termination-status) for details.

### C Interface

The public C API supports building LPs from CSR, CSC, COO, or dense matrices, configuring solver parameters, and retrieving solutions or infeasibility rays.

See the [C API Guide](docs/C_API.md) for a complete example, compilation instructions, warm starts, and termination-status details. The public headers are [`hopdlp.h`](include/hopdlp.h) and [`hopdlp_types.h`](include/hopdlp_types.h).

### Python Interface

The `hopdlp` package provides `Model`, `PDLP`, and `read` for building and solving LPs:

```python
import hopdlp

model = hopdlp.read("problem.mps.gz")
model.setParams(TimeLimit=60, OptimalityTol=1e-6, FeasibilityTol=1e-6)
model.optimize()
print(model.StatusName, model.ObjVal)
if model.Status == hopdlp.PDLP.OPTIMAL:
    print(model.X)
```

See the [Python API Guide](python/README.md) for matrix inputs, parameters,
warm starts, and the interpretation of infeasibility certificates.

## License
HoPDLP is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
