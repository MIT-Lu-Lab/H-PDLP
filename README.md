# **cuHSDLP: A GPU-Accelerated Homogeneous Self-Dual LP Solver**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**cuHSDLP** is a GPU-accelerated linear programming solver based on a Primal-Dual Hybrid Gradient (PDHG) method applied to a **Homogeneous Self-Dual (HSD) embedding**. It detects optimality, primal infeasibility, and dual infeasibility, and combines reflected Halpern updates, adaptive restart, PID-controlled primal weight, active-set step-size boosting, and rescaling.

cuHSDLP solves linear programs of the form
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
* **Target architectures:** `all-major` by default; configurable with `CMAKE_CUDA_ARCHITECTURES`.

The SpMV backend is selected automatically at compile time: `cusparseSpMVOp` for cuSPARSE ≥ 12.8.1 with experimental APIs enabled (the default build), and `cusparseSpMV` otherwise.

External dependencies (fetched automatically by CMake if not found):
* **ZLIB** — decompression of `.mps.gz` inputs.
* **[PSLP](https://github.com/dance858/PSLP)** (v0.0.11) — presolve, fetched by CMake.

### Build from Source
Clone the repository and compile the project using CMake.
```bash
git clone git@github.com:LucasBoTang/cuHSDLP.git
cd cuHSDLP
cmake -B build
cmake --build build --clean-first
```
On Linux, this creates `./build/cuhsdlp`, `./build/libcuhsdlp.so` (shared), and `./build/libcuhsdlp_core.a` (static). The default build type is Release.

Optional CMake flags:

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `-DCUHSDLP_BUILD_STATIC_LIB` | `ON` | Build static library `cuhsdlp_core` |
| `-DCUHSDLP_BUILD_SHARED_LIB` | `ON` | Build shared library |
| `-DCUHSDLP_BUILD_CLI` | `ON` | Build CLI executable |
| `-DCUHSDLP_BUILD_TESTS` | `OFF` | Build the test suite (run with `ctest`) |

The CLI and tests require the static library.

#### Verifying the Installation
Run a small test problem to confirm that the solver was built correctly.
```bash
# 1. Download a test instance from the MIPLIB library
mkdir -p test
wget -P test/ https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz

# 2. Solve its LP relaxation and write output to test/
./build/cuhsdlp test/2club200v15p5scn.mps.gz test/
```
Inspect `test/2club200v15p5scn_summary.txt` for the termination reason and residuals to verify the solve.

## Usage

### Command-line Interface

After building the project, the `./build/cuhsdlp` binary can be invoked from the command line as follows:

```bash
./build/cuhsdlp [OPTIONS] <mps_file> <output_directory>
```

#### Arguments
- `<mps_file>`: The path to the input linear programming problem. Both plain (`.mps`) and gzipped (`.mps.gz`) files are supported.
- `<output_directory>`: An existing directory where the output files will be saved.

For MPS inputs with integer variables, cuHSDLP solves the LP relaxation; integrality is not enforced.

#### Solver Options

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

#### Output Files
The solver writes a summary and any available primal/dual vectors to `<output_directory>`. For an input `INSTANCE.mps.gz`, the filenames are:
```
<output_directory>/
├── INSTANCE_summary.txt          # Statistics, timings, and termination status
├── INSTANCE_primal_solution.txt  # Candidate solution or primal ray, when available
└── INSTANCE_dual_solution.txt    # Candidate dual solution or dual ray, when available
```

The summary reports termination status, timing, iteration count, objective values, and residuals. Vector files are written when available; check the termination status to distinguish solution candidates from infeasibility rays.

### C Interface

The public C API is defined in the header files [`include/cuhsdlp.h`](include/cuhsdlp.h) and [`include/cuhsdlp_types.h`](include/cuhsdlp_types.h). The core entry points are:

```c
lp_problem_t      *create_lp_problem(const double *c,
                                     const matrix_desc_t *A,
                                     const double *l_c, const double *u_c,
                                     const double *l_v, const double *u_v,
                                     const double *objective_constant);
void               set_default_parameters(pdhg_parameters_t *params);
void               set_objective_sense(lp_problem_t *prob,
                                      objective_sense_t sense);
bool               set_start_values(lp_problem_t *prob,
                                   const double *primal, const double *dual);
cuhsdlp_result_t  *solve_lp_problem(const lp_problem_t *prob,
                                    const pdhg_parameters_t *params);
void               cuhsdlp_result_free(cuhsdlp_result_t *result);
void               lp_problem_free(lp_problem_t *prob);
```

Basic usage:

- **Matrix formats:** CSR, CSC, COO, and dense. Dense matrices use row-major storage; sparse indices are zero-based.
- **Objective and bounds:** Problems minimize by default; use `set_objective_sense` for maximization. Omitted variable bounds leave variables free.
- **Parameters:** Initialize with `set_default_parameters`, then customize before calling `solve_lp_problem`.
- **Warm starts:** Use `set_start_values` in original problem coordinates with presolve disabled.

Interpret the returned vectors according to `termination_reason`:

| Status                    | Returned vectors                         | Interpretation                           |
| :------------------------ | :--------------------------------------- | :--------------------------------------- |
| `OPTIMAL`                 | Primal/dual solutions and reduced costs  | Optimality conditions satisfied.         |
| `TIME_LIMIT`              | Primal/dual candidates and reduced costs | Time limit reached.                      |
| `ITERATION_LIMIT`         | Primal/dual candidates and reduced costs | Iteration limit reached.                 |
| `PRIMAL_INFEASIBLE`       | Dual and reduced-cost rays               | Primal infeasibility detected.           |
| `DUAL_INFEASIBLE`         | Primal ray                               | Dual infeasibility detected.             |
| `INFEASIBLE_OR_UNBOUNDED` | None                                     | Infeasibility or unboundedness detected. |
| `NUMERICAL_FAILURE`       | None                                     | Numerical failure detected.              |

Limit statuses do not guarantee feasibility or optimality. Presolve may detect infeasibility without returning a ray.

## License
cuHSDLP is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
