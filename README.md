# **cuHSDLP: A GPU-Accelerated Homogeneous Self-Dual LP Solver**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**cuHSDLP** is a GPU-accelerated linear programming solver based on a Primal-Dual Hybrid Gradient (PDHG) method applied to a **Homogeneous Self-Dual (HSD) embedding** of the LP. The HSD formulation lets a single first-order loop detect optimality, primal infeasibility, and dual infeasibility simultaneously, while retaining the massive parallelism of PDHG on GPU. This branch is a **plain PDHG baseline**: the iteration is vanilla Chambolle-Pock on the HSD embedding, with no Halpern anchoring, no reflection, no adaptive restart, no gauge control, and a fixed primal weight. Standard rescaling (Ruiz + Pock-Chambolle + bound-objective) and presolve are retained.

cuHSDLP solves linear programs of the form
```math
\begin{aligned}
\min_{x} \quad & c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & \ell_v \le x \le u_v.
\end{aligned}
```

## Installation

### Requirements
* **GPU:** NVIDIA GPU with CUDA 12.4+ (cuBLAS, cuSPARSE, cuRAND).
* **Build Tools:** CMake (≥ 3.20), a C99 / C++17 compiler, NVCC.
* **Supported architectures:** Pascal through Hopper (compute capability 60, 70, 75, 80, 86, 89, 90).

External dependencies (fetched automatically by CMake if not found):
* **ZLIB** — decompression of `.mps.gz` inputs.
* **[PSLP](https://github.com/dance858/PSLP)** (v0.0.8) — presolve.

### Build from Source
Clone the repository and compile the project using CMake.
```bash
git clone git@github.com:LucasBoTang/cuHSDLP.git
cd cuHSDLP
cmake -B build
cmake --build build --clean-first
```
This will create the solver binary at `./build/cuhsdlp`, together with `libcuhsdlp.{so,dll}` (shared) and `libcuhsdlp_core.a` (static).

Optional CMake flags:

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `-DCUHSDLP_BUILD_STATIC_LIB` | `ON` | Build static library `cuhsdlp_core` |
| `-DCUHSDLP_BUILD_SHARED_LIB` | `ON` | Build shared library |
| `-DCUHSDLP_BUILD_CLI` | `ON` | Build CLI executable |
| `-DCUHSDLP_BUILD_TESTS` | `OFF` | Build the test suite (run with `ctest`) |

#### Verifying the Installation
Run a small test problem to confirm that the solver was built correctly.
```bash
# 1. Download a test instance from the MIPLIB library
wget -P test/ https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz

# 2. Solve the problem and write output to the current directory (.)
./build/cuhsdlp test/2club200v15p5scn.mps.gz test/
```
If the solver runs and creates output files, your installation is successful.

## Usage

### Command-line Interface

After building the project, the `./build/cuhsdlp` binary can be invoked from the command line as follows:

```bash
./build/cuhsdlp [OPTIONS] <mps_file> <output_directory>
```

#### Arguments
- `<mps_file>`: The path to the input linear programming problem. Both plain (`.mps`) and gzipped (`.mps.gz`) files are supported.
- `<output_directory>`: The directory where the output files will be saved.

#### Solver Options

**General**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `-h`, `--help` | `flag` | Display the help message. | N/A |
| `-v`, `--verbose` | `flag` | Enable verbose logging. | `false` |
| `--basis_trace` | `flag` | Write one row per phase change (iteration, primal/dual basis changes) to `INSTANCE_basis_trace.csv`. | `false` |

**Preprocessing / Scaling**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--no_presolve` | `flag` | Disable presolve (PSLP). | presolve enabled |
| `--l_inf_ruiz_iter` | `int` | Iterations for L-∞ Ruiz rescaling. | `10` |
| `--no_pock_chambolle` | `flag` | Disable Pock-Chambolle rescaling. | enabled |
| `--pock_chambolle_alpha` | `float` | Pock-Chambolle α. | `1.0` |
| `--no_bound_obj_rescaling` | `flag` | Disable bound / objective rescaling. | enabled |

**Solver algorithm**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--step_size_factor` | `float` | Step-size scaling factor in (0, 1]. | `0.998` |
| `--singular_value_max_iter` | `int` | Max power iterations for σ_max(A). | `5000` |
| `--singular_value_tol` | `float` | Power-iteration tolerance. | `1e-4` |
| `--cone_newton_max_iter` | `int` | Max Newton iterations for cone projection. | `100` |
| `--cone_newton_tol` | `float` | Newton tolerance for cone projection. | `1e-6` |

**Termination**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--time_limit` | `double` | Time limit in seconds. | `3600.0` |
| `--iter_limit` | `int` | Iteration limit. | `2147483647` |
| `--eps_opt` | `double` | Relative optimality tolerance. | `1e-4` |
| `--eps_feas` | `double` | Relative feasibility tolerance. | `1e-4` |
| `--eval_freq` | `int` | Termination-check frequency (iterations). | `200` |
| `--opt_norm` | `string` | Norm for optimality criteria: `l2` or `linf`. | `l2` |

#### Output Files
The solver generates three text files in the specified `<output_directory>`. The filenames are derived from the input file's basename. For an input `INSTANCE.mps.gz`, the output will be:
```
<output_directory>/
├── INSTANCE_summary.txt          # Statistics, timings, and termination status
├── INSTANCE_primal_solution.txt  # Primal solution vector
└── INSTANCE_dual_solution.txt    # Dual solution vector
```

The summary file reports the termination reason (`OPTIMAL`, `PRIMAL_INFEASIBLE`, `DUAL_INFEASIBLE`, `INFEASIBLE_OR_UNBOUNDED`, `TIME_LIMIT`, `ITERATION_LIMIT`, or `NUMERICAL_FAILURE`), presolve statistics, runtime, iteration count, primal/dual objective values, relative residuals, objective gap, and basis statistics. The basis is the PDHG basis of Liu & Lu (arXiv:2409.14715) in the sense their Theorem 7 relies on: the branch taken by each cone projection, which differs from their Definition 3 only at single-step boundary grazes.

### C Interface
The public C API is defined in the header files [`include/cuhsdlp.h`](include/cuhsdlp.h) and [`include/cuhsdlp_types.h`](include/cuhsdlp_types.h). The core entry points are:

```c
lp_problem_t      *create_lp_problem(const double *c,
                                     const matrix_desc_t *A,
                                     const double *l_c, const double *u_c,
                                     const double *l_v, const double *u_v,
                                     const double *objective_constant);
void               set_default_parameters(pdhg_parameters_t *params);
cuhsdlp_result_t  *solve_lp_problem(lp_problem_t *prob,
                                    const pdhg_parameters_t *params);
void               cuhsdlp_result_free(cuhsdlp_result_t *result);
void               lp_problem_free(lp_problem_t *prob);
```

The matrix descriptor `matrix_desc_t` accepts CSR, CSC, COO, or dense input; warm-start values can be supplied through `set_start_values`. The result struct exposes primal/dual solutions, reduced costs, residuals, objective values, timing breakdown, and the final HSD homogeneous scale τ.

## License
cuHSDLP is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
