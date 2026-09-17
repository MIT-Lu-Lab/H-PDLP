# **cuHSDLP: A GPU-Accelerated Homogeneous Self-Dual LP Solver**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**cuHSDLP** is a GPU-accelerated linear programming solver based on a Primal-Dual Hybrid Gradient (PDHG) method applied to a **Homogeneous Self-Dual (HSD) embedding** of the LP. The HSD formulation lets a single first-order loop detect optimality, primal infeasibility, and dual infeasibility simultaneously, while retaining the massive parallelism of PDHG on GPU. It combines a reflected Halpern-style update, adaptive restart, PID-controlled primal weight, active-set step-size boosting, and standard rescaling (geometric-mean + Ruiz + Pock-Chambolle + bound-objective) to deliver robust convergence across a wide range of LP benchmarks.

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
* **Target architectures:** CMake defaults to `all-major`, covering the major architectures supported by the selected CUDA compiler. Override `CMAKE_CUDA_ARCHITECTURES` to select a specific target.

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
- `<output_directory>`: An existing directory where the output files will be saved; the CLI does not create it.

For MPS inputs with integer variables, cuHSDLP solves the LP relaxation; integrality is not enforced.

#### Solver Options

**General**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `-h`, `--help` | `flag` | Display the help message. | N/A |
| `-v`, `--verbose` | `flag` | Enable verbose logging (already enabled by default). | `true` |

The current CLI has no quiet flag. C callers can set `params.verbose = false` to disable regular progress logging.

**Preprocessing / Scaling**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--no_presolve` | `flag` | Disable presolve (PSLP). | presolve enabled |
| `--l_inf_ruiz_iter` | `int` | Iterations for L-∞ Ruiz rescaling. | `10` |
| `--no_pock_chambolle` | `flag` | Disable Pock-Chambolle rescaling. | enabled |
| `--pock_chambolle_alpha` | `float` | Pock-Chambolle α. | `1.0` |
| `--no_bound_obj_rescaling` | `flag` | Disable bound / objective rescaling. | enabled |

Geometric-mean scaling also runs for 12 iterations by default; its iteration count is configurable through `scaling_params.geometric_mean_iterations` in the C API. Input cleaning defaults to `matrix_zero_tol = 1e-9` and `infinite_bound = 1e20`: entries with magnitude at most the former are dropped, and lower/upper bounds beyond `-infinite_bound`/`+infinite_bound` (inclusive) become infinite. These operations can change the model.

**Solver algorithm**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--step_size_factor` | `float` | Step-size scaling factor in (0, 1). | `0.998` |
| `--no_active_set_step_size_boost` | `flag` | Disable active-set step-size boosting. | enabled |
| `--singular_value_max_iter` | `int` | Max power iterations for σ_max(A). | `5000` |
| `--singular_value_tol` | `float` | Power-iteration tolerance. | `1e-4` |
| `--cone_newton_max_iter` | `int` | Max Newton iterations for cone projection. | `100` |
| `--cone_newton_tol` | `float` | Newton tolerance for cone projection. | `1e-6` |

**Termination**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--time_limit` | `double` | Iteration-phase time limit; see timing semantics below. | `3600.0` |
| `--iter_limit` | `int` | Iteration limit, checked at batch boundaries. | `2147483647` |
| `--eps_opt` | `double` | Relative optimality tolerance. | `1e-4` |
| `--eps_feas` | `double` | Relative feasibility tolerance. | `1e-4` |
| `--eps_farkas_homogeneous` | `double` | Residual-to-margin tolerance for homogeneous infeasibility certificates. | `1e-9` |
| `--eps_farkas_displacement` | `double` | Residual-to-margin tolerance for displacement infeasibility certificates. | `1e-10` |
| `--eval_freq` | `int` | Termination-check frequency (iterations), at least 3. | `200` |
| `--opt_norm` | `string` | Norm for optimality criteria: `l2` or `linf`. | `l2` |

`Runtime (sec)` and `--time_limit` currently measure the main iteration phase using C `clock()` (process CPU time on Linux/WSL); presolve and preconditioning are reported separately. Limits are checked at batch boundaries, so the iteration count may exceed `--iter_limit` by up to `--eval_freq - 1`.

#### Output Files
The solver writes a summary and any available primal/dual vectors to `<output_directory>`. For an input `INSTANCE.mps.gz`, the filenames are:
```
<output_directory>/
├── INSTANCE_summary.txt          # Statistics, timings, and termination status
├── INSTANCE_primal_solution.txt  # Candidate solution or primal ray, when available
└── INSTANCE_dual_solution.txt    # Candidate dual solution or dual ray, when available
```

The summary file reports the termination reason (`OPTIMAL`, `PRIMAL_INFEASIBLE`, `DUAL_INFEASIBLE`, `INFEASIBLE_OR_UNBOUNDED`, `TIME_LIMIT`, `ITERATION_LIMIT`, or `NUMERICAL_FAILURE`), presolve statistics, runtime, iteration count, primal/dual objective values, relative residuals, objective gap, and restart count.

Solution files are written only when the corresponding vectors are available. Check the termination reason before using them: limit statuses return candidates, while infeasibility statuses may return rays. Presolve can prove infeasibility without returning a ray.

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
void               set_start_values(lp_problem_t *prob,
                                   const double *primal, const double *dual);
cuhsdlp_result_t  *solve_lp_problem(lp_problem_t *prob,
                                    const pdhg_parameters_t *params);
void               cuhsdlp_result_free(cuhsdlp_result_t *result);
void               lp_problem_free(lp_problem_t *prob);
```

The matrix descriptor `matrix_desc_t` accepts CSR, CSC, COO, or dense input. Dense matrices are row-major; sparse indices are zero-based, and duplicate entries are summed. For an m-by-n matrix, objective and variable-bound arrays have n entries, and constraint-bound arrays have m entries. Use finite coefficients and no NaN values.

`create_lp_problem` copies its inputs and minimizes by default. NULL objective coefficients and objective offset default to zero; NULL lower and upper bounds default to negative and positive infinity. In particular, omitted variable bounds create **free variables**, not nonnegative variables. Use `set_objective_sense` to request maximization.

Initialize parameters with `set_default_parameters` before changing individual fields, or pass NULL to `solve_lp_problem` for defaults. `set_start_values` copies primal/dual starts in original, unscaled coordinates and replaces both previous starts; NULL clears the corresponding start. Set `params.presolve = false` to use warm starts, which are otherwise ignored.

The result struct exposes primal/dual vectors, reduced costs, residuals, objective values, timing breakdown, restart count, and the final HSD homogeneous scale τ. Interpret its vectors according to `termination_reason`:

| Status | Available vectors and meaning |
| :--- | :--- |
| `OPTIMAL` | Primal/dual solution and reduced costs accepted by the solver |
| `TIME_LIMIT`, `ITERATION_LIMIT` | Current primal/dual candidate and reduced costs; feasibility and optimality are not guaranteed |
| `PRIMAL_INFEASIBLE` | Dual and reduced-cost rays after iterative detection; may be absent after presolve |
| `DUAL_INFEASIBLE` | Primal recession ray; implies primal unboundedness if the primal is feasible |
| `INFEASIBLE_OR_UNBOUNDED`, `NUMERICAL_FAILURE` | No solution vectors |

Check both the termination reason and pointer availability before accessing arrays. Returned vectors use the original variable/constraint indexing after postsolve. For minimization candidates, the dual convention is `A^T y + z = c`, where `y` is `dual_solution` and `z` is `reduced_cost`; maximization candidates are converted back to the requested objective sense. The CLI does not write reduced costs.

The problem and result own separate host arrays. Free them with `lp_problem_free` and `cuhsdlp_result_free`; do not free their fields individually. Both free functions accept NULL. Some input errors return NULL, while invalid parameters, allocation failures, and CUDA-library errors can currently terminate the calling process.

## License
cuHSDLP is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
