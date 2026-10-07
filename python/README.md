# Python interface for HoPDLP

The `hopdlp` package provides a Python interface to HoPDLP. It uses NumPy/SciPy for model data and pybind11 to call the C/CUDA solver directly.

## Installation

From the repository root, with Python 3.10+, CMake 3.24+, a C/C++ toolchain, CUDA 12.4+ (including NVCC), and an NVIDIA GPU:

```bash
python -m pip install .
```

The build uses `scikit-build-core` and `pybind11`, and installs NumPy/SciPy runtime dependencies. Set `CUDACXX` to select a CUDA compiler if needed. To override GPU architecture selection:

```bash
python -m pip install . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=89
```

Choose an architecture supported by your toolkit and GPU. Linux/WSL is the primary build environment. The package includes Windows DLL lookup support, but native Windows builds require a compatible CUDA/MSVC toolchain.

## Build and solve a model

```python
import numpy as np
from hopdlp import Model, PDLP

model = Model(
    objective_vector=[1., 1.],
    constraint_matrix=np.array([[1., 2.], [0., 1.], [3., 2.]]),
    constraint_lower_bound=[5., -np.inf, -np.inf],
    constraint_upper_bound=[5., 2., 8.],
    variable_lower_bound=[0., 0.],
    variable_upper_bound=None,
    objective_constant=0.,
)
model.Params.TimeLimit = 60
model.setParams(OutputFlag=False, OptimalityTol=1e-6, FeasibilityTol=1e-6)
model.optimize()
if model.Status == PDLP.OPTIMAL:
    print(model.ObjVal, model.X, model.Pi, model.RC)  # objective ≈ 3, X ≈ [1, 2]
```

The model represents `min c.T @ x + c0`, with `l <= A @ x <= u` and `lb <= x <= ub`. Set `model.ModelSense = PDLP.MAXIMIZE` to maximize. Omitted variable bounds are **-infinity and +infinity**, respectively; specify `variable_lower_bound=0 * np.ones(n)` for nonnegative variables. Constraint bounds may also be `None` for the corresponding infinite bound.

`A` accepts a two-dimensional NumPy array or SciPy sparse matrix/array. The high-level model normalizes sparse inputs to CSR. The compiled `hopdlp._core.solve_once` entry point also accepts CSC and COO directly. Values use float64; sparse indices must fit in int32. Model inputs are copied and exposed as read-only arrays.

## Read MPS and modify data

```python
import hopdlp

model = hopdlp.read("problem.mps.gz")  # also accepts .mps and pathlib.Path
model.optimize()
model.setObjectiveConstant(2.)
model.optimize()
```

MPS `OBJSENSE` is preserved. Integer variables are treated as continuous: the interface solves the LP relaxation.

Model setters include `setObjectiveVector`, `setObjectiveConstant`, `setConstraintMatrix`, `setConstraintLowerBound`, `setConstraintUpperBound`, `setVariableLowerBound`, and `setVariableUpperBound`. Assigning `c`, `c0`, `A`, `constr_lb`, `constr_ub`, `lb`, or `ub` calls the corresponding setter. Changing model data or objective sense clears cached results. Each `optimize()` creates a fresh native problem; it does not retain GPU state.

## Parameters

Use `model.Params.TimeLimit = 60`, `model.Params["TimeLimit"] = 60`, `model.setParam("TimeLimit", 60)`, or `model.setParams(TimeLimit=60, Presolve=False)`. `getParam()` reads values and `resetParams()` restores HoPDLP's C defaults. `dict(model.Params.items())` lists all supported canonical keys. Invalid names raise `KeyError`; invalid values raise `TypeError` or `ValueError`. Parameter updates are transactional, including checks involving multiple values. Python numeric parameters must be finite, including `TimeLimit` and `InfiniteBound`; use a large finite value when needed.

Common parameter aliases:

| Alias | Canonical key |
| --- | --- |
| `TimeLimit` | `time_sec_limit` |
| `IterationLimit` | `iteration_limit` |
| `OutputFlag`, `LogToConsole` | `verbose` |
| `Presolve` | `presolve` |
| `OptimalityTol` | `eps_optimal_relative` |
| `FeasibilityTol` | `eps_feasible_relative` |
| `OptimalityNorm` | `optimality_norm` (`l2` or `linf`) |
| `TermCheckFreq` | `termination_evaluation_frequency` |
| `ActiveSetBoost` | `active_set_boost` |

HoPDLP-specific controls:

| Alias | Canonical key |
| --- | --- |
| `Diagnose` | `diagnose` |
| `FarkasHomogeneousTol` | `eps_farkas_homogeneous` |
| `FarkasDisplacementTol` | `eps_farkas_displacement` |
| `ConeNewtonMaxIter` | `cone_newton_max_iter` |
| `ConeNewtonTol` | `cone_newton_tol` |
| `StepSizeFactor` | `step_size_factor` |
| `RestartKp` | `k_p` |
| `Ki`, `Kd`, `ISmooth` | `k_i`, `k_d`, `i_smooth` |
| `PrimalWeightSafeguard` | `primal_weight_safeguard` |
| `ASBRelativeTol` | `asb_relative_tol` |

TimeLimit follows the C solver's iteration CPU-time accounting; it is not a wall-clock budget for preprocessing and Python conversion. Defaults come from HoPDLP. Unknown parameters are rejected. The internal `tau_trace_path` debugging output is not exposed by this interface.

## Warm starts

```python
model.setParam("Presolve", False)
model.setWarmStart(primal=x0, dual=y0)
model.optimize()
model.setWarmStart(primal=x1)  # retains the existing dual start
model.setWarmStart(dual=None) # clears only the dual start
model.clearWarmStart()        # clears both
```

Vectors must have the original model dimensions and finite values. Set `Presolve=False` to use explicit warm starts. With presolve enabled, the C solver prints a warning and ignores the supplied starts. Previous solve results are not used as warm starts automatically.

## Results and certificates

`optimize()` returns the model itself. Results include `X`, `Pi`, `RC`, `ObjVal`, `DualObj`, `Gap`, `RelGap`, `Status`, `StatusName`, `IterCount`, `Runtime`, `RescalingTime`, `RelPrimalResidual`, and `RelDualResidual`. `PrimalInfeas` and `DualInfeas` are aliases for the relative residuals. Result arrays own their memory and remain valid across subsequent solves. Before a solve, cached result properties are `None`.

| Status | Interpretation of vector properties |
| --- | --- |
| `PDLP.OPTIMAL` | `X`, `Pi`, `RC` describe an optimal primal/dual solution within tolerances. |
| `PDLP.TIME_LIMIT`, `PDLP.ITERATION_LIMIT` | Candidate vectors; feasibility and optimality are not guaranteed. |
| `PDLP.PRIMAL_INFEASIBLE` | `Pi` and `RC` may contain a dual certificate; `X` is `None`. |
| `PDLP.DUAL_INFEASIBLE` | `X` may contain a primal ray; `Pi` and `RC` are `None`. |
| `PDLP.INFEASIBLE_OR_UNBOUNDED` | Presolve could not distinguish the two cases; vectors may be absent. |
| `PDLP.NUMERICAL_FAILURE`, `PDLP.UNSPECIFIED` | No solution guarantee; vectors may be absent. |

Presolve can identify infeasibility without returning a certificate. A missing vector is `None`, not a zero-filled array. Check status before interpreting objective values, residuals, or vectors. A zero-valued certificate statistic does not imply a certificate is available.

HoPDLP additionally exposes `FarkasMargin`, `FarkasResidual`, `HomogeneousScale`, `RestartCount`, `PresolveTime`, `PresolveStatus`, `ASBRaiseCount`, `ASBRevertCount`, `ASBPowerIterations`, and `FinalStepSizePrimal`, `FinalStepSizeDual`, `FinalStepSizeHomogeneous`. The Farkas statistics are the backend's certificate diagnostics. See [C API certificate semantics](../docs/C_API.md#results-and-termination-status).

The binding releases the Python GIL during native solves and serializes native solves within the process because the solver uses CUDA global graph capture. It raises Python exceptions for invalid inputs and unavailable CUDA devices. Fatal allocation or CUDA failures deeper in the C solver still use the existing process-exit error paths.

## Tests

With the package installed and a working CUDA GPU:

```bash
python -m pip install '.[test]'
python -m pytest --require-gpu
```

The suite checks model inputs, sparse conversions, parameter validation, warm
starts, MPS reading, result properties and ownership, and complete solves.
Solution and certificate checks independently use the original model data.

Run the checks that do not need a GPU device separately:

```bash
python -m pytest -m "not gpu"
```

The compiled package and its CUDA runtime dependencies must still be installed.
Without `--require-gpu`, tests marked `gpu` skip when no usable CUDA device is
available. The flag makes that condition fail during release verification.
An import failure or solver failure is reported as an error.
