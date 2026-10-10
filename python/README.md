# Python interface for H-PDLP

`hpdlp` is the Python interface to H-PDLP, a first-order LP solver for NVIDIA GPUs. It accepts NumPy and SciPy model data.

## Installation

Build from source with Python 3.10+, CMake 3.24+, Git, a C/C++ toolchain, CUDA 12.4+ (including NVCC), and an NVIDIA GPU supported by the toolkit:

```bash
git clone https://github.com/MIT-Lu-Lab/H-PDLP.git
cd H-PDLP
python -m pip install .
```

If you already have a checkout, run the install command from its root. The first build needs network access to fetch PSLP and, if not installed, ZLIB.

The install command also installs NumPy and SciPy. Set `CUDACXX` to select a CUDA compiler. To specify a GPU architecture:

```bash
python -m pip install . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=89
```

Choose an architecture supported by your toolkit and GPU. Linux and WSL2 are the primary build environments. Native Windows builds have not been validated.

## Build and solve a model

```python
import numpy as np
from hpdlp import Model, PDLP

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

The model represents `min c.T @ x + c0`, with `l <= A @ x <= u` and `lb <= x <= ub`. Set `model.ModelSense = PDLP.MAXIMIZE` to maximize. Omitted variable bounds are **-infinity and +infinity**, respectively; specify `variable_lower_bound=np.zeros(n)` for nonnegative variables. Constraint bounds may also be `None` for the corresponding infinite bound.

`A` accepts a two-dimensional NumPy array or SciPy sparse matrix/array. The high-level model normalizes sparse inputs to CSR. The compiled `hpdlp._core.solve_once` entry point also accepts CSC and COO directly. Values use float64; sparse indices must fit in int32. Model inputs are copied and exposed as read-only arrays.

## Read MPS and modify data

```python
import hpdlp

model = hpdlp.read("examples/2club200v15p5scn.mps.gz")  # also accepts .mps and pathlib.Path
model.optimize()
model.setObjectiveConstant(2.)
model.optimize()
```

Run this example from the repository root. The first solve should return `OPTIMAL` with an objective near -121.2; adding the objective constant shifts it by 2. See [example details](https://github.com/MIT-Lu-Lab/H-PDLP/blob/main/examples/README.md).

MPS `OBJSENSE` is preserved. Integer variables are treated as continuous: the interface solves the LP relaxation.

Model setters include `setObjectiveVector`, `setObjectiveConstant`, `setConstraintMatrix`, `setConstraintLowerBound`, `setConstraintUpperBound`, `setVariableLowerBound`, and `setVariableUpperBound`. Assigning `c`, `c0`, `A`, `constr_lb`, `constr_ub`, `lb`, or `ub` calls the corresponding setter. Changing model data or objective sense clears cached results. Each `optimize()` creates a fresh native problem; it does not retain GPU state.

## Parameters

Use `model.Params.TimeLimit = 60`, `model.Params["TimeLimit"] = 60`, `model.setParam("TimeLimit", 60)`, or `model.setParams(TimeLimit=60, Presolve=False)`. `getParam()` reads values and `resetParams()` restores H-PDLP's C defaults. `dict(model.Params.items())` lists all supported canonical keys. Invalid names raise `KeyError`; invalid values raise `TypeError` or `ValueError`. Parameter updates are transactional, including checks involving multiple values. Python numeric parameters must be finite, including `TimeLimit` and `InfiniteBound`; use a large finite value when needed.

Common parameter aliases:

| Alias | Canonical key |
| --- | --- |
| `Presolve` | `presolve` |
| `OutputFlag`, `LogToConsole` | `verbose` |
| `OptimalityNorm` | `optimality_norm` (`l2` or `linf`) |
| `TermCheckFreq` | `termination_evaluation_frequency` |
| `OptimalityTol` | `eps_optimal_relative` |
| `FeasibilityTol` | `eps_feasible_relative` |
| `TimeLimit` | `time_sec_limit` |
| `IterationLimit` | `iteration_limit` |
| `ActiveSetBoost` | `active_set_boost` |

H-PDLP-specific controls:

| Alias | Canonical key |
| --- | --- |
| `Diagnose` | `diagnose` |
| `StepSizeFactor` | `step_size_factor` |
| `RestartKp` | `k_p` |
| `Ki`, `Kd`, `ISmooth` | `k_i`, `k_d`, `i_smooth` |
| `PrimalWeightSafeguard` | `primal_weight_safeguard` |
| `ConeNewtonMaxIter` | `cone_newton_max_iter` |
| `ConeNewtonTol` | `cone_newton_tol` |
| `FarkasHomogeneousTol` | `eps_farkas_homogeneous` |
| `FarkasDisplacementTol` | `eps_farkas_displacement` |
| `ASBRelativeTol` | `asb_relative_tol` |

`TimeLimit` is the iteration CPU-time limit in seconds, measured by C `clock()`. It excludes presolve, initialization, and Python conversion and is checked at termination evaluations. It is not a wall-clock limit.

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

H-PDLP additionally exposes `FarkasMargin`, `FarkasResidual`, `HomogeneousScale`, `RestartCount`, `PresolveTime`, `PresolveStatus`, `ASBRaiseCount`, `ASBRevertCount`, `ASBPowerIterations`, and `FinalStepSizePrimal`, `FinalStepSizeDual`, `FinalStepSizeHomogeneous`. The Farkas statistics are the backend's certificate diagnostics. See [C API certificate semantics](https://github.com/MIT-Lu-Lab/H-PDLP/blob/main/docs/C_API.md#results-and-termination-status).

The binding releases the Python GIL during native solves and serializes native solves within the process because the solver uses CUDA global graph capture. It raises Python exceptions for invalid inputs and unavailable CUDA devices. Allocation and CUDA failures inside the C solver are reported as `RuntimeError` with the native error message.

## Tests

With the package installed and a working CUDA GPU:

```bash
python -m pip install '.[test]'
python -m pytest --require-gpu
```

The suite checks model inputs, sparse conversions, parameter validation, warm starts, MPS reading, result properties and ownership, and complete solves. Solution and certificate checks independently use the original model data.

Run the checks that do not need a GPU device separately:

```bash
python -m pytest -m "not gpu"
```

The compiled package and its CUDA runtime dependencies must still be installed. Without `--require-gpu`, tests marked `gpu` skip when no usable CUDA device is available. The flag makes that condition fail during release verification. An import failure or solver failure is reported as an error.
