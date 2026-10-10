# Command-line Interface

After [building H-PDLP](../README.md#installation), run from the repository root:

```bash
./build/hpdlp [OPTIONS] <mps_file> <output_directory>
```

## Arguments

- `<mps_file>`: Input file in `.mps` or `.mps.gz` format.
- `<output_directory>`: Existing directory for result files.

For MPS inputs with integer variables, H-PDLP solves the LP relaxation; integrality is not enforced.

To set a time limit and tighter tolerances:

```bash
mkdir -p result
./build/hpdlp --time_limit 60 --eps_opt 1e-6 --eps_feas 1e-6 \
    examples/2club200v15p5scn.mps.gz result/
```

Use `./build/hpdlp --help` to list the available options.

## Solver Options

**General**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `-h`, `--help` | `flag` | Display the help message. | N/A |
| `--no_presolve` | `flag` | Disable presolve (PSLP). | `false` |
| `-v`, `--verbose` | `flag` | Enable verbose logging. | `true` |
| `-q`, `--quiet` | `flag` | Disable verbose logging. | `false` |
| `-d`, `--diagnose` | `flag` | Add HSD scale, step-size, primal-weight and block-norm columns to the iteration log. | `false` |
| `--opt_norm` | `string` | Norm for optimality criteria: `l2` or `linf`. | `l2` |
| `--eval_freq` | `int` | Termination-check frequency (iterations), at least 3. | `200` |

**Scaling**

By default, PSLP presolve is followed by geometric-mean scaling (12 iterations), L-infinity Ruiz scaling (10 iterations), Pock-Chambolle scaling, and bound/objective rescaling. Geometric-mean scaling has no CLI flag; use `scaling_params.geometric_mean_iterations` in the C API to change it.

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--l_inf_ruiz_iter` | `int` | Iterations for L-∞ Ruiz rescaling. | `10` |
| `--no_pock_chambolle` | `flag` | Disable Pock-Chambolle rescaling. | `false` |
| `--pock_chambolle_alpha` | `double` | Pock-Chambolle α. | `1.0` |
| `--no_bound_obj_rescaling` | `flag` | Disable bound / objective rescaling. | `false` |

**Solver algorithm**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--sv_max_iter` | `int` | Max power iterations for σ_max(A). | `5000` |
| `--sv_tol` | `double` | Power-iteration tolerance. | `1e-4` |
| `--step_size_factor` | `double` | Step-size scaling factor in (0, 1). | `0.998` |
| `--no_primal_weight_safeguard` | `flag` | Disable primal-weight safeguarding. | `false` |
| `--cone_newton_max_iter` | `int` | Max Newton iterations for cone projection. | `100` |
| `--cone_newton_tol` | `double` | Newton tolerance for cone projection. | `1e-6` |

**Termination**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--eps_opt` | `double` | Relative optimality tolerance. | `1e-4` |
| `--eps_feas` | `double` | Relative feasibility tolerance. | `1e-4` |
| `--eps_farkas_homogeneous` | `double` | Residual-to-margin tolerance for homogeneous infeasibility certificates. | `1e-9` |
| `--eps_farkas_displacement` | `double` | Residual-to-margin tolerance for displacement infeasibility certificates. | `1e-10` |
| `--time_limit` | `double` | Iteration CPU-time limit in seconds. | `3600.0` |
| `--iter_limit` | `int` | Iteration limit. | `2147483647` |

`--time_limit` uses iteration CPU time measured by C `clock()`. It excludes presolve and initialization and is checked at termination evaluations. It is not a wall-clock limit.

**Active-set step boost**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--no_active_set_boost` | `flag` | Disable active-set step-size boosting. | `false` |

## Output Files

For an input `INSTANCE.mps.gz`, the solver writes:

```
<output_directory>/
├── INSTANCE_summary.txt          # Statistics, timings, and termination status
├── INSTANCE_primal_solution.txt  # Candidate solution or primal ray, when available
└── INSTANCE_dual_solution.txt    # Candidate dual solution or dual ray, when available
```

The summary reports status, timing, iteration count, objectives, and residuals. Vector files are written only when available. Check the status to distinguish candidate solutions from infeasibility rays.

`TIME_LIMIT` and `ITERATION_LIMIT` do not guarantee feasibility or optimality. Presolve may detect infeasibility without returning a ray. See the [termination-status reference](C_API.md#results-and-termination-status) for details.

## Iteration Log

The default log groups columns into iteration/time, primal/dual objectives, absolute residuals and gap, and relative residuals and gap. `R` marks a restart. The `time` column uses the iteration CPU-time accounting described above.

Add `--diagnose` to show these additional groups:

| Group | Columns | Meaning |
| :--- | :--- | :--- |
| HSD | `scale`, `gap`, `ratio` | Current primal homogeneous scale, normalized homogeneous gap, and normalized homogeneous ratio. |
| Step sizes | `pr ss`, `du ss`, `sc ss` | Primal, dual, and homogeneous step sizes. |
| Weight and scales | `omega`, `tau`, `alpha` | Primal weight, PDHG primal scale, and PDHG dual scale. |
| Block norms | `\|x\|`, `\|y\|` | Primal and dual solution norms. |

