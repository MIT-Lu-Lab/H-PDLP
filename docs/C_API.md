# C API Guide

The public API is defined in [`hpdlp.h`](../include/hpdlp.h), with matrix, parameter, and result types in [`hpdlp_types.h`](../include/hpdlp_types.h).

## Example

This example minimizes `x + 2y` subject to `x + y >= 1` and `x, y >= 0`. Its optimal solution is `(x, y) = (1, 0)`, with objective value `1`.

Save the following as `example.c` in the repository root:

```c
#include <stdio.h>
#include "hpdlp.h"

int main(void)
{
    const double c[] = {1.0, 2.0};
    const double values[] = {1.0, 1.0};
    const double con_lb[] = {1.0};
    const double var_lb[] = {0.0, 0.0};
    const matrix_desc_t A = {
        .m = 1, .n = 2, .fmt = matrix_dense,
        .data.dense = {.A = values}
    };

    /* NULL upper bounds mean +infinity; NULL objective constant means zero. */
    lp_problem_t *prob = create_lp_problem(
        c, &A, con_lb, NULL, var_lb, NULL, NULL);
    if (!prob)
        return 1;

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;

    hpdlp_result_t *result = solve_lp_problem(prob, &params);
    if (!result)
    {
        lp_problem_free(prob);
        return 1;
    }

    int exit_code = 0;
    if (result->termination_reason == TERMINATION_REASON_OPTIMAL)
    {
        printf("Objective: %.6f\n", result->primal_objective_value);
        printf("x = %.6f, y = %.6f\n",
               result->primal_solution[0], result->primal_solution[1]);
    }
    else
    {
        fprintf(stderr, "Stopped with status %d\n",
                (int)result->termination_reason);
        exit_code = 1;
    }

    hpdlp_result_free(result);
    lp_problem_free(prob);
    return exit_code;
}
```

After building H-PDLP with the default shared-library option, compile and run on Linux from the repository root:

```bash
cc -std=c99 example.c -Iinclude -Lbuild -lhpdlp -o example
LD_LIBRARY_PATH="$PWD/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./example
```

The reported objective should be approximately `1`, with `x` approximately `1` and `y` approximately `0`.

## Model Data and Parameters

- **Matrix formats:** CSR, CSC, COO, and dense. Dense matrices use row-major storage; sparse indices are zero-based. `create_lp_problem` copies the input data.
- **Bounds:** A `NULL` lower-bound array means negative infinity; a `NULL` upper-bound array means positive infinity. In particular, omitting variable bounds leaves variables free. Include `<math.h>` and use `-INFINITY` or `INFINITY` for individual unbounded entries.
- **Objective:** Problems minimize by default. Call `set_objective_sense(prob, OBJECTIVE_SENSE_MAXIMIZE)` for maximization. The last argument to `create_lp_problem` is a pointer to the objective constant, or `NULL` for zero.
- **Parameters:** Call `set_default_parameters` before changing individual fields. The CLI option `--eps_opt` corresponds to `params.termination_criteria.eps_optimal_relative`; `--eps_feas` corresponds to `eps_feasible_relative`.
- **Parameter validation:** `hpdlp_validate_parameters(&params, error, sizeof(error))` returns `0` on success and `-1` for invalid parameters, without GPU work. `error` is an optional character buffer for the diagnostic. The solver and Python interface use the same validation rules; Python additionally requires finite numeric values.
- **Warm starts:** Call `set_start_values(prob, primal, dual)` with vectors in original problem coordinates, check its Boolean return value, and set `params.presolve = false` before solving.
- **Ownership:** Release the result with `hpdlp_result_free` and the problem with `lp_problem_free`.

## Results and Termination Status

Check `result->termination_reason` before interpreting the returned vectors. The status names below omit the `TERMINATION_REASON_` prefix used by the C enum.

| Status | Returned vectors | Interpretation |
| :--- | :--- | :--- |
| `OPTIMAL` | Primal/dual solutions and reduced costs | Feasibility and optimality tolerances satisfied. |
| `TIME_LIMIT` | Primal/dual candidates and reduced costs | Time limit reached. |
| `ITERATION_LIMIT` | Primal/dual candidates and reduced costs | Iteration limit reached. |
| `PRIMAL_INFEASIBLE` | Dual and reduced-cost rays, when available | Primal infeasibility detected. |
| `DUAL_INFEASIBLE` | Primal ray, when available | Dual infeasibility detected. |
| `INFEASIBLE_OR_UNBOUNDED` | None | Infeasibility or unboundedness detected. |
| `NUMERICAL_FAILURE` | None | Numerical failure detected. |

Limit statuses do not guarantee feasibility or optimality. Presolve may detect infeasibility without returning a ray. A dual-infeasibility certificate establishes primal unboundedness only when the primal problem is feasible.

The vector fields are `primal_solution`, `dual_solution`, and `reduced_cost`; on an infeasibility return, these fields hold the corresponding rays. The result also includes objective values, residuals, timings, and certificate statistics. The CLI writes primal and dual vectors when available; reduced costs are accessible through the C API.
