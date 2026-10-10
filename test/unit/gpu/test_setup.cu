#include "gpu_support.h"
#include "hpdlp.h"
#include "setup.h"
#include "test_checks.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static lp_problem_t *create_problem(const double *values)
{
    const double objective[] = {1.0, 1.0};
    const int row_ptr[] = {0, 1, 2};
    const int col_ind[] = {0, 1};
    const double lower[] = {0.0, 0.0};
    const double upper[] = {1.0, 1.0};
    const double constant = 0.0;
    matrix_desc_t matrix = {};
    matrix.m = matrix.n = 2;
    matrix.fmt = matrix_csr;
    matrix.data.csr.nnz = 2;
    matrix.data.csr.row_ptr = row_ptr;
    matrix.data.csr.col_ind = col_ind;
    matrix.data.csr.vals = values;
    return create_lp_problem(objective, &matrix, lower, upper, lower, upper, &constant);
}

static void set_test_parameters(pdhg_parameters_t *params)
{
    set_default_parameters(params);
    params->verbose = false;
    params->presolve = false;
    disable_rescaling(params);
    params->power_iter_params.tol = 1e-10;
}

static double measure_initial_norm(const double *values, const pdhg_parameters_t *params)
{
    lp_problem_t *problem = create_problem(values);
    if (problem == NULL)
    {
        fprintf(stderr, "FAIL: could not build the problem\n");
        ++failures;
        return NAN;
    }
    pdhg_solver_state_t *state = create_solver_state(problem, params);
    const double norm = state->rescaled_constraint_norm;
    pdhg_solver_state_free(state);
    lp_problem_free(problem);
    return norm;
}

static void check_constraint_norm_estimate(void)
{
    const double values[] = {3.0, 4.0};
    const double zero_values[] = {0.0, 0.0};
    pdhg_parameters_t params;

    set_test_parameters(&params);
    const double converged = measure_initial_norm(values, &params);
    printf("converged %.17g  ", converged);
    check("a converged estimate is the largest singular value",
          fabs(converged - 4.0) < 1e-6);

    set_test_parameters(&params);
    params.power_iter_params.max_iter = 1;
    const double cut_off = measure_initial_norm(values, &params);
    printf("cut off %.17g  ", cut_off);
    check("an unconverged estimate remains usable",
          cut_off >= 3.0 && cut_off <= 4.0);

    set_test_parameters(&params);
    const double degenerate = measure_initial_norm(zero_values, &params);
    printf("degenerate %.17g\n", degenerate);
    check("stored zeros leave the norm at zero", degenerate == 0.0);
}

int main(void)
{
    const int status = test_gpu_status();
    if (status) return status;
    check_constraint_norm_estimate();
    return failures != 0;
}
