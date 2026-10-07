#include "gpu_support.h"
#include "hopdlp.h"
#include "setup.h"
#include "test_checks.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check_filtered_csr_is_consistent(const char *fixture,
                                             const double *values,
                                             int num_rows,
                                             int num_columns,
                                             const int *row_pointers,
                                             const int *col_indices)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    const double tol = params.scaling_params.matrix_zero_tol;
    const int nnz = row_pointers[num_rows];

    int expected_nnz = 0;
    for (int k = 0; k < nnz; ++k)
    {
        if (!(fabs(values[k]) <= tol))
            ++expected_nnz;
    }

    lp_problem_t original;
    memset(&original, 0, sizeof(original));
    original.num_constraints = num_rows;
    original.num_variables = num_columns;
    original.constraint_matrix_num_nonzeros = nnz;
    original.constraint_matrix_row_pointers = (int *)row_pointers;
    original.constraint_matrix_col_indices = (int *)col_indices;
    original.constraint_matrix_values = (double *)values;

    int *owned_row_pointers = NULL;
    int *owned_col_indices = NULL;
    double *owned_values = NULL;
    filter_constraint_matrix_entries(
        &original, &params,
        &owned_row_pointers, &owned_col_indices, &owned_values);

    lp_problem_t working_problem = original;
    if (owned_row_pointers)
    {
        working_problem.constraint_matrix_row_pointers = owned_row_pointers;
        working_problem.constraint_matrix_col_indices = owned_col_indices;
        working_problem.constraint_matrix_values = owned_values;
        working_problem.constraint_matrix_num_nonzeros = owned_row_pointers[num_rows];
    }

    printf("%s: expected %d of %d entries kept\n", fixture, expected_nnz, nnz);

    check("row_pointers[num_rows] equals the reported nonzero count",
          working_problem.constraint_matrix_row_pointers[num_rows] ==
              working_problem.constraint_matrix_num_nonzeros);
    check("reported nonzero count matches the predicate",
          working_problem.constraint_matrix_num_nonzeros == expected_nnz);

    int monotone = 1;
    for (int i = 0; i < num_rows; ++i)
    {
        if (working_problem.constraint_matrix_row_pointers[i] >
            working_problem.constraint_matrix_row_pointers[i + 1])
            monotone = 0;
    }
    check("row pointers are non-decreasing", monotone);

    int kept_are_not_near_zero = 1;
    for (int k = 0; k < working_problem.constraint_matrix_num_nonzeros; ++k)
    {
        if (fabs(working_problem.constraint_matrix_values[k]) <= tol)
            kept_are_not_near_zero = 0;
    }
    check("every written entry is above the tolerance", kept_are_not_near_zero);

    check("filtered matrix pointers are registered as owned",
          (expected_nnz == nnz &&
           !owned_row_pointers && !owned_col_indices && !owned_values) ||
          (expected_nnz < nnz &&
           owned_row_pointers && owned_col_indices && owned_values));

    free(owned_row_pointers);
    free(owned_col_indices);
    free(owned_values);
}

static int check_matrix_filter(void)
{
    const int row_pointers[] = {0, 2, 4};
    const int col_indices[] = {0, 1, 0, 1};

    const double with_non_finite[] = {1.0, 1e-15, NAN, 2.0};
    check_filtered_csr_is_consistent("non-finite entry", with_non_finite, 2, 2,
                                     row_pointers, col_indices);

    const double with_near_zero[] = {1.0, 1e-15, 3.0, 2.0};
    check_filtered_csr_is_consistent("near-zero entry", with_near_zero, 2, 2,
                                     row_pointers, col_indices);

    const double all_significant[] = {1.0, 4.0, 3.0, 2.0};
    check_filtered_csr_is_consistent("nothing to drop", all_significant, 2, 2,
                                     row_pointers, col_indices);

    printf("%s (%d failure(s))\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures != 0;
}

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
    params->scaling_params.geometric_mean_iterations = 0;
    params->scaling_params.l_inf_ruiz_iterations = 0;
    params->scaling_params.has_pock_chambolle_alpha = false;
    params->scaling_params.bound_objective_rescaling = false;
    params->power_iter_params.tol = 1e-10;
}

static double measure_initial_norm(const double *values, const pdhg_parameters_t *params)
{
    lp_problem_t *problem = create_problem(values);
    if (problem == NULL)
    {
        printf("  FAIL: could not build the problem\n");
        ++failures;
        return NAN;
    }
    pdhg_solver_state_t *state = initialize_solver_state(problem, problem, params);
    const double norm = state->rescaled_constraint_norm;
    pdhg_solver_state_free(state);
    lp_problem_free(problem);
    return norm;
}

static int check_constraint_norm_estimate(void)
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

    printf("test_initial_constraint_norm: %d failure(s)\n", failures);
    return failures != 0;
}

int main(void)
{
    const int status = test_gpu_status();
    if (status) return status;
    int failed = 0;
    failed |= check_matrix_filter();
    failed |= check_constraint_norm_estimate();
    return failed || failures != 0;
}
