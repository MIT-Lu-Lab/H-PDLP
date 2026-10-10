#include "hpdlp.h"
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

static void check_matrix_filter(void)
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
}

int main(void)
{
    check_matrix_filter();
    return failures != 0;
}
