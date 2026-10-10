#include "hpdlp.h"
#include "test_checks.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool is_same_value(double actual, double expected)
{
    return actual == expected || (isnan(actual) && isnan(expected));
}

static void check_filtered_csr(const char *fixture,
                               const double *values,
                               const int *expected_row_pointers,
                               const int *expected_col_indices,
                               const double *expected_values,
                               int expected_nnz)
{
    const int row_pointers[] = {0, 2, 4};
    const int col_indices[] = {0, 1, 0, 1};
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = false;
    params.scaling_params.matrix_zero_tol = 1e-12;

    lp_problem_t original = {0};
    original.num_constraints = 2;
    original.num_variables = 2;
    original.constraint_matrix_num_nonzeros = 4;
    original.constraint_matrix_row_pointers = (int *)row_pointers;
    original.constraint_matrix_col_indices = (int *)col_indices;
    original.constraint_matrix_values = (double *)values;

    int *owned_row_pointers = NULL;
    int *owned_col_indices = NULL;
    double *owned_values = NULL;
    filter_constraint_matrix_entries(
        &original, &params,
        &owned_row_pointers, &owned_col_indices, &owned_values);

    char message[160];
    if (expected_nnz == original.constraint_matrix_num_nonzeros)
    {
        snprintf(message, sizeof(message), "%s: an unchanged matrix is not copied", fixture);
        check(message, !owned_row_pointers && !owned_col_indices && !owned_values);
        return;
    }
    snprintf(message, sizeof(message), "%s: a filtered matrix is owned", fixture);
    check(message, owned_row_pointers && owned_col_indices && owned_values);
    if (!owned_row_pointers || !owned_col_indices || !owned_values)
        return;

    bool kept_entries_match =
        memcmp(owned_row_pointers, expected_row_pointers, 3 * sizeof(int)) == 0 &&
        memcmp(owned_col_indices, expected_col_indices, (size_t)expected_nnz * sizeof(int)) == 0;
    for (int k = 0; k < expected_nnz; ++k)
        kept_entries_match &= is_same_value(owned_values[k], expected_values[k]);
    snprintf(message, sizeof(message), "%s: exactly the expected entries are kept", fixture);
    check(message, kept_entries_match);

    free(owned_row_pointers);
    free(owned_col_indices);
    free(owned_values);
}

static void check_matrix_filter(void)
{
    const int expected_row_pointers[] = {0, 1, 3};
    const int expected_col_indices[] = {0, 0, 1};

    const double with_non_finite[] = {1.0, 1e-15, NAN, 2.0};
    const double kept_non_finite[] = {1.0, NAN, 2.0};
    check_filtered_csr("non-finite entry", with_non_finite,
                       expected_row_pointers, expected_col_indices, kept_non_finite, 3);

    const double with_near_zero[] = {1.0, 1e-15, 3.0, 2.0};
    const double kept_near_zero[] = {1.0, 3.0, 2.0};
    check_filtered_csr("near-zero entry", with_near_zero,
                       expected_row_pointers, expected_col_indices, kept_near_zero, 3);

    const double all_significant[] = {1.0, 4.0, 3.0, 2.0};
    check_filtered_csr("nothing to drop", all_significant, NULL, NULL, NULL, 4);
}

int main(void)
{
    check_matrix_filter();
    return failures != 0;
}
