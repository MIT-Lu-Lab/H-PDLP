#include "hpdlp.h"
#include "test_checks.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int check_canonical_csr(void)
{
    const int pointers[] = {0, 4, 4, 7};
    const int indices[] = {2, 0, 2, 0, 1, 1, 0};
    const double values[] = {3, 2, -3, 1, 4, -1, 0};
    const int expected_pointers[] = {0, 2, 2, 4};
    const int expected_indices[] = {0, 2, 0, 1};
    const double expected_values[] = {3, 0, 0, 3};
    matrix_desc_t matrix = {0};
    matrix.m = matrix.n = 3;
    matrix.fmt = matrix_csr;
    matrix.data.csr.nnz = 7;
    matrix.data.csr.row_ptr = pointers;
    matrix.data.csr.col_ind = indices;
    matrix.data.csr.vals = values;
    lp_problem_t *problem = create_lp_problem(NULL, &matrix, NULL, NULL, NULL, NULL, NULL);
    int failed = !problem;
    if (problem)
    {
        failed = problem->constraint_matrix_num_nonzeros != 4 ||
            memcmp(problem->constraint_matrix_row_pointers, expected_pointers, sizeof(expected_pointers)) ||
            memcmp(problem->constraint_matrix_col_indices, expected_indices, sizeof(expected_indices)) ||
            memcmp(problem->constraint_matrix_values, expected_values, sizeof(expected_values));
        lp_problem_free(problem);
    }
    const int original_pointers[] = {0, 4, 4, 7};
    const int original_indices[] = {2, 0, 2, 0, 1, 1, 0};
    const double original_values[] = {3, 2, -3, 1, 4, -1, 0};
    failed |= memcmp(pointers, original_pointers, sizeof(pointers)) != 0 ||
        memcmp(indices, original_indices, sizeof(indices)) != 0 ||
        memcmp(values, original_values, sizeof(values)) != 0;
    if (failed) fprintf(stderr, "CSR canonicalization or input ownership failed\n");
    return failed;
}

static int check_creation(matrix_format_t format, double first, double second,
                          int duplicate, int accepted)
{
    double values[] = {first, second};
    int pointers[] = {0, 2, 2};
    int indices[] = {0, duplicate ? 0 : 1};
    int rows[] = {0, 0};
    matrix_desc_t matrix = {0};
    matrix.m = 2;
    matrix.n = 2;
    matrix.fmt = format;
    double dense[] = {first, second, 0.0, 0.0};

    switch (format)
    {
    case matrix_dense:
        matrix.data.dense.A = dense;
        break;
    case matrix_csr:
        matrix.data.csr.nnz = 2;
        matrix.data.csr.row_ptr = pointers;
        matrix.data.csr.col_ind = indices;
        matrix.data.csr.vals = values;
        break;
    case matrix_csc:
        matrix.data.csc.nnz = 2;
        matrix.data.csc.col_ptr = pointers;
        matrix.data.csc.row_ind = indices;
        matrix.data.csc.vals = values;
        break;
    case matrix_coo:
        matrix.data.coo.nnz = 2;
        matrix.data.coo.row_ind = rows;
        matrix.data.coo.col_ind = indices;
        matrix.data.coo.vals = values;
        break;
    }

    lp_problem_t *problem = create_lp_problem(
        NULL, &matrix, NULL, NULL, NULL, NULL, NULL);
    int failed = (problem != NULL) != accepted;
    if (failed)
        fprintf(stderr, "format=%d values=(%g, %g) duplicate=%d: expected %s\n",
                format, first, second, duplicate,
                accepted ? "a problem" : "NULL");
    lp_problem_free(problem);
    return failed;
}

static int check_coefficients(void)
{
    int failures = check_canonical_csr();
    const matrix_format_t formats[] = {
        matrix_dense, matrix_csr, matrix_csc, matrix_coo};
    const double invalid[] = {NAN, INFINITY, -INFINITY};
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            failures += check_creation(formats[i], invalid[j], 1.0, 0, 0);
            failures += check_creation(formats[i], 1.0, invalid[j], 0, 0);
        }
        failures += check_creation(formats[i], 0.0, 0.0, 0, 1);
        failures += check_creation(formats[i], -2.0, 1e-12, 0, 1);
        if (formats[i] != matrix_dense)
        {
            failures += check_creation(formats[i], INFINITY, -INFINITY, 1, 0);
            failures += check_creation(formats[i], 2.0, -2.0, 1, 1);
        }
    }
    return failures ? 1 : 0;
}

static int check_dimensions(matrix_format_t format, int m, int n, int accepted)
{
    int pointers[] = {0, 0, 0};
    double values[] = {0.0, 0.0, 0.0, 0.0};
    matrix_desc_t matrix = {0};
    matrix.m = m;
    matrix.n = n;
    matrix.fmt = format;
    switch (format)
    {
    case matrix_dense:
        matrix.data.dense.A = values;
        break;
    case matrix_csr:
        matrix.data.csr.row_ptr = pointers;
        break;
    case matrix_csc:
        matrix.data.csc.col_ptr = pointers;
        break;
    case matrix_coo:
        break;
    }

    lp_problem_t *problem = create_lp_problem(
        NULL, &matrix, NULL, NULL, NULL, NULL, NULL);
    int failed = (problem != NULL) != accepted;
    if (failed)
        fprintf(stderr, "format=%d m=%d n=%d: expected %s\n",
                format, m, n, accepted ? "a problem" : "NULL");
    lp_problem_free(problem);
    return failed;
}

static int check_model_dimensions(void)
{
    const matrix_format_t formats[] = {
        matrix_dense, matrix_csr, matrix_csc, matrix_coo};
    int failures = 0;
    for (int i = 0; i < 4; ++i)
    {
        failures += check_dimensions(formats[i], 0, 0, 1);
        failures += check_dimensions(formats[i], 0, 2, 1);
        failures += check_dimensions(formats[i], 2, 0, 1);
        failures += check_dimensions(formats[i], 2, 2, 1);
    }
    lp_problem_t *problem = create_lp_problem(
        NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    if (problem != NULL)
    {
        fprintf(stderr, "NULL matrix descriptor: expected NULL\n");
        ++failures;
    }
    lp_problem_free(problem);
    return failures ? 1 : 0;
}

static int check_invalid_csr_structure(void)
{
    const double value = 1;
    int pointers[] = {0, 1, 1}, column = 0;
    matrix_desc_t matrix = {0};
    matrix.m = matrix.n = 2;
    matrix.fmt = matrix_csr;
    matrix.data.csr.nnz = 1;
    matrix.data.csr.row_ptr = pointers;
    matrix.data.csr.col_ind = &column;
    matrix.data.csr.vals = &value;
    const int invalid_pointers[][3] = {{1, 1, 1}, {0, 2, 1}, {0, 1, 0}};
    for (int i = 0; i < 3; ++i)
    {
        memcpy(pointers, invalid_pointers[i], sizeof(pointers));
        lp_problem_t *problem = create_lp_problem(NULL, &matrix, NULL, NULL, NULL, NULL, NULL);
        CHECK(problem == NULL);
        lp_problem_free(problem);
    }
    pointers[0] = 0; pointers[1] = pointers[2] = 1;
    const int invalid_columns[] = {-1, 2};
    for (int i = 0; i < 2; ++i)
    {
        column = invalid_columns[i];
        lp_problem_t *problem = create_lp_problem(NULL, &matrix, NULL, NULL, NULL, NULL, NULL);
        CHECK(problem == NULL);
        lp_problem_free(problem);
    }
    return failures != 0;
}

int main(void)
{
    int failed = 0;
    failed |= check_coefficients();
    failed |= check_model_dimensions();
    failed |= check_invalid_csr_structure();
    return failed || failures != 0;
}
