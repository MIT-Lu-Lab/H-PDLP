/*
Copyright 2026 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "matrix_formats.h"
#include "utils.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    int col;
    int position;
    double value;
} RowEntry;

static matrix_format_status_t validate_conversion_args(
    const matrix_desc_t *desc,
    int **row_ptr,
    int **col_ind,
    double **vals,
    int *nnz_out);
static void canonicalize_sorted_rows_csr(int m,
                                         const int *input_row_ptr,
                                         const int *input_col_ind,
                                         const double *input_vals,
                                         int **row_ptr, int **col_ind,
                                         double **vals, int *nnz_out);
static void init_empty_csr(int m, int **row_ptr, int **col_ind,
                           double **vals, int *nnz_out);
static void sort_unsorted_rows(int m, const int *row_ptr, int *col_ind,
                               double *vals);
static int compare_row_entries(const void *lhs, const void *rhs);
static matrix_format_status_t validate_csr_structure(
    int m, int n, int nnz_in, const int *row_ptr, const int *col_ind);

const char *matrix_format_status_message(matrix_format_status_t status)
{
    switch (status)
    {
    case MATRIX_FORMAT_OK:
        return "ok";
    case MATRIX_FORMAT_UNSUPPORTED:
        return "unsupported matrix format";
    case MATRIX_FORMAT_INVALID_COO_ROW:
        return "invalid COO row index";
    case MATRIX_FORMAT_INVALID_COO_COL:
        return "invalid COO column index";
    case MATRIX_FORMAT_INVALID_COO_DATA:
        return "invalid COO matrix data";
    case MATRIX_FORMAT_INVALID_CSR_DATA:
        return "invalid CSR matrix data";
    case MATRIX_FORMAT_INVALID_CSR_ROW_PTR:
        return "invalid CSR row pointer";
    case MATRIX_FORMAT_INVALID_CSR_COL:
        return "invalid CSR column index";
    case MATRIX_FORMAT_INVALID_CSC_ROW:
        return "invalid CSC row index";
    case MATRIX_FORMAT_INVALID_CSC_DATA:
        return "invalid CSC matrix data";
    case MATRIX_FORMAT_INVALID_DENSE_DATA:
        return "invalid dense matrix data";
    case MATRIX_FORMAT_INVALID_DESCRIPTOR:
        return "invalid matrix descriptor";
    default:
        return "unknown matrix format error";
    }
}

matrix_format_status_t matrix_desc_to_csr(const matrix_desc_t *desc,
                                          int **row_ptr,
                                          int **col_ind,
                                          double **vals,
                                          int *nnz_out)
{
    matrix_format_status_t status =
        validate_conversion_args(desc, row_ptr, col_ind, vals, nnz_out);
    if (status != MATRIX_FORMAT_OK)
        return status;

    switch (desc->fmt)
    {
    case matrix_dense:
        return dense_to_csr(desc, row_ptr, col_ind, vals, nnz_out);
    case matrix_csc:
        return csc_to_csr(desc, row_ptr, col_ind, vals, nnz_out);
    case matrix_coo:
        return coo_to_csr(desc, row_ptr, col_ind, vals, nnz_out);
    case matrix_csr:
        return canonicalize_csr(desc, row_ptr, col_ind, vals, nnz_out);
    default:
        return MATRIX_FORMAT_UNSUPPORTED;
    }
}

static matrix_format_status_t validate_conversion_args(
    const matrix_desc_t *desc,
    int **row_ptr,
    int **col_ind,
    double **vals,
    int *nnz_out)
{
    if (!desc || !row_ptr || !col_ind || !vals || !nnz_out)
    {
        return MATRIX_FORMAT_INVALID_DESCRIPTOR;
    }

    return MATRIX_FORMAT_OK;
}

matrix_format_status_t dense_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind,
                                    double **vals, int *nnz_out)
{
    matrix_format_status_t status =
        validate_conversion_args(desc, row_ptr, col_ind, vals, nnz_out);
    if (status != MATRIX_FORMAT_OK)
        return status;

    const int m = desc->m;
    const int n = desc->n;

    if (m < 0 || n < 0 ||
        ((size_t)m * (size_t)n > 0 && desc->data.dense.A == NULL))
    {
        return MATRIX_FORMAT_INVALID_DENSE_DATA;
    }

    size_t nnz = 0;
    for (size_t i = 0; i < (size_t)m * (size_t)n; ++i)
    {
        if (desc->data.dense.A[i] != 0.0)
            ++nnz;
    }
    if (nnz > (size_t)INT_MAX)
    {
        return MATRIX_FORMAT_INVALID_DENSE_DATA;
    }

    *row_ptr = (int *)safe_malloc((size_t)(m + 1) * sizeof(int));
    *col_ind = (int *)safe_malloc((size_t)((nnz > 0) ? nnz : 1) * sizeof(int));
    *vals = (double *)safe_malloc((size_t)((nnz > 0) ? nnz : 1) * sizeof(double));

    int nnz_filled = 0;
    for (int i = 0; i < m; ++i)
    {
        (*row_ptr)[i] = nnz_filled;
        for (int j = 0; j < n; ++j)
        {
            double val = desc->data.dense.A[(size_t)i * (size_t)n + (size_t)j];
            if (val == 0.0)
                continue;
            (*col_ind)[nnz_filled] = j;
            (*vals)[nnz_filled] = val;
            ++nnz_filled;
        }
    }
    (*row_ptr)[m] = nnz_filled;
    *nnz_out = nnz_filled;
    return MATRIX_FORMAT_OK;
}

matrix_format_status_t csc_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind,
                                  double **vals, int *nnz_out)
{
    matrix_format_status_t status =
        validate_conversion_args(desc, row_ptr, col_ind, vals, nnz_out);
    if (status != MATRIX_FORMAT_OK)
        return status;

    const int m = desc->m;
    const int n = desc->n;
    const int nnz_in = desc->data.csc.nnz;
    const int *col_ptr = desc->data.csc.col_ptr;
    const int *row_ind = desc->data.csc.row_ind;
    const double *csc_vals = desc->data.csc.vals;

    if (m < 0 || n < 0 || nnz_in < 0 || col_ptr == NULL ||
        (nnz_in > 0 && (row_ind == NULL || csc_vals == NULL)) ||
        col_ptr[0] != 0 || col_ptr[n] != nnz_in)
    {
        return MATRIX_FORMAT_INVALID_CSC_DATA;
    }
    for (int j = 0; j < n; ++j)
    {
        if (col_ptr[j] > col_ptr[j + 1])
        {
            return MATRIX_FORMAT_INVALID_CSC_DATA;
        }
    }

    int *raw_row_ptr = (int *)safe_calloc((size_t)(m + 1), sizeof(int));

    for (int j = 0; j < n; ++j)
    {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
        {
            int ri = row_ind[k];
            if (ri < 0 || ri >= m)
            {
                free(raw_row_ptr);
                return MATRIX_FORMAT_INVALID_CSC_ROW;
            }
            ++(raw_row_ptr[ri + 1]);
        }
    }

    for (int i = 0; i < m; ++i)
        raw_row_ptr[i + 1] += raw_row_ptr[i];

    const size_t alloc_nnz = (nnz_in > 0) ? (size_t)nnz_in : 1;
    int *raw_col_ind = (int *)safe_malloc(alloc_nnz * sizeof(int));
    double *raw_vals = (double *)safe_malloc(alloc_nnz * sizeof(double));

    int *next = (int *)safe_malloc((size_t)m * sizeof(int));
    for (int i = 0; i < m; ++i)
        next[i] = raw_row_ptr[i];

    for (int j = 0; j < n; ++j)
    {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
        {
            int ri = row_ind[k];
            int pos = next[ri]++;
            raw_col_ind[pos] = j;
            raw_vals[pos] = csc_vals[k];
        }
    }

    free(next);
    canonicalize_sorted_rows_csr(m, raw_row_ptr, raw_col_ind,
                                 raw_vals, row_ptr, col_ind, vals, nnz_out);
    free(raw_row_ptr);
    free(raw_col_ind);
    free(raw_vals);
    return MATRIX_FORMAT_OK;
}

static void canonicalize_sorted_rows_csr(int m,
                                         const int *input_row_ptr,
                                         const int *input_col_ind,
                                         const double *input_vals,
                                         int **row_ptr, int **col_ind,
                                         double **vals, int *nnz_out)
{
    *row_ptr = (int *)safe_calloc((size_t)(m + 1), sizeof(int));

    int canonical_nnz = 0;
    for (int i = 0; i < m; ++i)
    {
        int last_col = -1;
        for (int k = input_row_ptr[i]; k < input_row_ptr[i + 1]; ++k)
        {
            int col = input_col_ind[k];
            if (col == last_col)
                continue;
            last_col = col;
            ++(*row_ptr)[i + 1];
            ++canonical_nnz;
        }
    }

    for (int i = 0; i < m; ++i)
        (*row_ptr)[i + 1] += (*row_ptr)[i];

    const size_t alloc_nnz = (canonical_nnz > 0) ? (size_t)canonical_nnz : 1;
    *col_ind = (int *)safe_malloc(alloc_nnz * sizeof(int));
    *vals = (double *)safe_malloc(alloc_nnz * sizeof(double));

    int out = 0;
    for (int i = 0; i < m; ++i)
    {
        for (int k = input_row_ptr[i]; k < input_row_ptr[i + 1]; ++k)
        {
            int col = input_col_ind[k];
            if (k > input_row_ptr[i] && col == input_col_ind[k - 1])
            {
                (*vals)[out - 1] += input_vals[k];
            }
            else
            {
                (*col_ind)[out] = col;
                (*vals)[out] = input_vals[k];
                ++out;
            }
        }
    }

    *nnz_out = out;
}

matrix_format_status_t coo_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind,
                                  double **vals, int *nnz_out)
{
    matrix_format_status_t status =
        validate_conversion_args(desc, row_ptr, col_ind, vals, nnz_out);
    if (status != MATRIX_FORMAT_OK)
        return status;

    const int m = desc->m;
    const int n = desc->n;
    const int nnz_in = desc->data.coo.nnz;
    const int *coo_row_ind = desc->data.coo.row_ind;
    const int *coo_col_ind = desc->data.coo.col_ind;
    const double *coo_vals = desc->data.coo.vals;

    if (m < 0 || n < 0 || nnz_in < 0 ||
        (nnz_in > 0 &&
         (coo_row_ind == NULL || coo_col_ind == NULL || coo_vals == NULL)))
    {
        return MATRIX_FORMAT_INVALID_COO_DATA;
    }

    for (int k = 0; k < nnz_in; ++k)
    {
        int ri = coo_row_ind[k];
        int cj = coo_col_ind[k];
        if (ri < 0 || ri >= m)
        {
            return MATRIX_FORMAT_INVALID_COO_ROW;
        }
        if (cj < 0 || cj >= n)
        {
            return MATRIX_FORMAT_INVALID_COO_COL;
        }
    }

    if (nnz_in == 0)
    {
        init_empty_csr(m, row_ptr, col_ind, vals, nnz_out);
        return MATRIX_FORMAT_OK;
    }

    int *raw_row_ptr = (int *)safe_calloc((size_t)(m + 1), sizeof(int));
    for (int k = 0; k < nnz_in; ++k)
        ++(raw_row_ptr[coo_row_ind[k] + 1]);

    for (int i = 0; i < m; ++i)
        raw_row_ptr[i + 1] += raw_row_ptr[i];

    int *raw_col_ind = (int *)safe_malloc((size_t)nnz_in * sizeof(int));
    double *raw_vals = (double *)safe_malloc((size_t)nnz_in * sizeof(double));

    int *next = (int *)safe_malloc((size_t)m * sizeof(int));
    memcpy(next, raw_row_ptr, (size_t)m * sizeof(int));

    for (int k = 0; k < nnz_in; ++k)
    {
        int pos = next[coo_row_ind[k]]++;
        raw_col_ind[pos] = coo_col_ind[k];
        raw_vals[pos] = coo_vals[k];
    }

    free(next);
    sort_unsorted_rows(m, raw_row_ptr, raw_col_ind, raw_vals);
    canonicalize_sorted_rows_csr(m, raw_row_ptr, raw_col_ind,
                                 raw_vals, row_ptr, col_ind, vals, nnz_out);
    free(raw_row_ptr);
    free(raw_col_ind);
    free(raw_vals);
    return MATRIX_FORMAT_OK;
}

static void init_empty_csr(int m, int **row_ptr, int **col_ind,
                           double **vals, int *nnz_out)
{
    *row_ptr = (int *)safe_calloc((size_t)(m + 1), sizeof(int));
    *col_ind = (int *)safe_malloc(sizeof(int));
    *vals = (double *)safe_malloc(sizeof(double));
    *nnz_out = 0;
}

static void sort_unsorted_rows(int m, const int *row_ptr, int *col_ind,
                               double *vals)
{
    RowEntry *entries = NULL;
    size_t entries_capacity = 0;

    for (int i = 0; i < m; ++i)
    {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        int sorted_end = row_start + 1;
        while (sorted_end < row_end && col_ind[sorted_end - 1] <= col_ind[sorted_end])
            ++sorted_end;
        if (sorted_end >= row_end)
            continue;

        size_t row_nnz = (size_t)(row_end - row_start);
        if (row_nnz > entries_capacity)
        {
            entries_capacity = row_nnz;
            entries = (RowEntry *)safe_realloc(entries,
                                               entries_capacity * sizeof(RowEntry));
        }
        for (int k = row_start; k < row_end; ++k)
        {
            entries[k - row_start].col = col_ind[k];
            entries[k - row_start].position = k;
            entries[k - row_start].value = vals[k];
        }
        qsort(entries, row_nnz, sizeof(RowEntry), compare_row_entries);
        for (int k = row_start; k < row_end; ++k)
        {
            col_ind[k] = entries[k - row_start].col;
            vals[k] = entries[k - row_start].value;
        }
    }

    free(entries);
}

static int compare_row_entries(const void *lhs, const void *rhs)
{
    const RowEntry *a = (const RowEntry *)lhs;
    const RowEntry *b = (const RowEntry *)rhs;

    if (a->col != b->col)
        return (a->col < b->col) ? -1 : 1;
    if (a->position != b->position)
        return (a->position < b->position) ? -1 : 1;
    return 0;
}

matrix_format_status_t canonicalize_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind,
                                        double **vals, int *nnz_out)
{
    matrix_format_status_t status =
        validate_conversion_args(desc, row_ptr, col_ind, vals, nnz_out);
    if (status != MATRIX_FORMAT_OK)
        return status;

    const int m = desc->m;
    const int n = desc->n;
    const int nnz_in = desc->data.csr.nnz;
    const int *csr_row_ptr = desc->data.csr.row_ptr;
    const int *csr_col_ind = desc->data.csr.col_ind;
    const double *csr_vals = desc->data.csr.vals;

    if (m < 0 || n < 0 || nnz_in < 0 || csr_row_ptr == NULL ||
        (nnz_in > 0 && (csr_col_ind == NULL || csr_vals == NULL)) ||
        csr_row_ptr[0] != 0)
    {
        return MATRIX_FORMAT_INVALID_CSR_DATA;
    }

    status = validate_csr_structure(m, n, nnz_in, csr_row_ptr, csr_col_ind);
    if (status != MATRIX_FORMAT_OK)
    {
        return status;
    }
    if (nnz_in == 0)
    {
        init_empty_csr(m, row_ptr, col_ind, vals, nnz_out);
        return MATRIX_FORMAT_OK;
    }

    int *sorted_col_ind = safe_malloc((size_t)nnz_in * sizeof(int));
    double *sorted_vals = safe_malloc((size_t)nnz_in * sizeof(double));
    memcpy(sorted_col_ind, csr_col_ind, (size_t)nnz_in * sizeof(int));
    memcpy(sorted_vals, csr_vals, (size_t)nnz_in * sizeof(double));
    sort_unsorted_rows(m, csr_row_ptr, sorted_col_ind, sorted_vals);
    canonicalize_sorted_rows_csr(m, csr_row_ptr, sorted_col_ind, sorted_vals,
                                row_ptr, col_ind, vals, nnz_out);
    free(sorted_vals);
    free(sorted_col_ind);
    return MATRIX_FORMAT_OK;
}

static matrix_format_status_t validate_csr_structure(
    int m, int n, int nnz_in, const int *row_ptr, const int *col_ind)
{
    if (row_ptr[m] != nnz_in)
    {
        return MATRIX_FORMAT_INVALID_CSR_ROW_PTR;
    }
    for (int i = 0; i < m; ++i)
    {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        if (row_start > row_end || row_start < 0 || row_end > nnz_in)
        {
            return MATRIX_FORMAT_INVALID_CSR_ROW_PTR;
        }
        for (int k = row_start; k < row_end; ++k)
        {
            int cj = col_ind[k];
            if (cj < 0 || cj >= n)
            {
                return MATRIX_FORMAT_INVALID_CSR_COL;
            }
        }
    }

    return MATRIX_FORMAT_OK;
}

void transpose_csr_and_build_map(
    const lp_problem_t *problem,
    int **transpose_row_ptr,
    int **transpose_col_ind,
    double **transpose_val,
    int **transpose_map)
{
    const int num_rows = problem->num_constraints;
    const int num_cols = problem->num_variables;
    const int nnz = problem->constraint_matrix_num_nonzeros;
    const int *row_ptr = problem->constraint_matrix_row_pointers;
    const int *col_ind = problem->constraint_matrix_col_indices;
    const double *val = problem->constraint_matrix_values;
    const size_t alloc_nnz = nnz > 0 ? (size_t)nnz : 1;
    const size_t alloc_cols = num_cols > 0 ? (size_t)num_cols : 1;

    *transpose_row_ptr = (int *)safe_calloc((size_t)(num_cols + 1), sizeof(int));
    *transpose_col_ind = (int *)safe_malloc(alloc_nnz * sizeof(int));
    *transpose_val = (double *)safe_malloc(alloc_nnz * sizeof(double));
    *transpose_map = (int *)safe_malloc(alloc_nnz * sizeof(int));

    for (int k = 0; k < nnz; ++k)
        (*transpose_row_ptr)[col_ind[k] + 1]++;

    for (int j = 0; j < num_cols; ++j)
        (*transpose_row_ptr)[j + 1] += (*transpose_row_ptr)[j];

    int *next = (int *)safe_malloc(alloc_cols * sizeof(int));
    if (num_cols > 0)
        memcpy(next, *transpose_row_ptr, (size_t)num_cols * sizeof(int));

    for (int i = 0; i < num_rows; ++i)
    {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        {
            const int j = col_ind[k];
            const int pos = next[j]++;
            (*transpose_col_ind)[pos] = i;
            (*transpose_val)[pos] = val[k];
            (*transpose_map)[k] = pos;
        }
    }

    free(next);
}
