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
#include <stdlib.h>

typedef struct
{
    int row;
    int col;
    double value;
} CooEntry;

static int compare_coo_entries(const void *lhs, const void *rhs);
static int build_coo_entries(int m, int n,
                             int nnz_in,
                             const int *row_ind,
                             const int *col_ind,
                             const double *input_vals,
                             CooEntry **entries_out,
                             int *nnz_out,
                             matrix_format_status_t *status_out);
static int build_csr_entries(int m, int n,
                             int nnz_in,
                             const int *row_ptr,
                             const int *col_ind,
                             const double *input_vals,
                             CooEntry **entries_out,
                             int *nnz_out,
                             matrix_format_status_t *status_out);
static void sorted_coo_to_csr(const CooEntry *entries, int m, int nnz,
                              int **row_ptr, int **col_ind,
                              double **vals, int *nnz_out);
static void canonicalize_sorted_rows_csr(int m,
                                         const int *input_row_ptr,
                                         const int *input_col_ind,
                                         const double *input_vals,
                                         int **row_ptr, int **col_ind,
                                         double **vals, int *nnz_out);
static void init_empty_csr(int m, int **row_ptr, int **col_ind,
                           double **vals, int *nnz_out);

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
    case MATRIX_FORMAT_INVALID_CSR_DATA:
        return "invalid CSR matrix data";
    case MATRIX_FORMAT_INVALID_CSR_ROW_PTR:
        return "invalid CSR row pointer";
    case MATRIX_FORMAT_INVALID_CSR_COL:
        return "invalid CSR column index";
    case MATRIX_FORMAT_INVALID_CSC_ROW:
        return "invalid CSC row index";
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

matrix_format_status_t dense_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind,
                                    double **vals, int *nnz_out)
{
    const int m = desc->m;
    const int n = desc->n;

    int nnz = 0;
    for (int i = 0; i < m * n; ++i)
    {
        if (desc->data.dense.A[i] != 0.0)
            ++nnz;
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
            double val = desc->data.dense.A[i * n + j];
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
    const int m = desc->m;
    const int n = desc->n;
    const int *col_ptr = desc->data.csc.col_ptr;
    const int *row_ind = desc->data.csc.row_ind;
    const double *csc_vals = desc->data.csc.vals;

    int *raw_row_ptr = (int *)safe_malloc((size_t)(m + 1) * sizeof(int));
    for (int i = 0; i <= m; ++i)
        raw_row_ptr[i] = 0;

    int raw_nnz = 0;
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
            ++raw_nnz;
        }
    }

    for (int i = 0; i < m; ++i)
        raw_row_ptr[i + 1] += raw_row_ptr[i];

    const size_t alloc_nnz = (raw_nnz > 0) ? (size_t)raw_nnz : 1;
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

matrix_format_status_t coo_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind,
                                  double **vals, int *nnz_out)
{
    const int m = desc->m;
    const int n = desc->n;
    const int nnz_in = desc->data.coo.nnz;
    const int *coo_row_ind = desc->data.coo.row_ind;
    const int *coo_col_ind = desc->data.coo.col_ind;
    const double *coo_vals = desc->data.coo.vals;

    CooEntry *entries = NULL;
    int entry_count = 0;
    matrix_format_status_t status = MATRIX_FORMAT_OK;
    if (build_coo_entries(m, n, nnz_in, coo_row_ind, coo_col_ind,
                          coo_vals, &entries, &entry_count, &status) != 0)
    {
        return status;
    }
    if (entry_count == 0)
    {
        init_empty_csr(m, row_ptr, col_ind, vals, nnz_out);
        return MATRIX_FORMAT_OK;
    }

    qsort(entries, (size_t)entry_count, sizeof(CooEntry), compare_coo_entries);
    sorted_coo_to_csr(entries, m, entry_count, row_ptr, col_ind, vals, nnz_out);
    free(entries);
    return MATRIX_FORMAT_OK;
}

matrix_format_status_t canonicalize_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind,
                                        double **vals, int *nnz_out)
{
    const int m = desc->m;
    const int n = desc->n;
    const int nnz_in = desc->data.csr.nnz;
    const int *csr_row_ptr = desc->data.csr.row_ptr;
    const int *csr_col_ind = desc->data.csr.col_ind;
    const double *csr_vals = desc->data.csr.vals;

    if (m < 0 || n < 0 || nnz_in < 0 || csr_row_ptr == NULL ||
        csr_col_ind == NULL || csr_vals == NULL || csr_row_ptr[0] != 0)
    {
        return MATRIX_FORMAT_INVALID_CSR_DATA;
    }

    CooEntry *entries = NULL;
    int entry_count = 0;
    matrix_format_status_t status = MATRIX_FORMAT_OK;
    if (build_csr_entries(m, n, nnz_in, csr_row_ptr, csr_col_ind,
                          csr_vals, &entries, &entry_count, &status) != 0)
    {
        return status;
    }
    if (entry_count == 0)
    {
        init_empty_csr(m, row_ptr, col_ind, vals, nnz_out);
        return MATRIX_FORMAT_OK;
    }

    qsort(entries, (size_t)entry_count, sizeof(CooEntry), compare_coo_entries);
    sorted_coo_to_csr(entries, m, entry_count, row_ptr, col_ind, vals, nnz_out);
    free(entries);
    return MATRIX_FORMAT_OK;
}

static int compare_coo_entries(const void *lhs, const void *rhs)
{
    const CooEntry *a = (const CooEntry *)lhs;
    const CooEntry *b = (const CooEntry *)rhs;

    if (a->row != b->row)
        return (a->row < b->row) ? -1 : 1;
    if (a->col != b->col)
        return (a->col < b->col) ? -1 : 1;
    return 0;
}

static int build_coo_entries(int m, int n,
                             int nnz_in,
                             const int *row_ind,
                             const int *col_ind,
                             const double *input_vals,
                             CooEntry **entries_out,
                             int *nnz_out,
                             matrix_format_status_t *status_out)
{
    for (int k = 0; k < nnz_in; ++k)
    {
        int ri = row_ind[k];
        int cj = col_ind[k];
        if (ri < 0 || ri >= m)
        {
            *status_out = MATRIX_FORMAT_INVALID_COO_ROW;
            return -1;
        }
        if (cj < 0 || cj >= n)
        {
            *status_out = MATRIX_FORMAT_INVALID_COO_COL;
            return -1;
        }
    }

    if (nnz_in == 0)
    {
        *entries_out = NULL;
        *nnz_out = 0;
        return 0;
    }

    CooEntry *entries = (CooEntry *)safe_malloc((size_t)nnz_in * sizeof(CooEntry));
    for (int k = 0; k < nnz_in; ++k)
    {
        entries[k].row = row_ind[k];
        entries[k].col = col_ind[k];
        entries[k].value = input_vals[k];
    }

    *entries_out = entries;
    *nnz_out = nnz_in;
    return 0;
}

static int build_csr_entries(int m, int n,
                             int nnz_in,
                             const int *row_ptr,
                             const int *col_ind,
                             const double *input_vals,
                             CooEntry **entries_out,
                             int *nnz_out,
                             matrix_format_status_t *status_out)
{
    for (int i = 0; i < m; ++i)
    {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        if (row_start > row_end || row_start < 0 || row_end > nnz_in)
        {
            *status_out = MATRIX_FORMAT_INVALID_CSR_ROW_PTR;
            return -1;
        }
        for (int k = row_start; k < row_end; ++k)
        {
            int cj = col_ind[k];
            if (cj < 0 || cj >= n)
            {
                *status_out = MATRIX_FORMAT_INVALID_CSR_COL;
                return -1;
            }
        }
    }

    if (nnz_in == 0)
    {
        *entries_out = NULL;
        *nnz_out = 0;
        return 0;
    }

    CooEntry *entries = (CooEntry *)safe_malloc((size_t)nnz_in * sizeof(CooEntry));
    int pos = 0;
    for (int i = 0; i < m; ++i)
    {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        {
            entries[pos].row = i;
            entries[pos].col = col_ind[k];
            entries[pos].value = input_vals[k];
            ++pos;
        }
    }

    *entries_out = entries;
    *nnz_out = nnz_in;
    return 0;
}

static void sorted_coo_to_csr(const CooEntry *entries, int m, int nnz,
                              int **row_ptr, int **col_ind,
                              double **vals, int *nnz_out)
{
    *row_ptr = (int *)safe_calloc((size_t)(m + 1), sizeof(int));

    int canonical_nnz = 0;
    int last_row = -1;
    int last_col = -1;
    for (int k = 0; k < nnz; ++k)
    {
        if (entries[k].row == last_row && entries[k].col == last_col)
            continue;
        last_row = entries[k].row;
        last_col = entries[k].col;
        ++(*row_ptr)[entries[k].row + 1];
        ++canonical_nnz;
    }

    for (int i = 0; i < m; ++i)
        (*row_ptr)[i + 1] += (*row_ptr)[i];

    const size_t alloc_nnz = (canonical_nnz > 0) ? (size_t)canonical_nnz : 1;
    *col_ind = (int *)safe_malloc(alloc_nnz * sizeof(int));
    *vals = (double *)safe_malloc(alloc_nnz * sizeof(double));

    int out = 0;
    for (int k = 0; k < nnz; ++k)
    {
        if (out > 0 &&
            entries[k].row == entries[k - 1].row &&
            entries[k].col == entries[k - 1].col)
        {
            (*vals)[out - 1] += entries[k].value;
        }
        else
        {
            (*col_ind)[out] = entries[k].col;
            (*vals)[out] = entries[k].value;
            ++out;
        }
    }

    *nnz_out = out;
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

static void init_empty_csr(int m, int **row_ptr, int **col_ind,
                           double **vals, int *nnz_out)
{
    *row_ptr = (int *)safe_calloc((size_t)(m + 1), sizeof(int));
    *col_ind = (int *)safe_malloc(sizeof(int));
    *vals = (double *)safe_malloc(sizeof(double));
    *nnz_out = 0;
}
