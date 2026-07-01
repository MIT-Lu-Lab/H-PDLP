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

#pragma once

#include "internal_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum
    {
        MATRIX_FORMAT_OK = 0,
        MATRIX_FORMAT_UNSUPPORTED,
        MATRIX_FORMAT_INVALID_COO_ROW,
        MATRIX_FORMAT_INVALID_COO_COL,
        MATRIX_FORMAT_INVALID_COO_DATA,
        MATRIX_FORMAT_INVALID_CSR_DATA,
        MATRIX_FORMAT_INVALID_CSR_ROW_PTR,
        MATRIX_FORMAT_INVALID_CSR_COL,
        MATRIX_FORMAT_INVALID_CSC_ROW,
        MATRIX_FORMAT_INVALID_CSC_DATA,
        MATRIX_FORMAT_INVALID_DENSE_DATA,
        MATRIX_FORMAT_INVALID_DESCRIPTOR
    } matrix_format_status_t;

    const char *matrix_format_status_message(matrix_format_status_t status);
    matrix_format_status_t matrix_desc_to_csr(const matrix_desc_t *desc,
                                              int **row_ptr,
                                              int **col_ind,
                                              double **vals,
                                              int *nnz_out);

    matrix_format_status_t dense_to_csr(const matrix_desc_t *desc, int **row_ptr,
                                        int **col_ind, double **vals, int *nnz_out);
    matrix_format_status_t csc_to_csr(const matrix_desc_t *desc, int **row_ptr,
                                      int **col_ind, double **vals, int *nnz_out);
    matrix_format_status_t coo_to_csr(const matrix_desc_t *desc, int **row_ptr,
                                      int **col_ind, double **vals, int *nnz_out);
    matrix_format_status_t canonicalize_csr(const matrix_desc_t *desc,
                                            int **row_ptr, int **col_ind,
                                            double **vals, int *nnz_out);

#ifdef __cplusplus
}
#endif
