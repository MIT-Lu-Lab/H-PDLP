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
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum
    {
        SV_ESTIMATOR_CONVERGED  = 0,
        SV_ESTIMATOR_EARLY_EXIT = 1,
        SV_ESTIMATOR_MAX_ITER   = 2,
        SV_ESTIMATOR_DEGENERATE = 3,
    } sv_estimator_status_t;

    typedef struct
    {
        int max_iter;
        double tol;
        const bool *row_mask_d;
        const bool *column_mask_d;
        double early_exit_singular_value_threshold;
    } sv_estimator_opts_t;

    typedef struct
    {
        sv_estimator_status_t status;
        double max_singular_value;
        int iterations;
    } sv_estimator_result_t;

    sv_estimator_ctx_t *sv_estimator_create(
        cusparseHandle_t sparse_handle,
        cublasHandle_t blas_handle,
        cudaStream_t stream,
        curandGenerator_t curand_gen,
        const cu_sparse_matrix_csr_t *matrix,
        const cu_sparse_matrix_csr_t *transpose,
        int num_rows,
        int num_columns);

    sv_estimator_result_t sv_estimator_run(
        sv_estimator_ctx_t *ctx,
        const sv_estimator_opts_t *opts);

    void sv_estimator_free(sv_estimator_ctx_t *ctx);

#ifdef __cplusplus
}
#endif
