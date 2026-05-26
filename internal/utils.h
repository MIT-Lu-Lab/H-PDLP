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
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess)                                    \
        {                                                          \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorName(err));              \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                           \
    do                                                               \
    {                                                                \
        cublasStatus_t status = call;                                \
        if (status != CUBLAS_STATUS_SUCCESS)                         \
        {                                                            \
            fprintf(stderr, "cuBLAS Error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cublasGetStatusName(status));          \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

#define CUSPARSE_CHECK(call)                                           \
    do                                                                 \
    {                                                                  \
        cusparseStatus_t status = call;                                \
        if (status != CUSPARSE_STATUS_SUCCESS)                         \
        {                                                              \
            fprintf(stderr, "cuSPARSE Error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cusparseGetErrorName(status));           \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#define CURAND_CHECK(call)                                              \
    do                                                                  \
    {                                                                   \
        curandStatus_t status = call;                                   \
        if (status != CURAND_STATUS_SUCCESS)                            \
        {                                                               \
            fprintf(stderr, "cuRAND Error at %s:%d: %d\n", __FILE__,   \
                    __LINE__, (int)status);                             \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

#define THREADS_PER_BLOCK 256

    extern const double HOST_ONE;
    extern const double HOST_ZERO;

    void *safe_malloc(size_t size);
    void *safe_calloc(size_t num, size_t size);
    void *safe_realloc(void *ptr, size_t new_size);

    void reduce_device_sum(
        pdhg_solver_state_t *state,
        const double *input,
        int length,
        double *output);

    void reduce_device_max(
        pdhg_solver_state_t *state,
        const double *input,
        int length,
        double *output);

    void reduce_device_abs_max(
        pdhg_solver_state_t *state,
        const double *input,
        int length,
        double *output);

    void filter_constraint_matrix_entries(
        lp_problem_t *problem,
        const pdhg_parameters_t *params);

    void compute_lp_problem_norms(
        const lp_problem_t *problem,
        norm_type_t optimality_norm,
        double *out_objective_norm,
        double *out_bound_norm,
        double *out_constraint_frobenius_norm);

    void compute_lp_problem_norms_device(
        pdhg_solver_state_t *state,
        norm_type_t optimality_norm,
        double *out_objective_norm,
        double *out_bound_norm);

#ifdef __cplusplus
}
#endif
