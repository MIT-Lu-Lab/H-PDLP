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

#include "power_iteration.h"
#include "utils.h"
#include <math.h>

__global__ void normalize_power_iteration_vector_kernel(
    double *normalized_vector_d,
    const double *vector_d,
    const double *vector_norm_d,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double vector_norm = *vector_norm_d;
    normalized_vector_d[i] = (vector_norm > 0.0) ? (vector_d[i] / vector_norm) : 0.0;
}

__global__ void compute_power_iteration_residual_kernel(
    double *residual_d,
    const double *vector_d,
    const double *normalized_vector_d,
    const double *sigma_squared_d,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double sigma_squared = *sigma_squared_d;
    residual_d[i] = vector_d[i] - sigma_squared * normalized_vector_d[i];
}

double estimate_maximum_singular_value(
    pdhg_solver_state_t *state,
    const power_iteration_parameters_t *power_iter_params)
{
    const int residual_check_interval = 5;
    int m = state->num_constraints;
    double *power_iteration_vector_d = state->power_iteration_vector_d;
    double *power_iteration_normalized_vector_d = state->power_iteration_normalized_vector_d;

    size_t curand_count = ((size_t)m + 1) & ~(size_t)1;
    CURAND_CHECK(curandGenerateNormalDouble(
        state->curand_gen,
        power_iteration_vector_d,
        curand_count,
        0.0,
        1.0));

    *state->power_iter_sigma_squared_mp = 1.0;
    *state->power_iter_residual_norm_mp = INFINITY;

    for (int i = 0; i < power_iter_params->max_iter; ++i)
    {
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A,
            m,
            power_iteration_vector_d,
            1,
            state->power_iteration_vector_norm_scalar_d));

        normalize_power_iteration_vector_kernel<<<
            state->num_blocks_dual,
            THREADS_PER_BLOCK,
            0,
            state->stream_A>>>(
            power_iteration_normalized_vector_d,
            power_iteration_vector_d,
            state->power_iteration_vector_norm_scalar_d,
            m);
        CUDA_CHECK(cudaGetLastError());

        CUSPARSE_CHECK(cusparseSpMV(
            state->sparse_handle_A,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &HOST_ONE,
            state->matAt,
            state->spmv_power_iteration_normalized_vector,
            &HOST_ZERO,
            state->spmv_power_iteration_product,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG2,
            state->dual_spmv_buffer));

        CUSPARSE_CHECK(cusparseSpMV(
            state->sparse_handle_A,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &HOST_ONE,
            state->matA,
            state->spmv_power_iteration_product,
            &HOST_ZERO,
            state->spmv_power_iteration_vector,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG2,
            state->primal_spmv_buffer));

        CUBLAS_CHECK(cublasDdot(
            state->blas_handle_A,
            m,
            power_iteration_normalized_vector_d,
            1,
            power_iteration_vector_d,
            1,
            state->power_iter_sigma_squared_mp));

        compute_power_iteration_residual_kernel<<<
            state->num_blocks_dual,
            THREADS_PER_BLOCK,
            0,
            state->stream_A>>>(
            power_iteration_normalized_vector_d,
            power_iteration_vector_d,
            power_iteration_normalized_vector_d,
            state->power_iter_sigma_squared_mp,
            m);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A,
            m,
            power_iteration_normalized_vector_d,
            1,
            state->power_iter_residual_norm_mp));

        if (((i + 1) % residual_check_interval) == 0 || (i + 1) == power_iter_params->max_iter)
        {
            CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
            if (*state->power_iter_residual_norm_mp < power_iter_params->tol)
                break;
        }
    }

    return sqrt(*state->power_iter_sigma_squared_mp);
}
