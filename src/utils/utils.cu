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

#include "utils.h"
#include <cub/cub.cuh>
#include <math.h>

#ifndef CUHSDLP_VERSION
#define CUHSDLP_VERSION "unknown"
#endif

__global__ void normalize_power_iteration_vector_kernel(
    double *normalized_vector_d,
    const double *vector_d,
    const double *vector_norm_d,
    int n);
__global__ void compute_power_iteration_residual_kernel(
    double *residual_d,
    const double *vector_d,
    const double *normalized_vector_d,
    const double *sigma_squared_d,
    int n);
const double HOST_ONE = 1.0;
const double HOST_ZERO = 0.0;

void *safe_malloc(size_t size)
{
    if (size == 0)
        size = 1;

    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        perror("Fatal error: malloc failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_calloc(size_t num, size_t size)
{
    if (num == 0 || size == 0)
    {
        num = 1;
        size = 1;
    }

    void *ptr = calloc(num, size);
    if (ptr == NULL)
    {
        perror("Fatal error: calloc failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_realloc(void *ptr, size_t new_size)
{
    if (new_size == 0)
        new_size = 1;

    void *tmp = realloc(ptr, new_size);
    if (!tmp)
    {
        perror("Fatal error: realloc failed");
        exit(EXIT_FAILURE);
    }
    return tmp;
}

void set_default_parameters(pdhg_parameters_t *params)
{
    params->presolve = true;
    params->verbose = true;
    params->optimality_norm = NORM_TYPE_L2;
    params->reflection_coefficient = 1.0;
    params->termination_evaluation_frequency = 200;

    params->scaling_params.l_inf_ruiz_iterations = 10;
    params->scaling_params.has_pock_chambolle_alpha = true;
    params->scaling_params.pock_chambolle_alpha = 1.0;
    params->scaling_params.bound_objective_rescaling = true;
    params->scaling_params.matrix_zero_tol = 1e-9;

    params->power_iter_params.max_iter = 5000;
    params->power_iter_params.tol = 1e-4;

    params->step_size_params.step_size_factor = 0.998;
    params->step_size_params.objective_coupling_budget = 0.05;
    params->step_size_params.k_p = 0.99;
    params->step_size_params.k_i = 0.01;
    params->step_size_params.k_d = 0.0;
    params->step_size_params.i_smooth = 0.3;

    params->cone_params.max_iter = 30;
    params->cone_params.tol = 1e-6;

    params->termination_criteria.eps_optimal_relative = 1e-4;
    params->termination_criteria.eps_feasible_relative = 1e-4;
    params->termination_criteria.time_sec_limit = 3600.0;
    params->termination_criteria.iteration_limit = INT32_MAX;

    params->restart_params.artificial_restart_threshold = 0.36;
    params->restart_params.sufficient_reduction_for_restart = 0.2;
    params->restart_params.necessary_reduction_for_restart = 0.5;
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
    CURAND_CHECK(curandGenerateNormalDouble(state->curand_gen, power_iteration_vector_d, curand_count, 0.0, 1.0));

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

        normalize_power_iteration_vector_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            power_iteration_normalized_vector_d,
            power_iteration_vector_d,
            state->power_iteration_vector_norm_scalar_d,
            m);
        CUDA_CHECK(cudaGetLastError());

        CUSPARSE_CHECK(cusparseSpMV(
            state->sparse_handle_A, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &HOST_ONE, state->matAt, state->spmv_power_iteration_normalized_vector, &HOST_ZERO, state->spmv_power_iteration_product,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

        CUSPARSE_CHECK(cusparseSpMV(
            state->sparse_handle_A, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &HOST_ONE, state->matA, state->spmv_power_iteration_product, &HOST_ZERO, state->spmv_power_iteration_vector,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

        CUBLAS_CHECK(cublasDdot(
            state->blas_handle_A,
            m,
            power_iteration_normalized_vector_d,
            1,
            power_iteration_vector_d,
            1,
            state->power_iter_sigma_squared_mp));

        compute_power_iteration_residual_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
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

void reduce_device_vector(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output,
    bool use_linf)
{
    if (use_linf)
        CUDA_CHECK(cub::DeviceReduce::Max(state->cub_reduce_workspace,
                                          state->cub_reduce_workspace_bytes,
                                          input, output, length, state->stream_A));
    else
        CUDA_CHECK(cub::DeviceReduce::Sum(state->cub_reduce_workspace,
                                          state->cub_reduce_workspace_bytes,
                                          input, output, length, state->stream_A));
    CUDA_CHECK(cudaGetLastError());
}

__global__ void normalize_power_iteration_vector_kernel(
    double *normalized_vector_d,
    const double *vector_d,
    const double *vector_norm_d,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

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
    if (i >= n) return;

    double sigma_squared = *sigma_squared_d;
    residual_d[i] = vector_d[i] - sigma_squared * normalized_vector_d[i];
}

__global__ void abs_vector_inplace_kernel(double *__restrict__ arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        arr[i] = fabs(arr[i]);
}


__global__ void fill_ones_kernel(double *__restrict__ arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        arr[i] = 1.0;
}
