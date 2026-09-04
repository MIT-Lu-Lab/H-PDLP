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
#include "spmv_backend.h"
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

static void run_power_iteration_step(
    pdhg_solver_state_t *state,
    int m,
    bool evaluate_residual)
{
    double *power_iteration_vector_d = state->primal_product_d;
    double *power_iteration_normalized_vector_d = state->power_iteration_normalized_vector_d;

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

    execute_spmv(
        state->sparse_handle_A,
        state->spmv_power_iteration_normalized_vector,
        state->spmv_dual_product,
        &state->matAt_spmv_plan);

    execute_spmv(
        state->sparse_handle_A,
        state->spmv_dual_product,
        state->spmv_primal_product,
        &state->matA_spmv_plan);

    CUBLAS_CHECK(cublasDdot(
        state->blas_handle_A,
        m,
        power_iteration_normalized_vector_d,
        1,
        power_iteration_vector_d,
        1,
        state->power_iter_sigma_squared_mp));

    if (evaluate_residual)
    {
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

        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A,
            m,
            power_iteration_normalized_vector_d,
            1,
            state->power_iter_residual_norm_mp));
    }
}

double estimate_maximum_singular_value(
    pdhg_solver_state_t *state,
    const power_iteration_parameters_t *power_iter_params)
{
    const int window = 5;
    int m = state->num_constraints;

    if (m == 0)
        return 0.0;

    size_t curand_count = ((size_t)m + 1) & ~(size_t)1;
    CURAND_CHECK(curandGenerateNormalDouble(
        state->curand_gen,
        state->primal_product_d,
        curand_count,
        0.0,
        1.0));

    *state->power_iter_sigma_squared_mp = 1.0;
    *state->power_iter_residual_norm_mp = INFINITY;

    const int num_windows = power_iter_params->max_iter / window;
    const int remainder = power_iter_params->max_iter % window;

    if (num_windows > 0)
    {
        cudaGraph_t graph = NULL;
        cudaGraphExec_t graph_exec = NULL;
        CUDA_CHECK(cudaStreamBeginCapture(
            state->stream_A,
            cudaStreamCaptureModeGlobal));
        for (int k = 0; k < window - 1; ++k)
        {
            run_power_iteration_step(state, m, false);
        }
        run_power_iteration_step(state, m, true);
        CUDA_CHECK(cudaStreamEndCapture(state->stream_A, &graph));
        CUDA_CHECK(cudaGraphInstantiate(
            &graph_exec,
            graph,
            NULL,
            NULL,
            0));
        CUDA_CHECK(cudaGraphDestroy(graph));

        for (int j = 0; j < num_windows; ++j)
        {
            CUDA_CHECK(cudaGraphLaunch(
                graph_exec,
                state->stream_A));
            CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
            if (*state->power_iter_residual_norm_mp <
                power_iter_params->tol * fmin(1.0, *state->power_iter_sigma_squared_mp))
            {
                break;
            }
        }

        CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    }

    if (remainder > 0 &&
        *state->power_iter_residual_norm_mp >=
            power_iter_params->tol * fmin(1.0, *state->power_iter_sigma_squared_mp))
    {
        for (int k = 0; k < remainder - 1; ++k)
        {
            run_power_iteration_step(state, m, false);
        }
        run_power_iteration_step(state, m, true);
        CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    }

    return sqrt(*state->power_iter_sigma_squared_mp);
}
