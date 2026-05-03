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
