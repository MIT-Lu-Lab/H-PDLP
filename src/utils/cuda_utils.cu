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

#include "cuda_utils.h"
#include "cuda_macros.h"
#include <cub/device/device_reduce.cuh>
#include <math.h>
#include <thrust/iterator/transform_iterator.h>

struct absolute_value_transform_t
{
    __host__ __device__ double operator()(const double &value) const
    {
        return value < 0.0 ? -value : value;
    }
};

__global__ void compute_objective_norm_entries_kernel(
    const double *__restrict__ obj_vec,
    double *__restrict__ entries,
    int n_vars,
    int use_linf);
__global__ void compute_bound_norm_entries_kernel(
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    double *__restrict__ entries,
    int n_cons,
    int use_linf);

void allocate_zeroed_device_buffer(void **dst, size_t bytes)
{
    const size_t allocation_bytes = bytes > 0 ? bytes : 1;
    CUDA_CHECK(cudaMalloc(dst, allocation_bytes));
    CUDA_CHECK(cudaMemset(*dst, 0, allocation_bytes));
}

void allocate_and_copy_to_device(
    void **dst,
    const void *src,
    size_t bytes)
{
    const size_t allocation_bytes = bytes > 0 ? bytes : 1;
    CUDA_CHECK(cudaMalloc(dst, allocation_bytes));
    if (bytes > 0)
    {
        CUDA_CHECK(cudaMemcpy(
            *dst,
            src,
            bytes,
            cudaMemcpyHostToDevice));
    }
}

void allocate_cub_reduce_workspace(pdhg_solver_state_t *state)
{
    const int max_reduction_length =
        state->num_variables + state->num_constraints;
    size_t sum_bytes = 0;
    size_t max_bytes = 0;
    size_t abs_max_bytes = 0;
    const double *null_input = NULL;
    double *null_output = NULL;
    auto absolute_values =
        thrust::make_transform_iterator(
            null_input,
            absolute_value_transform_t{});

    CUDA_CHECK(cub::DeviceReduce::Sum(
        NULL,
        sum_bytes,
        null_input,
        null_output,
        max_reduction_length));
    CUDA_CHECK(cub::DeviceReduce::Max(
        NULL,
        max_bytes,
        null_input,
        null_output,
        max_reduction_length));
    CUDA_CHECK(cub::DeviceReduce::Max(
        NULL,
        abs_max_bytes,
        absolute_values,
        null_output,
        max_reduction_length));

    size_t workspace_bytes =
        sum_bytes > max_bytes ? sum_bytes : max_bytes;
    if (abs_max_bytes > workspace_bytes)
        workspace_bytes = abs_max_bytes;
    state->cub_reduce_workspace_bytes = workspace_bytes;
    CUDA_CHECK(cudaMalloc(
        &state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes));
}

void compute_lp_problem_norms_device(
    pdhg_solver_state_t *state,
    norm_type_t optimality_norm,
    double *out_objective_norm,
    double *out_bound_norm)
{
    const int use_linf = (optimality_norm == NORM_TYPE_L_INF);

    double *norm_entries_vars =
        state->variable_reduction_entries_d;
    double *norm_entries_cons =
        state->constraint_reduction_entries_d;

    compute_objective_norm_entries_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->objective_vector_d, norm_entries_vars, state->num_variables, use_linf);
    CUDA_CHECK(cudaGetLastError());

    compute_bound_norm_entries_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->constraint_lower_bound_d, state->constraint_upper_bound_d,
        norm_entries_cons, state->num_constraints, use_linf);
    CUDA_CHECK(cudaGetLastError());

    if (use_linf)
    {
        reduce_device_max(state, norm_entries_vars, state->num_variables,
                          state->objective_norm_workspace_mp);
        reduce_device_max(state, norm_entries_cons, state->num_constraints,
                          state->bound_norm_workspace_mp);
    }
    else
    {
        reduce_device_sum(state, norm_entries_vars, state->num_variables,
                          state->objective_norm_workspace_mp);
        reduce_device_sum(state, norm_entries_cons, state->num_constraints,
                          state->bound_norm_workspace_mp);
    }

    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));

    if (state->num_variables == 0)
        *state->objective_norm_workspace_mp = 0.0;
    if (state->num_constraints == 0)
        *state->bound_norm_workspace_mp = 0.0;

    if (!use_linf) {
        *state->objective_norm_workspace_mp =
            sqrt(*state->objective_norm_workspace_mp);
        *state->bound_norm_workspace_mp =
            sqrt(*state->bound_norm_workspace_mp);
    }

    *out_objective_norm = *state->objective_norm_workspace_mp;
    *out_bound_norm = *state->bound_norm_workspace_mp;
}

void reduce_device_sum(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output)
{
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        input,
        output,
        length,
        state->stream_A));
}

void reduce_device_max(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output)
{
    CUDA_CHECK(cub::DeviceReduce::Max(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        input,
        output,
        length,
        state->stream_A));
}

void reduce_device_abs_max(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output)
{
    if (length == 0)
    {
        CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(double), state->stream_A));
        return;
    }
    auto abs_input =
        thrust::make_transform_iterator(
            input,
            absolute_value_transform_t{});
    CUDA_CHECK(cub::DeviceReduce::Max(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        abs_input,
        output,
        length,
        state->stream_A));
}

__global__ void compute_objective_norm_entries_kernel(
    const double *__restrict__ obj_vec,
    double *__restrict__ entries,
    int n_vars,
    int use_linf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vars) return;

    double v = obj_vec[i];
    entries[i] = use_linf ? fabs(v) : (v * v);
}

__global__ void compute_bound_norm_entries_kernel(
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    double *__restrict__ entries,
    int n_cons,
    int use_linf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons) return;

    double Li = cons_lb[i];
    double Ui = cons_ub[i];
    double acc = 0.0;
    if (use_linf)
    {
        if (isfinite(Li) && Li != Ui) acc = fmax(acc, fabs(Li));
        if (isfinite(Ui)) acc = fmax(acc, fabs(Ui));
    }
    else
    {
        if (isfinite(Li) && Li != Ui) acc += Li * Li;
        if (isfinite(Ui)) acc += Ui * Ui;
    }
    entries[i] = acc;
}
