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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

__global__ void compute_abs_kernel(
    const double *__restrict__ input,
    int length,
    double *__restrict__ output);

template <typename EntryFunction>
static size_t query_sum_of_bytes(int length);

template <typename EntryFunction>
static size_t query_max_of_bytes(int length);

template <typename EntryFunction>
static void reduce_sum_of(
    pdhg_solver_state_t *state,
    EntryFunction entry,
    int length,
    double *output);

template <typename EntryFunction>
static void reduce_max_of(
    pdhg_solver_state_t *state,
    EntryFunction entry,
    int length,
    double *output);

__host__ __device__ __forceinline__ double compute_box_support_term(
    double direction,
    double lb,
    double ub);

__host__ __device__ __forceinline__ double compute_recession_violation(
    double value,
    double lb,
    double ub);

__host__ __device__ __forceinline__ double compute_multiplier_violation(
    double value,
    double lb,
    double ub);

struct absolute_value_transform_t
{
    __host__ __device__ double operator()(const double &value) const
    {
        return value < 0.0 ? -value : value;
    }
};

struct box_support_term_t
{
    const double *direction;
    const double *lb;
    const double *ub;
    double sign;

    __host__ __device__ double operator()(int i) const
    {
        return compute_box_support_term(sign * direction[i], lb[i], ub[i]);
    }
};

struct box_support_magnitude_t
{
    const double *direction;
    const double *lb;
    const double *ub;
    double sign;

    __host__ __device__ double operator()(int i) const
    {
        return fabs(compute_box_support_term(sign * direction[i], lb[i], ub[i]));
    }
};

struct absolute_product_t
{
    const double *left;
    const double *right;

    __host__ __device__ double operator()(int i) const
    {
        return fabs(left[i] * right[i]);
    }
};

struct recession_violation_t
{
    const double *value;
    const double *lb;
    const double *ub;
    const double *rescaling;

    __host__ __device__ double operator()(int i) const
    {
        const double violation = compute_recession_violation(value[i], lb[i], ub[i]);
        return rescaling ? violation * rescaling[i] : violation;
    }
};

struct reduced_cost_violation_t
{
    const double *dual_product;
    const double *lb;
    const double *ub;
    const double *rescaling;

    __host__ __device__ double operator()(int i) const
    {
        const double reduced_cost = -dual_product[i];
        const double violation = compute_multiplier_violation(reduced_cost, lb[i], ub[i]);
        return rescaling ? violation * rescaling[i] : violation;
    }
};

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
        abs_max_bytes,
        absolute_values,
        null_output,
        max_reduction_length));

    const size_t requirements[] = {
        sum_bytes,
        abs_max_bytes,
        query_sum_of_bytes<box_support_term_t>(max_reduction_length),
        query_sum_of_bytes<box_support_magnitude_t>(max_reduction_length),
        query_sum_of_bytes<absolute_product_t>(max_reduction_length),
        query_max_of_bytes<recession_violation_t>(max_reduction_length),
        query_max_of_bytes<reduced_cost_violation_t>(max_reduction_length)};
    size_t workspace_bytes = 0;
    for (size_t i = 0; i < sizeof(requirements) / sizeof(requirements[0]); ++i)
    {
        if (requirements[i] > workspace_bytes)
            workspace_bytes = requirements[i];
    }
    state->cub_reduce_workspace_bytes = workspace_bytes;
    CUDA_CHECK(cudaMalloc(
        &state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes));
}

template <typename EntryFunction>
static size_t query_sum_of_bytes(int length)
{
    size_t bytes = 0;
    double *null_output = NULL;
    auto entries =
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), EntryFunction{});
    CUDA_CHECK(cub::DeviceReduce::Sum(NULL, bytes, entries, null_output, length));
    return bytes;
}

template <typename EntryFunction>
static size_t query_max_of_bytes(int length)
{
    size_t bytes = 0;
    double *null_output = NULL;
    auto entries =
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), EntryFunction{});
    CUDA_CHECK(cub::DeviceReduce::Max(NULL, bytes, entries, null_output, length));
    return bytes;
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

void reduce_device_box_support(
    pdhg_solver_state_t *state,
    const double *direction,
    const double *lb,
    const double *ub,
    double sign,
    int length,
    double *output)
{
    reduce_sum_of(state, box_support_term_t{direction, lb, ub, sign}, length, output);
}

template <typename EntryFunction>
static void reduce_sum_of(
    pdhg_solver_state_t *state,
    EntryFunction entry,
    int length,
    double *output)
{
    auto entries = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), entry);
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        entries,
        output,
        length,
        state->stream_A));
}

void reduce_device_box_support_magnitude(
    pdhg_solver_state_t *state,
    const double *direction,
    const double *lb,
    const double *ub,
    double sign,
    int length,
    double *output)
{
    reduce_sum_of(state, box_support_magnitude_t{direction, lb, ub, sign}, length, output);
}

void reduce_device_absolute_product(
    pdhg_solver_state_t *state,
    const double *left,
    const double *right,
    int length,
    double *output)
{
    reduce_sum_of(state, absolute_product_t{left, right}, length, output);
}

void reduce_device_recession_violation(
    pdhg_solver_state_t *state,
    const double *value,
    const double *lb,
    const double *ub,
    const double *rescaling,
    int length,
    double *output)
{
    reduce_max_of(state, recession_violation_t{value, lb, ub, rescaling}, length, output);
}

template <typename EntryFunction>
static void reduce_max_of(
    pdhg_solver_state_t *state,
    EntryFunction entry,
    int length,
    double *output)
{
    if (length == 0)
    {
        CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(double), state->stream_A));
        return;
    }
    auto entries = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), entry);
    CUDA_CHECK(cub::DeviceReduce::Max(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        entries,
        output,
        length,
        state->stream_A));
}

void reduce_device_reduced_cost_violation(
    pdhg_solver_state_t *state,
    const double *dual_product,
    const double *lb,
    const double *ub,
    const double *rescaling,
    int length,
    double *output)
{
    reduce_max_of(state, reduced_cost_violation_t{dual_product, lb, ub, rescaling}, length, output);
}

void compute_device_abs(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output)
{
    const int num_blocks = (length + 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_abs_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        input, length, output);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void compute_abs_kernel(
    const double *__restrict__ input,
    int length,
    double *__restrict__ output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    output[i] = fabs(input[i]);
}

__host__ __device__ __forceinline__ double compute_box_support_term(
    double direction,
    double lb,
    double ub)
{
    if (direction > 0.0 && isfinite(lb))
        return lb * direction;
    if (direction < 0.0 && isfinite(ub))
        return ub * direction;
    return 0.0;
}

__host__ __device__ __forceinline__ double compute_recession_violation(
    double value,
    double lb,
    double ub)
{
    return isfinite(value) ? fabs(value - project_box_recession(value, lb, ub)) : INFINITY;
}

__host__ __device__ __forceinline__ double compute_multiplier_violation(
    double value,
    double lb,
    double ub)
{
    return isfinite(value) ? fabs(value - project_box_multiplier(value, lb, ub)) : INFINITY;
}
