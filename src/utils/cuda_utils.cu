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

__global__ void compute_abs_kernel(
    const double *__restrict__ input,
    int length,
    double *__restrict__ output);

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

    state->cub_reduce_workspace_bytes =
        sum_bytes > abs_max_bytes ? sum_bytes : abs_max_bytes;
    CUDA_CHECK(cudaMalloc(
        &state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes));
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
