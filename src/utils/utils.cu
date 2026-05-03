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

struct AbsTransform
{
    __host__ __device__ double operator()(const double &x) const
    {
        return fabs(x);
    }
};

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

void reduce_device_sum(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output)
{
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace,
        state->cub_reduce_workspace_bytes,
        input,
        output,
        length,
        state->stream_A));
    CUDA_CHECK(cudaGetLastError());
}

void reduce_device_max(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output)
{
    CUDA_CHECK(cub::DeviceReduce::Max(
        state->cub_reduce_workspace,
        state->cub_reduce_workspace_bytes,
        input,
        output,
        length,
        state->stream_A));
    CUDA_CHECK(cudaGetLastError());
}

void reduce_device_abs_max(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output)
{
    cub::TransformInputIterator<double, AbsTransform, const double *>
        abs_input(input, AbsTransform{});
    CUDA_CHECK(cub::DeviceReduce::Max(
        state->cub_reduce_workspace,
        state->cub_reduce_workspace_bytes,
        abs_input,
        output,
        length,
        state->stream_A));
    CUDA_CHECK(cudaGetLastError());
}

__global__ void fill_ones_kernel(double *__restrict__ arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        arr[i] = 1.0;
}
