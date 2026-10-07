#pragma once

#include <cuda_runtime_api.h>
#include <stdio.h>

static int test_gpu_status(void)
{
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && count == 0))
    {
        fprintf(stderr, "SKIP: no usable CUDA device\n");
        return 77;
    }
    if (status != cudaSuccess)
    {
        fprintf(stderr, "CUDA device query failed: %s\n", cudaGetErrorString(status));
        return 1;
    }
    return 0;
}
