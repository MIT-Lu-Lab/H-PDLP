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

#include "cone_projector.h"
#include "utils.h"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math.h>

namespace cg = cooperative_groups;

typedef cub::BlockReduce<double, THREADS_PER_BLOCK> ConeBlockReduce;

__global__ void solve_cone_newton_cooperative_kernel(
    const double *__restrict__ unprojected,
    cone_projection_state_t cone,
    int max_iter,
    double tol);
__global__ void solve_cone_newton_single_block_kernel(
    const double *__restrict__ unprojected,
    cone_projection_state_t cone,
    int max_iter,
    double tol);
__device__ __forceinline__ double select_active_cone_bound(
    double vi, double l, double u, double scale);
__device__ __forceinline__ void block_sum_denom_numer(
    ConeBlockReduce::TempStorage &temp,
    double thread_denom,
    double thread_numer,
    double &block_denom,
    double &block_numer);
__device__ __forceinline__ bool update_cone_newton_bracket(
    double total_denom,
    double total_numer,
    double unprojected_scale,
    double tol,
    double &scale,
    double &scale_low,
    double &scale_high);
__device__ inline void warp_reduce_denom_numer(
    double &denom,
    double &numer);

void solve_cone_newton(
    const cone_parameters_t *cone_params,
    cone_projection_state_t *cone_state,
    const double *unprojected_vector,
    cudaStream_t stream)
{
    const cone_projection_state_t cone = *cone_state;

    if (cone.num_blocks_newton == 1)
    {
        solve_cone_newton_single_block_kernel<<<
            1, THREADS_PER_BLOCK, 0, stream>>>(
            unprojected_vector,
            cone,
            cone_params->max_iter,
            cone_params->tol);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeCooperative;
    attr.val.cooperative = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(cone.num_blocks_newton);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        solve_cone_newton_cooperative_kernel,
        unprojected_vector,
        cone,
        cone_params->max_iter,
        cone_params->tol));
}

int max_cooperative_cone_grid()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    int max_active_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_per_sm,
        solve_cone_newton_cooperative_kernel,
        THREADS_PER_BLOCK,
        0));
    return max_active_per_sm * sm_count;
}

__global__ void solve_cone_newton_cooperative_kernel(
    const double *__restrict__ unprojected,
    cone_projection_state_t cone,
    int max_iter,
    double tol)
{
    cg::grid_group grid = cg::this_grid();

    __shared__ ConeBlockReduce::TempStorage temp;
    __shared__ double s_scale;
    __shared__ int s_converged;

    double scale = *cone.scale_d;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double unprojected_scale = *cone.unprojected_scale_d;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        newton_denom_numer_t *partials = cone.newton_partials_d + (iter & 1) * cone.num_blocks_newton;

        double thread_denom = 0.0, thread_numer = 0.0;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < cone.num_elements;
             i += cone.num_blocks_newton * blockDim.x)
        {
            const double vi = -unprojected[i];
            const double bound = select_active_cone_bound(
                vi, cone.lower_bound_d[i], cone.upper_bound_d[i], scale);
            thread_denom += bound * bound;
            thread_numer += bound * vi;
        }

        double bd, bn;
        block_sum_denom_numer(temp, thread_denom, thread_numer, bd, bn);

        if (threadIdx.x == 0)
        {
            partials[blockIdx.x] = newton_denom_numer_t{bd, bn};
        }

        grid.sync();

        if (threadIdx.x < warpSize)
        {
            double td = 0.0, tn = 0.0;
            for (int bk = threadIdx.x; bk < cone.num_blocks_newton; bk += warpSize)
            {
                td += partials[bk].denom;
                tn += partials[bk].numer;
            }
            warp_reduce_denom_numer(td, tn);

            if (threadIdx.x == 0)
            {
                const bool new_converged = update_cone_newton_bracket(
                    td, tn, unprojected_scale, tol, scale, scale_low, scale_high);
                s_scale = scale;
                s_converged = new_converged ? 1 : 0;
            }
        }

        __syncthreads();
        scale = s_scale;

        if (s_converged)
        {
            break;
        }

        grid.sync();
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *cone.scale_d = scale;
    }
}

__global__ void solve_cone_newton_single_block_kernel(
    const double *__restrict__ unprojected,
    cone_projection_state_t cone,
    int max_iter,
    double tol)
{
    __shared__ ConeBlockReduce::TempStorage temp;
    __shared__ double s_scale;
    __shared__ int s_converged;

    double scale = *cone.scale_d;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double unprojected_scale = *cone.unprojected_scale_d;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        double thread_denom = 0.0, thread_numer = 0.0;
        for (int i = threadIdx.x; i < cone.num_elements; i += blockDim.x)
        {
            const double vi = -unprojected[i];
            const double bound = select_active_cone_bound(
                vi, cone.lower_bound_d[i], cone.upper_bound_d[i], scale);
            thread_denom += bound * bound;
            thread_numer += bound * vi;
        }

        double total_denom, total_numer;
        block_sum_denom_numer(temp, thread_denom, thread_numer, total_denom, total_numer);

        if (threadIdx.x == 0)
        {
            const bool new_converged = update_cone_newton_bracket(
                total_denom, total_numer, unprojected_scale, tol,
                scale, scale_low, scale_high);
            s_scale = scale;
            s_converged = new_converged ? 1 : 0;
        }

        __syncthreads();
        scale = s_scale;

        if (s_converged)
        {
            break;
        }
    }

    if (threadIdx.x == 0)
    {
        *cone.scale_d = scale;
    }
}

__device__ __forceinline__ double select_active_cone_bound(
    double vi, double l, double u, double scale)
{
    double b = 0.0;
    b = (vi > u * scale) ? u : b;
    b = (vi < l * scale) ? l : b;
    return b;
}

__device__ __forceinline__ void block_sum_denom_numer(
    ConeBlockReduce::TempStorage &temp,
    double thread_denom,
    double thread_numer,
    double &block_denom,
    double &block_numer)
{
    block_denom = ConeBlockReduce(temp).Sum(thread_denom);
    __syncthreads();
    block_numer = ConeBlockReduce(temp).Sum(thread_numer);
    __syncthreads();
}

__device__ __forceinline__ bool update_cone_newton_bracket(
    double total_denom,
    double total_numer,
    double unprojected_scale,
    double tol,
    double &scale,
    double &scale_low,
    double &scale_high)
{
    const double newton_step =
        fmax(0.0, (total_numer - unprojected_scale) / (1.0 + total_denom));

    if (newton_step > scale)
        scale_low = scale;
    else
        scale_high = scale;

    double new_scale;
    if (newton_step >= scale_low && newton_step <= scale_high)
        new_scale = newton_step;
    else if (isfinite(scale_high))
        new_scale = 0.5 * (scale_low + scale_high);
    else
        new_scale = scale_low;

    const bool converged =
        (fabs(new_scale - scale) < tol * (1.0 + fabs(new_scale)));
    scale = new_scale;
    return converged;
}

__device__ inline void warp_reduce_denom_numer(double &denom, double &numer)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        denom += __shfl_down_sync(0xFFFFFFFFu, denom, offset);
        numer += __shfl_down_sync(0xFFFFFFFFu, numer, offset);
    }
}
