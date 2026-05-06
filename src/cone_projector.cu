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
#include <cuda_runtime.h>
#include <math.h>

namespace cg = cooperative_groups;

__global__ void solve_cone_and_project_kernel(
    cone_projection_state_t cone,
    const double *__restrict__ unprojected_vector,
    double *__restrict__ projected_vector,
    int max_iter,
    double tol);
__global__ void solve_single_block_cone_and_project_kernel(
    cone_projection_state_t cone,
    const double *__restrict__ unprojected_vector,
    double *__restrict__ projected_vector,
    int max_iter,
    double tol);
static void launch_cooperative_cone_projection(
    cone_projection_state_t cone,
    const double *unprojected_vector,
    double *projected_vector,
    int max_iter,
    double tol,
    cudaStream_t stream);
static void launch_single_block_cone_projection(
    cone_projection_state_t cone,
    const double *unprojected_vector,
    double *projected_vector,
    int max_iter,
    double tol,
    cudaStream_t stream);
__device__ inline void block_reduce_denom_numer(
    double *s_denom,
    double *s_numer);
__device__ inline void warp_reduce_denom_numer(
    double &denom,
    double &numer);

int max_cooperative_cone_grid()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    int max_active_per_sm = 0;
    const size_t smem_bytes = 2 * THREADS_PER_BLOCK * sizeof(double);
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_per_sm,
        solve_cone_and_project_kernel,
        THREADS_PER_BLOCK,
        smem_bytes));
    return max_active_per_sm * sm_count;
}

void solve_cone_newton(
    const cone_parameters_t *cone_params,
    cone_projection_state_t *cone_state,
    const double *unprojected_vector,
    double *projected_vector,
    cudaStream_t stream)
{
    if (cone_state->num_blocks == 1)
    {
        launch_single_block_cone_projection(
            *cone_state,
            unprojected_vector,
            projected_vector,
            cone_params->max_iter,
            cone_params->tol,
            stream);
        return;
    }

    launch_cooperative_cone_projection(
        *cone_state,
        unprojected_vector,
        projected_vector,
        cone_params->max_iter,
        cone_params->tol,
        stream);
}

static void launch_cooperative_cone_projection(
    cone_projection_state_t cone,
    const double *unprojected_vector,
    double *projected_vector,
    int max_iter,
    double tol,
    cudaStream_t stream)
{
    const size_t smem_bytes = 2 * THREADS_PER_BLOCK * sizeof(double);

    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeCooperative;
    attr.val.cooperative = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(cone.num_blocks);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        solve_cone_and_project_kernel,
        cone,
        unprojected_vector,
        projected_vector,
        max_iter,
        tol));
}

static void launch_single_block_cone_projection(
    cone_projection_state_t cone,
    const double *unprojected_vector,
    double *projected_vector,
    int max_iter,
    double tol,
    cudaStream_t stream)
{
    const size_t smem_bytes = 2 * THREADS_PER_BLOCK * sizeof(double);
    solve_single_block_cone_and_project_kernel<<<
        1,
        THREADS_PER_BLOCK,
        smem_bytes,
        stream>>>(
        cone,
        unprojected_vector,
        projected_vector,
        max_iter,
        tol);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void solve_cone_and_project_kernel(
    cone_projection_state_t cone,
    const double *__restrict__ unprojected_vector,
    double *__restrict__ projected_vector,
    int max_iter,
    double tol)
{
    cg::grid_group grid = cg::this_grid();

    extern __shared__ double smem[];
    double *s_denom = smem;
    double *s_numer = smem + blockDim.x;

    double scale = *cone.scale_d;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double unprojected_scale = *cone.unprojected_scale_d;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        newton_denom_numer_t *partials = cone.newton_partials_d + (iter & 1) * cone.num_blocks;

        double thread_denom = 0.0, thread_numer = 0.0;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < cone.num_elements;
             i += cone.num_blocks * blockDim.x)
        {
            const double vi = -unprojected_vector[i];
            const double li = cone.lower_bound_d[i];
            const double ui = cone.upper_bound_d[i];
            double bound = 0.0;
            if (isfinite(li) && vi < li * scale)
            {
                bound = li;
            }
            else if (isfinite(ui) && vi > ui * scale)
            {
                bound = ui;
            }
            thread_denom += bound * bound;
            thread_numer += bound * vi;
        }

        s_denom[threadIdx.x] = thread_denom;
        s_numer[threadIdx.x] = thread_numer;
        block_reduce_denom_numer(s_denom, s_numer);

        if (threadIdx.x == 0)
        {
            partials[blockIdx.x] =
                newton_denom_numer_t{s_denom[0], s_numer[0]};
        }

        grid.sync();

        if (threadIdx.x < warpSize)
        {
            double td = 0.0, tn = 0.0;
            for (int bk = threadIdx.x; bk < cone.num_blocks; bk += warpSize)
            {
                td += partials[bk].denom;
                tn += partials[bk].numer;
            }
            warp_reduce_denom_numer(td, tn);

            if (threadIdx.x == 0)
            {
                const double total_denom = td;
                const double total_numer = tn;

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

                const bool new_converged =
                    (fabs(new_scale - scale) < tol * (1.0 + fabs(new_scale)));

                scale = new_scale;
                s_denom[0] = scale;
                s_numer[0] = new_converged ? 1.0 : 0.0;
            }
        }

        __syncthreads();
        scale = s_denom[0];

        if (s_numer[0] != 0.0)
        {
            break;
        }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *cone.scale_d = scale;
    }

    const double tau_star = scale;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < cone.num_elements;
         i += cone.num_blocks * blockDim.x)
    {
        const double vi = -unprojected_vector[i];
        const double li = cone.lower_bound_d[i];
        const double ui = cone.upper_bound_d[i];
        const double box_lower = isfinite(li) ? li * tau_star : -INFINITY;
        const double box_upper = isfinite(ui) ? ui * tau_star : INFINITY;
        projected_vector[i] = unprojected_vector[i] + fmin(fmax(vi, box_lower), box_upper);
    }
}

__global__ void solve_single_block_cone_and_project_kernel(
    cone_projection_state_t cone,
    const double *__restrict__ unprojected_vector,
    double *__restrict__ projected_vector,
    int max_iter,
    double tol)
{
    extern __shared__ double smem[];
    double *s_denom = smem;
    double *s_numer = smem + blockDim.x;

    double scale = *cone.scale_d;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double unprojected_scale = *cone.unprojected_scale_d;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        double thread_denom = 0.0, thread_numer = 0.0;
        for (int i = threadIdx.x; i < cone.num_elements; i += blockDim.x)
        {
            const double vi = -unprojected_vector[i];
            const double li = cone.lower_bound_d[i];
            const double ui = cone.upper_bound_d[i];
            double bound = 0.0;
            if (isfinite(li) && vi < li * scale)
            {
                bound = li;
            }
            else if (isfinite(ui) && vi > ui * scale)
            {
                bound = ui;
            }
            thread_denom += bound * bound;
            thread_numer += bound * vi;
        }

        s_denom[threadIdx.x] = thread_denom;
        s_numer[threadIdx.x] = thread_numer;
        block_reduce_denom_numer(s_denom, s_numer);

        if (threadIdx.x == 0)
        {
            const double total_denom = s_denom[0];
            const double total_numer = s_numer[0];
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

            const bool new_converged =
                (fabs(new_scale - scale) < tol * (1.0 + fabs(new_scale)));
            scale = new_scale;
            s_denom[0] = scale;
            s_numer[0] = new_converged ? 1.0 : 0.0;
        }

        __syncthreads();
        scale = s_denom[0];

        if (s_numer[0] != 0.0)
        {
            break;
        }
    }

    if (threadIdx.x == 0)
    {
        *cone.scale_d = scale;
    }

    const double tau_star = scale;
    for (int i = threadIdx.x; i < cone.num_elements; i += blockDim.x)
    {
        const double vi = -unprojected_vector[i];
        const double li = cone.lower_bound_d[i];
        const double ui = cone.upper_bound_d[i];
        const double box_lower = isfinite(li) ? li * tau_star : -INFINITY;
        const double box_upper = isfinite(ui) ? ui * tau_star : INFINITY;
        projected_vector[i] = unprojected_vector[i] + fmin(fmax(vi, box_lower), box_upper);
    }
}

__device__ inline void block_reduce_denom_numer(double *s_denom, double *s_numer)
{
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            s_denom[threadIdx.x] += s_denom[threadIdx.x + s];
            s_numer[threadIdx.x] += s_numer[threadIdx.x + s];
        }
        __syncthreads();
    }
}

__device__ inline void warp_reduce_denom_numer(double &denom, double &numer)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        denom += __shfl_down_sync(0xFFFFFFFFu, denom, offset);
        numer += __shfl_down_sync(0xFFFFFFFFu, numer, offset);
    }
}
