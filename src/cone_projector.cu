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

void solve_cone_newton(
    const cone_parameters_t *cone_params,
    cone_projection_state_t *cone_state,
    const double *unprojected_vector,
    double *projected_vector,
    cudaStream_t stream)
{
    if (cone_state->blocks == 1)
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
    config.gridDim = dim3(cone.blocks);
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

    const bool is_owner = (blockIdx.x == 0);
    double scale = *cone.scale_d;
    bool converged = false;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double unprojected_scale = *cone.unprojected_scale_d;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        if (!converged)
        {
            double thread_denom = 0.0, thread_numer = 0.0;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < cone.num;
                 i += cone.blocks * blockDim.x)
            {
                const double vi = -unprojected_vector[i];
                const double li = cone.lower_bound[i];
                const double ui = cone.upper_bound[i];
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
                cone.newton_partials_d[blockIdx.x] =
                    newton_denom_numer_t{s_denom[0], s_numer[0]};
            }
        }

        grid.sync();

        if (is_owner && !converged)
        {
            double thread_denom = 0.0, thread_numer = 0.0;
            for (int bk = threadIdx.x; bk < cone.blocks; bk += blockDim.x)
            {
                thread_denom += cone.newton_partials_d[bk].denom;
                thread_numer += cone.newton_partials_d[bk].numer;
            }
            s_denom[threadIdx.x] = thread_denom;
            s_numer[threadIdx.x] = thread_numer;
            block_reduce_denom_numer(s_denom, s_numer);

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
            converged = new_converged;
            *cone.scale_d = scale;
            *cone.converged_d = converged ? 1 : 0;
        }

        grid.sync();

        scale = *cone.scale_d;
        converged = (*cone.converged_d != 0);

        if (converged)
        {
            break;
        }
    }

    const double tau_star = scale;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < cone.num;
         i += cone.blocks * blockDim.x)
    {
        const double vi = -unprojected_vector[i];
        const double li = cone.lower_bound[i];
        const double ui = cone.upper_bound[i];
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
    bool converged = false;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double unprojected_scale = *cone.unprojected_scale_d;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        if (!converged)
        {
            double thread_denom = 0.0, thread_numer = 0.0;
            for (int i = threadIdx.x; i < cone.num; i += blockDim.x)
            {
                const double vi = -unprojected_vector[i];
                const double li = cone.lower_bound[i];
                const double ui = cone.upper_bound[i];
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

                converged =
                    (fabs(new_scale - scale) < tol * (1.0 + fabs(new_scale)));
                scale = new_scale;
                *cone.scale_d = scale;
                *cone.converged_d = converged ? 1 : 0;
            }

            __syncthreads();
            scale = *cone.scale_d;
            converged = (*cone.converged_d != 0);
        }

        if (converged)
        {
            break;
        }
    }

    const double tau_star = scale;
    for (int i = threadIdx.x; i < cone.num; i += blockDim.x)
    {
        const double vi = -unprojected_vector[i];
        const double li = cone.lower_bound[i];
        const double ui = cone.upper_bound[i];
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
