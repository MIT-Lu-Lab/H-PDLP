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

__global__ void solve_dual_cones_and_project_kernel(
    cone_projection_state_t c,
    const double *__restrict__ c_unprojected_vector,
    double *__restrict__ c_projected_vector,
    cone_projection_state_t b,
    const double *__restrict__ b_unprojected_vector,
    double *__restrict__ b_projected_vector,
    int max_iter,
    double tol);
__device__ inline void block_reduce_denom_numer(
    double *s_denom,
    double *s_numer);

void solve_cone_newton(
    const cone_parameters_t *cone_params,
    cone_projection_state_t *constraint_cone_state,
    const double *constraint_unprojected_vector,
    double *constraint_projected_vector,
    cone_projection_state_t *bound_cone_state,
    const double *bound_unprojected_vector,
    double *bound_projected_vector,
    cudaStream_t stream)
{
    const int total_blocks = constraint_cone_state->blocks + bound_cone_state->blocks;
    const size_t smem_bytes = 2 * THREADS_PER_BLOCK * sizeof(double);

    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeCooperative;
    attr.val.cooperative = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(total_blocks);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        solve_dual_cones_and_project_kernel,
        *constraint_cone_state,
        constraint_unprojected_vector,
        constraint_projected_vector,
        *bound_cone_state,
        bound_unprojected_vector,
        bound_projected_vector,
        cone_params->max_iter,
        cone_params->tol));
}

__global__ void solve_dual_cones_and_project_kernel(
    cone_projection_state_t c,
    const double *__restrict__ c_unprojected_vector,
    double *__restrict__ c_projected_vector,
    cone_projection_state_t b,
    const double *__restrict__ b_unprojected_vector,
    double *__restrict__ b_projected_vector,
    int max_iter,
    double tol)
{
    cg::grid_group grid = cg::this_grid();

    extern __shared__ double smem[];
    double *s_denom = smem;
    double *s_numer = smem + blockDim.x;

    const bool is_constraint_cone = (blockIdx.x < c.blocks);
    const int block_idx_in_cone =
        is_constraint_cone ? blockIdx.x : (blockIdx.x - c.blocks);
    const bool is_owner = (block_idx_in_cone == 0);

    double c_scale = *c.scale_d;
    bool c_converged = false;
    double b_scale = *b.scale_d;
    bool b_converged = false;

    double c_scale_low = 0.0;
    double c_scale_high = INFINITY;
    double b_scale_low = 0.0;
    double b_scale_high = INFINITY;

    const double c_unprojected_scale = *c.unprojected_scale_d;
    const double b_unprojected_scale = *b.unprojected_scale_d;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        const cone_projection_state_t &my_cone =
            is_constraint_cone ? c : b;
        const double *my_unprojected =
            is_constraint_cone ? c_unprojected_vector : b_unprojected_vector;
        const double my_scale =
            is_constraint_cone ? c_scale : b_scale;
        const bool my_cone_converged =
            is_constraint_cone ? c_converged : b_converged;

        if (!my_cone_converged)
        {
            double thread_denom = 0.0, thread_numer = 0.0;
            for (int i = block_idx_in_cone * blockDim.x + threadIdx.x;
                 i < my_cone.num;
                 i += my_cone.blocks * blockDim.x)
            {
                const double vi = -my_unprojected[i];
                const double li = my_cone.lower_bound[i];
                const double ui = my_cone.upper_bound[i];
                double bound = 0.0;
                if (isfinite(li) && vi < li * my_scale)
                {
                    bound = li;
                }
                else if (isfinite(ui) && vi > ui * my_scale)
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
                my_cone.newton_partials_d[block_idx_in_cone] =
                    newton_denom_numer_t{s_denom[0], s_numer[0]};
            }
        }

        grid.sync();

        if (is_owner && is_constraint_cone && !c_converged)
        {
            double thread_denom = 0.0, thread_numer = 0.0;
            for (int bk = threadIdx.x; bk < c.blocks; bk += blockDim.x)
            {
                thread_denom += c.newton_partials_d[bk].denom;
                thread_numer += c.newton_partials_d[bk].numer;
            }
            s_denom[threadIdx.x] = thread_denom;
            s_numer[threadIdx.x] = thread_numer;
            block_reduce_denom_numer(s_denom, s_numer);

            const double total_denom = s_denom[0];
            const double total_numer = s_numer[0];

            const double newton_step =
                fmax(0.0, (total_numer - c_unprojected_scale) / (1.0 + total_denom));

            if (newton_step > c_scale)
                c_scale_low = c_scale;
            else
                c_scale_high = c_scale;

            double new_scale;
            if (newton_step >= c_scale_low && newton_step <= c_scale_high)
                new_scale = newton_step;
            else if (isfinite(c_scale_high))
                new_scale = 0.5 * (c_scale_low + c_scale_high);
            else
                new_scale = c_scale_low;

            const bool new_converged =
                (fabs(new_scale - c_scale) < tol * (1.0 + fabs(new_scale)));

            c_scale = new_scale;
            c_converged = new_converged;

            if (threadIdx.x == 0)
            {
                *c.scale_d = c_scale;
                *c.converged_d = c_converged ? 1 : 0;
            }
        }

        if (is_owner && !is_constraint_cone && !b_converged)
        {
            double thread_denom = 0.0, thread_numer = 0.0;
            for (int bk = threadIdx.x; bk < b.blocks; bk += blockDim.x)
            {
                thread_denom += b.newton_partials_d[bk].denom;
                thread_numer += b.newton_partials_d[bk].numer;
            }
            s_denom[threadIdx.x] = thread_denom;
            s_numer[threadIdx.x] = thread_numer;
            block_reduce_denom_numer(s_denom, s_numer);

            const double total_denom = s_denom[0];
            const double total_numer = s_numer[0];

            const double newton_step =
                fmax(0.0, (total_numer - b_unprojected_scale) / (1.0 + total_denom));

            if (newton_step > b_scale)
                b_scale_low = b_scale;
            else
                b_scale_high = b_scale;

            double new_scale;
            if (newton_step >= b_scale_low && newton_step <= b_scale_high)
                new_scale = newton_step;
            else if (isfinite(b_scale_high))
                new_scale = 0.5 * (b_scale_low + b_scale_high);
            else
                new_scale = b_scale_low;

            const bool new_converged =
                (fabs(new_scale - b_scale) < tol * (1.0 + fabs(new_scale)));

            b_scale = new_scale;
            b_converged = new_converged;

            if (threadIdx.x == 0)
            {
                *b.scale_d = b_scale;
                *b.converged_d = b_converged ? 1 : 0;
            }
        }

        grid.sync();

        c_scale = *c.scale_d;
        c_converged = (*c.converged_d != 0);
        b_scale = *b.scale_d;
        b_converged = (*b.converged_d != 0);

        if (c_converged && b_converged)
        {
            break;
        }
    }

    const cone_projection_state_t &my_cone = is_constraint_cone ? c : b;
    const double *my_unprojected = is_constraint_cone ? c_unprojected_vector : b_unprojected_vector;
    double *my_projected = is_constraint_cone ? c_projected_vector : b_projected_vector;
    const double tau_star = is_constraint_cone ? c_scale : b_scale;

    for (int i = block_idx_in_cone * blockDim.x + threadIdx.x;
         i < my_cone.num;
         i += my_cone.blocks * blockDim.x)
    {
        const double vi = -my_unprojected[i];
        const double li = my_cone.lower_bound[i];
        const double ui = my_cone.upper_bound[i];
        const double box_lower = isfinite(li) ? li * tau_star : -INFINITY;
        const double box_upper = isfinite(ui) ? ui * tau_star : INFINITY;
        my_projected[i] = my_unprojected[i] + fmin(fmax(vi, box_lower), box_upper);
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
