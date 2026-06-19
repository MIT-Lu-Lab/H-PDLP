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

struct SumNewtonTerms {
    __device__ __forceinline__ newton_terms_t operator()(
        const newton_terms_t &a, const newton_terms_t &b) const
    {
        return newton_terms_t{a.derivative + b.derivative,
                              a.function_value + b.function_value};
    }
};

typedef cub::BlockReduce<newton_terms_t, THREADS_PER_BLOCK> ConeBlockReduce;

__global__ void solve_dual_cone_newton_cooperative_kernel(
    const double *__restrict__ unprojected,
    box_newton_state_t cone,
    int max_iter,
    double tol);
__global__ void solve_dual_cone_newton_single_block_kernel(
    const double *__restrict__ unprojected,
    box_newton_state_t cone,
    int max_iter,
    double tol);
__global__ void primal_box_resolvent_cooperative_kernel(
    const double *__restrict__ unprojected_vector,
    const double *__restrict__ objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    box_newton_state_t cone,
    int max_iter,
    double tol);
__global__ void primal_box_resolvent_single_block_kernel(
    const double *__restrict__ unprojected_vector,
    const double *__restrict__ objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    box_newton_state_t cone,
    int max_iter,
    double tol);
__device__ __forceinline__ double select_active_dual_bound(
    double vi, double l, double u, double scale);
__device__ __forceinline__ void evaluate_dual_cone_newton_terms(
    double unprojected_value, double l, double u, double scale,
    double &function_contribution, double &derivative_contribution);
__device__ __forceinline__ void evaluate_primal_box_newton_terms(
    double vi, double ci, double li, double ui,
    double homogeneous_scale, double primal_metric,
    double &function_contribution, double &derivative_contribution);
__device__ __forceinline__ void block_sum_newton_terms(
    ConeBlockReduce::TempStorage &temp,
    double thread_derivative,
    double thread_function,
    double &block_derivative,
    double &block_function);
__device__ inline void warp_reduce_newton_terms(
    double &derivative,
    double &function_value);
__device__ __forceinline__ bool update_newton_bracket(
    double newton_step,
    double unprojected_scale,
    double tol,
    double &scale,
    double &scale_low,
    double &scale_high);

void solve_dual_cone_newton(
    const cone_parameters_t *cone_params,
    box_newton_state_t *cone_state,
    const double *unprojected_vector,
    cudaStream_t stream)
{
    const box_newton_state_t cone = *cone_state;

    if (cone.num_blocks_newton == 1)
    {
        solve_dual_cone_newton_single_block_kernel<<<
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
        solve_dual_cone_newton_cooperative_kernel,
        unprojected_vector,
        cone,
        cone_params->max_iter,
        cone_params->tol));
}

void solve_primal_box_resolvent_scale(
    const cone_parameters_t *cone_params,
    box_newton_state_t *cone_state,
    const double *unprojected_vector,
    const double *objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    cudaStream_t stream)
{
    const box_newton_state_t cone = *cone_state;

    if (cone.num_blocks_newton == 1)
    {
        primal_box_resolvent_single_block_kernel<<<
            1, THREADS_PER_BLOCK, 0, stream>>>(
            unprojected_vector,
            objective_d,
            step_size_d,
            step_size_homogeneous_d,
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
        primal_box_resolvent_cooperative_kernel,
        unprojected_vector,
        objective_d,
        step_size_d,
        step_size_homogeneous_d,
        cone,
        cone_params->max_iter,
        cone_params->tol));
}

int max_cooperative_dual_cone_grid()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    int active = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active,
        solve_dual_cone_newton_cooperative_kernel,
        THREADS_PER_BLOCK,
        0));
    return active * sm_count;
}

int max_cooperative_primal_box_grid()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    int active = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active,
        primal_box_resolvent_cooperative_kernel,
        THREADS_PER_BLOCK,
        0));
    return active * sm_count;
}

__global__ void solve_dual_cone_newton_cooperative_kernel(
    const double *__restrict__ unprojected,
    box_newton_state_t cone,
    int max_iter,
    double tol)
{
    cg::grid_group grid = cg::this_grid();

    __shared__ ConeBlockReduce::TempStorage temp;
    __shared__ double s_scale;
    __shared__ int s_converged;

    double scale = *cone.scalar_d;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double unprojected_scale = *cone.unprojected_scalar_d;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        newton_terms_t *partials = cone.newton_partials_d + (iter & 1) * cone.num_blocks_newton;

        double thread_derivative = 0.0, thread_function = 0.0;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < cone.num_elements;
             i += cone.num_blocks_newton * blockDim.x)
        {
            double function_contribution, derivative_contribution;
            evaluate_dual_cone_newton_terms(
                unprojected[i], cone.lower_bound_d[i], cone.upper_bound_d[i], scale,
                function_contribution, derivative_contribution);
            thread_derivative += derivative_contribution;
            thread_function += function_contribution;
        }

        double block_derivative, block_function;
        block_sum_newton_terms(temp, thread_derivative, thread_function, block_derivative, block_function);

        if (threadIdx.x == 0)
        {
            partials[blockIdx.x] = newton_terms_t{block_derivative, block_function};
        }

        grid.sync();

        if (threadIdx.x < warpSize)
        {
            double total_derivative = 0.0, total_function = 0.0;
            for (int bk = threadIdx.x; bk < cone.num_blocks_newton; bk += warpSize)
            {
                total_derivative += partials[bk].derivative;
                total_function += partials[bk].function_value;
            }
            warp_reduce_newton_terms(total_derivative, total_function);

            if (threadIdx.x == 0)
            {
                const double newton_step =
                    fmax(0.0, (total_function - unprojected_scale) / (1.0 + total_derivative));
                const bool new_converged = update_newton_bracket(
                    newton_step, unprojected_scale, tol, scale, scale_low, scale_high);
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
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *cone.scalar_d = scale;
    }
}

__global__ void solve_dual_cone_newton_single_block_kernel(
    const double *__restrict__ unprojected,
    box_newton_state_t cone,
    int max_iter,
    double tol)
{
    __shared__ ConeBlockReduce::TempStorage temp;
    __shared__ double s_scale;
    __shared__ int s_converged;

    double scale = *cone.scalar_d;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double unprojected_scale = *cone.unprojected_scalar_d;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        double thread_derivative = 0.0, thread_function = 0.0;
        for (int i = threadIdx.x; i < cone.num_elements; i += blockDim.x)
        {
            double function_contribution, derivative_contribution;
            evaluate_dual_cone_newton_terms(
                unprojected[i], cone.lower_bound_d[i], cone.upper_bound_d[i], scale,
                function_contribution, derivative_contribution);
            thread_derivative += derivative_contribution;
            thread_function += function_contribution;
        }

        double total_derivative, total_function;
        block_sum_newton_terms(temp, thread_derivative, thread_function, total_derivative, total_function);

        if (threadIdx.x == 0)
        {
            const double newton_step =
                fmax(0.0, (total_function - unprojected_scale) / (1.0 + total_derivative));
            const bool new_converged = update_newton_bracket(
                newton_step, unprojected_scale, tol, scale, scale_low, scale_high);
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
        *cone.scalar_d = scale;
    }
}

__global__ void primal_box_resolvent_cooperative_kernel(
    const double *__restrict__ unprojected_vector,
    const double *__restrict__ objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    box_newton_state_t cone,
    int max_iter,
    double tol)
{
    cg::grid_group grid = cg::this_grid();

    __shared__ ConeBlockReduce::TempStorage temp;
    __shared__ double s_scale;
    __shared__ int s_converged;

    double homogeneous_scale = *cone.scalar_d;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double unprojected_homogeneous_scale = *cone.unprojected_scalar_d;
    const double primal_metric = 1.0 / (*step_size_d);
    const double tau_metric = 1.0 / (*step_size_homogeneous_d);

    for (int iter = 0; iter < max_iter; ++iter)
    {
        newton_terms_t *partials = cone.newton_partials_d + (iter & 1) * cone.num_blocks_newton;

        double thread_derivative = 0.0, thread_function = 0.0;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < cone.num_elements;
             i += cone.num_blocks_newton * blockDim.x)
        {
            double function_contribution, derivative_contribution;
            evaluate_primal_box_newton_terms(
                unprojected_vector[i], objective_d[i],
                cone.lower_bound_d[i], cone.upper_bound_d[i],
                homogeneous_scale, primal_metric, function_contribution, derivative_contribution);
            thread_derivative += derivative_contribution;
            thread_function += function_contribution;
        }

        double block_derivative, block_function;
        block_sum_newton_terms(temp, thread_derivative, thread_function, block_derivative, block_function);

        if (threadIdx.x == 0)
        {
            partials[blockIdx.x] = newton_terms_t{block_derivative, block_function};
        }

        grid.sync();

        if (threadIdx.x < warpSize)
        {
            double total_derivative = 0.0, total_function = 0.0;
            for (int bk = threadIdx.x; bk < cone.num_blocks_newton; bk += warpSize)
            {
                total_derivative += partials[bk].derivative;
                total_function += partials[bk].function_value;
            }
            warp_reduce_newton_terms(total_derivative, total_function);

            if (threadIdx.x == 0)
            {
                const double derivative = total_derivative + tau_metric;
                const double function_value =
                    total_function + tau_metric * (unprojected_homogeneous_scale - homogeneous_scale);
                const double newton_step =
                    fmax(0.0, homogeneous_scale + function_value / derivative);
                const bool new_converged = update_newton_bracket(
                    newton_step, unprojected_homogeneous_scale, tol,
                    homogeneous_scale, scale_low, scale_high);
                s_scale = homogeneous_scale;
                s_converged = new_converged ? 1 : 0;
            }
        }

        __syncthreads();
        homogeneous_scale = s_scale;

        if (s_converged)
        {
            break;
        }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *cone.scalar_d = homogeneous_scale;
    }
}

__global__ void primal_box_resolvent_single_block_kernel(
    const double *__restrict__ unprojected_vector,
    const double *__restrict__ objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    box_newton_state_t cone,
    int max_iter,
    double tol)
{
    __shared__ ConeBlockReduce::TempStorage temp;
    __shared__ double s_scale;
    __shared__ int s_converged;

    double homogeneous_scale = *cone.scalar_d;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double unprojected_homogeneous_scale = *cone.unprojected_scalar_d;
    const double primal_metric = 1.0 / (*step_size_d);
    const double tau_metric = 1.0 / (*step_size_homogeneous_d);

    for (int iter = 0; iter < max_iter; ++iter)
    {
        double thread_derivative = 0.0, thread_function = 0.0;
        for (int i = threadIdx.x; i < cone.num_elements; i += blockDim.x)
        {
            double function_contribution, derivative_contribution;
            evaluate_primal_box_newton_terms(
                unprojected_vector[i], objective_d[i],
                cone.lower_bound_d[i], cone.upper_bound_d[i],
                homogeneous_scale, primal_metric, function_contribution, derivative_contribution);
            thread_derivative += derivative_contribution;
            thread_function += function_contribution;
        }

        double total_derivative, total_function;
        block_sum_newton_terms(temp, thread_derivative, thread_function, total_derivative, total_function);

        if (threadIdx.x == 0)
        {
            const double derivative = total_derivative + tau_metric;
            const double function_value =
                total_function + tau_metric * (unprojected_homogeneous_scale - homogeneous_scale);
            const double newton_step =
                fmax(0.0, homogeneous_scale + function_value / derivative);
            const bool new_converged = update_newton_bracket(
                newton_step, unprojected_homogeneous_scale, tol,
                homogeneous_scale, scale_low, scale_high);
            s_scale = homogeneous_scale;
            s_converged = new_converged ? 1 : 0;
        }

        __syncthreads();
        homogeneous_scale = s_scale;

        if (s_converged)
        {
            break;
        }
    }

    if (threadIdx.x == 0)
    {
        *cone.scalar_d = homogeneous_scale;
    }
}

__device__ __forceinline__ double select_active_dual_bound(
    double vi, double l, double u, double scale)
{
    double b = 0.0;
    b = (vi > u * scale) ? u : b;
    b = (vi < l * scale) ? l : b;
    return b;
}

__device__ __forceinline__ void evaluate_dual_cone_newton_terms(
    double unprojected_value, double l, double u, double scale,
    double &function_contribution, double &derivative_contribution)
{
    const double vi = -unprojected_value;
    const double bound = select_active_dual_bound(vi, l, u, scale);
    derivative_contribution = bound * bound;
    function_contribution = bound * vi;
}

__device__ __forceinline__ void evaluate_primal_box_newton_terms(
    double vi, double ci, double li, double ui,
    double homogeneous_scale, double primal_metric,
    double &function_contribution, double &derivative_contribution)
{
    const double xhat = vi - ci * homogeneous_scale / primal_metric;
    const double box_lower = isfinite(li) ? li * homogeneous_scale : -INFINITY;
    const double box_upper = isfinite(ui) ? ui * homogeneous_scale : INFINITY;

    if (xhat > box_upper)
    {
        const double n_i = primal_metric * (vi - box_upper) - ci * homogeneous_scale;
        function_contribution = ci * box_upper + ui * n_i;
        derivative_contribution = primal_metric * ui * ui;
    }
    else if (xhat < box_lower)
    {
        const double n_i = primal_metric * (vi - box_lower) - ci * homogeneous_scale;
        function_contribution = ci * box_lower + li * n_i;
        derivative_contribution = primal_metric * li * li;
    }
    else
    {
        function_contribution = ci * xhat;
        derivative_contribution = ci * ci / primal_metric;
    }
}

__device__ __forceinline__ void block_sum_newton_terms(
    ConeBlockReduce::TempStorage &temp,
    double thread_derivative,
    double thread_function,
    double &block_derivative,
    double &block_function)
{
    const newton_terms_t result = ConeBlockReduce(temp).Reduce(
        newton_terms_t{thread_derivative, thread_function}, SumNewtonTerms{});
    block_derivative = result.derivative;
    block_function = result.function_value;
}

__device__ inline void warp_reduce_newton_terms(double &derivative, double &function_value)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        derivative += __shfl_down_sync(0xFFFFFFFFu, derivative, offset);
        function_value += __shfl_down_sync(0xFFFFFFFFu, function_value, offset);
    }
}

__device__ __forceinline__ bool update_newton_bracket(
    double newton_step,
    double unprojected_scale,
    double tol,
    double &scale,
    double &scale_low,
    double &scale_high)
{
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
        (fabs(new_scale - scale) < tol * fabs(new_scale) + 1.0e-14 * fabs(unprojected_scale));
    scale = new_scale;
    return converged;
}
