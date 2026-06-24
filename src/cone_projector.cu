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

__global__ void update_dual_box_cone_projection_cooperative_kernel(
    double *__restrict__ current_dual,
    double *__restrict__ reflected_dual,
    double *__restrict__ evaluation_dual,
    const double *__restrict__ initial_dual,
    const double *__restrict__ gradient,
    const double *step_size_d,
    const double *primal_scale_current,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *dual_scale_initial,
    double *dual_scale_current,
    double *dual_scale_evaluation,
    double *dual_scale_reflected,
    bool save_evaluation_iterate,
    box_newton_state_t state,
    int max_iter,
    double tol);
__global__ void update_dual_box_cone_projection_single_block_kernel(
    double *__restrict__ current_dual,
    double *__restrict__ reflected_dual,
    double *__restrict__ evaluation_dual,
    const double *__restrict__ initial_dual,
    const double *__restrict__ gradient,
    const double *step_size_d,
    const double *primal_scale_current,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *dual_scale_initial,
    double *dual_scale_current,
    double *dual_scale_evaluation,
    double *dual_scale_reflected,
    bool save_evaluation_iterate,
    box_newton_state_t state,
    int max_iter,
    double tol);
__global__ void update_primal_box_cone_resolvent_cooperative_kernel(
    double *__restrict__ current_primal,
    double *__restrict__ reflected_primal,
    double *__restrict__ evaluation_primal,
    const double *__restrict__ initial_primal,
    const double *__restrict__ gradient,
    const double *__restrict__ objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    const double *homogeneous_dual_gradient,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    double *initial_primal_scale,
    double *current_primal_scale,
    double *evaluation_primal_scale,
    double *reflected_primal_scale,
    bool save_evaluation_iterate,
    box_newton_state_t state,
    int max_iter,
    double tol);
__global__ void update_primal_box_cone_resolvent_single_block_kernel(
    double *__restrict__ current_primal,
    double *__restrict__ reflected_primal,
    double *__restrict__ evaluation_primal,
    const double *__restrict__ initial_primal,
    const double *__restrict__ gradient,
    const double *__restrict__ objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    const double *homogeneous_dual_gradient,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    double *initial_primal_scale,
    double *current_primal_scale,
    double *evaluation_primal_scale,
    double *reflected_primal_scale,
    bool save_evaluation_iterate,
    box_newton_state_t state,
    int max_iter,
    double tol);
__device__ __forceinline__ double select_active_dual_bound(
    double vi, double l, double u, double scale);
__device__ __forceinline__ void evaluate_dual_box_cone_newton_terms(
    double unprojected_value, double l, double u, double scale,
    double &function_contribution, double &derivative_contribution);
__device__ __forceinline__ void evaluate_primal_box_cone_newton_terms(
    double vi, double ci, double li, double ui,
    double primal_scale, double primal_metric,
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
__device__ __forceinline__ bool update_newton_bracket_safeguarded(
    double newton_step,
    double function_value,
    double unprojected_scale,
    double tol,
    double &scale,
    double &scale_low,
    double &scale_high,
    double &previous_step);

void update_dual_box_cone_projection(
    const cone_parameters_t *cone_params,
    box_newton_state_t *dual_state,
    double *current_dual,
    double *reflected_dual,
    double *evaluation_dual,
    const double *initial_dual,
    const double *gradient,
    const double *step_size_d,
    const double *primal_scale_current,
    const int *base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *dual_scale_initial,
    double *dual_scale_current,
    double *dual_scale_evaluation,
    double *dual_scale_reflected,
    bool save_evaluation_iterate,
    cudaStream_t stream)
{
    const box_newton_state_t state = *dual_state;

    if (state.num_blocks_newton == 1)
    {
        update_dual_box_cone_projection_single_block_kernel<<<
            1, THREADS_PER_BLOCK, 0, stream>>>(
            current_dual,
            reflected_dual,
            evaluation_dual,
            initial_dual,
            gradient,
            step_size_d,
            primal_scale_current,
            base_count_d,
            k_offset,
            reflection_coefficient,
            dual_scale_initial,
            dual_scale_current,
            dual_scale_evaluation,
            dual_scale_reflected,
            save_evaluation_iterate,
            state,
            cone_params->max_iter,
            cone_params->tol);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeCooperative;
    attr.val.cooperative = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(state.num_blocks_newton);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        update_dual_box_cone_projection_cooperative_kernel,
        current_dual,
        reflected_dual,
        evaluation_dual,
        initial_dual,
        gradient,
        step_size_d,
        primal_scale_current,
        base_count_d,
        k_offset,
        reflection_coefficient,
        dual_scale_initial,
        dual_scale_current,
        dual_scale_evaluation,
        dual_scale_reflected,
        save_evaluation_iterate,
        state,
        cone_params->max_iter,
        cone_params->tol));
}

void update_primal_box_cone_resolvent(
    const cone_parameters_t *cone_params,
    box_newton_state_t *primal_state,
    double *current_primal,
    double *reflected_primal,
    double *evaluation_primal,
    const double *initial_primal,
    const double *gradient,
    const double *objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    const double *homogeneous_dual_gradient,
    const int *base_count_d,
    int k_offset,
    double reflection_coefficient,
    double *initial_primal_scale,
    double *current_primal_scale,
    double *evaluation_primal_scale,
    double *reflected_primal_scale,
    bool save_evaluation_iterate,
    cudaStream_t stream)
{
    const box_newton_state_t state = *primal_state;

    if (state.num_blocks_newton == 1)
    {
        update_primal_box_cone_resolvent_single_block_kernel<<<
            1, THREADS_PER_BLOCK, 0, stream>>>(
            current_primal,
            reflected_primal,
            evaluation_primal,
            initial_primal,
            gradient,
            objective_d,
            step_size_d,
            step_size_homogeneous_d,
            homogeneous_dual_gradient,
            base_count_d,
            k_offset,
            reflection_coefficient,
            initial_primal_scale,
            current_primal_scale,
            evaluation_primal_scale,
            reflected_primal_scale,
            save_evaluation_iterate,
            state,
            cone_params->max_iter,
            cone_params->tol);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeCooperative;
    attr.val.cooperative = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(state.num_blocks_newton);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        update_primal_box_cone_resolvent_cooperative_kernel,
        current_primal,
        reflected_primal,
        evaluation_primal,
        initial_primal,
        gradient,
        objective_d,
        step_size_d,
        step_size_homogeneous_d,
        homogeneous_dual_gradient,
        base_count_d,
        k_offset,
        reflection_coefficient,
        initial_primal_scale,
        current_primal_scale,
        evaluation_primal_scale,
        reflected_primal_scale,
        save_evaluation_iterate,
        state,
        cone_params->max_iter,
        cone_params->tol));
}

int compute_max_cooperative_dual_box_cone_blocks()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    int active = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active,
        update_dual_box_cone_projection_cooperative_kernel,
        THREADS_PER_BLOCK,
        0));
    return active * sm_count;
}

int compute_max_cooperative_primal_box_cone_blocks()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    int active = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active,
        update_primal_box_cone_resolvent_cooperative_kernel,
        THREADS_PER_BLOCK,
        0));
    return active * sm_count;
}

__global__ void update_dual_box_cone_projection_cooperative_kernel(
    double *__restrict__ current_dual,
    double *__restrict__ reflected_dual,
    double *__restrict__ evaluation_dual,
    const double *__restrict__ initial_dual,
    const double *__restrict__ gradient,
    const double *step_size_d,
    const double *primal_scale_current,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *dual_scale_initial,
    double *dual_scale_current,
    double *dual_scale_evaluation,
    double *dual_scale_reflected,
    bool save_evaluation_iterate,
    box_newton_state_t state,
    int max_iter,
    double tol)
{
    cg::grid_group grid = cg::this_grid();

    __shared__ ConeBlockReduce::TempStorage temp;
    __shared__ double s_scale;
    __shared__ int s_converged;

    double scale = *state.scalar_d;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double step_size = *step_size_d;
    const double unprojected_scale =
        *dual_scale_current - step_size * (*primal_scale_current);

    for (int iter = 0; iter < max_iter; ++iter)
    {
        newton_terms_t *partials = state.newton_partials_d + (iter & 1) * state.num_blocks_newton;

        double thread_derivative = 0.0, thread_function = 0.0;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < state.num_elements;
             i += state.num_blocks_newton * blockDim.x)
        {
            double function_contribution, derivative_contribution;
            // Recompute instead of materializing an n-element intermediate.
            // This keeps the fused kernel valid for arbitrarily large cones
            // without fixed-size per-thread register arrays.
            const double unprojected =
                current_dual[i] - step_size * gradient[i];
            evaluate_dual_box_cone_newton_terms(
                unprojected, state.lower_bound_d[i], state.upper_bound_d[i], scale,
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
            for (int bk = threadIdx.x; bk < state.num_blocks_newton; bk += warpSize)
            {
                total_derivative += partials[bk].derivative;
                total_function += partials[bk].function_value;
            }
            warp_reduce_newton_terms(total_derivative, total_function);

            if (threadIdx.x == 0)
            {
                if (iter == 0)
                {
                    const double function_value =
                        scale * (1.0 + total_derivative) - total_function + unprojected_scale;
                    const double scale_bound = fmax(0.0, scale - function_value);
                    if (function_value < 0.0)
                    {
                        scale_high = scale_bound;
                    }
                    else
                    {
                        scale_low = scale_bound;
                        scale_high = fmax(0.0, scale);
                    }
                }
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

    const int base_count = *base_count_d;
    const int k = base_count + k_offset;
    const double weight = (double)k / (double)(k + 1);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < state.num_elements;
         i += state.num_blocks_newton * blockDim.x)
    {
        const double current = current_dual[i];
        const double unprojected = current - step_size * gradient[i];
        const double lower = state.lower_bound_d[i];
        const double upper = state.upper_bound_d[i];
        const double box_lower = isfinite(lower) ? lower * scale : -INFINITY;
        const double box_upper = isfinite(upper) ? upper * scale : INFINITY;
        const double projected =
            unprojected + fmin(fmax(-unprojected, box_lower), box_upper);
        if (save_evaluation_iterate)
            evaluation_dual[i] = projected;
        const double reflected = 2.0 * projected - current;
        reflected_dual[i] = reflected;
        const double relaxed =
            reflection_coefficient * reflected +
            (1.0 - reflection_coefficient) * current;
        current_dual[i] =
            weight * relaxed + (1.0 - weight) * initial_dual[i];
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *state.scalar_d = scale;
        const double projected = unprojected_scale + scale;
        if (save_evaluation_iterate)
            *dual_scale_evaluation = projected;
        const double current = *dual_scale_current;
        const double reflected = 2.0 * projected - current;
        const double relaxed =
            reflection_coefficient * reflected +
            (1.0 - reflection_coefficient) * current;
        *dual_scale_current =
            weight * relaxed + (1.0 - weight) * (*dual_scale_initial);
        *dual_scale_reflected = reflected;
    }
}

__global__ void update_dual_box_cone_projection_single_block_kernel(
    double *__restrict__ current_dual,
    double *__restrict__ reflected_dual,
    double *__restrict__ evaluation_dual,
    const double *__restrict__ initial_dual,
    const double *__restrict__ gradient,
    const double *step_size_d,
    const double *primal_scale_current,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *dual_scale_initial,
    double *dual_scale_current,
    double *dual_scale_evaluation,
    double *dual_scale_reflected,
    bool save_evaluation_iterate,
    box_newton_state_t state,
    int max_iter,
    double tol)
{
    __shared__ ConeBlockReduce::TempStorage temp;
    __shared__ double s_scale;
    __shared__ int s_converged;

    double scale = *state.scalar_d;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    const double step_size = *step_size_d;
    const double unprojected_scale =
        *dual_scale_current - step_size * (*primal_scale_current);

    for (int iter = 0; iter < max_iter; ++iter)
    {
        double thread_derivative = 0.0, thread_function = 0.0;
        for (int i = threadIdx.x; i < state.num_elements; i += blockDim.x)
        {
            double function_contribution, derivative_contribution;
            const double unprojected =
                current_dual[i] - step_size * gradient[i];
            evaluate_dual_box_cone_newton_terms(
                unprojected, state.lower_bound_d[i], state.upper_bound_d[i], scale,
                function_contribution, derivative_contribution);
            thread_derivative += derivative_contribution;
            thread_function += function_contribution;
        }

        double total_derivative, total_function;
        block_sum_newton_terms(temp, thread_derivative, thread_function, total_derivative, total_function);

        if (threadIdx.x == 0)
        {
            if (iter == 0)
            {
                const double function_value =
                    scale * (1.0 + total_derivative) - total_function + unprojected_scale;
                const double scale_bound = fmax(0.0, scale - function_value);
                if (function_value < 0.0)
                {
                    scale_high = scale_bound;
                }
                else
                {
                    scale_low = scale_bound;
                    scale_high = fmax(0.0, scale);
                }
            }
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

    const int base_count = *base_count_d;
    const int k = base_count + k_offset;
    const double weight = (double)k / (double)(k + 1);
    for (int i = threadIdx.x; i < state.num_elements; i += blockDim.x)
    {
        const double current = current_dual[i];
        const double unprojected = current - step_size * gradient[i];
        const double lower = state.lower_bound_d[i];
        const double upper = state.upper_bound_d[i];
        const double box_lower = isfinite(lower) ? lower * scale : -INFINITY;
        const double box_upper = isfinite(upper) ? upper * scale : INFINITY;
        const double projected =
            unprojected + fmin(fmax(-unprojected, box_lower), box_upper);
        if (save_evaluation_iterate)
            evaluation_dual[i] = projected;
        const double reflected = 2.0 * projected - current;
        reflected_dual[i] = reflected;
        const double relaxed =
            reflection_coefficient * reflected +
            (1.0 - reflection_coefficient) * current;
        current_dual[i] =
            weight * relaxed + (1.0 - weight) * initial_dual[i];
    }

    if (threadIdx.x == 0)
    {
        *state.scalar_d = scale;
        const double projected = unprojected_scale + scale;
        if (save_evaluation_iterate)
            *dual_scale_evaluation = projected;
        const double current = *dual_scale_current;
        const double reflected = 2.0 * projected - current;
        const double relaxed =
            reflection_coefficient * reflected +
            (1.0 - reflection_coefficient) * current;
        *dual_scale_current =
            weight * relaxed + (1.0 - weight) * (*dual_scale_initial);
        *dual_scale_reflected = reflected;
    }
}

__global__ void update_primal_box_cone_resolvent_cooperative_kernel(
    double *__restrict__ current_primal,
    double *__restrict__ reflected_primal,
    double *__restrict__ evaluation_primal,
    const double *__restrict__ initial_primal,
    const double *__restrict__ gradient,
    const double *__restrict__ objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    const double *homogeneous_dual_gradient,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    double *initial_primal_scale,
    double *current_primal_scale,
    double *evaluation_primal_scale,
    double *reflected_primal_scale,
    bool save_evaluation_iterate,
    box_newton_state_t state,
    int max_iter,
    double tol)
{
    cg::grid_group grid = cg::this_grid();

    __shared__ ConeBlockReduce::TempStorage temp;
    __shared__ double s_scale;
    __shared__ int s_converged;

    const double primal_step = *step_size_d;
    const double current_scale = *current_primal_scale;
    double primal_scale = current_scale;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    double newton_previous_step = INFINITY;
    const double unprojected_primal_scale =
        current_scale +
        (*step_size_homogeneous_d) * (*homogeneous_dual_gradient);
    const double primal_metric = 1.0 / primal_step;
    const double homogeneous_metric =
        1.0 / (*step_size_homogeneous_d);

    for (int iter = 0; iter < max_iter; ++iter)
    {
        newton_terms_t *partials =
            state.newton_partials_d +
            (iter & 1) * state.num_blocks_newton;

        double thread_derivative = 0.0, thread_function = 0.0;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < state.num_elements;
             i += state.num_blocks_newton * blockDim.x)
        {
            double function_contribution, derivative_contribution;
            const double unprojected =
                current_primal[i] + primal_step * gradient[i];
            evaluate_primal_box_cone_newton_terms(
                unprojected, objective_d[i],
                state.lower_bound_d[i], state.upper_bound_d[i],
                primal_scale, primal_metric, function_contribution, derivative_contribution);
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
            for (int bk = threadIdx.x;
                 bk < state.num_blocks_newton;
                 bk += warpSize)
            {
                total_derivative += partials[bk].derivative;
                total_function += partials[bk].function_value;
            }
            warp_reduce_newton_terms(total_derivative, total_function);

            if (threadIdx.x == 0)
            {
                const double derivative =
                    total_derivative + homogeneous_metric;
                const double function_value =
                    total_function +
                    homogeneous_metric *
                        (unprojected_primal_scale -
                         primal_scale);
                if (iter == 0)
                {
                    const double scale_bound =
                        fmax(0.0, primal_scale + function_value / homogeneous_metric);
                    if (function_value > 0.0)
                    {
                        scale_high = scale_bound;
                    }
                    else
                    {
                        scale_low = scale_bound;
                        scale_high = fmax(0.0, primal_scale);
                    }
                }
                const double newton_step =
                    fmax(0.0, primal_scale + function_value / derivative);
                const bool new_converged = update_newton_bracket_safeguarded(
                    newton_step, function_value, unprojected_primal_scale, tol,
                    primal_scale, scale_low, scale_high, newton_previous_step);
                s_scale = primal_scale;
                s_converged = new_converged ? 1 : 0;
            }
        }

        __syncthreads();
        primal_scale = s_scale;

        if (s_converged)
        {
            break;
        }
    }

    const int base_count = *base_count_d;
    const int k = base_count + k_offset;
    const double weight = (double)k / (double)(k + 1);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < state.num_elements;
         i += state.num_blocks_newton * blockDim.x)
    {
        const double unprojected =
            current_primal[i] + primal_step * gradient[i];
        const double shifted =
            unprojected - primal_step * objective_d[i] * primal_scale;
        const double lower =
            isfinite(state.lower_bound_d[i])
                ? state.lower_bound_d[i] * primal_scale
                : -INFINITY;
        const double upper =
            isfinite(state.upper_bound_d[i])
                ? state.upper_bound_d[i] * primal_scale
                : INFINITY;
        const double projected = fmin(fmax(shifted, lower), upper);

        if (save_evaluation_iterate)
            evaluation_primal[i] = projected;

        const double reflected = 2.0 * projected - current_primal[i];
        reflected_primal[i] = reflected;
        const double relaxed =
            reflection_coefficient * reflected +
            (1.0 - reflection_coefficient) * current_primal[i];
        current_primal[i] =
            weight * relaxed + (1.0 - weight) * initial_primal[i];
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if (save_evaluation_iterate)
            *evaluation_primal_scale = primal_scale;

        const double reflected_scale =
            2.0 * primal_scale - (*current_primal_scale);
        const double relaxed_scale =
            reflection_coefficient * reflected_scale +
            (1.0 - reflection_coefficient) *
                (*current_primal_scale);
        *current_primal_scale =
            weight * relaxed_scale +
            (1.0 - weight) * (*initial_primal_scale);
        *reflected_primal_scale = reflected_scale;
    }
}

__global__ void update_primal_box_cone_resolvent_single_block_kernel(
    double *__restrict__ current_primal,
    double *__restrict__ reflected_primal,
    double *__restrict__ evaluation_primal,
    const double *__restrict__ initial_primal,
    const double *__restrict__ gradient,
    const double *__restrict__ objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    const double *homogeneous_dual_gradient,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    double *initial_primal_scale,
    double *current_primal_scale,
    double *evaluation_primal_scale,
    double *reflected_primal_scale,
    bool save_evaluation_iterate,
    box_newton_state_t state,
    int max_iter,
    double tol)
{
    __shared__ ConeBlockReduce::TempStorage temp;
    __shared__ double s_scale;
    __shared__ int s_converged;

    const double primal_step = *step_size_d;
    const double current_scale = *current_primal_scale;
    double primal_scale = current_scale;
    double scale_low = 0.0;
    double scale_high = INFINITY;
    double newton_previous_step = INFINITY;
    const double unprojected_primal_scale =
        current_scale +
        (*step_size_homogeneous_d) * (*homogeneous_dual_gradient);
    const double primal_metric = 1.0 / primal_step;
    const double homogeneous_metric =
        1.0 / (*step_size_homogeneous_d);

    for (int iter = 0; iter < max_iter; ++iter)
    {
        double thread_derivative = 0.0, thread_function = 0.0;
        for (int i = threadIdx.x;
             i < state.num_elements;
             i += blockDim.x)
        {
            double function_contribution, derivative_contribution;
            const double unprojected =
                current_primal[i] + primal_step * gradient[i];
            evaluate_primal_box_cone_newton_terms(
                unprojected, objective_d[i],
                state.lower_bound_d[i], state.upper_bound_d[i],
                primal_scale, primal_metric, function_contribution, derivative_contribution);
            thread_derivative += derivative_contribution;
            thread_function += function_contribution;
        }

        double total_derivative, total_function;
        block_sum_newton_terms(temp, thread_derivative, thread_function, total_derivative, total_function);

        if (threadIdx.x == 0)
        {
            const double derivative =
                total_derivative + homogeneous_metric;
            const double function_value =
                total_function +
                homogeneous_metric *
                    (unprojected_primal_scale -
                     primal_scale);
            if (iter == 0)
            {
                const double scale_bound =
                    fmax(0.0, primal_scale + function_value / homogeneous_metric);
                if (function_value > 0.0)
                {
                    scale_high = scale_bound;
                }
                else
                {
                    scale_low = scale_bound;
                    scale_high = fmax(0.0, primal_scale);
                }
            }
            const double newton_step =
                fmax(0.0, primal_scale + function_value / derivative);
            const bool new_converged = update_newton_bracket_safeguarded(
                newton_step, function_value, unprojected_primal_scale, tol,
                primal_scale, scale_low, scale_high, newton_previous_step);
            s_scale = primal_scale;
            s_converged = new_converged ? 1 : 0;
        }

        __syncthreads();
        primal_scale = s_scale;

        if (s_converged)
        {
            break;
        }
    }

    const int base_count = *base_count_d;
    const int k = base_count + k_offset;
    const double weight = (double)k / (double)(k + 1);
    for (int i = threadIdx.x; i < state.num_elements; i += blockDim.x)
    {
        const double unprojected =
            current_primal[i] + primal_step * gradient[i];
        const double shifted =
            unprojected - primal_step * objective_d[i] * primal_scale;
        const double lower =
            isfinite(state.lower_bound_d[i])
                ? state.lower_bound_d[i] * primal_scale
                : -INFINITY;
        const double upper =
            isfinite(state.upper_bound_d[i])
                ? state.upper_bound_d[i] * primal_scale
                : INFINITY;
        const double projected = fmin(fmax(shifted, lower), upper);

        if (save_evaluation_iterate)
            evaluation_primal[i] = projected;

        const double reflected = 2.0 * projected - current_primal[i];
        reflected_primal[i] = reflected;
        const double relaxed =
            reflection_coefficient * reflected +
            (1.0 - reflection_coefficient) * current_primal[i];
        current_primal[i] =
            weight * relaxed + (1.0 - weight) * initial_primal[i];
    }

    if (threadIdx.x == 0)
    {
        if (save_evaluation_iterate)
            *evaluation_primal_scale = primal_scale;

        const double reflected_scale =
            2.0 * primal_scale - (*current_primal_scale);
        const double relaxed_scale =
            reflection_coefficient * reflected_scale +
            (1.0 - reflection_coefficient) *
                (*current_primal_scale);
        *current_primal_scale =
            weight * relaxed_scale +
            (1.0 - weight) * (*initial_primal_scale);
        *reflected_primal_scale = reflected_scale;
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

__device__ __forceinline__ void evaluate_dual_box_cone_newton_terms(
    double unprojected_value, double l, double u, double scale,
    double &function_contribution, double &derivative_contribution)
{
    const double vi = -unprojected_value;
    const double bound = select_active_dual_bound(vi, l, u, scale);
    derivative_contribution = bound * bound;
    function_contribution = bound * vi;
}

__device__ __forceinline__ void evaluate_primal_box_cone_newton_terms(
    double vi, double ci, double li, double ui,
    double primal_scale, double primal_metric,
    double &function_contribution, double &derivative_contribution)
{
    const double xhat = vi - ci * primal_scale / primal_metric;
    const double box_lower = isfinite(li) ? li * primal_scale : -INFINITY;
    const double box_upper = isfinite(ui) ? ui * primal_scale : INFINITY;

    if (xhat > box_upper)
    {
        const double n_i = primal_metric * (vi - box_upper) - ci * primal_scale;
        function_contribution = ci * box_upper + ui * n_i;
        derivative_contribution = primal_metric * ui * ui;
    }
    else if (xhat < box_lower)
    {
        const double n_i = primal_metric * (vi - box_lower) - ci * primal_scale;
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

__device__ __forceinline__ bool update_newton_bracket_safeguarded(
    double newton_step,
    double function_value,
    double unprojected_scale,
    double tol,
    double &scale,
    double &scale_low,
    double &scale_high,
    double &previous_step)
{
    if (function_value > 0.0)
        scale_low = scale;
    else
        scale_high = scale;

    const double newton_delta = newton_step - scale;
    const bool in_bracket =
        (newton_step >= scale_low) && (newton_step <= scale_high);

    double new_scale;
    if (isfinite(scale_high) && (!in_bracket || fabs(newton_delta) > 0.5 * fabs(previous_step)))
    {
        previous_step = 0.5 * (scale_high - scale_low);
        new_scale = scale_low + previous_step;
    }
    else
    {
        previous_step = newton_delta;
        new_scale = newton_step;
    }

    const bool converged =
        (fabs(new_scale - scale) < tol * fabs(new_scale) + 1.0e-14 * fabs(unprojected_scale));
    scale = new_scale;
    return converged;
}
