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
#include "cuda_macros.h"
#include <cooperative_groups.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <math.h>

namespace cg = cooperative_groups;

struct combine_newton_terms_t
{
    __device__ __forceinline__ newton_terms_t operator()(
        const newton_terms_t &a, const newton_terms_t &b) const
    {
        return newton_terms_t{
            a.negative_derivative + b.negative_derivative,
            a.function_value + b.function_value};
    }
};

typedef cub::BlockReduce<newton_terms_t, THREADS_PER_BLOCK> cone_block_reduce_t;

struct newton_bracket_t
{
    double scale_low;
    double scale_high;
    double previous_step;
};

template <bool COOPERATIVE>
__global__ void update_dual_box_cone_projection_kernel(
    double *__restrict__ dual_current,
    double *__restrict__ dual_reflected,
    double *__restrict__ dual_evaluation,
    double *__restrict__ constraint_dual_projection_input,
    const double *__restrict__ dual_initial,
    const double *__restrict__ gradient,
    const double *step_size_d,
    const double *primal_current_scale,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *dual_initial_scale,
    double *dual_current_scale,
    double *dual_evaluation_scale,
    double *dual_reflected_scale,
    bool is_major,
    box_newton_state_t state,
    int max_iter,
    double tol);
__device__ __forceinline__ double compute_newton_start_scale(const double *history);
template <bool COOPERATIVE, typename EvaluateTerms, typename UpdateScale>
__device__ __forceinline__ double solve_scale_newton(
    double scale,
    int newton_max_iter,
    int num_elements,
    newton_terms_t *newton_partials,
    EvaluateTerms evaluate_terms,
    UpdateScale update_scale);
template <bool COOPERATIVE>
__device__ __forceinline__ newton_terms_t reduce_newton_terms(
    const newton_terms_t &thread_terms,
    newton_terms_t *newton_partials,
    int iter);
__device__ __forceinline__ void warp_reduce_newton_terms(newton_terms_t &terms);
__device__ __forceinline__ newton_terms_t evaluate_dual_box_cone_newton_terms(
    double unprojected,
    double lb,
    double ub,
    double scale);
__device__ __forceinline__ double select_active_dual_bound(
    double vi, double lb, double ub, double scale);
__device__ __forceinline__ bool update_dual_scale_newton_step(
    const newton_terms_t &total,
    double unprojected_scale,
    double tol,
    double &scale,
    newton_bracket_t &bracket);
__device__ __forceinline__ bool update_newton_bracket_safeguarded(
    double newton_step,
    double function_value,
    double unprojected_scale,
    double tol,
    double &scale,
    newton_bracket_t &bracket);
__device__ __forceinline__ double compute_halpern_update(
    double reflected,
    double current,
    double initial,
    double reflection_coefficient,
    double weight);
__device__ __forceinline__ void update_scale_history(double *history, double scale);
template <bool COOPERATIVE>
__global__ void update_primal_box_cone_resolvent_kernel(
    double *__restrict__ primal_current,
    double *__restrict__ primal_reflected,
    double *__restrict__ primal_evaluation,
    double *__restrict__ variable_primal_projection_multiplier,
    const double *__restrict__ primal_initial,
    const double *__restrict__ gradient,
    const double *__restrict__ objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    const double *homogeneous_dual_gradient,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *primal_initial_scale,
    double *primal_current_scale,
    double *primal_evaluation_scale,
    double *primal_reflected_scale,
    bool is_major,
    box_newton_state_t state,
    int max_iter,
    double tol);
__device__ __forceinline__ newton_terms_t evaluate_primal_box_cone_newton_terms(
    double unprojected,
    double ci,
    double lb,
    double ub,
    double scale,
    double primal_step,
    double inv_primal_step);
__device__ __forceinline__ bool update_primal_scale_newton_step(
    const newton_terms_t &total,
    int iter,
    double inv_homogeneous_step,
    double unprojected_scale,
    double tol,
    double &scale,
    newton_bracket_t &bracket);
__device__ __forceinline__ void initialize_newton_bracket(
    double function_value,
    double scale,
    double slope_floor,
    newton_bracket_t &bracket);
template <typename Kernel, typename... Args>
static void launch_box_cone_kernel(
    Kernel kernel, int num_blocks, cudaStream_t stream, Args... args);
static int compute_max_cooperative_blocks(const void *kernel);

void update_dual_box_cone_projection(
    const cone_parameters_t *cone_params,
    const box_newton_state_t *dual_state,
    double *dual_current,
    double *dual_reflected,
    double *dual_evaluation,
    double *constraint_dual_projection_input,
    const double *dual_initial,
    const double *gradient,
    const double *step_size_d,
    const double *primal_current_scale,
    const int *base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *dual_initial_scale,
    double *dual_current_scale,
    double *dual_evaluation_scale,
    double *dual_reflected_scale,
    bool is_major,
    cudaStream_t stream)
{
    launch_box_cone_kernel(
        dual_state->num_blocks_newton > 1
            ? update_dual_box_cone_projection_kernel<true>
            : update_dual_box_cone_projection_kernel<false>,
        dual_state->num_blocks_newton,
        stream,
        dual_current,
        dual_reflected,
        dual_evaluation,
        constraint_dual_projection_input,
        dual_initial,
        gradient,
        step_size_d,
        primal_current_scale,
        base_count_d,
        k_offset,
        reflection_coefficient,
        dual_initial_scale,
        dual_current_scale,
        dual_evaluation_scale,
        dual_reflected_scale,
        is_major,
        *dual_state,
        cone_params->max_iter,
        cone_params->tol);
}

template <typename Kernel, typename... Args>
static void launch_box_cone_kernel(
    Kernel kernel, int num_blocks, cudaStream_t stream, Args... args)
{
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeCooperative;
    attr.val.cooperative = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_blocks);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = num_blocks > 1 ? 1 : 0;

    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

void update_primal_box_cone_resolvent(
    const cone_parameters_t *cone_params,
    const box_newton_state_t *primal_state,
    double *primal_current,
    double *primal_reflected,
    double *primal_evaluation,
    double *variable_primal_projection_multiplier,
    const double *primal_initial,
    const double *gradient,
    const double *objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    const double *homogeneous_dual_gradient,
    const int *base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *primal_initial_scale,
    double *primal_current_scale,
    double *primal_evaluation_scale,
    double *primal_reflected_scale,
    bool is_major,
    cudaStream_t stream)
{
    launch_box_cone_kernel(
        primal_state->num_blocks_newton > 1
            ? update_primal_box_cone_resolvent_kernel<true>
            : update_primal_box_cone_resolvent_kernel<false>,
        primal_state->num_blocks_newton,
        stream,
        primal_current,
        primal_reflected,
        primal_evaluation,
        variable_primal_projection_multiplier,
        primal_initial,
        gradient,
        objective_d,
        step_size_d,
        step_size_homogeneous_d,
        homogeneous_dual_gradient,
        base_count_d,
        k_offset,
        reflection_coefficient,
        primal_initial_scale,
        primal_current_scale,
        primal_evaluation_scale,
        primal_reflected_scale,
        is_major,
        *primal_state,
        cone_params->max_iter,
        cone_params->tol);
}

int compute_max_cooperative_dual_box_cone_blocks()
{
    return compute_max_cooperative_blocks(
        (const void *)update_dual_box_cone_projection_kernel<true>);
}

static int compute_max_cooperative_blocks(const void *kernel)
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    int active = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active,
        kernel,
        THREADS_PER_BLOCK,
        0));
    return active * sm_count;
}

int compute_max_cooperative_primal_box_cone_blocks()
{
    return compute_max_cooperative_blocks(
        (const void *)update_primal_box_cone_resolvent_kernel<true>);
}

template <bool COOPERATIVE>
__global__ void update_dual_box_cone_projection_kernel(
    double *__restrict__ dual_current,
    double *__restrict__ dual_reflected,
    double *__restrict__ dual_evaluation,
    double *__restrict__ constraint_dual_projection_input,
    const double *__restrict__ dual_initial,
    const double *__restrict__ gradient,
    const double *step_size_d,
    const double *primal_current_scale,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *dual_initial_scale,
    double *dual_current_scale,
    double *dual_evaluation_scale,
    double *dual_reflected_scale,
    bool is_major,
    box_newton_state_t state,
    int max_iter,
    double tol)
{
    const double step_size = *step_size_d;
    const double current_scale = *dual_current_scale;
    const double unprojected_scale =
        current_scale - step_size * (*primal_current_scale);
    const int newton_max_iter =
        state.newton_terms_are_zero ? 0 : max_iter;
    double scale = state.newton_terms_are_zero
        ? fmax(0.0, -unprojected_scale)
        : compute_newton_start_scale(state.scale_history_d);

    newton_bracket_t bracket = {0.0, INFINITY, INFINITY};
    scale = solve_scale_newton<COOPERATIVE>(
        scale,
        newton_max_iter,
        state.num_elements,
        state.newton_partials_d,
        [&](int i, double trial_scale)
        {
            return evaluate_dual_box_cone_newton_terms(
                dual_current[i] - step_size * gradient[i],
                state.lower_bound_d[i],
                state.upper_bound_d[i],
                trial_scale);
        },
        [&](const newton_terms_t &total, int, double &trial_scale)
        {
            return update_dual_scale_newton_step(
                total, unprojected_scale, tol, trial_scale, bracket);
        });

    const int k = *base_count_d + k_offset;
    const double weight = (double)k / (double)(k + 1);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < state.num_elements;
         i += gridDim.x * blockDim.x)
    {
        const double current = dual_current[i];
        const double unprojected = current - step_size * gradient[i];
        const double lb = state.lower_bound_d[i];
        const double ub = state.upper_bound_d[i];
        const double box_lower = isfinite(lb) ? lb * scale : -INFINITY;
        const double box_upper = isfinite(ub) ? ub * scale : INFINITY;
        const double projected =
            unprojected + fmin(fmax(-unprojected, box_lower), box_upper);
        if (is_major)
        {
            dual_evaluation[i] = projected;
            if (constraint_dual_projection_input)
                constraint_dual_projection_input[i] = unprojected;
        }
        const double reflected = 2.0 * projected - current;
        dual_reflected[i] = reflected;
        dual_current[i] = compute_halpern_update(
            reflected, current, dual_initial[i], reflection_coefficient, weight);
    }

    if (newton_max_iter == 0)
    {
        if constexpr (COOPERATIVE)
            cg::this_grid().sync();
        else
            __syncthreads();
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        update_scale_history(state.scale_history_d, scale);
        const double projected_scale = unprojected_scale + scale;
        const double reflected_scale = 2.0 * projected_scale - current_scale;
        if (is_major)
            *dual_evaluation_scale = projected_scale;
        *dual_reflected_scale = reflected_scale;
        *dual_current_scale = compute_halpern_update(
            reflected_scale, current_scale, *dual_initial_scale,
            reflection_coefficient, weight);
    }
}

__device__ __forceinline__ double compute_newton_start_scale(const double *history)
{
    const double latest_scale   = history[BOX_CONE_SCALE_LATEST];
    const double previous_scale = history[BOX_CONE_SCALE_PREVIOUS];
    const double latest_delta   = latest_scale - previous_scale;
    const double previous_delta = previous_scale - history[BOX_CONE_SCALE_OLDEST];
    const bool deltas_agree =
        latest_delta * previous_delta > 0.0 &&
        fabs(latest_delta) <= 2.0 * fabs(previous_delta) &&
        fabs(previous_delta) <= 2.0 * fabs(latest_delta);
    return deltas_agree ? fmax(0.0, latest_scale + latest_delta) : latest_scale;
}

template <bool COOPERATIVE, typename EvaluateTerms, typename UpdateScale>
__device__ __forceinline__ double solve_scale_newton(
    double scale,
    int newton_max_iter,
    int num_elements,
    newton_terms_t *newton_partials,
    EvaluateTerms evaluate_terms,
    UpdateScale update_scale)
{
    __shared__ double s_scale;
    __shared__ bool s_converged;

    for (int iter = 0; iter < newton_max_iter; ++iter)
    {
        newton_terms_t thread_terms = {0.0, 0.0};
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < num_elements;
             i += gridDim.x * blockDim.x)
        {
            thread_terms =
                combine_newton_terms_t{}(thread_terms, evaluate_terms(i, scale));
        }

        const newton_terms_t total =
            reduce_newton_terms<COOPERATIVE>(thread_terms, newton_partials, iter);

        if (threadIdx.x == 0)
        {
            s_converged = update_scale(total, iter, scale);
            s_scale = scale;
        }

        __syncthreads();
        scale = s_scale;

        if (s_converged)
        {
            break;
        }
    }
    return scale;
}

template <bool COOPERATIVE>
__device__ __forceinline__ newton_terms_t reduce_newton_terms(
    const newton_terms_t &thread_terms,
    newton_terms_t *newton_partials,
    int iter)
{
    __shared__ cone_block_reduce_t::TempStorage temp;
    const newton_terms_t block_terms =
        cone_block_reduce_t(temp).Reduce(thread_terms, combine_newton_terms_t{});
    if constexpr (!COOPERATIVE)
    {
        return block_terms;
    }
    else
    {
        const int num_blocks = (int)gridDim.x;
        newton_terms_t *partials = newton_partials + (iter & 1) * num_blocks;
        if (threadIdx.x == 0)
        {
            partials[blockIdx.x] = block_terms;
        }

        cg::this_grid().sync();

        newton_terms_t total = {0.0, 0.0};
        if (threadIdx.x < warpSize)
        {
            for (int bk = threadIdx.x; bk < num_blocks; bk += warpSize)
            {
                total = combine_newton_terms_t{}(total, partials[bk]);
            }
            warp_reduce_newton_terms(total);
        }
        return total;
    }
}

__device__ __forceinline__ void warp_reduce_newton_terms(newton_terms_t &terms)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        const newton_terms_t other = {
            __shfl_down_sync(0xFFFFFFFFu, terms.negative_derivative, offset),
            __shfl_down_sync(0xFFFFFFFFu, terms.function_value, offset)};
        terms = combine_newton_terms_t{}(terms, other);
    }
}

__device__ __forceinline__ newton_terms_t evaluate_dual_box_cone_newton_terms(
    double unprojected,
    double lb,
    double ub,
    double scale)
{
    const double vi = -unprojected;
    const double bound =
        (lb == ub) ? ub : select_active_dual_bound(vi, lb, ub, scale);
    return newton_terms_t{bound * bound, bound * vi};
}

__device__ __forceinline__ double select_active_dual_bound(
    double vi, double lb, double ub, double scale)
{
    double bound = 0.0;
    bound = (vi > ub * scale) ? ub : bound;
    bound = (vi < lb * scale) ? lb : bound;
    return bound;
}

__device__ __forceinline__ bool update_dual_scale_newton_step(
    const newton_terms_t &total,
    double unprojected_scale,
    double tol,
    double &scale,
    newton_bracket_t &bracket)
{
    const double negative_derivative = 1.0 + total.negative_derivative;
    const double function_value =
        total.function_value -
        unprojected_scale -
        scale * negative_derivative;
    const double newton_step =
        fmax(0.0, (total.function_value - unprojected_scale) / negative_derivative);
    return update_newton_bracket_safeguarded(
        newton_step, function_value, unprojected_scale, tol, scale, bracket);
}

__device__ __forceinline__ bool update_newton_bracket_safeguarded(
    double newton_step,
    double function_value,
    double unprojected_scale,
    double tol,
    double &scale,
    newton_bracket_t &bracket)
{
    if (function_value > 0.0)
        bracket.scale_low = scale;
    else
        bracket.scale_high = scale;

    const double newton_delta = newton_step - scale;
    const bool in_bracket =
        (newton_step >= bracket.scale_low) && (newton_step <= bracket.scale_high);

    double new_scale;
    if (isfinite(bracket.scale_high) &&
        (!in_bracket || fabs(newton_delta) > 0.5 * fabs(bracket.previous_step)))
    {
        bracket.previous_step = 0.5 * (bracket.scale_high - bracket.scale_low);
        new_scale = bracket.scale_low + bracket.previous_step;
    }
    else
    {
        bracket.previous_step = newton_delta;
        new_scale = newton_step;
    }

    const double step_tolerance = tol * fabs(new_scale) + 1.0e-14 * fabs(unprojected_scale);
    const bool converged = fabs(new_scale - scale) <= step_tolerance;
    scale = new_scale;
    return converged;
}

__device__ __forceinline__ double compute_halpern_update(
    double reflected,
    double current,
    double initial,
    double reflection_coefficient,
    double weight)
{
    const double relaxed =
        reflection_coefficient * reflected +
        (1.0 - reflection_coefficient) * current;
    return weight * relaxed + (1.0 - weight) * initial;
}

__device__ __forceinline__ void update_scale_history(double *history, double scale)
{
    history[BOX_CONE_SCALE_OLDEST]   = history[BOX_CONE_SCALE_PREVIOUS];
    history[BOX_CONE_SCALE_PREVIOUS] = history[BOX_CONE_SCALE_LATEST];
    history[BOX_CONE_SCALE_LATEST]   = scale;
}

template <bool COOPERATIVE>
__global__ void update_primal_box_cone_resolvent_kernel(
    double *__restrict__ primal_current,
    double *__restrict__ primal_reflected,
    double *__restrict__ primal_evaluation,
    double *__restrict__ variable_primal_projection_multiplier,
    const double *__restrict__ primal_initial,
    const double *__restrict__ gradient,
    const double *__restrict__ objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    const double *homogeneous_dual_gradient,
    const int *__restrict__ base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *primal_initial_scale,
    double *primal_current_scale,
    double *primal_evaluation_scale,
    double *primal_reflected_scale,
    bool is_major,
    box_newton_state_t state,
    int max_iter,
    double tol)
{
    const double primal_step = *step_size_d;
    const double inv_primal_step = 1.0 / primal_step;
    const double homogeneous_step = *step_size_homogeneous_d;
    const double inv_homogeneous_step = 1.0 / homogeneous_step;
    const double current_scale = *primal_current_scale;
    const double unprojected_scale =
        current_scale + homogeneous_step * (*homogeneous_dual_gradient);
    const int newton_max_iter =
        state.newton_terms_are_zero ? 0 : max_iter;
    double scale = state.newton_terms_are_zero
        ? fmax(0.0, unprojected_scale)
        : compute_newton_start_scale(state.scale_history_d);

    newton_bracket_t bracket = {0.0, INFINITY, INFINITY};
    scale = solve_scale_newton<COOPERATIVE>(
        scale,
        newton_max_iter,
        state.num_elements,
        state.newton_partials_d,
        [&](int i, double trial_scale)
        {
            return evaluate_primal_box_cone_newton_terms(
                primal_current[i] + primal_step * gradient[i],
                objective_d[i],
                state.lower_bound_d[i],
                state.upper_bound_d[i],
                trial_scale,
                primal_step,
                inv_primal_step);
        },
        [&](const newton_terms_t &total, int iter, double &trial_scale)
        {
            return update_primal_scale_newton_step(
                total, iter, inv_homogeneous_step, unprojected_scale,
                tol, trial_scale, bracket);
        });

    const int k = *base_count_d + k_offset;
    const double weight = (double)k / (double)(k + 1);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < state.num_elements;
         i += gridDim.x * blockDim.x)
    {
        const double current = primal_current[i];
        const double unprojected = current + primal_step * gradient[i];
        const double shifted =
            unprojected - primal_step * objective_d[i] * scale;
        const double lb = state.lower_bound_d[i];
        const double ub = state.upper_bound_d[i];
        const double box_lower = isfinite(lb) ? lb * scale : -INFINITY;
        const double box_upper = isfinite(ub) ? ub * scale : INFINITY;
        const double projected = fmin(fmax(shifted, box_lower), box_upper);
        const double reflected = 2.0 * projected - current;
        if (is_major)
        {
            primal_evaluation[i] = projected;
            primal_reflected[i] = reflected;
            variable_primal_projection_multiplier[i] =
                (projected - shifted) * inv_primal_step;
        }
        primal_current[i] = compute_halpern_update(
            reflected, current, primal_initial[i], reflection_coefficient, weight);
    }

    if (newton_max_iter == 0)
    {
        if constexpr (COOPERATIVE)
            cg::this_grid().sync();
        else
            __syncthreads();
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        update_scale_history(state.scale_history_d, scale);
        const double reflected_scale = 2.0 * scale - current_scale;
        if (is_major)
        {
            *primal_evaluation_scale = scale;
            *primal_reflected_scale = reflected_scale;
        }
        *primal_current_scale = compute_halpern_update(
            reflected_scale, current_scale, *primal_initial_scale,
            reflection_coefficient, weight);
    }
}

__device__ __forceinline__ newton_terms_t evaluate_primal_box_cone_newton_terms(
    double unprojected,
    double ci,
    double lb,
    double ub,
    double scale,
    double primal_step,
    double inv_primal_step)
{
    const double shifted = unprojected - ci * scale * primal_step;
    const double box_lower = isfinite(lb) ? lb * scale : -INFINITY;
    const double box_upper = isfinite(ub) ? ub * scale : INFINITY;

    if (lb == ub || shifted > box_upper)
        return newton_terms_t{
            inv_primal_step * ub * ub,
            inv_primal_step * ub * (unprojected - box_upper)};
    if (shifted < box_lower)
        return newton_terms_t{
            inv_primal_step * lb * lb,
            inv_primal_step * lb * (unprojected - box_lower)};
    return newton_terms_t{ci * ci * primal_step, ci * shifted};
}

__device__ __forceinline__ bool update_primal_scale_newton_step(
    const newton_terms_t &total,
    int iter,
    double inv_homogeneous_step,
    double unprojected_scale,
    double tol,
    double &scale,
    newton_bracket_t &bracket)
{
    const double negative_derivative =
        total.negative_derivative + inv_homogeneous_step;
    const double function_value =
        total.function_value +
        inv_homogeneous_step * (unprojected_scale - scale);
    if (iter == 0)
        initialize_newton_bracket(
            function_value,
            scale,
            inv_homogeneous_step,
            bracket);
    const double newton_step =
        fmax(0.0, scale + function_value / negative_derivative);
    return update_newton_bracket_safeguarded(
        newton_step, function_value, unprojected_scale, tol,
        scale, bracket);
}

__device__ __forceinline__ void initialize_newton_bracket(
    double function_value,
    double scale,
    double slope_floor,
    newton_bracket_t &bracket)
{
    const double scale_bound = fmax(0.0, scale + function_value / slope_floor);
    if (function_value > 0.0)
    {
        bracket.scale_high = scale_bound;
    }
    else
    {
        bracket.scale_low = scale_bound;
        bracket.scale_high = fmax(0.0, scale);
    }
}
