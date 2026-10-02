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
#include <cub/cub.cuh>
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

template <typename Kernel, typename... Args>
static void launch_box_cone_kernel(
    Kernel kernel, int num_blocks, cudaStream_t stream, Args... args);
template <bool COOPERATIVE>
__global__ void update_dual_box_cone_projection_kernel(
    double *__restrict__ current_dual,
    double *__restrict__ reflected_dual,
    double *__restrict__ evaluation_dual,
    double *__restrict__ constraint_dual_projection_input,
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
    bool is_major,
    box_newton_state_t state,
    int max_iter,
    double tol);
template <bool COOPERATIVE, typename EvaluateTerms, typename UpdateScale>
__device__ __forceinline__ double solve_scale_newton(
    double scale,
    int newton_max_iter,
    int num_elements,
    newton_terms_t *newton_partials,
    EvaluateTerms evaluate_terms,
    UpdateScale update_scale);
__device__ __forceinline__ newton_terms_t make_empty_newton_terms();
template <bool COOPERATIVE>
__device__ __forceinline__ newton_terms_t reduce_newton_terms(
    const newton_terms_t &thread_terms,
    newton_terms_t *newton_partials,
    int iter);
__device__ inline void warp_reduce_newton_terms(newton_terms_t &terms);
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
template <bool COOPERATIVE>
__global__ void update_primal_box_cone_resolvent_kernel(
    double *__restrict__ current_primal,
    double *__restrict__ reflected_primal,
    double *__restrict__ evaluation_primal,
    double *__restrict__ variable_primal_projection_multiplier,
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
static int compute_max_cooperative_blocks(const void *kernel);

void update_dual_box_cone_projection(
    const cone_parameters_t *cone_params,
    box_newton_state_t *dual_state,
    double *current_dual,
    double *reflected_dual,
    double *evaluation_dual,
    double *constraint_dual_projection_input,
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
    bool is_major,
    cudaStream_t stream)
{
    const box_newton_state_t state = *dual_state;
    launch_box_cone_kernel(
        state.num_blocks_newton > 1
            ? update_dual_box_cone_projection_kernel<true>
            : update_dual_box_cone_projection_kernel<false>,
        state.num_blocks_newton,
        stream,
        current_dual,
        reflected_dual,
        evaluation_dual,
        constraint_dual_projection_input,
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
        is_major,
        state,
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

template <bool COOPERATIVE>
__global__ void update_dual_box_cone_projection_kernel(
    double *__restrict__ current_dual,
    double *__restrict__ reflected_dual,
    double *__restrict__ evaluation_dual,
    double *__restrict__ constraint_dual_projection_input,
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
    bool is_major,
    box_newton_state_t state,
    int max_iter,
    double tol)
{
    const double step_size = *step_size_d;
    const double current_scale = *dual_scale_current;
    const double unprojected_scale =
        current_scale - step_size * (*primal_scale_current);
    const int newton_max_iter =
        state.newton_terms_are_zero ? 0 : max_iter;
    double scale = *state.scalar_d;
    if (state.newton_terms_are_zero)
        scale = fmax(0.0, -unprojected_scale);

    newton_bracket_t bracket = {0.0, INFINITY, INFINITY};
    scale = solve_scale_newton<COOPERATIVE>(
        scale,
        newton_max_iter,
        state.num_elements,
        state.newton_partials_d,
        [&](int i, double trial_scale)
        {
            return evaluate_dual_box_cone_newton_terms(
                current_dual[i] - step_size * gradient[i],
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
        const double current = current_dual[i];
        const double unprojected = current - step_size * gradient[i];
        const double lb = state.lower_bound_d[i];
        const double ub = state.upper_bound_d[i];
        const double box_lower = isfinite(lb) ? lb * scale : -INFINITY;
        const double box_upper = isfinite(ub) ? ub * scale : INFINITY;
        const double projected =
            unprojected + fmin(fmax(-unprojected, box_lower), box_upper);
        if (is_major)
        {
            evaluation_dual[i] = projected;
            if (constraint_dual_projection_input)
                constraint_dual_projection_input[i] = unprojected;
        }
        const double reflected = 2.0 * projected - current;
        reflected_dual[i] = reflected;
        current_dual[i] = compute_halpern_update(
            reflected, current, initial_dual[i], reflection_coefficient, weight);
    }

    if constexpr (COOPERATIVE)
    {
        if (newton_max_iter == 0)
            cg::this_grid().sync();
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *state.scalar_d = scale;
        const double projected = unprojected_scale + scale;
        if (is_major)
            *dual_scale_evaluation = projected;
        const double reflected = 2.0 * projected - current_scale;
        *dual_scale_current = compute_halpern_update(
            reflected, current_scale, *dual_scale_initial,
            reflection_coefficient, weight);
        *dual_scale_reflected = reflected;
    }
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
        newton_terms_t thread_terms = make_empty_newton_terms();
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

__device__ __forceinline__ newton_terms_t make_empty_newton_terms()
{
    return newton_terms_t{0.0, 0.0};
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

        newton_terms_t total = make_empty_newton_terms();
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

__device__ inline void warp_reduce_newton_terms(newton_terms_t &terms)
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
    const double bound = select_active_dual_bound(vi, lb, ub, scale);
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

void update_primal_box_cone_resolvent(
    const cone_parameters_t *cone_params,
    box_newton_state_t *primal_state,
    double *current_primal,
    double *reflected_primal,
    double *evaluation_primal,
    double *variable_primal_projection_multiplier,
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
    bool is_major,
    cudaStream_t stream)
{
    const box_newton_state_t state = *primal_state;
    launch_box_cone_kernel(
        state.num_blocks_newton > 1
            ? update_primal_box_cone_resolvent_kernel<true>
            : update_primal_box_cone_resolvent_kernel<false>,
        state.num_blocks_newton,
        stream,
        current_primal,
        reflected_primal,
        evaluation_primal,
        variable_primal_projection_multiplier,
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
        is_major,
        state,
        cone_params->max_iter,
        cone_params->tol);
}

template <bool COOPERATIVE>
__global__ void update_primal_box_cone_resolvent_kernel(
    double *__restrict__ current_primal,
    double *__restrict__ reflected_primal,
    double *__restrict__ evaluation_primal,
    double *__restrict__ variable_primal_projection_multiplier,
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
    bool is_major,
    box_newton_state_t state,
    int max_iter,
    double tol)
{
    const double primal_step = *step_size_d;
    const double inv_primal_step = 1.0 / primal_step;
    const double homogeneous_step = *step_size_homogeneous_d;
    const double inv_homogeneous_step = 1.0 / homogeneous_step;
    const double current_scale = *current_primal_scale;
    const double unprojected_scale =
        current_scale + homogeneous_step * (*homogeneous_dual_gradient);
    const int newton_max_iter =
        state.newton_terms_are_zero ? 0 : max_iter;
    double scale = fmax(0.0, current_scale);
    if (state.newton_terms_are_zero)
        scale = fmax(0.0, unprojected_scale);

    newton_bracket_t bracket = {0.0, INFINITY, INFINITY};
    scale = solve_scale_newton<COOPERATIVE>(
        scale,
        newton_max_iter,
        state.num_elements,
        state.newton_partials_d,
        [&](int i, double trial_scale)
        {
            return evaluate_primal_box_cone_newton_terms(
                current_primal[i] + primal_step * gradient[i],
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
        const double current = current_primal[i];
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
            evaluation_primal[i] = projected;
            reflected_primal[i] = reflected;
            variable_primal_projection_multiplier[i] =
                (projected - shifted) * inv_primal_step;
        }
        current_primal[i] = compute_halpern_update(
            reflected, current, initial_primal[i], reflection_coefficient, weight);
    }

    if constexpr (COOPERATIVE)
    {
        if (newton_max_iter == 0)
            cg::this_grid().sync();
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const double reflected_scale = 2.0 * scale - current_scale;
        if (is_major)
        {
            *evaluation_primal_scale = scale;
            *reflected_primal_scale = reflected_scale;
        }
        *current_primal_scale = compute_halpern_update(
            reflected_scale, current_scale, *initial_primal_scale,
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

    if (shifted > box_upper)
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
