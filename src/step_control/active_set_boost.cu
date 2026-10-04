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

#include "active_set_boost.h"
#include "cuda_macros.h"
#include "cuda_utils.h"
#include "display.h"
#include "power_iteration.h"
#include "termination_checker.h"
#include <math.h>

__global__ void update_binding_constraint_mask_kernel(
    const double *__restrict__ constraint_dual_projection_input,
    const double *dual_cone_scale,
    const double *__restrict__ constraint_lower_bound,
    const double *__restrict__ constraint_upper_bound,
    int *__restrict__ constraint_last_binding_iter,
    bool *__restrict__ binding_constraint_mask,
    active_set_mask_changes_t *mask_changes,
    double relative_tol,
    int current_iter,
    int window_start_iter,
    int num_constraints);
__global__ void update_free_variable_mask_kernel(
    const double *__restrict__ variable_primal_projection_multiplier,
    const double *__restrict__ objective_vector,
    const double *primal_scale_pdhg,
    const double *__restrict__ variable_lower_bound,
    const double *__restrict__ variable_upper_bound,
    int *__restrict__ variable_last_free_iter,
    bool *__restrict__ free_variable_mask,
    active_set_mask_changes_t *mask_changes,
    double relative_tol,
    int current_iter,
    int window_start_iter,
    int num_variables);
static bool has_active_set_revert_budget(
    const pdhg_solver_state_t *state,
    const pdhg_parameters_t *params);
static __device__ __forceinline__ bool is_safely_between_bounds(
    double value,
    double lower,
    double upper,
    double relative_tol);
static __device__ __forceinline__ void update_window_mask_entry(
    bool observed_in_mask,
    int current_iter,
    int window_start_iter,
    int *last_observed_iter,
    bool *mask,
    int *entries_added,
    int *entries_removed);
static void collect_active_set_mask_changes(pdhg_solver_state_t *state);
static bool is_raise_possible(
    double target_step_size,
    double current_step_size,
    double min_raise_ratio);
static void mark_raise_checked(active_set_state_t *active_set);
static sv_estimator_result_t estimate_masked_constraint_norm(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params);
static void save_active_set_anchor(pdhg_solver_state_t *state);
static void restore_active_set_anchor(pdhg_solver_state_t *state);

void initialize_active_set(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    if (!is_active_set_boost_enabled(params))
    {
        return;
    }

    active_set_state_t *active_set = &state->active_set;
    const size_t num_variables = (size_t)state->num_variables;
    const size_t num_constraints = (size_t)state->num_constraints;
    allocate_zeroed_device_buffer(
        (void **)&active_set->constraint_dual_projection_input_d,
        state->con_bytes);
    allocate_zeroed_device_buffer(
        (void **)&active_set->variable_last_free_iter_d,
        num_variables * sizeof(int));
    allocate_zeroed_device_buffer(
        (void **)&active_set->constraint_last_binding_iter_d,
        num_constraints * sizeof(int));
    allocate_zeroed_device_buffer(
        (void **)&active_set->free_variable_mask_d,
        num_variables * sizeof(bool));
    allocate_zeroed_device_buffer(
        (void **)&active_set->binding_constraint_mask_d,
        num_constraints * sizeof(bool));
    allocate_zeroed_device_buffer(
        (void **)&active_set->mask_changes_d,
        sizeof(active_set_mask_changes_t));
    allocate_zeroed_device_buffer(
        (void **)&active_set->anchor.primal_solution_d,
        state->var_bytes);
    allocate_zeroed_device_buffer(
        (void **)&active_set->anchor.dual_solution_d,
        state->con_bytes);
    allocate_zeroed_device_buffer(
        (void **)&active_set->anchor.primal_projection_multiplier_d,
        state->var_bytes);
    allocate_zeroed_device_buffer(
        (void **)&active_set->anchor.hsd_scalars_d,
        HSD_SCALARS_COUNT * sizeof(double));

    CUDA_CHECK(cudaMemset(
        active_set->free_variable_mask_d,
        1,
        num_variables * sizeof(bool)));
    CUDA_CHECK(cudaMemset(
        active_set->binding_constraint_mask_d,
        1,
        num_constraints * sizeof(bool)));
    active_set->free_variable_count = state->num_variables;
    active_set->binding_constraint_count = state->num_constraints;
    active_set->step_size_ceiling = INFINITY;
}

void active_set_free(active_set_state_t *active_set)
{
    sv_estimator_free(active_set->sv_estimator);
    if (active_set->constraint_dual_projection_input_d)
        CUDA_CHECK(cudaFree(active_set->constraint_dual_projection_input_d));
    if (active_set->variable_last_free_iter_d)
        CUDA_CHECK(cudaFree(active_set->variable_last_free_iter_d));
    if (active_set->constraint_last_binding_iter_d)
        CUDA_CHECK(cudaFree(active_set->constraint_last_binding_iter_d));
    if (active_set->free_variable_mask_d)
        CUDA_CHECK(cudaFree(active_set->free_variable_mask_d));
    if (active_set->binding_constraint_mask_d)
        CUDA_CHECK(cudaFree(active_set->binding_constraint_mask_d));
    if (active_set->mask_changes_d)
        CUDA_CHECK(cudaFree(active_set->mask_changes_d));
    if (active_set->anchor.primal_solution_d)
        CUDA_CHECK(cudaFree(active_set->anchor.primal_solution_d));
    if (active_set->anchor.dual_solution_d)
        CUDA_CHECK(cudaFree(active_set->anchor.dual_solution_d));
    if (active_set->anchor.primal_projection_multiplier_d)
        CUDA_CHECK(cudaFree(active_set->anchor.primal_projection_multiplier_d));
    if (active_set->anchor.hsd_scalars_d)
        CUDA_CHECK(cudaFree(active_set->anchor.hsd_scalars_d));

    *active_set = {};
}

void update_active_set_masks(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    if (!has_active_set_revert_budget(state, params))
    {
        return;
    }

    active_set_state_t *active_set = &state->active_set;
    const int current_iter = state->control.total_count;
    const int window_start_iter =
        current_iter - params->active_set_params.window_iterations;

    update_binding_constraint_mask_kernel<<<
        state->num_blocks_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        active_set->constraint_dual_projection_input_d,
        &state->dual_box_cone_state.scale_history_d[BOX_CONE_SCALE_LATEST],
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        active_set->constraint_last_binding_iter_d,
        active_set->binding_constraint_mask_d,
        active_set->mask_changes_d,
        params->active_set_params.relative_tol,
        current_iter,
        window_start_iter,
        state->num_constraints);
    update_free_variable_mask_kernel<<<
        state->num_blocks_primal,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->pdhg_primal_projection_multiplier_d,
        state->objective_vector_d,
        state->hsd_primal_scale_pdhg_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        active_set->variable_last_free_iter_d,
        active_set->free_variable_mask_d,
        active_set->mask_changes_d,
        params->active_set_params.relative_tol,
        current_iter,
        window_start_iter,
        state->num_variables);
}

static bool has_active_set_revert_budget(
    const pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    return is_active_set_boost_enabled(params) &&
           state->active_set.revert_count < params->active_set_params.max_reverts;
}

static __device__ __forceinline__ bool is_safely_between_bounds(
    double value,
    double lower,
    double upper,
    double relative_tol)
{
    const bool below_upper =
        !isfinite(upper) ||
        upper - value >
            relative_tol *
                fmax(1.0, fmax(fabs(value), fabs(upper)));
    const bool above_lower =
        !isfinite(lower) ||
        value - lower >
            relative_tol *
                fmax(1.0, fmax(fabs(value), fabs(lower)));
    return below_upper && above_lower;
}

static __device__ __forceinline__ void update_window_mask_entry(
    bool observed_in_mask,
    int current_iter,
    int window_start_iter,
    int *last_observed_iter,
    bool *mask,
    int *entries_added,
    int *entries_removed)
{
    if (observed_in_mask)
    {
        *last_observed_iter = current_iter;
    }
    const bool retained = *last_observed_iter >= window_start_iter;
    if (retained != *mask)
    {
        *mask = retained;
        atomicAdd(retained ? entries_added : entries_removed, 1);
    }
}

bool should_revert_active_set_boost(
    const pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    const bool is_boosted =
        state->control.step_size > state->control.step_size_base;
    if (!is_boosted)
    {
        return false;
    }
    const termination_reason_t reason = state->control.termination_reason;
    if (reason == TERMINATION_REASON_NUMERICAL_FAILURE)
    {
        return true;
    }
    const bool is_uncertified =
        reason == TERMINATION_REASON_UNSPECIFIED ||
        reason == TERMINATION_REASON_TIME_LIMIT ||
        reason == TERMINATION_REASON_ITERATION_LIMIT;
    if (!is_uncertified)
    {
        return false;
    }
    const double fixed_point_error = *state->control.fixed_point_error_mp;
    const bool metrics_are_finite =
        isfinite(state->metrics_h->relative_primal_residual) &&
        isfinite(state->metrics_h->relative_dual_residual) &&
        isfinite(state->metrics_h->relative_objective_gap) &&
        isfinite(fixed_point_error);
    if (!metrics_are_finite)
    {
        return true;
    }
    const double initial_fixed_point_error =
        state->control.initial_fixed_point_error;
    return initial_fixed_point_error > 0.0 &&
           fixed_point_error >
               (1.0 + params->active_set_params.divergence_margin) *
                   initial_fixed_point_error;
}

void try_raise_active_set_step_size(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    if (!has_active_set_revert_budget(state, params))
    {
        return;
    }

    collect_active_set_mask_changes(state);

    active_set_state_t *active_set = &state->active_set;
    const bool either_mask_is_empty =
        active_set->free_variable_count == 0 ||
        active_set->binding_constraint_count == 0;
    if (either_mask_is_empty)
    {
        return;
    }
    const bool masks_are_full_and_unchanged =
        active_set->free_variable_count == state->num_variables &&
        active_set->binding_constraint_count == state->num_constraints &&
        active_set->mask_changes_since_estimate == 0;
    if (masks_are_full_and_unchanged)
    {
        return;
    }
    const bool masks_changed_enough =
        active_set->mask_changes_since_estimate > 0 &&
        active_set->mask_changes_since_estimate >=
            params->active_set_params.reestimate_change_ratio *
                (active_set->free_variable_count +
                 active_set->binding_constraint_count);
    if (active_set->raise_checked && !masks_changed_enough)
    {
        return;
    }

    const double step_size_base = state->control.step_size_base;
    const double current_step_size = state->control.step_size;
    const double min_raise_ratio = params->active_set_params.min_raise_ratio;
    const double largest_allowed_step_size =
        fmax(step_size_base, active_set->step_size_ceiling);
    if (!is_raise_possible(
            largest_allowed_step_size,
            current_step_size,
            min_raise_ratio))
    {
        mark_raise_checked(active_set);
        return;
    }

    const sv_estimator_result_t estimate =
        estimate_masked_constraint_norm(state, params);
    if (estimate.status == SV_ESTIMATOR_MAX_ITER)
    {
        return;
    }
    mark_raise_checked(active_set);
    if (estimate.status != SV_ESTIMATOR_CONVERGED)
    {
        return;
    }

    const double target_step_size =
        fmin(fmax(step_size_base,
                  params->active_set_params.safety_factor /
                      estimate.max_singular_value),
             largest_allowed_step_size);
    if (!is_raise_possible(target_step_size, current_step_size, min_raise_ratio))
    {
        return;
    }
    state->control.step_size = target_step_size;
    active_set->raise_count++;
    save_active_set_anchor(state);
    display_active_set_boost_raise(
        params, current_step_size, target_step_size,
        estimate.max_singular_value);
}

static void collect_active_set_mask_changes(pdhg_solver_state_t *state)
{
    active_set_state_t *active_set = &state->active_set;
    active_set_mask_changes_t mask_changes = {};
    CUDA_CHECK(cudaMemcpyAsync(
        &mask_changes,
        active_set->mask_changes_d,
        sizeof(mask_changes),
        cudaMemcpyDeviceToHost,
        state->stream_A));
    CUDA_CHECK(cudaMemsetAsync(
        active_set->mask_changes_d,
        0,
        sizeof(mask_changes),
        state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));

    active_set->free_variable_count +=
        mask_changes.variables_added - mask_changes.variables_removed;
    active_set->binding_constraint_count +=
        mask_changes.constraints_added - mask_changes.constraints_removed;
    active_set->mask_changes_since_estimate +=
        mask_changes.variables_added + mask_changes.variables_removed +
        mask_changes.constraints_added + mask_changes.constraints_removed;
}

static bool is_raise_possible(
    double target_step_size,
    double current_step_size,
    double min_raise_ratio)
{
    return target_step_size > current_step_size &&
           target_step_size >= min_raise_ratio * current_step_size;
}

static void mark_raise_checked(active_set_state_t *active_set)
{
    active_set->raise_checked = true;
    active_set->mask_changes_since_estimate = 0;
}

static sv_estimator_result_t estimate_masked_constraint_norm(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    active_set_state_t *active_set = &state->active_set;
    if (!active_set->sv_estimator)
    {
        active_set->sv_estimator = sv_estimator_create(
            state->sparse_handle_A,
            state->blas_handle_A,
            state->stream_A,
            state->curand_gen,
            state->constraint_matrix,
            state->constraint_matrix_t,
            state->num_constraints,
            state->num_variables);
    }

    sv_estimator_opts_t opts = {};
    opts.max_iter = params->power_iter_params.max_iter;
    opts.tol = params->power_iter_params.tol;
    opts.row_mask_d = active_set->binding_constraint_mask_d;
    opts.column_mask_d = active_set->free_variable_mask_d;
    opts.early_exit_singular_value_threshold =
        params->active_set_params.safety_factor /
        (params->active_set_params.min_raise_ratio * state->control.step_size);

    const sv_estimator_result_t estimate =
        sv_estimator_run(active_set->sv_estimator, &opts);
    active_set->power_iteration_count += estimate.iterations;
    return estimate;
}

static void save_active_set_anchor(pdhg_solver_state_t *state)
{
    active_set_state_t *active_set = &state->active_set;
    CUDA_CHECK(cudaMemcpyAsync(
        active_set->anchor.primal_solution_d,
        state->initial_primal_solution_d,
        state->var_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        active_set->anchor.dual_solution_d,
        state->initial_dual_solution_d,
        state->con_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        active_set->anchor.primal_projection_multiplier_d,
        state->pdhg_primal_projection_multiplier_d,
        state->var_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        active_set->anchor.hsd_scalars_d,
        state->hsd_scalars_d,
        HSD_SCALARS_COUNT * sizeof(double),
        cudaMemcpyDeviceToDevice,
        state->stream_A));
    active_set->anchor.primal_weight = state->control.primal_weight;
    active_set->anchor.pid = state->control.pid;
}

void revert_active_set_step_size(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    active_set_state_t *active_set = &state->active_set;
    const double failed_step_size = state->control.step_size;
    active_set->step_size_ceiling =
        params->active_set_params.divergence_ceiling_ratio * failed_step_size;
    state->control.step_size = state->control.step_size_base;
    restore_active_set_anchor(state);
    active_set->raise_checked = false;
    active_set->revert_count++;
}

static void restore_active_set_anchor(pdhg_solver_state_t *state)
{
    active_set_state_t *active_set = &state->active_set;
    double *primal_destinations[] = {
        state->initial_primal_solution_d,
        state->current_primal_solution_d,
        state->pdhg_primal_solution_d};
    double *dual_destinations[] = {
        state->initial_dual_solution_d,
        state->current_dual_solution_d,
        state->pdhg_dual_solution_d};
    for (int i = 0; i < 3; ++i)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            primal_destinations[i],
            active_set->anchor.primal_solution_d,
            state->var_bytes,
            cudaMemcpyDeviceToDevice,
            state->stream_A));
        CUDA_CHECK(cudaMemcpyAsync(
            dual_destinations[i],
            active_set->anchor.dual_solution_d,
            state->con_bytes,
            cudaMemcpyDeviceToDevice,
            state->stream_A));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        state->pdhg_primal_projection_multiplier_d,
        active_set->anchor.primal_projection_multiplier_d,
        state->var_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        state->hsd_scalars_d,
        active_set->anchor.hsd_scalars_d,
        HSD_SCALARS_COUNT * sizeof(double),
        cudaMemcpyDeviceToDevice,
        state->stream_A));
    state->control.primal_weight = active_set->anchor.primal_weight;
    state->control.pid = active_set->anchor.pid;
}

__global__ void update_binding_constraint_mask_kernel(
    const double *__restrict__ constraint_dual_projection_input,
    const double *dual_cone_scale,
    const double *__restrict__ constraint_lower_bound,
    const double *__restrict__ constraint_upper_bound,
    int *__restrict__ constraint_last_binding_iter,
    bool *__restrict__ binding_constraint_mask,
    active_set_mask_changes_t *mask_changes,
    double relative_tol,
    int current_iter,
    int window_start_iter,
    int num_constraints)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_constraints)
    {
        return;
    }

    const double lower = constraint_lower_bound[i];
    const double upper = constraint_upper_bound[i];
    const bool is_binding =
        lower == upper ||
        !(*dual_cone_scale > 0.0) ||
        !is_safely_between_bounds(
            -(constraint_dual_projection_input[i] / (*dual_cone_scale)),
            lower,
            upper,
            relative_tol);
    update_window_mask_entry(
        is_binding,
        current_iter,
        window_start_iter,
        &constraint_last_binding_iter[i],
        &binding_constraint_mask[i],
        &mask_changes->constraints_added,
        &mask_changes->constraints_removed);
}

__global__ void update_free_variable_mask_kernel(
    const double *__restrict__ variable_primal_projection_multiplier,
    const double *__restrict__ objective_vector,
    const double *primal_scale_pdhg,
    const double *__restrict__ variable_lower_bound,
    const double *__restrict__ variable_upper_bound,
    int *__restrict__ variable_last_free_iter,
    bool *__restrict__ free_variable_mask,
    active_set_mask_changes_t *mask_changes,
    double relative_tol,
    int current_iter,
    int window_start_iter,
    int num_variables)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_variables)
    {
        return;
    }

    const double lower = variable_lower_bound[j];
    const double upper = variable_upper_bound[j];
    const double multiplier_tolerance =
        relative_tol *
        fmax(1.0, fabs(objective_vector[j])) *
        (*primal_scale_pdhg);
    const double multiplier = variable_primal_projection_multiplier[j];
    const bool multiplier_is_finite = isfinite(multiplier);
    const bool fixed_at_zero = lower == 0.0 && upper == 0.0;
    const bool clamped_at_lower =
        lower == 0.0 &&
        multiplier_is_finite &&
        multiplier > multiplier_tolerance;
    const bool clamped_at_upper =
        upper == 0.0 &&
        multiplier_is_finite &&
        multiplier < -multiplier_tolerance;
    const bool is_free =
        !(fixed_at_zero || clamped_at_lower || clamped_at_upper);
    update_window_mask_entry(
        is_free,
        current_iter,
        window_start_iter,
        &variable_last_free_iter[j],
        &free_variable_mask[j],
        &mask_changes->variables_added,
        &mask_changes->variables_removed);
}
