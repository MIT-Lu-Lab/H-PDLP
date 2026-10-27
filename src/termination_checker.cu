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

#include "termination_checker.h"
#include "cuda_macros.h"
#include "cublas_checks.h"
#include "cuda_utils.h"
#include "spmv_backend.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <time.h>

__global__ void compute_residual_kernel(
    double *__restrict__ primal_res,
    const double *__restrict__ primal_prod,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    double *__restrict__ dual_res,
    const double *__restrict__ dual_prod,
    const double *__restrict__ dual_sol,
    const double *__restrict__ obj_vec,
    const double *__restrict__ cons_rescaling,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    const double *__restrict__ primal_sol,
    const double *primal_pdhg_scale,
    double *__restrict__ primal_infeasibility_reduced_cost,
    double *__restrict__ primal_infeasibility_stationarity_res,
    double *__restrict__ constraint_recession_res,
    double *__restrict__ variable_recession_res,
    double *__restrict__ original_dual_ray,
    double *__restrict__ original_reduced_cost_ray,
    double *__restrict__ original_primal_ray,
    double *__restrict__ reduced_cost_out,
    const double *__restrict__ projection_multiplier,
    double *__restrict__ bound_multiplier_out,
    double *__restrict__ bound_multiplier_dual_res,
    int n_cons,
    int n_vars);
__global__ void compute_projected_difference_kernel(
    const double *__restrict__ primal_from,
    const double *__restrict__ primal_to,
    double *__restrict__ primal_delta,
    const double *__restrict__ dual_from,
    const double *__restrict__ dual_to,
    double *__restrict__ dual_delta,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    int n_vars,
    int n_cons);
__global__ void compute_termination_metrics_kernel(
    pdhg_metrics_t *metrics_d,
    const termination_reductions_t *__restrict__ reductions,
    const double *__restrict__ primal_current_scale,
    const double *__restrict__ primal_pdhg_scale,
    const double *__restrict__ dual_pdhg_scale,
    double constraint_bound_rescaling,
    double objective_vector_rescaling,
    double original_bound_norm,
    double original_objective_norm,
    double objective_constant,
    norm_type_t optimality_norm);
__device__ __forceinline__ void fill_certificate_metrics(
    certificate_metrics_t *metrics,
    double margin,
    double residual,
    double scaled_margin,
    double absolute_terms,
    double scaled_residual,
    double relative_bound_residual);
__device__ __forceinline__ double compute_relative_margin(double margin, double absolute_terms);
__host__ __device__ __forceinline__ double compute_relative_error(double residual, double activity);
__device__ __forceinline__ void fill_difference_certificate_metrics(
    certificate_metrics_t *primal_infeasibility,
    certificate_metrics_t *dual_infeasibility,
    const candidate_reductions_t *candidate,
    double bound_objective_scale,
    double constraint_bound_rescaling,
    double objective_vector_rescaling);
static void compute_matrix_products(
    pdhg_solver_state_t *state,
    cusparseDnVecDescr_t primal_vector,
    cusparseDnVecDescr_t dual_vector);
static void fork_transpose_stream(pdhg_solver_state_t *state);
static void join_transpose_stream(pdhg_solver_state_t *state);
static void compute_residual_vectors(pdhg_solver_state_t *state);
static void compute_residual_norms(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params);
static void compute_objective_and_support_terms(
    pdhg_solver_state_t *state,
    const double *primal_direction_d,
    const double *dual_direction_d,
    const double *variable_direction_d,
    double variable_direction_sign,
    double *objective_dot_d,
    double *constraint_support_d,
    double *variable_support_d);
static void compute_variable_support(
    pdhg_solver_state_t *state,
    const double *multiplier_d,
    double *variable_support_d);
static void compute_infeasibility_reductions(
    pdhg_solver_state_t *state,
    certificate_source_t source);
static void compute_difference_reductions(
    pdhg_solver_state_t *state,
    certificate_source_t source);
static void compute_projected_difference(
    pdhg_solver_state_t *state,
    certificate_source_t source);
static void compute_difference_residual_norms(
    pdhg_solver_state_t *state,
    candidate_reductions_t *candidate_reductions_d);
static bool passes_objective_margin(const certificate_metrics_t *metrics, double objective_margin);
static certificate_metrics_t *select_certificate_metrics(
    pdhg_metrics_t *metrics,
    certificate_source_t source,
    bool primal_infeasibility);
static void compute_ray_activity(
    pdhg_solver_state_t *state,
    const double *direction_d,
    bool primal_direction,
    double *activity_d);
static void finalize_relative_residual(certificate_metrics_t *metrics, bool activity_computed);
static bool certify_optimality(
    pdhg_solver_state_t *state,
    const termination_criteria_t *criteria);
static void adopt_bound_multiplier_certificate(pdhg_solver_state_t *state);
static bool certify_infeasibility(
    pdhg_solver_state_t *state,
    const certificate_metrics_t *metrics,
    double eps,
    double objective_margin,
    termination_reason_t reason,
    certificate_source_t source);

void compute_termination_metrics(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    compute_device_termination_metrics(state, params);
    copy_termination_metrics_to_host(state);
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    compute_certificate_residuals(state, params->termination_criteria.farkas_objective_margin);
}

void compute_device_termination_metrics(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    termination_reductions_t *reductions_d = state->termination_reductions_d;

    compute_matrix_products(state,
                            state->spmv_primal_pdhg_solution,
                            state->spmv_dual_pdhg_solution);
    compute_residual_vectors(state);
    compute_residual_norms(state, params);
    compute_objective_and_support_terms(
        state,
        state->primal_pdhg_solution_d,
        state->dual_pdhg_solution_d,
        state->pdhg_reduced_cost_d,
        1.0,
        &reductions_d->iterate.objective_dot,
        &reductions_d->iterate.constraint_support,
        &reductions_d->variable_support);
    compute_variable_support(state,
                             state->bound_multiplier_d,
                             &reductions_d->bound_multiplier_variable_support);
    compute_variable_support(state,
                             state->primal_infeasibility_reduced_cost_d,
                             &reductions_d->iterate.variable_support);
    compute_infeasibility_reductions(state, CERTIFICATE_SOURCE_ITERATE);
    compute_difference_reductions(state, CERTIFICATE_SOURCE_STEP);
    compute_difference_reductions(state, CERTIFICATE_SOURCE_RESTART);

    compute_termination_metrics_kernel<<<1, 1, 0, state->stream_A>>>(
        state->metrics_d,
        reductions_d,
        state->hsd_primal_current_scale_d,
        state->hsd_primal_pdhg_scale_d,
        state->hsd_dual_pdhg_scale_d,
        state->constraint_bound_rescaling,
        state->objective_vector_rescaling,
        state->original_bound_norm,
        state->original_objective_norm,
        state->objective_constant,
        params->optimality_norm);
    CUDA_CHECK(cudaGetLastError());
}

static void compute_matrix_products(
    pdhg_solver_state_t *state,
    cusparseDnVecDescr_t primal_vector,
    cusparseDnVecDescr_t dual_vector)
{
    fork_transpose_stream(state);
    execute_spmv(
        state->sparse_handle_A,
        primal_vector,
        state->spmv_primal_product,
        &state->constraint_matrix_spmv_plan);

    execute_spmv(
        state->sparse_handle_At,
        dual_vector,
        state->spmv_dual_product,
        &state->constraint_matrix_transpose_spmv_plan);
    join_transpose_stream(state);
}

static void fork_transpose_stream(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));
}

static void join_transpose_stream(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));
}

static void compute_residual_vectors(pdhg_solver_state_t *state)
{
    double *stationarity_residual_entries_d = state->variable_reduction_entries_d;
    double *constraint_recession_residual_entries_d = state->constraint_reduction_entries_d;

    compute_residual_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->primal_residual_d,
        state->primal_product_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        state->dual_residual_d,
        state->dual_product_d,
        state->dual_pdhg_solution_d,
        state->objective_vector_d,
        state->constraint_rescaling_d,
        state->variable_rescaling_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        state->primal_pdhg_solution_d,
        state->hsd_primal_pdhg_scale_d,
        state->primal_infeasibility_reduced_cost_d,
        stationarity_residual_entries_d,
        constraint_recession_residual_entries_d,
        state->variable_recession_residual_d,
        state->dual_iterate_ray_d,
        state->reduced_cost_iterate_ray_d,
        state->primal_iterate_ray_d,
        state->pdhg_reduced_cost_d,
        state->primal_pdhg_projection_multiplier_d,
        state->bound_multiplier_d,
        state->bound_multiplier_dual_residual_d,
        state->num_constraints,
        state->num_variables);
    CUDA_CHECK(cudaGetLastError());
}

static void compute_residual_norms(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    termination_reductions_t *reductions_d = state->termination_reductions_d;
    const double *stationarity_residual_entries_d = state->variable_reduction_entries_d;
    const double *constraint_recession_residual_entries_d = state->constraint_reduction_entries_d;

    reduce_device_abs_max(state, stationarity_residual_entries_d, state->num_variables,
                          &reductions_d->iterate.stationarity_residual);
    reduce_device_abs_max(state, constraint_recession_residual_entries_d, state->num_constraints,
                          &reductions_d->iterate.constraint_recession_residual);

    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        reduce_device_abs_max(state, state->primal_residual_d, state->num_constraints + state->num_variables,
                              &reductions_d->primal_residual_norm);
        reduce_device_abs_max(state, state->dual_residual_d, state->num_variables,
                              &reductions_d->dual_residual_norm);
        reduce_device_abs_max(state, state->bound_multiplier_dual_residual_d, state->num_variables,
                              &reductions_d->bound_multiplier_dual_residual_norm);
        reduce_device_abs_max(state, state->primal_pdhg_solution_d, state->num_variables,
                              &state->metrics_d->primal_solution_norm);
        reduce_device_abs_max(state, state->dual_pdhg_solution_d, state->num_constraints,
                              &state->metrics_d->dual_solution_norm);
    }
    else
    {
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_constraints + state->num_variables,
            state->primal_residual_d, 1,
            &reductions_d->primal_residual_norm));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_variables,
            state->dual_residual_d, 1,
            &reductions_d->dual_residual_norm));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_variables,
            state->bound_multiplier_dual_residual_d, 1,
            &reductions_d->bound_multiplier_dual_residual_norm));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_variables,
            state->primal_pdhg_solution_d, 1,
            &state->metrics_d->primal_solution_norm));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_constraints,
            state->dual_pdhg_solution_d, 1,
            &state->metrics_d->dual_solution_norm));
    }

    reduce_device_abs_max(state, state->variable_recession_residual_d, state->num_variables,
                          &reductions_d->variable_recession_residual_norm);
    reduce_device_abs_max(state, state->dual_iterate_ray_d, state->num_constraints,
                          &reductions_d->original_dual_ray_norm);
    reduce_device_abs_max(state, state->reduced_cost_iterate_ray_d, state->num_variables,
                          &reductions_d->original_reduced_cost_ray_norm);
    reduce_device_abs_max(state, state->primal_iterate_ray_d, state->num_variables,
                          &reductions_d->original_primal_ray_norm);
}

static void compute_objective_and_support_terms(
    pdhg_solver_state_t *state,
    const double *primal_direction_d,
    const double *dual_direction_d,
    const double *variable_direction_d,
    double variable_direction_sign,
    double *objective_dot_d,
    double *constraint_support_d,
    double *variable_support_d)
{
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle_A,
        state->num_variables,
        state->objective_vector_d, 1,
        primal_direction_d, 1,
        objective_dot_d));

    reduce_device_box_support(
        state,
        dual_direction_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        1.0,
        state->num_constraints,
        constraint_support_d);
    reduce_device_box_support(
        state,
        variable_direction_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        variable_direction_sign,
        state->num_variables,
        variable_support_d);
}

static void compute_variable_support(
    pdhg_solver_state_t *state,
    const double *multiplier_d,
    double *variable_support_d)
{
    reduce_device_box_support(
        state,
        multiplier_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        1.0,
        state->num_variables,
        variable_support_d);
}

static void compute_infeasibility_reductions(
    pdhg_solver_state_t *state,
    certificate_source_t source)
{
    candidate_reductions_t *candidate_reductions_d =
        select_candidate_reductions(state->termination_reductions_d, source);
    const candidate_vectors_t candidate = select_candidate_vectors(state, source);
    const bool iterate = source == CERTIFICATE_SOURCE_ITERATE;
    const double *variable_direction_d =
        iterate ? state->primal_infeasibility_reduced_cost_d : state->dual_product_d;
    const double variable_direction_sign = iterate ? 1.0 : -1.0;

    reduce_device_absolute_product(
        state,
        state->objective_vector_d,
        candidate.primal_d,
        state->num_variables,
        &candidate_reductions_d->sum_absolute_objective_terms);
    reduce_device_box_support_magnitude(
        state,
        candidate.dual_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        1.0,
        state->num_constraints,
        &candidate_reductions_d->sum_absolute_constraint_support_terms);
    reduce_device_box_support_magnitude(
        state,
        variable_direction_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        variable_direction_sign,
        state->num_variables,
        &candidate_reductions_d->sum_absolute_variable_support_terms);

    reduce_device_recession_violation(
        state,
        state->primal_product_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        NULL,
        state->num_constraints,
        &candidate_reductions_d->scaled_constraint_recession_residual);
    if (iterate)
    {
        termination_reductions_t *reductions_d = state->termination_reductions_d;
        reduce_device_recession_violation(
            state,
            candidate.primal_d,
            state->variable_lower_bound_d,
            state->variable_upper_bound_d,
            NULL,
            state->num_variables,
            &reductions_d->scaled_variable_recession_residual);
        reduce_device_abs_max(state, candidate.primal_d, state->num_variables,
                              &reductions_d->scaled_primal_ray_norm);
    }
    reduce_device_reduced_cost_violation(
        state,
        state->dual_product_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        NULL,
        state->num_variables,
        &candidate_reductions_d->scaled_stationarity_residual);
}

static void compute_difference_reductions(
    pdhg_solver_state_t *state,
    certificate_source_t source)
{
    candidate_reductions_t *candidate_reductions_d =
        select_candidate_reductions(state->termination_reductions_d, source);
    const candidate_vectors_t candidate = select_candidate_vectors(state, source);

    compute_projected_difference(state, source);
    compute_matrix_products(state, candidate.spmv_primal, candidate.spmv_dual);
    compute_objective_and_support_terms(
        state,
        candidate.primal_d,
        candidate.dual_d,
        state->dual_product_d,
        -1.0,
        &candidate_reductions_d->objective_dot,
        &candidate_reductions_d->constraint_support,
        &candidate_reductions_d->variable_support);
    compute_difference_residual_norms(state, candidate_reductions_d);
    compute_infeasibility_reductions(state, source);
}

static void compute_projected_difference(
    pdhg_solver_state_t *state,
    certificate_source_t source)
{
    const bool step = source == CERTIFICATE_SOURCE_STEP;
    const candidate_vectors_t candidate = select_candidate_vectors(state, source);

    compute_projected_difference_kernel<<<
        state->num_blocks_primal_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        step ? state->primal_pdhg_solution_d : state->primal_initial_solution_d,
        step ? state->primal_reflected_solution_d : state->primal_pdhg_solution_d,
        candidate.primal_d,
        step ? state->dual_pdhg_solution_d : state->dual_initial_solution_d,
        step ? state->dual_reflected_solution_d : state->dual_pdhg_solution_d,
        candidate.dual_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        state->num_variables,
        state->num_constraints);
    CUDA_CHECK(cudaGetLastError());
}

static void compute_difference_residual_norms(
    pdhg_solver_state_t *state,
    candidate_reductions_t *candidate_reductions_d)
{
    reduce_device_recession_violation(
        state,
        state->primal_product_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        state->constraint_rescaling_d,
        state->num_constraints,
        &candidate_reductions_d->constraint_recession_residual);
    reduce_device_reduced_cost_violation(
        state,
        state->dual_product_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        state->variable_rescaling_d,
        state->num_variables,
        &candidate_reductions_d->stationarity_residual);
}

void copy_termination_metrics_to_host(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpyAsync(state->metrics_h, state->metrics_d,
                               sizeof(pdhg_metrics_t),
                               cudaMemcpyDeviceToHost,
                               state->stream_A));
}

void compute_certificate_residuals(pdhg_solver_state_t *state, double objective_margin)
{
    pdhg_metrics_t *metrics_h = state->metrics_h;
    pdhg_metrics_t *metrics_d = state->metrics_d;
    bool primal_passes[CERTIFICATE_SOURCE_COUNT];
    bool dual_passes[CERTIFICATE_SOURCE_COUNT];
    bool any_passes = false;

    for (int s = 0; s < CERTIFICATE_SOURCE_COUNT; ++s)
    {
        const certificate_source_t source = (certificate_source_t)s;
        const candidate_vectors_t candidate = select_candidate_vectors(state, source);
        primal_passes[s] = passes_objective_margin(
            select_certificate_metrics(metrics_h, source, true), objective_margin);
        dual_passes[s] = passes_objective_margin(
            select_certificate_metrics(metrics_h, source, false), objective_margin);
        if (primal_passes[s])
        {
            certificate_metrics_t *primal_metrics_d = select_certificate_metrics(metrics_d, source, true);
            compute_ray_activity(state, candidate.dual_d, false, &primal_metrics_d->scaled_activity);
        }
        if (dual_passes[s])
        {
            certificate_metrics_t *dual_metrics_d = select_certificate_metrics(metrics_d, source, false);
            compute_ray_activity(state, candidate.primal_d, true, &dual_metrics_d->scaled_activity);
        }
        any_passes = any_passes || primal_passes[s] || dual_passes[s];
    }
    if (any_passes)
    {
        CUDA_CHECK(cudaMemcpyAsync(metrics_h, metrics_d,
                                   sizeof(pdhg_metrics_t),
                                   cudaMemcpyDeviceToHost,
                                   state->stream_A));
        CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    }

    for (int s = 0; s < CERTIFICATE_SOURCE_COUNT; ++s)
    {
        const certificate_source_t source = (certificate_source_t)s;
        finalize_relative_residual(select_certificate_metrics(metrics_h, source, true),
                                   primal_passes[s]);
        finalize_relative_residual(select_certificate_metrics(metrics_h, source, false),
                                   dual_passes[s]);
    }
}

static bool passes_objective_margin(const certificate_metrics_t *metrics, double objective_margin)
{
    return metrics->relative_margin > 0.0 && metrics->relative_margin >= objective_margin;
}

static certificate_metrics_t *select_certificate_metrics(
    pdhg_metrics_t *metrics,
    certificate_source_t source,
    bool primal_infeasibility)
{
    switch (source)
    {
    case CERTIFICATE_SOURCE_ITERATE:
        return primal_infeasibility ? &metrics->primal_infeasibility_iterate
                                    : &metrics->dual_infeasibility_iterate;
    case CERTIFICATE_SOURCE_STEP:
        return primal_infeasibility ? &metrics->primal_infeasibility_step
                                    : &metrics->dual_infeasibility_step;
    case CERTIFICATE_SOURCE_RESTART:
        return primal_infeasibility ? &metrics->primal_infeasibility_restart
                                    : &metrics->dual_infeasibility_restart;
    case CERTIFICATE_SOURCE_COUNT:
        break;
    }
    raise_error("invalid certificate source %d", (int)source);
    return NULL;
}

static void compute_ray_activity(
    pdhg_solver_state_t *state,
    const double *direction_d,
    bool primal_direction,
    double *activity_d)
{
    if (primal_direction)
    {
        compute_device_abs(state, direction_d, state->num_variables,
                           state->variable_reduction_entries_d);
        execute_spmv(
            state->sparse_handle_A,
            state->spmv_variable_reduction_entries,
            state->spmv_constraint_reduction_entries,
            &state->absolute_constraint_matrix_spmv_plan);
        reduce_device_abs_max(state, state->constraint_reduction_entries_d,
                              state->num_constraints, activity_d);
    }
    else
    {
        compute_device_abs(state, direction_d, state->num_constraints,
                           state->constraint_reduction_entries_d);
        execute_spmv(
            state->sparse_handle_A,
            state->spmv_constraint_reduction_entries,
            state->spmv_variable_reduction_entries,
            &state->absolute_constraint_matrix_transpose_spmv_plan);
        reduce_device_abs_max(state, state->variable_reduction_entries_d,
                              state->num_variables, activity_d);
    }
}

static void finalize_relative_residual(certificate_metrics_t *metrics, bool activity_computed)
{
    metrics->relative_residual =
        activity_computed
            ? fmax(compute_relative_error(metrics->scaled_residual, metrics->scaled_activity),
                   metrics->relative_bound_residual)
            : INFINITY;
}

void check_termination_criteria(
    pdhg_solver_state_t *state,
    const termination_criteria_t *criteria)
{
    state->control.cumulative_time_sec = (double)(clock() - state->control.start_time) / CLOCKS_PER_SEC;

    if (!isfinite(state->metrics_h->primal_solution_norm) ||
        !isfinite(state->metrics_h->dual_solution_norm) ||
        !isfinite(state->metrics_h->primal_pdhg_scale))
    {
        state->control.termination_reason = TERMINATION_REASON_NUMERICAL_FAILURE;
        return;
    }

    double ratio = state->metrics_h->normalized_homogeneous_ratio;
    double normalized_primal_scale = state->metrics_h->primal_pdhg_scale
                                   / fmax(state->metrics_h->primal_solution_norm,
                                          state->metrics_h->dual_solution_norm);
    if (ratio < 0.5)
    {
        if (certify_optimality(state, criteria))
        {
            return;
        }
    }
    if (ratio > 0.9 || normalized_primal_scale < 1e-10)
    {
        if (certify_infeasibility(state,
                                  &state->metrics_h->primal_infeasibility_iterate,
                                  criteria->eps_farkas_iterate,
                                  criteria->farkas_objective_margin,
                                  TERMINATION_REASON_PRIMAL_INFEASIBLE,
                                  CERTIFICATE_SOURCE_ITERATE))
        {
            return;
        }
        if (certify_infeasibility(state,
                                  &state->metrics_h->dual_infeasibility_iterate,
                                  criteria->eps_farkas_iterate,
                                  criteria->farkas_objective_margin,
                                  TERMINATION_REASON_DUAL_INFEASIBLE,
                                  CERTIFICATE_SOURCE_ITERATE))
        {
            return;
        }
    }

    if (certify_infeasibility(state,
                              &state->metrics_h->primal_infeasibility_restart,
                              criteria->eps_farkas_restart,
                              criteria->farkas_objective_margin,
                              TERMINATION_REASON_PRIMAL_INFEASIBLE,
                              CERTIFICATE_SOURCE_RESTART))
    {
        return;
    }
    if (certify_infeasibility(state,
                              &state->metrics_h->dual_infeasibility_restart,
                              criteria->eps_farkas_restart,
                              criteria->farkas_objective_margin,
                              TERMINATION_REASON_DUAL_INFEASIBLE,
                              CERTIFICATE_SOURCE_RESTART))
    {
        return;
    }
    if (certify_infeasibility(state,
                              &state->metrics_h->primal_infeasibility_step,
                              criteria->eps_farkas_step,
                              criteria->farkas_objective_margin,
                              TERMINATION_REASON_PRIMAL_INFEASIBLE,
                              CERTIFICATE_SOURCE_STEP))
    {
        return;
    }
    if (certify_infeasibility(state,
                              &state->metrics_h->dual_infeasibility_step,
                              criteria->eps_farkas_step,
                              criteria->farkas_objective_margin,
                              TERMINATION_REASON_DUAL_INFEASIBLE,
                              CERTIFICATE_SOURCE_STEP))
    {
        return;
    }

    if (state->control.total_count >= criteria->iteration_limit)
    {
        state->control.termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        return;
    }
    if (state->control.cumulative_time_sec >= criteria->time_sec_limit)
    {
        state->control.termination_reason = TERMINATION_REASON_TIME_LIMIT;
        return;
    }
}

static bool certify_optimality(
    pdhg_solver_state_t *state,
    const termination_criteria_t *criteria)
{
    if (!(state->metrics_h->primal_pdhg_scale > 1e-50 &&
          state->metrics_h->relative_primal_residual < criteria->eps_feasible_relative))
    {
        return false;
    }
    if (state->metrics_h->relative_dual_residual   < criteria->eps_feasible_relative &&
        state->metrics_h->relative_objective_gap   < criteria->eps_optimal_relative)
    {
        state->control.termination_reason = TERMINATION_REASON_OPTIMAL;
        return true;
    }
    if (state->metrics_h->bound_multiplier_relative_dual_residual < criteria->eps_feasible_relative &&
        state->metrics_h->bound_multiplier_relative_objective_gap < criteria->eps_optimal_relative)
    {
        adopt_bound_multiplier_certificate(state);
        state->control.termination_reason = TERMINATION_REASON_OPTIMAL;
        return true;
    }
    return false;
}

static void adopt_bound_multiplier_certificate(pdhg_solver_state_t *state)
{
    double *reduced_cost_d = state->pdhg_reduced_cost_d;
    state->pdhg_reduced_cost_d = state->bound_multiplier_d;
    state->bound_multiplier_d = reduced_cost_d;
    state->metrics_h->absolute_dual_residual = state->metrics_h->bound_multiplier_absolute_dual_residual;
    state->metrics_h->relative_dual_residual = state->metrics_h->bound_multiplier_relative_dual_residual;
    state->metrics_h->dual_objective = state->metrics_h->bound_multiplier_dual_objective;
    state->metrics_h->objective_gap = state->metrics_h->bound_multiplier_objective_gap;
    state->metrics_h->relative_objective_gap = state->metrics_h->bound_multiplier_relative_objective_gap;
}

static bool certify_infeasibility(
    pdhg_solver_state_t *state,
    const certificate_metrics_t *metrics,
    double eps,
    double objective_margin,
    termination_reason_t reason,
    certificate_source_t source)
{
    if (isfinite(metrics->margin) && metrics->margin > 0.0 &&
        passes_objective_margin(metrics, objective_margin) &&
        metrics->relative_residual <= eps)
    {
        state->control.certificate_margin = metrics->margin;
        state->control.certificate_residual = metrics->residual;
        state->control.certificate_source = source;
        state->control.termination_reason = reason;
        return true;
    }
    return false;
}

void compute_certificate_dual_product(pdhg_solver_state_t *state)
{
    const candidate_vectors_t candidate =
        select_candidate_vectors(state, state->control.certificate_source);
    fork_transpose_stream(state);
    execute_spmv(
        state->sparse_handle_At,
        candidate.spmv_dual,
        state->spmv_dual_product,
        &state->constraint_matrix_transpose_spmv_plan);
    join_transpose_stream(state);
}

candidate_vectors_t select_candidate_vectors(
    pdhg_solver_state_t *state,
    certificate_source_t source)
{
    candidate_vectors_t candidate = {};
    switch (source)
    {
    case CERTIFICATE_SOURCE_ITERATE:
        candidate.primal_d = state->primal_pdhg_solution_d;
        candidate.dual_d = state->dual_pdhg_solution_d;
        candidate.spmv_primal = state->spmv_primal_pdhg_solution;
        candidate.spmv_dual = state->spmv_dual_pdhg_solution;
        return candidate;
    case CERTIFICATE_SOURCE_STEP:
        candidate.primal_d = state->primal_step_delta_d;
        candidate.dual_d = state->dual_step_delta_d;
        candidate.spmv_primal = state->spmv_primal_step_delta;
        candidate.spmv_dual = state->spmv_dual_step_delta;
        return candidate;
    case CERTIFICATE_SOURCE_RESTART:
        candidate.primal_d = state->primal_restart_delta_d;
        candidate.dual_d = state->dual_restart_delta_d;
        candidate.spmv_primal = state->spmv_primal_restart_delta;
        candidate.spmv_dual = state->spmv_dual_restart_delta;
        return candidate;
    case CERTIFICATE_SOURCE_COUNT:
        break;
    }
    raise_error("invalid certificate source %d", (int)source);
    return candidate;
}

candidate_reductions_t *select_candidate_reductions(
    termination_reductions_t *reductions,
    certificate_source_t source)
{
    switch (source)
    {
    case CERTIFICATE_SOURCE_ITERATE:
        return &reductions->iterate;
    case CERTIFICATE_SOURCE_STEP:
        return &reductions->step;
    case CERTIFICATE_SOURCE_RESTART:
        return &reductions->restart;
    case CERTIFICATE_SOURCE_COUNT:
        break;
    }
    raise_error("invalid certificate source %d", (int)source);
    return NULL;
}

__global__ void compute_residual_kernel(
    double *__restrict__ primal_res,
    const double *__restrict__ primal_prod,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    double *__restrict__ dual_res,
    const double *__restrict__ dual_prod,
    const double *__restrict__ dual_sol,
    const double *__restrict__ obj_vec,
    const double *__restrict__ cons_rescaling,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    const double *__restrict__ primal_sol,
    const double *primal_pdhg_scale,
    double *__restrict__ primal_infeasibility_reduced_cost,
    double *__restrict__ primal_infeasibility_stationarity_res,
    double *__restrict__ constraint_recession_res,
    double *__restrict__ variable_recession_res,
    double *__restrict__ original_dual_ray,
    double *__restrict__ original_reduced_cost_ray,
    double *__restrict__ original_primal_ray,
    double *__restrict__ reduced_cost_out,
    const double *__restrict__ projection_multiplier,
    double *__restrict__ bound_multiplier_out,
    double *__restrict__ bound_multiplier_dual_res,
    int n_cons,
    int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double primal_scale = *primal_pdhg_scale;
    double inverse_primal_scale =
        (fabs(primal_scale) > 1e-50)
            ? 1.0 / primal_scale
            : 0.0;

    if (i < n_cons)
    {
        double ax = primal_prod[i];
        double ax_hat = ax * inverse_primal_scale;
        double clamped = fmax(cons_lb[i], fmin(ax_hat, cons_ub[i]));
        primal_res[i] = (ax_hat - clamped) * cons_rescaling[i];

        constraint_recession_res[i] = (ax - project_box_recession(ax, cons_lb[i], cons_ub[i]))
                                    * cons_rescaling[i];

        original_dual_ray[i] = dual_sol[i] / cons_rescaling[i];
    }
    else if (i < n_cons + n_vars)
    {
        int idx = i - n_cons;
        double aty = dual_prod[idx];
        const double reduced_cost = obj_vec[idx] * primal_scale - aty;
        const bool bound_active =
            (projection_multiplier[idx] > 0.0 && reduced_cost > 0.0) ||
            (projection_multiplier[idx] < 0.0 && reduced_cost < 0.0);
        reduced_cost_out[idx] = bound_active ? reduced_cost : 0.0;
        dual_res[idx] = bound_active
                      ? 0.0
                      : reduced_cost * inverse_primal_scale * var_rescaling[idx];

        double bound_multiplier = project_box_multiplier(reduced_cost, var_lb[idx], var_ub[idx]);
        bound_multiplier_out[idx] = bound_multiplier;
        bound_multiplier_dual_res[idx] = (reduced_cost - bound_multiplier)
                                       * inverse_primal_scale * var_rescaling[idx];

        double farkas_reduced_cost = project_box_multiplier(-aty, var_lb[idx], var_ub[idx]);
        primal_infeasibility_reduced_cost[idx] = farkas_reduced_cost;
        primal_infeasibility_stationarity_res[idx] =
            (aty + farkas_reduced_cost) * var_rescaling[idx];

        double x = primal_sol[idx];

        double x_hat = x * inverse_primal_scale;
        double x_box_clamped = fmax(var_lb[idx], fmin(x_hat, var_ub[idx]));
        primal_res[n_cons + idx] = (x_hat - x_box_clamped) / var_rescaling[idx];

        variable_recession_res[idx] = (x - project_box_recession(x, var_lb[idx], var_ub[idx]))
                                    / var_rescaling[idx];

        original_reduced_cost_ray[idx] = farkas_reduced_cost * var_rescaling[idx];
        original_primal_ray[idx] = x / var_rescaling[idx];
    }
}

__global__ void compute_projected_difference_kernel(
    const double *__restrict__ primal_from,
    const double *__restrict__ primal_to,
    double *__restrict__ primal_delta,
    const double *__restrict__ dual_from,
    const double *__restrict__ dual_to,
    double *__restrict__ dual_delta,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    int n_vars,
    int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_vars)
    {
        primal_delta[i] =
            project_box_recession(primal_to[i] - primal_from[i], var_lb[i], var_ub[i]);
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        dual_delta[idx] =
            project_box_multiplier(dual_to[idx] - dual_from[idx], cons_lb[idx], cons_ub[idx]);
    }
}

__global__ void compute_termination_metrics_kernel(
    pdhg_metrics_t *metrics_d,
    const termination_reductions_t *__restrict__ reductions,
    const double *__restrict__ primal_current_scale,
    const double *__restrict__ primal_pdhg_scale,
    const double *__restrict__ dual_pdhg_scale,
    double constraint_bound_rescaling,
    double objective_vector_rescaling,
    double original_bound_norm,
    double original_objective_norm,
    double objective_constant,
    norm_type_t optimality_norm)
{
    double primal_scale = *primal_pdhg_scale;
    double inv_primal_scale = (fabs(primal_scale) > 1e-50) ? 1.0 / primal_scale : 0.0;
    double bound_objective_scale = constraint_bound_rescaling * objective_vector_rescaling;
    double dual_scale = *dual_pdhg_scale;
    double absolute_primal_residual = reductions->primal_residual_norm / constraint_bound_rescaling;
    double absolute_dual_residual = reductions->dual_residual_norm / objective_vector_rescaling;
    double objective_dot_primal_value = reductions->iterate.objective_dot;
    double support_dual_objective = reductions->iterate.constraint_support
                                  + reductions->variable_support;

    double primal_objective_value = objective_dot_primal_value * inv_primal_scale
                                  / bound_objective_scale
                                  + objective_constant;
    double dual_objective_value = support_dual_objective * inv_primal_scale
                                / bound_objective_scale
                                + objective_constant;
    double objective_gap = fabs(primal_objective_value - dual_objective_value);
    double relative_primal_residual = absolute_primal_residual / (1.0 + original_bound_norm);
    double relative_dual_residual = absolute_dual_residual / (1.0 + original_objective_norm);
    double relative_objective_gap = objective_gap / (1.0 + fabs(primal_objective_value) + fabs(dual_objective_value));

    double bound_multiplier_absolute_dual_residual =
        reductions->bound_multiplier_dual_residual_norm / objective_vector_rescaling;
    double bound_multiplier_dual_objective =
        (reductions->iterate.constraint_support + reductions->bound_multiplier_variable_support)
        * inv_primal_scale / bound_objective_scale
        + objective_constant;
    double bound_multiplier_objective_gap =
        fabs(primal_objective_value - bound_multiplier_dual_objective);

    metrics_d->absolute_primal_residual = absolute_primal_residual;
    metrics_d->relative_primal_residual = relative_primal_residual;
    metrics_d->absolute_dual_residual   = absolute_dual_residual;
    metrics_d->relative_dual_residual   = relative_dual_residual;
    metrics_d->primal_objective = primal_objective_value;
    metrics_d->dual_objective   = dual_objective_value;
    metrics_d->objective_gap    = objective_gap;
    metrics_d->relative_objective_gap = relative_objective_gap;
    metrics_d->bound_multiplier_absolute_dual_residual = bound_multiplier_absolute_dual_residual;
    metrics_d->bound_multiplier_relative_dual_residual =
        bound_multiplier_absolute_dual_residual / (1.0 + original_objective_norm);
    metrics_d->bound_multiplier_dual_objective = bound_multiplier_dual_objective;
    metrics_d->bound_multiplier_objective_gap = bound_multiplier_objective_gap;
    metrics_d->bound_multiplier_relative_objective_gap = bound_multiplier_objective_gap
        / (1.0 + fabs(primal_objective_value) + fabs(bound_multiplier_dual_objective));
    metrics_d->primal_current_scale = *primal_current_scale;
    metrics_d->primal_pdhg_scale = primal_scale;
    metrics_d->dual_pdhg_scale = dual_scale;

    double primal_solution_norm = metrics_d->primal_solution_norm;
    double dual_solution_norm = metrics_d->dual_solution_norm;

    double homogeneous_gap = objective_dot_primal_value
                           - reductions->iterate.constraint_support
                           - reductions->bound_multiplier_variable_support;
    double iterate_norm = optimality_norm == NORM_TYPE_L_INF
                        ? fmax(fmax(primal_solution_norm, dual_solution_norm),
                               fmax(fabs(primal_scale), fabs(dual_scale)))
                        : sqrt(primal_solution_norm * primal_solution_norm
                             + dual_solution_norm * dual_solution_norm
                             + primal_scale * primal_scale
                             + dual_scale * dual_scale);
    double inv_iterate_norm = (iterate_norm > 1e-30) ? 1.0 / iterate_norm : 0.0;

    double homogeneous_gap_positive = fmax(-homogeneous_gap, 0.0);
    metrics_d->normalized_homogeneous_gap = homogeneous_gap_positive * inv_iterate_norm;

    double ratio_denominator = homogeneous_gap_positive + primal_scale;
    metrics_d->normalized_homogeneous_ratio = (ratio_denominator > 0.0)
                                            ? homogeneous_gap_positive / ratio_denominator
                                            : 0.0;

    double dual_certificate_norm = fmax(reductions->original_dual_ray_norm,
                                        reductions->original_reduced_cost_ray_norm);
    double inv_dual_certificate_norm = (dual_certificate_norm > 1e-30)
                                     ? 1.0 / dual_certificate_norm
                                     : 0.0;
    const candidate_reductions_t *iterate = &reductions->iterate;
    double support_total = (iterate->constraint_support + iterate->variable_support)
                         / constraint_bound_rescaling;
    fill_certificate_metrics(
        &metrics_d->primal_infeasibility_iterate,
        support_total * inv_dual_certificate_norm,
        iterate->stationarity_residual * inv_dual_certificate_norm,
        iterate->constraint_support + iterate->variable_support,
        iterate->sum_absolute_constraint_support_terms
            + iterate->sum_absolute_variable_support_terms,
        iterate->scaled_stationarity_residual,
        0.0);

    double original_primal_ray_norm = reductions->original_primal_ray_norm;
    double inv_original_primal_ray_norm = (original_primal_ray_norm > 1e-30)
                                        ? 1.0 / original_primal_ray_norm
                                        : 0.0;
    fill_certificate_metrics(
        &metrics_d->dual_infeasibility_iterate,
        -(iterate->objective_dot * inv_original_primal_ray_norm)
            / objective_vector_rescaling,
        fmax(iterate->constraint_recession_residual
                 * inv_original_primal_ray_norm,
             reductions->variable_recession_residual_norm
                 * inv_original_primal_ray_norm),
        -iterate->objective_dot,
        iterate->sum_absolute_objective_terms,
        iterate->scaled_constraint_recession_residual,
        compute_relative_error(reductions->scaled_variable_recession_residual,
                               reductions->scaled_primal_ray_norm));

    fill_difference_certificate_metrics(
        &metrics_d->primal_infeasibility_step,
        &metrics_d->dual_infeasibility_step,
        &reductions->step,
        bound_objective_scale,
        constraint_bound_rescaling,
        objective_vector_rescaling);
    fill_difference_certificate_metrics(
        &metrics_d->primal_infeasibility_restart,
        &metrics_d->dual_infeasibility_restart,
        &reductions->restart,
        bound_objective_scale,
        constraint_bound_rescaling,
        objective_vector_rescaling);
}

__device__ __forceinline__ void fill_certificate_metrics(
    certificate_metrics_t *metrics,
    double margin,
    double residual,
    double scaled_margin,
    double absolute_terms,
    double scaled_residual,
    double relative_bound_residual)
{
    metrics->margin = margin;
    metrics->residual = residual;
    metrics->relative_margin = compute_relative_margin(scaled_margin, absolute_terms);
    metrics->scaled_residual = scaled_residual;
    metrics->relative_bound_residual = relative_bound_residual;
    metrics->scaled_activity = 0.0;
    metrics->relative_residual = INFINITY;
}

__device__ __forceinline__ double compute_relative_margin(double margin, double absolute_terms)
{
    if (!isfinite(margin) || !isfinite(absolute_terms) ||
        margin <= 0.0 || absolute_terms <= 0.0)
        return 0.0;
    return margin / absolute_terms;
}

__host__ __device__ __forceinline__ double compute_relative_error(double residual, double activity)
{
    if (!isfinite(residual) || !isfinite(activity)) return INFINITY;
    if (activity > 0.0) return residual / activity;
    return residual == 0.0 ? 0.0 : INFINITY;
}

__device__ __forceinline__ void fill_difference_certificate_metrics(
    certificate_metrics_t *primal_infeasibility,
    certificate_metrics_t *dual_infeasibility,
    const candidate_reductions_t *candidate,
    double bound_objective_scale,
    double constraint_bound_rescaling,
    double objective_vector_rescaling)
{
    const double support = candidate->constraint_support + candidate->variable_support;
    fill_certificate_metrics(
        primal_infeasibility,
        support / bound_objective_scale,
        candidate->stationarity_residual / objective_vector_rescaling,
        support,
        candidate->sum_absolute_constraint_support_terms
            + candidate->sum_absolute_variable_support_terms,
        candidate->scaled_stationarity_residual,
        0.0);
    fill_certificate_metrics(
        dual_infeasibility,
        -candidate->objective_dot / bound_objective_scale,
        candidate->constraint_recession_residual / constraint_bound_rescaling,
        -candidate->objective_dot,
        candidate->sum_absolute_objective_terms,
        candidate->scaled_constraint_recession_residual,
        0.0);
}
