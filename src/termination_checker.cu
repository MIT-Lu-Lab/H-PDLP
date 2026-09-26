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
#include "cuda_utils.h"
#include "spmv_backend.h"
#include <cub/cub.cuh>
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
    const double *primal_scale_pdhg,
    double *__restrict__ primal_infeasibility_reduced_cost,
    double *__restrict__ primal_infeasibility_stationarity_res,
    double *__restrict__ constraint_recession_res,
    double *__restrict__ variable_recession_res,
    double *__restrict__ original_dual_ray,
    double *__restrict__ original_reduced_cost_ray,
    double *__restrict__ original_primal_ray,
    double *__restrict__ reduced_cost_out,
    int n_cons,
    int n_vars);
__global__ void compute_box_support_contribution_kernel(
    double *__restrict__ contrib,
    const double *__restrict__ direction,
    const double *__restrict__ lb,
    const double *__restrict__ ub,
    double direction_sign,
    int n);
__global__ void compute_projected_displacement_kernel(
    const double *__restrict__ initial_primal,
    const double *__restrict__ pdhg_primal,
    double *__restrict__ delta_primal,
    const double *__restrict__ initial_dual,
    const double *__restrict__ pdhg_dual,
    double *__restrict__ delta_dual,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    int n_vars,
    int n_cons);
__global__ void compute_ray_infeasibility_kernel(
    const double *__restrict__ primal_prod,
    const double *__restrict__ dual_prod,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    double *__restrict__ dual_infeasibility_res,
    double *__restrict__ primal_infeasibility_res,
    const double *__restrict__ cons_rescaling,
    const double *__restrict__ var_rescaling,
    int n_vars,
    int n_cons);
__global__ void compute_termination_metrics_kernel(
    pdhg_metrics_t *metrics_d,
    const termination_reductions_t *__restrict__ reductions,
    const double *__restrict__ primal_scale_current,
    const double *__restrict__ primal_scale_pdhg,
    const double *__restrict__ dual_scale_pdhg,
    double constraint_bound_rescaling,
    double objective_vector_rescaling,
    double original_bound_norm,
    double original_objective_norm,
    double objective_constant,
    norm_type_t optimality_norm);
__global__ void install_primal_infeasibility_ray_kernel(
    double *__restrict__ dual_sol,
    double *__restrict__ reduced_cost,
    const double *__restrict__ delta_dual,
    const double *__restrict__ dual_prod,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    int n_cons,
    int n_vars);
static void compute_matrix_products(
    pdhg_solver_state_t *state,
    cusparseDnVecDescr_t primal_vector,
    cusparseDnVecDescr_t dual_vector);
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
static void compute_homogeneous_variable_support(pdhg_solver_state_t *state);
static void compute_ray_infeasibility(pdhg_solver_state_t *state);
static void compute_projected_displacement(pdhg_solver_state_t *state);
static void compute_ray_infeasibility_norms(pdhg_solver_state_t *state);
static bool certify_optimality(
    pdhg_solver_state_t *state,
    const termination_criteria_t *criteria);
static bool certify_infeasibility(
    pdhg_solver_state_t *state,
    double margin,
    double residual,
    double eps,
    termination_reason_t reason,
    bool from_homogeneous);
static void prepare_homogeneous_primal_infeasibility_ray_result(pdhg_solver_state_t *state);
static void prepare_primal_infeasibility_ray_result(pdhg_solver_state_t *state);
static void prepare_dual_infeasibility_ray_result(pdhg_solver_state_t *state);

void compute_termination_metrics(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    termination_reductions_t *reductions_d = state->termination_reductions_d;

    compute_matrix_products(state,
                            state->spmv_pdhg_primal_solution,
                            state->spmv_pdhg_dual_solution);
    compute_residual_vectors(state);
    compute_residual_norms(state, params);
    compute_objective_and_support_terms(
        state,
        state->pdhg_primal_solution_d,
        state->pdhg_dual_solution_d,
        state->pdhg_reduced_cost_d,
        1.0,
        &reductions_d->objective_dot_primal,
        &reductions_d->constraint_support,
        &reductions_d->variable_support);
    compute_homogeneous_variable_support(state);
    compute_ray_infeasibility(state);

    compute_termination_metrics_kernel<<<1, 1, 0, state->stream_A>>>(
        state->metrics_d,
        reductions_d,
        state->hsd_primal_scale_current_d,
        state->hsd_primal_scale_pdhg_d,
        state->hsd_dual_scale_pdhg_d,
        state->constraint_bound_rescaling,
        state->objective_vector_rescaling,
        state->original_bound_norm,
        state->original_objective_norm,
        state->objective_constant,
        params->optimality_norm);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(state->metrics_h, state->metrics_d,
                               sizeof(pdhg_metrics_t),
                               cudaMemcpyDeviceToHost,
                               state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
}

static void compute_matrix_products(
    pdhg_solver_state_t *state,
    cusparseDnVecDescr_t primal_vector,
    cusparseDnVecDescr_t dual_vector)
{
    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));
    execute_spmv(
        state->sparse_handle_A,
        primal_vector,
        state->spmv_primal_product,
        &state->matA_spmv_plan);

    execute_spmv(
        state->sparse_handle_At,
        dual_vector,
        state->spmv_dual_product,
        &state->matAt_spmv_plan);
    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));
}

static void compute_residual_vectors(pdhg_solver_state_t *state)
{
    compute_residual_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->primal_residual_d,
        state->primal_product_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        state->dual_residual_d,
        state->dual_product_d,
        state->pdhg_dual_solution_d,
        state->objective_vector_d,
        state->constraint_rescaling_d,
        state->variable_rescaling_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        state->pdhg_primal_solution_d,
        state->hsd_primal_scale_pdhg_d,
        state->primal_infeasibility_reduced_cost_d,
        state->primal_infeasibility_stationarity_residual_d,
        state->constraint_recession_residual_d,
        state->variable_recession_residual_d,
        state->dual_ray_d,
        state->reduced_cost_ray_d,
        state->primal_ray_d,
        state->pdhg_reduced_cost_d,
        state->num_constraints,
        state->num_variables);
    CUDA_CHECK(cudaGetLastError());
}

static void compute_residual_norms(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    termination_reductions_t *reductions_d = state->termination_reductions_d;

    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        reduce_device_abs_max(state, state->primal_residual_d, state->num_constraints + state->num_variables,
                              &reductions_d->primal_residual_norm);
        reduce_device_abs_max(state, state->dual_residual_d, state->num_variables,
                              &reductions_d->dual_residual_norm);
        reduce_device_abs_max(state, state->pdhg_primal_solution_d, state->num_variables,
                              &state->metrics_d->primal_solution_norm);
        reduce_device_abs_max(state, state->pdhg_dual_solution_d, state->num_constraints,
                              &reductions_d->dual_solution_norm);
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
            state->pdhg_primal_solution_d, 1,
            &state->metrics_d->primal_solution_norm));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_constraints,
            state->pdhg_dual_solution_d, 1,
            &reductions_d->dual_solution_norm));
    }

    reduce_device_abs_max(state, state->primal_infeasibility_stationarity_residual_d, state->num_variables,
                          &reductions_d->primal_infeasibility_stationarity_residual_norm);
    reduce_device_abs_max(state, state->constraint_recession_residual_d, state->num_constraints,
                          &reductions_d->constraint_recession_residual_norm);
    reduce_device_abs_max(state, state->variable_recession_residual_d, state->num_variables,
                          &reductions_d->variable_recession_residual_norm);
    reduce_device_abs_max(state, state->dual_ray_d, state->num_constraints,
                          &reductions_d->original_dual_ray_norm);
    reduce_device_abs_max(state, state->reduced_cost_ray_d, state->num_variables,
                          &reductions_d->original_reduced_cost_ray_norm);
    reduce_device_abs_max(state, state->primal_ray_d, state->num_variables,
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
    double *constraint_support_entries_d = state->constraint_reduction_entries_d;
    double *variable_support_entries_d = state->variable_reduction_entries_d;

    CUBLAS_CHECK(cublasDdot(
        state->blas_handle_A,
        state->num_variables,
        state->objective_vector_d, 1,
        primal_direction_d, 1,
        objective_dot_d));

    compute_box_support_contribution_kernel<<<
        state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        constraint_support_entries_d,
        dual_direction_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        1.0,
        state->num_constraints);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        constraint_support_entries_d,
        constraint_support_d,
        state->num_constraints, state->stream_A));

    compute_box_support_contribution_kernel<<<
        state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        variable_support_entries_d,
        variable_direction_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        variable_direction_sign,
        state->num_variables);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        variable_support_entries_d,
        variable_support_d,
        state->num_variables, state->stream_A));
}

static void compute_homogeneous_variable_support(pdhg_solver_state_t *state)
{
    termination_reductions_t *reductions_d = state->termination_reductions_d;
    double *variable_support_entries_d = state->variable_reduction_entries_d;

    compute_box_support_contribution_kernel<<<
        state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        variable_support_entries_d,
        state->primal_infeasibility_reduced_cost_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        1.0,
        state->num_variables);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        variable_support_entries_d,
        &reductions_d->homogeneous_variable_support,
        state->num_variables, state->stream_A));
}

static void compute_ray_infeasibility(pdhg_solver_state_t *state)
{
    termination_reductions_t *reductions_d = state->termination_reductions_d;

    compute_projected_displacement(state);
    compute_matrix_products(state,
                            state->spmv_delta_primal_solution,
                            state->spmv_delta_dual_solution);
    compute_objective_and_support_terms(
        state,
        state->delta_primal_solution_d,
        state->delta_dual_solution_d,
        state->dual_product_d,
        -1.0,
        &reductions_d->dual_infeasibility_objective,
        &reductions_d->primal_infeasibility_constraint_support,
        &reductions_d->primal_infeasibility_variable_support);
    compute_ray_infeasibility_norms(state);
}

static void compute_projected_displacement(pdhg_solver_state_t *state)
{
    compute_projected_displacement_kernel<<<
        state->num_blocks_primal_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->initial_primal_solution_d,
        state->pdhg_primal_solution_d,
        state->delta_primal_solution_d,
        state->initial_dual_solution_d,
        state->pdhg_dual_solution_d,
        state->delta_dual_solution_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        state->num_variables,
        state->num_constraints);
    CUDA_CHECK(cudaGetLastError());
}

static void compute_ray_infeasibility_norms(pdhg_solver_state_t *state)
{
    termination_reductions_t *reductions_d = state->termination_reductions_d;
    double *dual_infeasibility_entries_d = state->constraint_reduction_entries_d;
    double *primal_infeasibility_entries_d = state->variable_reduction_entries_d;

    compute_ray_infeasibility_kernel<<<
        state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->primal_product_d,
        state->dual_product_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        dual_infeasibility_entries_d,
        primal_infeasibility_entries_d,
        state->constraint_rescaling_d,
        state->variable_rescaling_d,
        state->num_variables,
        state->num_constraints);
    CUDA_CHECK(cudaGetLastError());

    reduce_device_abs_max(state, dual_infeasibility_entries_d, state->num_constraints,
                          &reductions_d->dual_infeasibility_residual);
    reduce_device_abs_max(state, primal_infeasibility_entries_d, state->num_variables,
                          &reductions_d->primal_infeasibility_residual);
}

void check_termination_criteria(
    pdhg_solver_state_t *solver_state,
    const termination_criteria_t *criteria)
{
    solver_state->control.cumulative_time_sec = (double)(clock() - solver_state->control.start_time) / CLOCKS_PER_SEC;

    if (!isfinite(solver_state->metrics_h->primal_solution_norm) ||
        !isfinite(solver_state->metrics_h->dual_solution_norm) ||
        !isfinite(solver_state->metrics_h->pdhg_primal_scale))
    {
        solver_state->control.termination_reason = TERMINATION_REASON_NUMERICAL_FAILURE;
        return;
    }

    double ratio = solver_state->metrics_h->normalized_homogeneous_ratio;
    if (ratio < 0.5)
    {
        if (certify_optimality(solver_state, criteria))
        {
            return;
        }
    }
    else if (ratio > 0.9)
    {
        if (certify_infeasibility(solver_state,
                                  solver_state->metrics_h->primal_infeasibility_homogeneous_margin,
                                  solver_state->metrics_h->primal_infeasibility_homogeneous_residual,
                                  criteria->eps_farkas_homogeneous,
                                  TERMINATION_REASON_PRIMAL_INFEASIBLE,
                                  true))
        {
            prepare_homogeneous_primal_infeasibility_ray_result(solver_state);
            return;
        }
        if (certify_infeasibility(solver_state,
                                  solver_state->metrics_h->dual_infeasibility_homogeneous_margin,
                                  solver_state->metrics_h->dual_infeasibility_homogeneous_residual,
                                  criteria->eps_farkas_homogeneous,
                                  TERMINATION_REASON_DUAL_INFEASIBLE,
                                  true))
        {
            return;
        }
    }

    if (certify_infeasibility(solver_state,
                              solver_state->metrics_h->primal_infeasibility_displacement_margin,
                              solver_state->metrics_h->primal_infeasibility_displacement_residual,
                              criteria->eps_farkas_displacement,
                              TERMINATION_REASON_PRIMAL_INFEASIBLE,
                              false))
    {
        prepare_primal_infeasibility_ray_result(solver_state);
        return;
    }
    if (certify_infeasibility(solver_state,
                              solver_state->metrics_h->dual_infeasibility_displacement_margin,
                              solver_state->metrics_h->dual_infeasibility_displacement_residual,
                              criteria->eps_farkas_displacement,
                              TERMINATION_REASON_DUAL_INFEASIBLE,
                              false))
    {
        prepare_dual_infeasibility_ray_result(solver_state);
        return;
    }

    if (solver_state->control.total_count >= criteria->iteration_limit)
    {
        solver_state->control.termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        return;
    }
    if (solver_state->control.cumulative_time_sec >= criteria->time_sec_limit)
    {
        solver_state->control.termination_reason = TERMINATION_REASON_TIME_LIMIT;
        return;
    }
}

static bool certify_optimality(
    pdhg_solver_state_t *state,
    const termination_criteria_t *criteria)
{
    if (state->metrics_h->pdhg_primal_scale > 1e-50 &&
        state->metrics_h->relative_primal_residual < criteria->eps_feasible_relative &&
        state->metrics_h->relative_dual_residual   < criteria->eps_feasible_relative &&
        state->metrics_h->relative_objective_gap   < criteria->eps_optimal_relative)
    {
        state->control.termination_reason = TERMINATION_REASON_OPTIMAL;
        return true;
    }
    return false;
}

static bool certify_infeasibility(
    pdhg_solver_state_t *state,
    double margin,
    double residual,
    double eps,
    termination_reason_t reason,
    bool from_homogeneous)
{
    if (margin > 0.0 && residual < eps * margin)
    {
        state->control.certificate_margin = margin;
        state->control.certificate_residual = residual;
        state->control.certificate_from_homogeneous = from_homogeneous;
        state->control.termination_reason = reason;
        return true;
    }
    return false;
}

static void prepare_homogeneous_primal_infeasibility_ray_result(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpyAsync(
        state->pdhg_reduced_cost_d,
        state->primal_infeasibility_reduced_cost_d,
        state->var_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));
}

static void prepare_primal_infeasibility_ray_result(pdhg_solver_state_t *state)
{
    install_primal_infeasibility_ray_kernel<<<
        state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->pdhg_dual_solution_d,
        state->pdhg_reduced_cost_d,
        state->delta_dual_solution_d,
        state->dual_product_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        state->num_constraints,
        state->num_variables);
    CUDA_CHECK(cudaGetLastError());
}

static void prepare_dual_infeasibility_ray_result(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpyAsync(
        state->pdhg_primal_solution_d,
        state->delta_primal_solution_d,
        state->var_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));
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
    const double *primal_scale_pdhg,
    double *__restrict__ primal_infeasibility_reduced_cost,
    double *__restrict__ primal_infeasibility_stationarity_res,
    double *__restrict__ constraint_recession_res,
    double *__restrict__ variable_recession_res,
    double *__restrict__ original_dual_ray,
    double *__restrict__ original_reduced_cost_ray,
    double *__restrict__ original_primal_ray,
    double *__restrict__ reduced_cost_out,
    int n_cons,
    int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double primal_scale = *primal_scale_pdhg;
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

        double recession_lower_bound = isfinite(cons_lb[i]) ? 0.0 : -INFINITY;
        double recession_upper_bound = isfinite(cons_ub[i]) ? 0.0 : INFINITY;
        double projected_recession_value = fmax(recession_lower_bound,
                                                fmin(ax, recession_upper_bound));
        constraint_recession_res[i] = (ax - projected_recession_value) 
                                    * cons_rescaling[i];

        original_dual_ray[i] = dual_sol[i] / cons_rescaling[i];
    }
    else if (i < n_cons + n_vars)
    {
        int idx = i - n_cons;
        double aty = dual_prod[idx];
        double normalized_reduced_cost = (obj_vec[idx] - aty * inverse_primal_scale) 
                                       * var_rescaling[idx];
        double dual_cone_violation = 0.0;
        if (!isfinite(var_ub[idx]))
        {
            dual_cone_violation += fmax(0.0, -normalized_reduced_cost);
        }
        if (!isfinite(var_lb[idx]))
        {
            dual_cone_violation += fmax(0.0, normalized_reduced_cost);
        }
        dual_res[idx] = dual_cone_violation;

        double reduced_cost = obj_vec[idx] * primal_scale - aty;
        reduced_cost_out[idx] = reduced_cost;

        double farkas_lower_bound = !isfinite(var_ub[idx]) ? 0.0 : -INFINITY;
        double farkas_upper_bound = !isfinite(var_lb[idx]) ? 0.0 : INFINITY;
        double farkas_reduced_cost = fmax(farkas_lower_bound,
                                          fmin(-aty, farkas_upper_bound));
        primal_infeasibility_reduced_cost[idx] = farkas_reduced_cost;
        primal_infeasibility_stationarity_res[idx] =
            (aty + farkas_reduced_cost) * var_rescaling[idx];

        double x = primal_sol[idx];

        double x_hat = x * inverse_primal_scale;
        double x_box_clamped = fmax(var_lb[idx], fmin(x_hat, var_ub[idx]));
        primal_res[n_cons + idx] = (x_hat - x_box_clamped) / var_rescaling[idx];

        double recession_lower_bound = isfinite(var_lb[idx]) ? 0.0 : -INFINITY;
        double recession_upper_bound = isfinite(var_ub[idx]) ? 0.0 : INFINITY;
        double projected_recession_value = fmax(recession_lower_bound,
                                                fmin(x, recession_upper_bound));
        variable_recession_res[idx] = (x - projected_recession_value)
                                    / var_rescaling[idx];

        original_reduced_cost_ray[idx] = farkas_reduced_cost * var_rescaling[idx];
        original_primal_ray[idx] = x / var_rescaling[idx];
    }
}

__global__ void compute_box_support_contribution_kernel(
    double *__restrict__ contrib,
    const double *__restrict__ direction,
    const double *__restrict__ lb,
    const double *__restrict__ ub,
    double direction_sign,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double yi = direction_sign * direction[i];
    double c = 0.0;
    if (yi > 0.0 && isfinite(lb[i]))
    {
        c = lb[i] * yi;
    }
    else if (yi < 0.0 && isfinite(ub[i]))
    {
        c = ub[i] * yi;
    }
    contrib[i] = c;
}

__global__ void compute_projected_displacement_kernel(
    const double *__restrict__ initial_primal,
    const double *__restrict__ pdhg_primal,
    double *__restrict__ delta_primal,
    const double *__restrict__ initial_dual,
    const double *__restrict__ pdhg_dual,
    double *__restrict__ delta_dual,
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
        double displacement = pdhg_primal[i] - initial_primal[i];
        if (isfinite(var_lb[i]))
        {
            displacement = fmax(displacement, 0.0);
        }
        if (isfinite(var_ub[i]))
        {
            displacement = fmin(displacement, 0.0);
        }
        delta_primal[i] = displacement;
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        double displacement = pdhg_dual[idx] - initial_dual[idx];
        if (!isfinite(cons_lb[idx]))
        {
            displacement = fmin(displacement, 0.0);
        }
        if (!isfinite(cons_ub[idx]))
        {
            displacement = fmax(displacement, 0.0);
        }
        delta_dual[idx] = displacement;
    }
}

__global__ void compute_ray_infeasibility_kernel(
    const double *__restrict__ primal_prod,
    const double *__restrict__ dual_prod,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    double *__restrict__ dual_infeasibility_res,
    double *__restrict__ primal_infeasibility_res,
    const double *__restrict__ cons_rescaling,
    const double *__restrict__ var_rescaling,
    int n_vars,
    int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_cons)
    {
        double v = primal_prod[i];
        double lower_bound = isfinite(cons_lb[i]) ? 0.0 : -INFINITY;
        double upper_bound = isfinite(cons_ub[i]) ? 0.0 : INFINITY;
        double projected_value = fmax(lower_bound, fmin(v, upper_bound));
        dual_infeasibility_res[i] = (v - projected_value) * cons_rescaling[i];
    }
    else if (i < n_cons + n_vars)
    {
        int idx = i - n_cons;
        double v = -dual_prod[idx];
        double lower_bound = !isfinite(var_ub[idx]) ? 0.0 : -INFINITY;
        double upper_bound = !isfinite(var_lb[idx]) ? 0.0 : INFINITY;
        double projected_value = fmax(lower_bound, fmin(v, upper_bound));
        primal_infeasibility_res[idx] = (v - projected_value) * var_rescaling[idx];
    }
}

__global__ void compute_termination_metrics_kernel(
    pdhg_metrics_t *metrics_d,
    const termination_reductions_t *__restrict__ reductions,
    const double *__restrict__ primal_scale_current,
    const double *__restrict__ primal_scale_pdhg,
    const double *__restrict__ dual_scale_pdhg,
    double constraint_bound_rescaling,
    double objective_vector_rescaling,
    double original_bound_norm,
    double original_objective_norm,
    double objective_constant,
    norm_type_t optimality_norm)
{
    double primal_scale = *primal_scale_pdhg;
    double inv_primal_scale = (fabs(primal_scale) > 1e-50) ? 1.0 / primal_scale : 0.0;
    double pdhg_dual_scale = *dual_scale_pdhg;
    double absolute_primal_residual = reductions->primal_residual_norm / constraint_bound_rescaling;
    double absolute_dual_residual = reductions->dual_residual_norm / objective_vector_rescaling;
    double objective_dot_primal_value = reductions->objective_dot_primal;
    double support_dual_objective = reductions->constraint_support
                                  + reductions->variable_support;

    double primal_objective_value = objective_dot_primal_value * inv_primal_scale
                                  / (constraint_bound_rescaling * objective_vector_rescaling)
                                  + objective_constant;
    double dual_objective_value = support_dual_objective * inv_primal_scale
                                / (constraint_bound_rescaling * objective_vector_rescaling)
                                + objective_constant;
    double objective_gap = fabs(primal_objective_value - dual_objective_value);
    double relative_primal_residual = absolute_primal_residual / (1.0 + original_bound_norm);
    double relative_dual_residual = absolute_dual_residual / (1.0 + original_objective_norm);
    double relative_objective_gap = objective_gap / (1.0 + fabs(primal_objective_value) + fabs(dual_objective_value));

    metrics_d->absolute_primal_residual = absolute_primal_residual;
    metrics_d->relative_primal_residual = relative_primal_residual;
    metrics_d->absolute_dual_residual   = absolute_dual_residual;
    metrics_d->relative_dual_residual   = relative_dual_residual;
    metrics_d->primal_objective = primal_objective_value;
    metrics_d->dual_objective   = dual_objective_value;
    metrics_d->objective_gap    = objective_gap;
    metrics_d->relative_objective_gap = relative_objective_gap;
    metrics_d->current_primal_scale   = *primal_scale_current;
    metrics_d->pdhg_primal_scale      = primal_scale;
    metrics_d->pdhg_dual_scale = pdhg_dual_scale;

    double dual_solution_norm = reductions->dual_solution_norm;
    double primal_solution_norm = metrics_d->primal_solution_norm;

    metrics_d->dual_solution_norm = dual_solution_norm;

    double homogeneous_gap = objective_dot_primal_value
                           - reductions->constraint_support
                           - reductions->variable_support;
    double iterate_norm = optimality_norm == NORM_TYPE_L_INF
                        ? fmax(fmax(primal_solution_norm, dual_solution_norm),
                               fmax(fabs(primal_scale), fabs(pdhg_dual_scale)))
                        : sqrt(primal_solution_norm * primal_solution_norm
                             + dual_solution_norm * dual_solution_norm
                             + primal_scale * primal_scale
                             + pdhg_dual_scale * pdhg_dual_scale);
    double inv_iterate_norm = (iterate_norm > 1e-30) ? 1.0 / iterate_norm : 0.0;

    double homogeneous_gap_positive = fmax(-homogeneous_gap, 0.0);
    metrics_d->normalized_homogeneous_gap = homogeneous_gap_positive * inv_iterate_norm;

    double ratio_denominator = homogeneous_gap_positive + primal_scale;
    metrics_d->normalized_homogeneous_ratio = (ratio_denominator > 0.0)
                                            ? homogeneous_gap_positive / ratio_denominator
                                            : 0.0;

    double stationarity_residual_norm = reductions->primal_infeasibility_stationarity_residual_norm;
    double original_dual_norm = reductions->original_dual_ray_norm;
    double original_reduced_cost_norm = reductions->original_reduced_cost_ray_norm;
    double original_dual_ray_norm = fmax(original_dual_norm, original_reduced_cost_norm);
    double inv_original_dual_ray_norm = (original_dual_ray_norm > 1e-30)
                                      ? 1.0 / original_dual_ray_norm
                                      : 0.0;
    double support_total = (reductions->constraint_support
                          + reductions->homogeneous_variable_support)
                         / constraint_bound_rescaling;
    metrics_d->primal_infeasibility_homogeneous_margin =
        support_total * inv_original_dual_ray_norm;
    metrics_d->primal_infeasibility_homogeneous_residual =
        stationarity_residual_norm * inv_original_dual_ray_norm;

    double original_primal_ray_norm = reductions->original_primal_ray_norm;
    double inv_original_primal_ray_norm = (original_primal_ray_norm > 1e-30)
                                        ? 1.0 / original_primal_ray_norm
                                        : 0.0;
    metrics_d->dual_infeasibility_homogeneous_margin =
        -(objective_dot_primal_value * inv_original_primal_ray_norm)
        / objective_vector_rescaling;
    metrics_d->dual_infeasibility_homogeneous_residual =
        fmax(reductions->constraint_recession_residual_norm
                 * inv_original_primal_ray_norm,
             reductions->variable_recession_residual_norm
                 * inv_original_primal_ray_norm);

    double ray_scale = constraint_bound_rescaling * objective_vector_rescaling;
    metrics_d->primal_infeasibility_displacement_margin =
        (reductions->primal_infeasibility_constraint_support
         + reductions->primal_infeasibility_variable_support)
        / ray_scale;
    metrics_d->primal_infeasibility_displacement_residual =
        reductions->primal_infeasibility_residual
        / objective_vector_rescaling;
    metrics_d->dual_infeasibility_displacement_margin =
        -reductions->dual_infeasibility_objective / ray_scale;
    metrics_d->dual_infeasibility_displacement_residual =
        reductions->dual_infeasibility_residual
        / constraint_bound_rescaling;
}

__global__ void install_primal_infeasibility_ray_kernel(
    double *__restrict__ dual_sol,
    double *__restrict__ reduced_cost,
    const double *__restrict__ delta_dual,
    const double *__restrict__ dual_prod,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    int n_cons,
    int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_cons)
    {
        dual_sol[i] = delta_dual[i];
    }
    else if (i < n_cons + n_vars)
    {
        int idx = i - n_cons;
        double farkas_lower_bound = !isfinite(var_ub[idx]) ? 0.0 : -INFINITY;
        double farkas_upper_bound = !isfinite(var_lb[idx]) ? 0.0 : INFINITY;
        reduced_cost[idx] = fmax(farkas_lower_bound,
                                 fmin(-dual_prod[idx], farkas_upper_bound));
    }
}
