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
#include "utils.h"
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
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
    double *__restrict__ homogeneous_dual_res,
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
__global__ void compute_projected_ray_delta_kernel(
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
    const double *__restrict__ primal_product,
    const double *__restrict__ dual_product,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    double *__restrict__ primal_infeasibility,
    double *__restrict__ dual_infeasibility,
    const double *__restrict__ cons_rescaling,
    const double *__restrict__ var_rescaling,
    int n_vars,
    int n_cons);
__global__ void install_primal_infeasibility_ray_kernel(
    double *__restrict__ dual_solution,
    double *__restrict__ reduced_cost,
    double *__restrict__ original_dual_ray,
    double *__restrict__ original_reduced_cost_ray,
    const double *__restrict__ delta_dual,
    const double *__restrict__ dual_product,
    const double *__restrict__ cons_rescaling,
    const double *__restrict__ var_rescaling,
    int n_cons,
    int n_vars);
__global__ void install_dual_infeasibility_ray_kernel(
    double *__restrict__ primal_solution,
    double *__restrict__ original_primal_ray,
    const double *__restrict__ delta_primal,
    const double *__restrict__ var_rescaling,
    int n_vars);
__global__ void compute_ray_metrics_kernel(
    pdhg_metrics_t *metrics_d,
    const double *__restrict__ primal_ray_linear_objective,
    const double *__restrict__ constraint_support,
    const double *__restrict__ variable_support,
    const double *__restrict__ primal_ray_infeasibility,
    const double *__restrict__ dual_ray_infeasibility,
    double constraint_bound_rescaling,
    double objective_vector_rescaling);
__global__ void compute_residual_metrics_kernel(
    pdhg_metrics_t *metrics_d,
    const double *primal_norm,
    const double *dual_norm,
    const double *objective_dot_primal,
    const double *farkas_dual_solution_norm,
    const double *farkas_homogeneous_dual_norm,
    const double *farkas_constraint_recession,
    const double *farkas_variable_recession,
    const double *constraint_support,
    const double *variable_support,
    const double *farkas_original_dual_norm,
    const double *farkas_original_reduced_cost_norm,
    const double *farkas_original_primal_norm,
    const double *primal_scale_current,
    const double *primal_scale_pdhg,
    const double *dual_scale_pdhg,
    double constraint_bound_rescaling,
    double objective_vector_rescaling,
    double original_bound_norm,
    double original_objective_norm,
    double original_constraint_norm,
    double objective_constant);
__global__ void rescale_solution_kernel(
    double *__restrict__ primal_sol,
    double *__restrict__ dual_sol,
    double *__restrict__ reduced_cost,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ cons_rescaling,
    const double obj_vec_rescaling,
    const double cons_bound_rescaling,
    double divisor,
    int n_vars,
    int n_cons);
static void compute_residual_products(pdhg_solver_state_t *state);
static void compute_residual_vectors(pdhg_solver_state_t *state);
static void compute_linf_residual_norms(pdhg_solver_state_t *state);
static void compute_l2_residual_norms(pdhg_solver_state_t *state);
static void compute_objective_and_support_terms(pdhg_solver_state_t *state);
static void update_residual_metrics(pdhg_solver_state_t *state);
static void compute_ray_infeasibility(pdhg_solver_state_t *state);
static void compute_projected_ray_delta(pdhg_solver_state_t *state);
static void compute_ray_products(pdhg_solver_state_t *state);
static void compute_ray_objectives(pdhg_solver_state_t *state);
static void compute_ray_infeasibility_norms(pdhg_solver_state_t *state);
static bool optimality_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria);
static bool primal_infeasibility_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria);
static bool dual_infeasibility_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria);
static bool primal_infeasibility_ray_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria);
static bool dual_infeasibility_ray_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria);
static void prepare_primal_infeasibility_ray_result(pdhg_solver_state_t *state);
static void prepare_dual_infeasibility_ray_result(pdhg_solver_state_t *state);
static void rescale_solution(pdhg_solver_state_t *state);

void compute_termination_metrics(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    compute_residual_products(state);
    compute_residual_vectors(state);

    if (params->optimality_norm == NORM_TYPE_L_INF)
        compute_linf_residual_norms(state);
    else
        compute_l2_residual_norms(state);

    compute_objective_and_support_terms(state);
    update_residual_metrics(state);
    if (params->termination_criteria.enable_ray_infeas)
    {
        compute_ray_infeasibility(state);
    }
    CUDA_CHECK(cudaMemcpyAsync(state->metrics_h, state->metrics_d,
                               sizeof(pdhg_metrics_t),
                               cudaMemcpyDeviceToHost,
                               state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
}

static void compute_residual_products(pdhg_solver_state_t *state)
{
    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle_A,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE,
        state->matA,
        state->spmv_pdhg_primal_solution,
        &HOST_ZERO,
        state->spmv_primal_product,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2,
        state->matA_spmv_workspace_d));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle_At,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE,
        state->matAt,
        state->spmv_pdhg_dual_solution,
        &HOST_ZERO,
        state->spmv_dual_product,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2,
        state->matAt_spmv_workspace_d));

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
        state->homogeneous_dual_residual_d,
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

static void compute_linf_residual_norms(pdhg_solver_state_t *state)
{
    reduce_device_abs_max(state, state->primal_residual_d, state->num_constraints + state->num_variables,
                          state->pdhg_primal_norm_d);
    reduce_device_abs_max(state, state->dual_residual_d, state->num_variables,
                          state->pdhg_dual_norm_d);
    reduce_device_abs_max(state, state->homogeneous_dual_residual_d, state->num_variables,
                          state->farkas_homogeneous_dual_norm_d);
    reduce_device_abs_max(state, state->constraint_recession_residual_d, state->num_constraints,
                          state->farkas_constraint_recession_d);
    reduce_device_abs_max(state, state->variable_recession_residual_d, state->num_variables,
                          state->farkas_variable_recession_d);
    reduce_device_abs_max(state, state->pdhg_primal_solution_d, state->num_variables,
                          &state->metrics_d->primal_solution_norm);
    reduce_device_abs_max(state, state->pdhg_dual_solution_d, state->num_constraints,
                          state->farkas_dual_solution_norm_d);
    reduce_device_abs_max(state, state->dual_ray_d, state->num_constraints,
                          state->farkas_original_dual_norm_d);
    reduce_device_abs_max(state, state->reduced_cost_ray_d, state->num_variables,
                          state->farkas_original_reduced_cost_norm_d);
    reduce_device_abs_max(state, state->primal_ray_d, state->num_variables,
                          state->farkas_original_primal_norm_d);
}

static void compute_l2_residual_norms(pdhg_solver_state_t *state)
{
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_constraints + state->num_variables,
        state->primal_residual_d, 1,
        state->pdhg_primal_norm_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables,
        state->dual_residual_d, 1,
        state->pdhg_dual_norm_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables,
        state->homogeneous_dual_residual_d, 1,
        state->farkas_homogeneous_dual_norm_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_constraints,
        state->constraint_recession_residual_d, 1,
        state->farkas_constraint_recession_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables,
        state->variable_recession_residual_d, 1,
        state->farkas_variable_recession_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables,
        state->pdhg_primal_solution_d, 1,
        &state->metrics_d->primal_solution_norm));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_constraints,
        state->pdhg_dual_solution_d, 1,
        state->farkas_dual_solution_norm_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_constraints,
        state->dual_ray_d, 1,
        state->farkas_original_dual_norm_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables,
        state->reduced_cost_ray_d, 1,
        state->farkas_original_reduced_cost_norm_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables,
        state->primal_ray_d, 1,
        state->farkas_original_primal_norm_d));
}

static void compute_objective_and_support_terms(pdhg_solver_state_t *state)
{
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle_A,
        state->num_variables,
        state->objective_vector_d, 1,
        state->pdhg_primal_solution_d, 1,
        state->objective_dot_primal_d));

    compute_box_support_contribution_kernel<<<
        state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->constraint_reduction_entries_d,
        state->pdhg_dual_solution_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        1.0,
        state->num_constraints);
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        state->constraint_reduction_entries_d,
        state->constraint_support_value_d,
        state->num_constraints, state->stream_A));

    compute_box_support_contribution_kernel<<<
        state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->variable_reduction_entries_d,
        state->pdhg_reduced_cost_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        1.0,
        state->num_variables);
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        state->variable_reduction_entries_d,
        state->variable_support_value_d,
        state->num_variables, state->stream_A));
}

static void update_residual_metrics(pdhg_solver_state_t *state)
{
    compute_residual_metrics_kernel<<<1, 1, 0, state->stream_A>>>(
        state->metrics_d,
        state->pdhg_primal_norm_d,
        state->pdhg_dual_norm_d,
        state->objective_dot_primal_d,
        state->farkas_dual_solution_norm_d,
        state->farkas_homogeneous_dual_norm_d,
        state->farkas_constraint_recession_d,
        state->farkas_variable_recession_d,
        state->constraint_support_value_d,
        state->variable_support_value_d,
        state->farkas_original_dual_norm_d,
        state->farkas_original_reduced_cost_norm_d,
        state->farkas_original_primal_norm_d,
        state->hsd_primal_scale_current_d,
        state->hsd_primal_scale_pdhg_d,
        state->hsd_dual_scale_pdhg_d,
        state->constraint_bound_rescaling,
        state->objective_vector_rescaling,
        state->original_bound_norm,
        state->original_objective_norm,
        state->original_constraint_norm,
        state->objective_constant);

}

static void compute_ray_infeasibility(pdhg_solver_state_t *state)
{
    compute_projected_ray_delta(state);
    compute_ray_products(state);
    compute_ray_objectives(state);
    compute_ray_infeasibility_norms(state);

    compute_ray_metrics_kernel<<<1, 1, 0, state->stream_A>>>(
        state->metrics_d,
        state->primal_ray_linear_objective_d,
        state->dual_ray_constraint_support_d,
        state->dual_ray_variable_support_d,
        state->max_primal_ray_infeasibility_d,
        state->max_dual_ray_infeasibility_d,
        state->constraint_bound_rescaling,
        state->objective_vector_rescaling);
}

static void compute_projected_ray_delta(pdhg_solver_state_t *state)
{
    compute_projected_ray_delta_kernel<<<
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

static void compute_ray_products(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));
    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle_A,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE,
        state->matA,
        state->spmv_delta_primal_solution,
        &HOST_ZERO,
        state->spmv_primal_product,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2,
        state->matA_spmv_workspace_d));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle_At,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE,
        state->matAt,
        state->spmv_delta_dual_solution,
        &HOST_ZERO,
        state->spmv_dual_product,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2,
        state->matAt_spmv_workspace_d));
    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));
}

static void compute_ray_objectives(pdhg_solver_state_t *state)
{
    double *constraint_support_entries_d = state->constraint_reduction_entries_d;
    double *variable_support_entries_d = state->variable_reduction_entries_d;

    CUBLAS_CHECK(cublasDdot(
        state->blas_handle_A,
        state->num_variables,
        state->objective_vector_d, 1,
        state->delta_primal_solution_d, 1,
        state->primal_ray_linear_objective_d));

    compute_box_support_contribution_kernel<<<
        state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        constraint_support_entries_d,
        state->delta_dual_solution_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        1.0,
        state->num_constraints);
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        constraint_support_entries_d,
        state->dual_ray_constraint_support_d,
        state->num_constraints, state->stream_A));

    compute_box_support_contribution_kernel<<<
        state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        variable_support_entries_d,
        state->dual_product_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        -1.0,
        state->num_variables);
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        variable_support_entries_d,
        state->dual_ray_variable_support_d,
        state->num_variables, state->stream_A));
}

static void compute_ray_infeasibility_norms(pdhg_solver_state_t *state)
{
    double *primal_ray_infeasibility_entries_d = state->constraint_reduction_entries_d;
    double *dual_ray_infeasibility_entries_d = state->variable_reduction_entries_d;

    compute_ray_infeasibility_kernel<<<
        state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->primal_product_d,
        state->dual_product_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        primal_ray_infeasibility_entries_d,
        dual_ray_infeasibility_entries_d,
        state->constraint_rescaling_d,
        state->variable_rescaling_d,
        state->num_variables,
        state->num_constraints);
    CUDA_CHECK(cudaGetLastError());

    reduce_device_abs_max(state, primal_ray_infeasibility_entries_d,
                          state->num_constraints,
                          state->max_primal_ray_infeasibility_d);
    reduce_device_abs_max(state, dual_ray_infeasibility_entries_d,
                          state->num_variables,
                          state->max_dual_ray_infeasibility_d);
}

void check_termination_criteria(
    pdhg_solver_state_t *solver_state,
    const termination_criteria_t *criteria)
{
    solver_state->cumulative_time_sec = (double)(clock() - solver_state->start_time) / CLOCKS_PER_SEC;

    double ratio = solver_state->metrics_h->normalized_homogeneous_ratio;
    if (ratio < 0.01)
    {
        if (optimality_criteria_met(solver_state, criteria))
        {
            solver_state->termination_reason = TERMINATION_REASON_OPTIMAL;
            return;
        }
    }
    else if (ratio > 0.9)
    {
        if (primal_infeasibility_criteria_met(solver_state, criteria))
        {
            solver_state->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
            return;
        }
        if (dual_infeasibility_criteria_met(solver_state, criteria))
        {
            solver_state->termination_reason = TERMINATION_REASON_DUAL_INFEASIBLE;
            return;
        }
    }

    if (criteria->enable_ray_infeas)
    {
        if (primal_infeasibility_ray_criteria_met(solver_state, criteria))
        {
            prepare_primal_infeasibility_ray_result(solver_state);
            solver_state->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
            return;
        }
        if (dual_infeasibility_ray_criteria_met(solver_state, criteria))
        {
            prepare_dual_infeasibility_ray_result(solver_state);
            solver_state->termination_reason = TERMINATION_REASON_DUAL_INFEASIBLE;
            return;
        }
    }

    if (solver_state->total_count >= criteria->iteration_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        return;
    }
    if (solver_state->cumulative_time_sec >= criteria->time_sec_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_TIME_LIMIT;
        return;
    }
}

static bool optimality_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria)
{
    return state->metrics_h->relative_primal_residual < criteria->eps_feasible_relative &&
           state->metrics_h->relative_dual_residual   < criteria->eps_feasible_relative &&
           state->metrics_h->relative_objective_gap   < criteria->eps_optimal_relative;
}

static bool primal_infeasibility_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria)
{
    return state->metrics_h->relative_primal_infeasibility_residual  <  criteria->eps_infeas_residual &&
           state->metrics_h->relative_primal_infeasibility_objective >  criteria->eps_infeas_objective;
}

static bool dual_infeasibility_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria)
{
    return state->metrics_h->relative_dual_infeasibility_residual  <  criteria->eps_infeas_residual &&
           state->metrics_h->relative_dual_infeasibility_objective < -criteria->eps_infeas_objective;
}

static bool primal_infeasibility_ray_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria)
{
    if (state->metrics_h->dual_ray_objective <= 0.0)
    {
        return false;
    }
    return state->metrics_h->max_dual_ray_infeasibility /
           state->metrics_h->dual_ray_objective <=
           criteria->eps_infeas_ray;
}

static bool dual_infeasibility_ray_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria)
{
    if (state->metrics_h->primal_ray_linear_objective >= 0.0)
    {
        return false;
    }
    return state->metrics_h->max_primal_ray_infeasibility /
           (-state->metrics_h->primal_ray_linear_objective) <=
           criteria->eps_infeas_ray;
}

static void prepare_primal_infeasibility_ray_result(pdhg_solver_state_t *state)
{
    install_primal_infeasibility_ray_kernel<<<
        state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->pdhg_dual_solution_d,
        state->pdhg_reduced_cost_d,
        state->dual_ray_d,
        state->reduced_cost_ray_d,
        state->delta_dual_solution_d,
        state->dual_product_d,
        state->constraint_rescaling_d,
        state->variable_rescaling_d,
        state->num_constraints,
        state->num_variables);
    CUDA_CHECK(cudaGetLastError());

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A,
        state->num_constraints,
        state->dual_ray_d, 1,
        state->farkas_original_dual_norm_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A,
        state->num_variables,
        state->reduced_cost_ray_d, 1,
        state->farkas_original_reduced_cost_norm_d));
}

static void prepare_dual_infeasibility_ray_result(pdhg_solver_state_t *state)
{
    install_dual_infeasibility_ray_kernel<<<
        state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->pdhg_primal_solution_d,
        state->primal_ray_d,
        state->delta_primal_solution_d,
        state->variable_rescaling_d,
        state->num_variables);
    CUDA_CHECK(cudaGetLastError());

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A,
        state->num_variables,
        state->primal_ray_d, 1,
        state->farkas_original_primal_norm_d));
}

bool should_do_adaptive_restart(pdhg_solver_state_t *solver_state,
                                const pdhg_parameters_t *params)
{
    const restart_parameters_t *restart_params = &params->restart_params;
    int termination_evaluation_frequency = params->termination_evaluation_frequency;

    bool do_restart = false;
    if (solver_state->total_count == termination_evaluation_frequency)
    {
        do_restart = true;
    }
    else if (solver_state->total_count > termination_evaluation_frequency)
    {
        if (*solver_state->fixed_point_error_mp <= restart_params->sufficient_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            do_restart = true;
        }
        if (*solver_state->fixed_point_error_mp <= restart_params->necessary_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            if (*solver_state->fixed_point_error_mp > solver_state->last_trial_fixed_point_error)
            {
                do_restart = true;
            }
        }
        if (solver_state->inner_count >= restart_params->artificial_restart_threshold * solver_state->total_count)
        {
            do_restart = true;
        }
    }
    solver_state->last_trial_fixed_point_error = *solver_state->fixed_point_error_mp;
    return do_restart;
}

cuhsdlp_result_t *create_result_from_state(
    pdhg_solver_state_t *state,
    const lp_problem_t *original_problem)
{
    cuhsdlp_result_t *results = (cuhsdlp_result_t *)safe_calloc(1, sizeof(cuhsdlp_result_t));

    rescale_solution(state);

    results->primal_solution = (double *)safe_malloc(state->var_bytes);
    results->dual_solution = (double *)safe_malloc(state->con_bytes);
    results->reduced_cost = (double *)safe_malloc(state->var_bytes);

    CUDA_CHECK(cudaMemcpyAsync(
        results->primal_solution,
        state->pdhg_primal_solution_d,
        state->var_bytes,
        cudaMemcpyDeviceToHost,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        results->dual_solution,
        state->pdhg_dual_solution_d,
        state->con_bytes,
        cudaMemcpyDeviceToHost,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        results->reduced_cost,
        state->pdhg_reduced_cost_d,
        state->var_bytes,
        cudaMemcpyDeviceToHost,
        state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));

    results->num_variables = original_problem->num_variables;
    results->num_constraints = original_problem->num_constraints;
    results->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;
    results->total_count = state->total_count;
    results->rescaling_time_sec = state->rescaling_time_sec;
    results->cumulative_time_sec = state->cumulative_time_sec;
    results->absolute_primal_residual = state->metrics_h->absolute_primal_residual;
    results->relative_primal_residual = state->metrics_h->relative_primal_residual;
    results->absolute_dual_residual = state->metrics_h->absolute_dual_residual;
    results->relative_dual_residual = state->metrics_h->relative_dual_residual;
    results->primal_objective_value = state->objective_sign * state->metrics_h->primal_objective;
    results->dual_objective_value = state->objective_sign * state->metrics_h->dual_objective;
    results->objective_gap = state->metrics_h->objective_gap;
    results->relative_objective_gap = state->metrics_h->relative_objective_gap;
    results->max_primal_ray_infeasibility = state->metrics_h->max_primal_ray_infeasibility;
    results->max_dual_ray_infeasibility = state->metrics_h->max_dual_ray_infeasibility;
    results->primal_ray_linear_objective = state->metrics_h->primal_ray_linear_objective;
    results->dual_ray_objective = state->metrics_h->dual_ray_objective;
    results->termination_reason = state->termination_reason;

    results->restart_count = state->restart_count;
    results->homogeneous_scale = state->metrics_h->pdhg_primal_scale;
    results->final_step_size_primal = state->step_size_primal;
    results->final_step_size_dual = state->step_size_dual;
    results->final_step_size_homogeneous = state->step_size_homogeneous;

    return results;
}

static void rescale_solution(pdhg_solver_state_t *state)
{
    double farkas_h[FARKAS_COUNT];
    CUDA_CHECK(cudaMemcpy(farkas_h, state->farkas_reduce_d,
                          FARKAS_COUNT * sizeof(double), cudaMemcpyDeviceToHost));

    double divisor;
    switch (state->termination_reason)
    {
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
        divisor = sqrt(
            farkas_h[FARKAS_ORIGINAL_DUAL_NORM] *
                farkas_h[FARKAS_ORIGINAL_DUAL_NORM] +
            farkas_h[FARKAS_ORIGINAL_REDUCED_COST_NORM] *
                farkas_h[FARKAS_ORIGINAL_REDUCED_COST_NORM]);
        break;
    case TERMINATION_REASON_DUAL_INFEASIBLE:
        divisor = farkas_h[FARKAS_ORIGINAL_PRIMAL_NORM];
        break;
    default:
        divisor = state->metrics_h->pdhg_primal_scale;
        break;
    }

    rescale_solution_kernel<<<
        state->num_blocks_rescale,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->pdhg_primal_solution_d,
        state->pdhg_dual_solution_d,
        state->pdhg_reduced_cost_d,
        state->variable_rescaling_d,
        state->constraint_rescaling_d,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
        divisor,
        state->num_variables,
        state->num_constraints);
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
    double *__restrict__ homogeneous_dual_res,
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
        double scaled_reduced_cost = reduced_cost * var_rescaling[idx];
        reduced_cost_out[idx] = reduced_cost;

        double farkas_stationarity_residual = obj_vec[idx] * primal_scale * var_rescaling[idx];
        double reduced_cost_cone_violation = 0.0;
        if (!isfinite(var_ub[idx]))
        {
            reduced_cost_cone_violation += fmax(0.0, -scaled_reduced_cost);
        }
        if (!isfinite(var_lb[idx]))
        {
            reduced_cost_cone_violation += fmax(0.0, scaled_reduced_cost);
        }
        homogeneous_dual_res[idx] = sqrt(farkas_stationarity_residual * farkas_stationarity_residual 
                                       + reduced_cost_cone_violation * reduced_cost_cone_violation);

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

        original_reduced_cost_ray[idx] = scaled_reduced_cost;
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

__global__ void compute_projected_ray_delta_kernel(
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
        double ray_value = pdhg_primal[i] - initial_primal[i];
        if (isfinite(var_lb[i]))
        {
            ray_value = fmax(ray_value, 0.0);
        }
        if (isfinite(var_ub[i]))
        {
            ray_value = fmin(ray_value, 0.0);
        }
        delta_primal[i] = ray_value;
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        double ray_value = pdhg_dual[idx] - initial_dual[idx];
        if (!isfinite(cons_lb[idx]))
        {
            ray_value = fmin(ray_value, 0.0);
        }
        if (!isfinite(cons_ub[idx]))
        {
            ray_value = fmax(ray_value, 0.0);
        }
        delta_dual[idx] = ray_value;
    }
}

__global__ void compute_ray_infeasibility_kernel(
    const double *__restrict__ primal_product,
    const double *__restrict__ dual_product,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    double *__restrict__ primal_infeasibility,
    double *__restrict__ dual_infeasibility,
    const double *__restrict__ cons_rescaling,
    const double *__restrict__ var_rescaling,
    int n_vars,
    int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_cons)
    {
        double v = primal_product[i];
        double lower_bound = isfinite(cons_lb[i]) ? 0.0 : -INFINITY;
        double upper_bound = isfinite(cons_ub[i]) ? 0.0 : INFINITY;
        double projected_value = fmax(lower_bound, fmin(v, upper_bound));
        primal_infeasibility[i] = (v - projected_value) * cons_rescaling[i];
    }
    else if (i < n_cons + n_vars)
    {
        int idx = i - n_cons;
        double v = -dual_product[idx];
        double lower_bound = !isfinite(var_ub[idx]) ? 0.0 : -INFINITY;
        double upper_bound = !isfinite(var_lb[idx]) ? 0.0 : INFINITY;
        double projected_value = fmax(lower_bound, fmin(v, upper_bound));
        dual_infeasibility[idx] = (v - projected_value) * var_rescaling[idx];
    }
}

__global__ void install_primal_infeasibility_ray_kernel(
    double *__restrict__ dual_solution,
    double *__restrict__ reduced_cost,
    double *__restrict__ original_dual_ray,
    double *__restrict__ original_reduced_cost_ray,
    const double *__restrict__ delta_dual,
    const double *__restrict__ dual_product,
    const double *__restrict__ cons_rescaling,
    const double *__restrict__ var_rescaling,
    int n_cons,
    int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_cons)
    {
        double ray_value = delta_dual[i];
        dual_solution[i] = ray_value;
        original_dual_ray[i] = ray_value / cons_rescaling[i];
    }
    else if (i < n_cons + n_vars)
    {
        int idx = i - n_cons;
        double reduced_cost_value = -dual_product[idx];
        reduced_cost[idx] = reduced_cost_value;
        original_reduced_cost_ray[idx] =
            reduced_cost_value * var_rescaling[idx];
    }
}

__global__ void install_dual_infeasibility_ray_kernel(
    double *__restrict__ primal_solution,
    double *__restrict__ original_primal_ray,
    const double *__restrict__ delta_primal,
    const double *__restrict__ var_rescaling,
    int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vars) return;

    double ray_value = delta_primal[i];
    primal_solution[i] = ray_value;
    original_primal_ray[i] = ray_value / var_rescaling[i];
}

__global__ void compute_ray_metrics_kernel(
    pdhg_metrics_t *metrics_d,
    const double *__restrict__ primal_ray_linear_objective,
    const double *__restrict__ constraint_support,
    const double *__restrict__ variable_support,
    const double *__restrict__ primal_ray_infeasibility,
    const double *__restrict__ dual_ray_infeasibility,
    double constraint_bound_rescaling,
    double objective_vector_rescaling)
{
    double scale = constraint_bound_rescaling * objective_vector_rescaling;
    metrics_d->primal_ray_linear_objective = *primal_ray_linear_objective / scale;
    metrics_d->max_primal_ray_infeasibility =
        *primal_ray_infeasibility / constraint_bound_rescaling;

    metrics_d->max_dual_ray_infeasibility =
        *dual_ray_infeasibility / objective_vector_rescaling;
    metrics_d->dual_ray_objective =
        ((*constraint_support) + (*variable_support)) / scale;
}

__global__ void compute_residual_metrics_kernel(
    pdhg_metrics_t *metrics_d,
    const double *primal_norm,
    const double *dual_norm,
    const double *objective_dot_primal,
    const double *farkas_dual_solution_norm,
    const double *farkas_homogeneous_dual_norm,
    const double *farkas_constraint_recession,
    const double *farkas_variable_recession,
    const double *constraint_support,
    const double *variable_support,
    const double *farkas_original_dual_norm,
    const double *farkas_original_reduced_cost_norm,
    const double *farkas_original_primal_norm,
    const double *primal_scale_current,
    const double *primal_scale_pdhg,
    const double *dual_scale_pdhg,
    double constraint_bound_rescaling,
    double objective_vector_rescaling,
    double original_bound_norm,
    double original_objective_norm,
    double original_constraint_norm,
    double objective_constant)
{
    double primal_scale = *primal_scale_pdhg;
    double inv_primal_scale = (fabs(primal_scale) > 1e-50) ? 1.0 / primal_scale : 0.0;
    double pdhg_dual_scale = *dual_scale_pdhg;
    double absolute_primal_residual = (*primal_norm) / constraint_bound_rescaling;
    double absolute_dual_residual = (*dual_norm) / objective_vector_rescaling;
    double objective_dot_primal_value = *objective_dot_primal;
    double support_dual_objective = (*constraint_support) + (*variable_support);

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

    double yc_norm = *farkas_dual_solution_norm;
    double dual_solution_norm = yc_norm;
    double primal_solution_norm = metrics_d->primal_solution_norm;

    metrics_d->dual_solution_norm = dual_solution_norm;

    double homogeneous_gap = objective_dot_primal_value + pdhg_dual_scale - (*variable_support);
    double iterate_norm = sqrt(primal_solution_norm * primal_solution_norm
                             + dual_solution_norm * dual_solution_norm
                             + primal_scale * primal_scale
                             + pdhg_dual_scale * pdhg_dual_scale);
    double inv_iterate_norm = (iterate_norm > 1e-30) ? 1.0 / iterate_norm : 0.0;

    double primal_scale_hat = primal_scale * inv_iterate_norm;
    double homogeneous_gap_hat = fmax(-homogeneous_gap, 0.0) * inv_iterate_norm;
    double ratio_denominator = homogeneous_gap_hat + primal_scale_hat;
    metrics_d->normalized_homogeneous_gap   = homogeneous_gap_hat;
    metrics_d->normalized_homogeneous_ratio = (ratio_denominator > 0.0)
                                            ? homogeneous_gap_hat / ratio_denominator
                                            : 0.0;

    double homogeneous_dual_norm = *farkas_homogeneous_dual_norm;
    double original_dual_norm = *farkas_original_dual_norm;
    double original_reduced_cost_norm = *farkas_original_reduced_cost_norm;
    double original_dual_ray_norm = sqrt(original_dual_norm * original_dual_norm + original_reduced_cost_norm 
                                       * original_reduced_cost_norm);
    double inv_orig_dual_norm = (original_dual_ray_norm > 1.0e-30)
                              ? 1.0 / original_dual_ray_norm
                              : 0.0;
    double abs_p_infeas_residual = homogeneous_dual_norm * inv_orig_dual_norm;
    double rel_p_infeas_residual = abs_p_infeas_residual / (1.0 + original_constraint_norm);
    metrics_d->relative_primal_infeasibility_residual = rel_p_infeas_residual;

    double support_total = ((*constraint_support) + (*variable_support)) / constraint_bound_rescaling;
    double support_normalized = support_total * inv_orig_dual_norm;
    double rel_p_infeas_objective = support_normalized / (1.0 + original_bound_norm);
    metrics_d->relative_primal_infeasibility_objective = rel_p_infeas_objective;

    double orig_primal_ray_norm = *farkas_original_primal_norm;
    double inv_orig_primal_norm = (orig_primal_ray_norm > 1.0e-30) 
                                ? 1.0 / orig_primal_ray_norm 
                                : 0.0;
    double relative_constraint_recession_residual = ((*farkas_constraint_recession) * inv_orig_primal_norm)
                                                  / (1.0 + original_bound_norm);
    double relative_variable_recession_residual = ((*farkas_variable_recession) * inv_orig_primal_norm) 
                                                / (1.0 + original_bound_norm);
    double rel_d_infeas_residual = fmax(relative_constraint_recession_residual,
                                        relative_variable_recession_residual);
    metrics_d->relative_dual_infeasibility_residual = rel_d_infeas_residual;

    double c_dot_x = *objective_dot_primal;
    double rel_d_infeas_objective = (c_dot_x * inv_orig_primal_norm)
                                  / ((1.0 + original_objective_norm) * objective_vector_rescaling);
    metrics_d->relative_dual_infeasibility_objective = rel_d_infeas_objective;
}

__global__ void rescale_solution_kernel(
    double *__restrict__ primal_sol,
    double *__restrict__ dual_sol,
    double *__restrict__ reduced_cost,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ cons_rescaling,
    const double obj_vec_rescaling,
    const double cons_bound_rescaling,
    double divisor,
    int n_vars,
    int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double inv_divisor = (fabs(divisor) > 1e-50) ? 1.0 / divisor : 0.0;
    if (i < n_vars)
    {
        primal_sol[i] = primal_sol[i] * inv_divisor / var_rescaling[i] / cons_bound_rescaling;
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        dual_sol[idx] = dual_sol[idx] * inv_divisor / cons_rescaling[idx] / obj_vec_rescaling;
    }
    else if (i < 2 * n_vars + n_cons)
    {
        int idx = i - n_vars - n_cons;
        reduced_cost[idx] = reduced_cost[idx] * inv_divisor * var_rescaling[idx] / obj_vec_rescaling;
    }
}
