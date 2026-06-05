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
#include "cuhsdlp.h"
#include "display.h"
#include "internal_types.h"
#include "presolver.h"
#include "setup.h"
#include "solver.h"
#include "termination_checker.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__global__ void compute_hsd_unprojected_dual_kernel(
    double *__restrict__ out,
    const double *__restrict__ current,
    const double *__restrict__ gradient,
    double *scalar_out,
    const double *scalar_current,
    const double *scalar_gradient,
    const double *__restrict__ step_size_d,
    int n);
__global__ void update_hsd_dual_solution_kernel(
    double *__restrict__ cur,
    double *__restrict__ refl,
    double *__restrict__ pdhg,
    const double *__restrict__ init,
    const double *__restrict__ unprojected,
    cone_projection_state_t cone,
    const int *__restrict__ base_count_d,
    int k_offset,
    double refl_coeff,
    int n,
    double *offset_initial,
    double *offset_current,
    double *offset_pdhg,
    double *offset_reflected,
    bool is_major);
__global__ void update_hsd_homogeneous_scale_kernel(
    double *scale_initial,
    double *scale_current,
    double *scale_pdhg,
    double *scale_reflected,
    const double *dual_constraint_current,
    const double *dual_bound_current,
    const double *c_dot_x_d,
    const double *__restrict__ primal_step_d,
    const int *__restrict__ inner_count_d,
    int k_offset,
    double reflection_coefficient,
    bool is_major);
__global__ void update_hsd_primal_solution_kernel(
    double *__restrict__ cur,
    double *__restrict__ pdhg,
    double *__restrict__ refl,
    const double *__restrict__ init,
    const double *__restrict__ dual_prod,
    const double *__restrict__ refl_dual_slack,
    const double *__restrict__ obj_vec,
    const double *scale_reflected,
    const double *__restrict__ step_size_d,
    const int *__restrict__ base_count_d,
    int k_offset,
    double refl_coeff,
    int n,
    bool is_major);
__global__ void rescale_solution_kernel(
    double *__restrict__ primal_sol,
    double *__restrict__ dual_sol,
    double *__restrict__ dual_slack,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ cons_rescaling,
    const double obj_vec_rescaling,
    const double cons_bound_rescaling,
    double tau,
    int n_vars,
    int n_cons);
__global__ void compute_delta_kernel(
    const double *__restrict__ initial_primal,
    const double *__restrict__ pdhg_primal,
    double *__restrict__ delta_primal,
    const double *__restrict__ initial_dual,
    const double *__restrict__ pdhg_dual,
    double *__restrict__ delta_dual,
    const double *__restrict__ initial_dual_slack,
    const double *__restrict__ pdhg_dual_slack,
    double *__restrict__ delta_dual_slack,
    int n_vars,
    int n_cons);
__global__ void compute_fixed_point_error_kernel(
    double *fixed_point_error_d,
    const double *primal_norm,
    const double *dual_norm,
    const double *dual_cross,
    const double *slack_norm,
    const double *slack_cross,
    const double *scale_pdhg,
    const double *scale_reflected,
    const double *dual_constraint_pdhg,
    const double *dual_constraint_reflected,
    const double *dual_bound_pdhg,
    const double *dual_bound_reflected,
    double step_size_primal,
    double step_size_dual,
    double step_size_homogeneous);
__global__ void restart_hsd_scalars_kernel(
    double *scale_initial,           double *scale_current,           double *scale_pdhg,           double *scale_reflected,
    double *dual_constraint_initial, double *dual_constraint_current, double *dual_constraint_pdhg, double *dual_constraint_reflected,
    double *dual_bound_initial,      double *dual_bound_current,      double *dual_bound_pdhg,      double *dual_bound_reflected,
    double *constraint_cone_scale_d,
    double *bound_cone_scale_d,
    double  cone_warm_start_scale);
__global__ void normalize_hsd_scalars_kernel(
    double *scale_pdhg,
    double *dual_constraint_pdhg,
    double *dual_bound_pdhg,
    double  inv_scale);
static void compute_fixed_point_error(pdhg_solver_state_t *state);
static void hsd_run_iteration_step(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    int k_offset,
    bool is_major);
static void initialize_step_size_and_primal_weight(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params);
void hsd_update_homogeneous_scale(
    pdhg_solver_state_t *state,
    const int k_offset,
    const double reflection_coefficient,
    bool is_major);
void hsd_update_dual_solution(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    const int k_offset,
    bool is_major);
void hsd_update_primal_solution(
    pdhg_solver_state_t *state,
    const int k_offset,
    const double reflection_coefficient,
    bool is_major);
static void sync_inner_count_to_gpu(pdhg_solver_state_t *state);
static void sync_step_sizes_to_gpu(pdhg_solver_state_t *state);
static void perform_restart(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params);
static void update_primal_weight_via_restart_pid(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params);
static void normalize_hsd_variables(pdhg_solver_state_t *state);
static void rescale_solution(pdhg_solver_state_t *state);
static cuhsdlp_result_t *create_result_from_state(
    pdhg_solver_state_t *state,
    const lp_problem_t *original_problem);

cuhsdlp_result_t *optimize(
    const pdhg_parameters_t *params,
    lp_problem_t *original_problem)
{
    check_params_validity(params);
    filter_constraint_matrix_entries(original_problem, params);
    print_initial_info(params, original_problem);

    cuhsdlp_presolve_info_t *presolve_info = NULL;
    lp_problem_t *working_problem = original_problem;

    if (params->presolve)
    {
        presolve_info = pslp_presolve(original_problem, params->verbose);
        if (presolve_info->problem_solved_during_presolve)
        {
            cuhsdlp_result_t *result = create_result_from_presolve(presolve_info, original_problem);
            cuhsdlp_presolve_info_free(presolve_info);
            pdhg_final_log(result, params);
            return result;
        }
        working_problem = presolve_info->reduced_problem;
    }

    pdhg_solver_state_t *state = initialize_solver_state(working_problem, original_problem, params);
    display_iteration_stats(state, params->verbose, false);

    initialize_step_size_and_primal_weight(state, params);

    state->start_time = clock();
    bool do_restart = false;

    while (state->total_count < params->termination_criteria.iteration_limit)
    {
        sync_inner_count_to_gpu(state);
        sync_step_sizes_to_gpu(state);

        hsd_run_iteration_step(state, params, 1, true);

        if (do_restart)
        {
            compute_fixed_point_error(state);
            CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
            state->initial_fixed_point_error = *state->fixed_point_error_mp;
            do_restart = false;
        }

        if (!state->graph_created)
        {
            cudaGraph_t graph = NULL;
            CUDA_CHECK(cudaStreamBeginCapture(state->stream_A,
                                              cudaStreamCaptureModeGlobal));

            for (int i = 2; i <= params->termination_evaluation_frequency - 1; i++)
            {
                hsd_run_iteration_step(state, params, i, false);
            }
            hsd_run_iteration_step(state, params, params->termination_evaluation_frequency, true);

            CUDA_CHECK(cudaStreamEndCapture(state->stream_A, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&state->graphExec, graph,
                                            NULL, NULL, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            state->graph_created = true;
        }
        CUDA_CHECK(cudaGraphLaunch(state->graphExec, state->stream_A));

        compute_fixed_point_error(state);
        compute_residual(state, params->optimality_norm);
        state->inner_count += params->termination_evaluation_frequency;
        state->total_count += params->termination_evaluation_frequency;

        check_termination_criteria(state, &params->termination_criteria);
        if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
        {
            display_iteration_stats(state, params->verbose, false);
            break;
        }

        do_restart = should_do_adaptive_restart(state, params);
        if (do_restart)
        {
            state->restart_count++;
            display_iteration_stats(state, params->verbose, true);
            perform_restart(state, &params->step_size_params);
        }
        else
        {
            display_iteration_stats(state, params->verbose, false);
        }
    }

    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_residual(state, params->optimality_norm);
        display_iteration_stats(state, params->verbose, false);
    }

    cuhsdlp_result_t *result = create_result_from_state(state, original_problem);

    if (params->presolve && presolve_info)
    {
        pslp_postsolve(presolve_info, result, original_problem);
        cuhsdlp_presolve_info_free(presolve_info);
    }

    pdhg_final_log(result, params);
    pdhg_solver_state_free(state);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

static void sync_inner_count_to_gpu(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpyAsync(state->inner_count_d,
                               &state->inner_count,
                               sizeof(int),
                               cudaMemcpyHostToDevice,
                               state->stream_A));
}

static void sync_step_sizes_to_gpu(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpyAsync(state->step_size_primal_d,
                               &state->step_size_primal,
                               sizeof(double),
                               cudaMemcpyHostToDevice,
                               state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(state->step_size_dual_d,
                               &state->step_size_dual,
                               sizeof(double),
                               cudaMemcpyHostToDevice,
                               state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(state->step_size_homogeneous_d,
                               &state->step_size_homogeneous,
                               sizeof(double),
                               cudaMemcpyHostToDevice,
                               state->stream_A));
}

static void compute_step_sizes(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params)
{
    double omega = state->primal_weight;
    double base = state->step_size_base;
    double c_norm = state->rescaled_objective_norm;
    double factor = step_size_params->step_size_factor;
    double xi = step_size_params->objective_coupling_budget;

    state->step_size_primal = base / omega;
    state->step_size_dual = base * omega;
    state->step_size_homogeneous =
        factor * xi / (c_norm * c_norm * base / omega + 2.0 * base * omega);
}

static void initialize_step_size_and_primal_weight(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    double sigma = state->rescaled_constraint_norm;
    double factor = params->step_size_params.step_size_factor;
    double xi = params->step_size_params.objective_coupling_budget;

    state->step_size_base = factor * sqrt((1.0 - xi) / (sigma * sigma + 1.0));

    if (params->scaling_params.bound_objective_rescaling)
    {
        state->primal_weight = 1.0;
    }
    else
    {
        state->primal_weight =
            (state->rescaled_objective_norm + 1.0) / (state->rescaled_bound_norm + 1.0);
    }

    compute_step_sizes(state, &params->step_size_params);

    state->best_primal_weight = state->primal_weight;
}

static void hsd_run_iteration_step(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    int k_offset,
    bool is_major)
{
    double reflection_coefficient = params->reflection_coefficient;
    hsd_update_homogeneous_scale(state, k_offset, reflection_coefficient, is_major);
    hsd_update_dual_solution(state, params, k_offset, is_major);
    hsd_update_primal_solution(state, k_offset, reflection_coefficient, is_major);
}

void hsd_update_homogeneous_scale(
    pdhg_solver_state_t *state,
    const int k_offset,
    const double reflection_coefficient,
    bool is_major)
{
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle_A,
        state->num_variables,
        state->objective_vector_d,
        1,
        state->current_primal_solution_d,
        1,
        state->pdhg_primal_cross_d));

    update_hsd_homogeneous_scale_kernel<<<1, 1, 0, state->stream_A>>>(
        state->hsd_scale_initial_d,
        state->hsd_scale_current_d,
        state->hsd_scale_pdhg_d,
        state->hsd_scale_reflected_d,
        state->hsd_dual_constraint_current_d,
        state->hsd_dual_bound_current_d,
        state->pdhg_primal_cross_d,
        state->step_size_homogeneous_d,
        state->inner_count_d,
        k_offset,
        reflection_coefficient,
        is_major);

    CUDA_CHECK(cudaEventRecord(state->event_s, state->stream_A));
}

void hsd_update_dual_solution(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    const int k_offset,
    bool is_major)
{
    const cone_parameters_t *cone_params = &params->cone_params;
    const double reflection_coefficient = params->reflection_coefficient;

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle_A,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE,
        state->matA,
        state->spmv_current_primal_solution,
        &HOST_ZERO,
        state->spmv_primal_product,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2,
        state->primal_spmv_buffer_d));

    compute_hsd_unprojected_dual_kernel<<<
        state->num_blocks_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->dual_vec_buffer_d,
        state->current_dual_solution_d,
        state->primal_product_d,
        state->constraint_cone_state.unprojected_scale_d,
        state->hsd_dual_constraint_current_d,
        state->hsd_scale_reflected_d,
        state->step_size_dual_d,
        state->num_constraints);

    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_s, 0));

    compute_hsd_unprojected_dual_kernel<<<
        state->num_blocks_primal,
        THREADS_PER_BLOCK,
        0,
        state->stream_At>>>(
        state->primal_vec_buffer_d,
        state->current_dual_slack_d,
        state->current_primal_solution_d,
        state->bound_cone_state.unprojected_scale_d,
        state->hsd_dual_bound_current_d,
        state->hsd_scale_reflected_d,
        state->step_size_dual_d,
        state->num_variables);

    solve_cone_newton(
        cone_params,
        &state->constraint_cone_state,
        state->dual_vec_buffer_d,
        state->stream_A);

    solve_cone_newton(
        cone_params,
        &state->bound_cone_state,
        state->primal_vec_buffer_d,
        state->stream_At);

    update_hsd_dual_solution_kernel<<<
        state->num_blocks_primal,
        THREADS_PER_BLOCK,
        0,
        state->stream_At>>>(
        state->current_dual_slack_d,
        state->reflected_dual_slack_d,
        state->pdhg_dual_slack_d,
        state->initial_dual_slack_d,
        state->primal_vec_buffer_d,
        state->bound_cone_state,
        state->inner_count_d,
        k_offset,
        reflection_coefficient,
        state->num_variables,
        state->hsd_dual_bound_initial_d,
        state->hsd_dual_bound_current_d,
        state->hsd_dual_bound_pdhg_d,
        state->hsd_dual_bound_reflected_d,
        is_major);

    update_hsd_dual_solution_kernel<<<
        state->num_blocks_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->current_dual_solution_d,
        state->reflected_dual_solution_d,
        state->pdhg_dual_solution_d,
        state->initial_dual_solution_d,
        state->dual_vec_buffer_d,
        state->constraint_cone_state,
        state->inner_count_d,
        k_offset,
        reflection_coefficient,
        state->num_constraints,
        state->hsd_dual_constraint_initial_d,
        state->hsd_dual_constraint_current_d,
        state->hsd_dual_constraint_pdhg_d,
        state->hsd_dual_constraint_reflected_d,
        is_major);

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
}

void hsd_update_primal_solution(
    pdhg_solver_state_t *state,
    const int k_offset,
    const double reflection_coefficient,
    bool is_major)
{
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle_At,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE,
        state->matAt,
        state->spmv_reflected_dual_solution,
        &HOST_ZERO,
        state->spmv_dual_product,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2,
        state->dual_spmv_buffer_d));

    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));

    update_hsd_primal_solution_kernel<<<
        state->num_blocks_primal,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->current_primal_solution_d,
        state->pdhg_primal_solution_d,
        state->reflected_primal_solution_d,
        state->initial_primal_solution_d,
        state->dual_product_d,
        state->reflected_dual_slack_d,
        state->objective_vector_d,
        state->hsd_scale_reflected_d,
        state->step_size_primal_d,
        state->inner_count_d,
        k_offset,
        reflection_coefficient,
        state->num_variables,
        is_major);
}

static void compute_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_kernel<<<
        state->num_blocks_primal_dual_slack,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->pdhg_primal_solution_d,
        state->reflected_primal_solution_d,
        state->delta_primal_solution_d,
        state->pdhg_dual_solution_d,
        state->reflected_dual_solution_d,
        state->delta_dual_solution_d,
        state->pdhg_dual_slack_d,
        state->reflected_dual_slack_d,
        state->delta_dual_slack_d,
        state->num_variables,
        state->num_constraints);

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));

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
        state->dual_spmv_buffer_d));

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A,
        state->num_variables,
        state->delta_primal_solution_d,
        1,
        state->pdhg_primal_norm_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A,
        state->num_constraints,
        state->delta_dual_solution_d,
        1,
        state->pdhg_dual_norm_d));

    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));

    CUBLAS_CHECK(cublasDdot(
        state->blas_handle_A,
        state->num_variables,
        state->dual_product_d,
        1,
        state->delta_primal_solution_d,
        1,
        state->pdhg_dual_cross_d));

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A,
        state->num_variables,
        state->delta_dual_slack_d,
        1,
        state->pdhg_slack_norm_d));

    CUBLAS_CHECK(cublasDdot(
        state->blas_handle_A,
        state->num_variables,
        state->delta_dual_slack_d,
        1,
        state->delta_primal_solution_d,
        1,
        state->pdhg_slack_cross_d));

    compute_fixed_point_error_kernel<<<1, 1, 0, state->stream_A>>>(
        state->fixed_point_error_mp,
        state->pdhg_primal_norm_d,
        state->pdhg_dual_norm_d,
        state->pdhg_dual_cross_d,
        state->pdhg_slack_norm_d,
        state->pdhg_slack_cross_d,
        state->hsd_scale_pdhg_d,
        state->hsd_scale_reflected_d,
        state->hsd_dual_constraint_pdhg_d,
        state->hsd_dual_constraint_reflected_d,
        state->hsd_dual_bound_pdhg_d,
        state->hsd_dual_bound_reflected_d,
        state->step_size_primal,
        state->step_size_dual,
        state->step_size_homogeneous);

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));
}

static void perform_restart(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params)
{
    compute_delta_kernel<<<
        state->num_blocks_primal_dual_slack,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->initial_primal_solution_d,
        state->pdhg_primal_solution_d,
        state->delta_primal_solution_d,
        state->initial_dual_solution_d,
        state->pdhg_dual_solution_d,
        state->delta_dual_solution_d,
        state->initial_dual_slack_d,
        state->pdhg_dual_slack_d,
        state->delta_dual_slack_d,
        state->num_variables,
        state->num_constraints);

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A,
        state->num_variables,
        state->delta_primal_solution_d,
        1,
        state->pdhg_primal_norm_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A,
        state->num_constraints,
        state->delta_dual_solution_d,
        1,
        state->pdhg_dual_norm_d));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A,
        state->num_variables,
        state->delta_dual_slack_d,
        1,
        state->pdhg_slack_norm_d));

    update_primal_weight_via_restart_pid(state, step_size_params);

    compute_step_sizes(state, step_size_params);

    normalize_hsd_variables(state);

    CUDA_CHECK(cudaMemcpyAsync(
        state->initial_primal_solution_d,
        state->pdhg_primal_solution_d,
        state->var_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        state->current_primal_solution_d,
        state->pdhg_primal_solution_d,
        state->var_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        state->initial_dual_solution_d,
        state->pdhg_dual_solution_d,
        state->con_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        state->current_dual_solution_d,
        state->pdhg_dual_solution_d,
        state->con_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        state->initial_dual_slack_d,
        state->pdhg_dual_slack_d,
        state->var_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        state->current_dual_slack_d,
        state->pdhg_dual_slack_d,
        state->var_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));

    state->metrics_h->current_homogeneous_scale = state->metrics_h->pdhg_homogeneous_scale;

    restart_hsd_scalars_kernel<<<1, 1, 0, state->stream_A>>>(
        state->hsd_scale_initial_d,             state->hsd_scale_current_d,             state->hsd_scale_pdhg_d,             state->hsd_scale_reflected_d,
        state->hsd_dual_constraint_initial_d,   state->hsd_dual_constraint_current_d,   state->hsd_dual_constraint_pdhg_d,   state->hsd_dual_constraint_reflected_d,
        state->hsd_dual_bound_initial_d,        state->hsd_dual_bound_current_d,        state->hsd_dual_bound_pdhg_d,        state->hsd_dual_bound_reflected_d,
        state->constraint_cone_state.scale_d,
        state->bound_cone_state.scale_d,
        state->cone_warm_start_scale);

    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

static void apply_pid_step(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params,
    double primal_dist,
    double dual_dist,
    bool clamp_step)
{
    double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);
    state->primal_weight_error_sum *= step_size_params->i_smooth;
    state->primal_weight_error_sum += error;
    double delta_error = error - state->primal_weight_last_error;
    double step = step_size_params->k_p * error +
                  step_size_params->k_i * state->primal_weight_error_sum +
                  step_size_params->k_d * delta_error;
    if (clamp_step)
    {
        const double max_log_step = log(10.0);
        step = fmin(fmax(step, -max_log_step), max_log_step);
    }
    state->primal_weight *= exp(step);
    state->primal_weight_last_error = error;
}

static void update_primal_weight_via_restart_pid(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params)
{
    double restart_scalars_h[3];
    double hsd_scalars_h[HSD_SCALARS_COUNT];

    CUDA_CHECK(cudaMemcpyAsync(
        restart_scalars_h,
        state->pdhg_primal_norm_d,
        3 * sizeof(double),
        cudaMemcpyDeviceToHost,
        state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(
        hsd_scalars_h,
        state->hsd_scalars_d,
        HSD_SCALARS_COUNT * sizeof(double),
        cudaMemcpyDeviceToHost,
        state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));

    double dx2     = restart_scalars_h[0] * restart_scalars_h[0];
    double dy2_ds2 = restart_scalars_h[1] * restart_scalars_h[1]
                   + restart_scalars_h[2] * restart_scalars_h[2];

    double d_tau   = hsd_scalars_h[HSD_SCALE_PDHG]
                   - hsd_scalars_h[HSD_SCALE_INITIAL];
    double d_ac    = hsd_scalars_h[HSD_DUAL_CONSTRAINT_PDHG]
                   - hsd_scalars_h[HSD_DUAL_CONSTRAINT_INITIAL];
    double d_av    = hsd_scalars_h[HSD_DUAL_BOUND_PDHG]
                   - hsd_scalars_h[HSD_DUAL_BOUND_INITIAL];

    double c_norm  = state->rescaled_objective_norm;
    double xi      = step_size_params->objective_coupling_budget;
    double base2   = state->step_size_base * state->step_size_base;
    double alpha_p = 2.0 * base2;
    double alpha_d = c_norm * c_norm * base2 / xi;
    double d_tau2  = d_tau * d_tau;

    double primal_dist = sqrt(dx2 + alpha_p * d_tau2);
    double dual_dist   = sqrt(dy2_ds2 + d_ac * d_ac + d_av * d_av + alpha_d * d_tau2);

    double ratio_infeas = state->metrics_h->relative_dual_residual 
                        / state->metrics_h->relative_primal_residual;

    bool dist_ok  = primal_dist > 1e-16 && dual_dist > 1e-16 &&
                    primal_dist < 1e12  && dual_dist < 1e12;
    bool ratio_ok = ratio_infeas > 1e-8 && ratio_infeas < 1e8;
    bool signal_bad = !dist_ok || !ratio_ok;

    const int PID_OOB_STREAK_THRESHOLD = 3;
    if (!signal_bad)
    {
        apply_pid_step(state, step_size_params, primal_dist, dual_dist, false);
        state->pid_has_succeeded = true;
        state->pid_oob_streak = 0;
    }
    else if (!state->pid_has_succeeded)
    {
        apply_pid_step(state, step_size_params, primal_dist, dual_dist, true);
    }
    else
    {
        state->pid_oob_streak++;
        if (state->pid_oob_streak >= PID_OOB_STREAK_THRESHOLD)
        {
            apply_pid_step(state, step_size_params, primal_dist, dual_dist, true);
        }
        else
        {
            state->primal_weight = state->best_primal_weight;
            state->primal_weight_error_sum = 0.0;
            state->primal_weight_last_error = 0.0;
        }
    }

    double primal_dual_residual_gap = fabs(log10(ratio_infeas));
    if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
    {
        state->best_primal_dual_residual_gap = primal_dual_residual_gap;
        state->best_primal_weight = state->primal_weight;
    }
}

static void normalize_hsd_variables(pdhg_solver_state_t *state)
{
    double tau = state->metrics_h->pdhg_homogeneous_scale;
    if (tau >= 0.1 && tau <= 10.0)
    {
        state->cone_warm_start_scale = 1.0;
        return;
    }

    double inv_scale;
    double iterate_block_norm = fmax(fmax(state->metrics_h->primal_solution_norm,
                                          state->metrics_h->dual_solution_norm),
                                fmax(fabs(tau),
                                fmax(fabs(state->metrics_h->pdhg_dual_constraint_offset),
                                     fabs(state->metrics_h->pdhg_dual_bound_offset))));
    if (tau > 1.0e-3)
        inv_scale = 1.0 / tau;
    else if (iterate_block_norm > 1.0e12)
        inv_scale = 1.0 / iterate_block_norm;
    else
        inv_scale = 10.0;

    CUDA_CHECK(cudaMemcpyAsync(
        state->hsd_inverse_scale_d,
        &inv_scale, sizeof(double),
        cudaMemcpyHostToDevice,
        state->stream_A));
    CUBLAS_CHECK(cublasDscal(
        state->blas_handle_A,
        state->num_variables,
        state->hsd_inverse_scale_d,
        state->pdhg_primal_solution_d, 1));
    CUBLAS_CHECK(cublasDscal(
        state->blas_handle_A,
        state->num_constraints,
        state->hsd_inverse_scale_d,
        state->pdhg_dual_solution_d, 1));
    CUBLAS_CHECK(cublasDscal(
        state->blas_handle_A,
        state->num_variables,
        state->hsd_inverse_scale_d,
        state->pdhg_dual_slack_d, 1));
    normalize_hsd_scalars_kernel<<<1, 1, 0, state->stream_A>>>(
        state->hsd_scale_pdhg_d,
        state->hsd_dual_constraint_pdhg_d,
        state->hsd_dual_bound_pdhg_d,
        inv_scale);
    state->metrics_h->pdhg_homogeneous_scale      *= inv_scale;
    state->metrics_h->pdhg_dual_constraint_offset *= inv_scale;
    state->metrics_h->pdhg_dual_bound_offset      *= inv_scale;
    state->cone_warm_start_scale = inv_scale;
}

static cuhsdlp_result_t *create_result_from_state(
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
        state->pdhg_dual_slack_d,
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
    results->primal_objective_value = state->metrics_h->primal_objective;
    results->dual_objective_value = state->metrics_h->dual_objective;
    results->objective_gap = state->metrics_h->objective_gap;
    results->relative_objective_gap = state->metrics_h->relative_objective_gap;
    results->termination_reason = state->termination_reason;

    results->restart_count = state->restart_count;
    results->homogeneous_scale = state->metrics_h->pdhg_homogeneous_scale;
    results->final_step_size_primal = state->step_size_primal;
    results->final_step_size_dual = state->step_size_dual;
    results->final_step_size_homogeneous = state->step_size_homogeneous;

    return results;
}

static void rescale_solution(pdhg_solver_state_t *state)
{
    rescale_solution_kernel<<<
        state->num_blocks_primal_dual_slack,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->pdhg_primal_solution_d,
        state->pdhg_dual_solution_d,
        state->pdhg_dual_slack_d,
        state->variable_rescaling_d,
        state->constraint_rescaling_d,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
        state->metrics_h->pdhg_homogeneous_scale,
        state->num_variables,
        state->num_constraints);
}

__global__ void compute_hsd_unprojected_dual_kernel(
    double *__restrict__ out,
    const double *__restrict__ current,
    const double *__restrict__ gradient,
    double *scalar_out,
    const double *scalar_current,
    const double *scalar_gradient,
    const double *__restrict__ step_size_d,
    int n)
{
    double step_size = *step_size_d;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = current[i] - step_size * gradient[i];
    else if (i == n)
        *scalar_out = (*scalar_current) - step_size * (*scalar_gradient);
}

__global__ void update_hsd_dual_solution_kernel(
    double *__restrict__ cur,
    double *__restrict__ refl,
    double *__restrict__ pdhg,
    const double *__restrict__ init,
    const double *__restrict__ unprojected,
    cone_projection_state_t cone,
    const int *__restrict__ base_count_d,
    int k_offset,
    double refl_coeff,
    int n,
    double *offset_initial,
    double *offset_current,
    double *offset_pdhg,
    double *offset_reflected,
    bool is_major)
{
    int base_count = *base_count_d;
    double tau_star = *cone.scale_d;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int k = base_count + k_offset;
        double weight = (double)(k) / (double)(k + 1);

        double u = unprojected[i];
        double li = cone.lower_bound_d[i];
        double ui = cone.upper_bound_d[i];
        double box_lower = isfinite(li) ? li * tau_star : -INFINITY;
        double box_upper = isfinite(ui) ? ui * tau_star : INFINITY;
        double proj_i = u + fmin(fmax(-u, box_lower), box_upper);

        if (is_major)
        {
            pdhg[i] = proj_i;
        }

        refl[i] = 2.0 * proj_i - cur[i];
        double mixed = refl_coeff * refl[i] + (1.0 - refl_coeff) * cur[i];
        cur[i] = weight * mixed + (1.0 - weight) * init[i];
    }
    if (i == n)
    {
        int k = base_count + k_offset;
        double weight = (double)(k) / (double)(k + 1);
        double unprojected_scalar = *cone.unprojected_scale_d;
        double projected = unprojected_scalar + tau_star;
        if (is_major)
        {
            *offset_pdhg = projected;
        }
        double reflected = 2.0 * projected - (*offset_current);
        double mixed = refl_coeff * reflected + (1.0 - refl_coeff) * (*offset_current);
        *offset_current = weight * mixed + (1.0 - weight) * (*offset_initial);
        *offset_reflected = reflected;
    }
}

__global__ void update_hsd_homogeneous_scale_kernel(
    double *scale_initial,
    double *scale_current,
    double *scale_pdhg,
    double *scale_reflected,
    const double *dual_constraint_current,
    const double *dual_bound_current,
    const double *c_dot_x_d,
    const double *__restrict__ primal_step_d,
    const int *__restrict__ inner_count_d,
    int k_offset,
    double reflection_coefficient,
    bool is_major)
{
    double primal_step = *primal_step_d;
    int    inner_count = *inner_count_d;
    double c_dot_x = c_dot_x_d[0];
    double gradient = c_dot_x + (*dual_constraint_current) + (*dual_bound_current);
    double projected = fmax(0.0, (*scale_current) + primal_step * gradient);

    if (is_major)
    {
        *scale_pdhg = projected;
    }

    int k = inner_count + k_offset;
    double weight = (double)(k) / (double)(k + 1);
    double reflected = 2.0 * projected - (*scale_current);
    double mixed =
        reflection_coefficient * reflected + (1.0 - reflection_coefficient) * (*scale_current);
    *scale_current = weight * mixed + (1.0 - weight) * (*scale_initial);
    *scale_reflected = reflected;
}

__global__ void update_hsd_primal_solution_kernel(
    double *__restrict__ cur,
    double *__restrict__ pdhg,
    double *__restrict__ refl,
    const double *__restrict__ init,
    const double *__restrict__ dual_prod,
    const double *__restrict__ refl_dual_slack,
    const double *__restrict__ obj_vec,
    const double *scale_reflected,
    const double *__restrict__ step_size_d,
    const int *__restrict__ base_count_d,
    int k_offset,
    double refl_coeff,
    int n,
    bool is_major)
{
    double step_size = *step_size_d;
    int    base_count = *base_count_d;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double tau_refl = *scale_reflected;
    int k = base_count + k_offset;
    double weight = (double)(k) / (double)(k + 1);

    double grad = obj_vec[i] * tau_refl - dual_prod[i] - refl_dual_slack[i];
    double next = cur[i] - step_size * grad;

    if (is_major)
    {
        pdhg[i] = next;
    }

    refl[i] = 2.0 * next - cur[i];
    double mixed = refl_coeff * refl[i] + (1.0 - refl_coeff) * cur[i];
    cur[i] = weight * mixed + (1.0 - weight) * init[i];
}

__global__ void rescale_solution_kernel(
    double *__restrict__ primal_sol,
    double *__restrict__ dual_sol,
    double *__restrict__ dual_slack,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ cons_rescaling,
    const double obj_vec_rescaling,
    const double cons_bound_rescaling,
    double tau,
    int n_vars,
    int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double inv_tau = (fabs(tau) > 1e-50) ? 1.0 / tau : 0.0;
    if (i < n_vars)
    {
        primal_sol[i] = primal_sol[i] * inv_tau / var_rescaling[i] / cons_bound_rescaling;
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        dual_sol[idx] = dual_sol[idx] * inv_tau / cons_rescaling[idx] / obj_vec_rescaling;
    }
    else if (i < 2 * n_vars + n_cons)
    {
        int idx = i - n_vars - n_cons;
        dual_slack[idx] = dual_slack[idx] * inv_tau * var_rescaling[idx] / obj_vec_rescaling;
    }
}

__global__ void compute_delta_kernel(
    const double *__restrict__ initial_primal,
    const double *__restrict__ pdhg_primal,
    double *__restrict__ delta_primal,
    const double *__restrict__ initial_dual,
    const double *__restrict__ pdhg_dual,
    double *__restrict__ delta_dual,
    const double *__restrict__ initial_dual_slack,
    const double *__restrict__ pdhg_dual_slack,
    double *__restrict__ delta_dual_slack,
    int n_vars,
    int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = pdhg_primal[i] - initial_primal[i];
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        delta_dual[idx] = pdhg_dual[idx] - initial_dual[idx];
    }
    else if (i < 2 * n_vars + n_cons)
    {
        int idx = i - n_vars - n_cons;
        delta_dual_slack[idx] = pdhg_dual_slack[idx] - initial_dual_slack[idx];
    }
}

__global__ void compute_fixed_point_error_kernel(
    double *fixed_point_error_d,
    const double *primal_norm,
    const double *dual_norm,
    const double *dual_cross,
    const double *slack_norm,
    const double *slack_cross,
    const double *scale_pdhg,
    const double *scale_reflected,
    const double *dual_constraint_pdhg,
    const double *dual_constraint_reflected,
    const double *dual_bound_pdhg,
    const double *dual_bound_reflected,
    double step_size_primal,
    double step_size_dual,
    double step_size_homogeneous)
{
    double delta_primal_scale = (*scale_reflected) - (*scale_pdhg);
    double delta_constraint_offset = (*dual_constraint_reflected) - (*dual_constraint_pdhg);
    double delta_bound_offset = (*dual_bound_reflected) - (*dual_bound_pdhg);
    double primal_norm_v = *primal_norm;
    double dual_norm_v = *dual_norm;
    double cross_term = *dual_cross;
    double slack_norm_v = *slack_norm;
    double slack_cross_v = *slack_cross;

    double dual_movement =
        (dual_norm_v * dual_norm_v +
         slack_norm_v * slack_norm_v +
         delta_constraint_offset * delta_constraint_offset +
         delta_bound_offset * delta_bound_offset) / step_size_dual;
    double movement =
        primal_norm_v * primal_norm_v / step_size_primal +
        dual_movement +
        delta_primal_scale * delta_primal_scale / step_size_homogeneous;
    double interaction = 2.0 * (cross_term - slack_cross_v);

    fixed_point_error_d[0] = sqrt(fmax(0.0, movement + interaction));
}

__global__ void restart_hsd_scalars_kernel(
    double *scale_initial,           double *scale_current,           double *scale_pdhg,           double *scale_reflected,
    double *dual_constraint_initial, double *dual_constraint_current, double *dual_constraint_pdhg, double *dual_constraint_reflected,
    double *dual_bound_initial,      double *dual_bound_current,      double *dual_bound_pdhg,      double *dual_bound_reflected,
    double *constraint_cone_scale_d,
    double *bound_cone_scale_d,
    double  cone_warm_start_scale)
{
    double s = *scale_pdhg;
    *scale_initial   = s;
    *scale_current   = s;
    *scale_reflected = s;

    double dc = *dual_constraint_pdhg;
    *dual_constraint_initial   = dc;
    *dual_constraint_current   = dc;
    *dual_constraint_reflected = dc;

    double db = *dual_bound_pdhg;
    *dual_bound_initial   = db;
    *dual_bound_current   = db;
    *dual_bound_reflected = db;

    *constraint_cone_scale_d *= cone_warm_start_scale;
    *bound_cone_scale_d      *= cone_warm_start_scale;
}

__global__ void normalize_hsd_scalars_kernel(
    double *scale_pdhg,
    double *dual_constraint_pdhg,
    double *dual_bound_pdhg,
    double  inv_scale)
{
    *scale_pdhg           *= inv_scale;
    *dual_constraint_pdhg *= inv_scale;
    *dual_bound_pdhg      *= inv_scale;
}
