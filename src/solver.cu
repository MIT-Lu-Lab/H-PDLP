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
#include "pid_controller.h"
#include "presolver.h"
#include "result.h"
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

__global__ void compute_fixed_point_error_kernel(
    double *fixed_point_error_d,
    const double *primal_norm,
    const double *dual_norm,
    const double *dual_cross,
    const double *primal_scale_pdhg,
    const double *primal_scale_reflected,
    const double *dual_scale_pdhg,
    const double *dual_scale_reflected,
    double step_size_primal,
    double step_size_dual,
    double step_size_homogeneous);
__global__ void restart_hsd_scalars_kernel(
    double *primal_scale_initial,           double *primal_scale_current,           double *primal_scale_pdhg,           double *primal_scale_reflected,
    double *dual_scale_initial, double *dual_scale_current, double *dual_scale_pdhg, double *dual_scale_reflected,
    double *constraint_cone_scale_d,
    double  cone_warm_start_scale);
__global__ void scale_hsd_scalars_kernel(
    double *primal_scale_pdhg,
    double *dual_scale_pdhg,
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
static void hsd_update_dual_solution(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    const int k_offset,
    bool is_major);
static void update_hsd_primal_block(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    const int k_offset,
    const double reflection_coefficient,
    bool is_major);
static void sync_inner_count_to_gpu(pdhg_solver_state_t *state);
static void sync_step_sizes_to_gpu(pdhg_solver_state_t *state);
static bool should_do_adaptive_restart(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params);
static void perform_restart(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params);
static void update_primal_weight_via_restart_pid(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params);
static double apply_gauge_control(pdhg_solver_state_t *state);

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

    state->control.start_time = clock();
    initialize_solutions(state, working_problem);
    bool do_restart = false;

    while (state->control.total_count < params->termination_criteria.iteration_limit)
    {
        sync_inner_count_to_gpu(state);
        sync_step_sizes_to_gpu(state);

        hsd_run_iteration_step(state, params, 1, true);

        if (do_restart)
        {
            compute_fixed_point_error(state);
            CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
            state->control.initial_fixed_point_error = *state->control.fixed_point_error_mp;
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
        compute_termination_metrics(state, params);
        state->control.inner_count += params->termination_evaluation_frequency;
        state->control.total_count += params->termination_evaluation_frequency;

        check_termination_criteria(state, &params->termination_criteria);
        if (state->control.termination_reason != TERMINATION_REASON_UNSPECIFIED)
        {
            display_iteration_stats(state, params->verbose, false);
            break;
        }

        do_restart = should_do_adaptive_restart(state, params);
        if (do_restart)
        {
            state->control.restart_count++;
            display_iteration_stats(state, params->verbose, true);
            perform_restart(state, &params->step_size_params);
        }
        else
        {
            display_iteration_stats(state, params->verbose, false);
        }
    }

    if (state->control.termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->control.termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_termination_metrics(state, params);
        display_iteration_stats(state, params->verbose, false);
    }

    cuhsdlp_result_t *result = create_result_from_state(state, original_problem);

    if (params->presolve && presolve_info)
    {
        if (termination_has_solution(result->termination_reason))
        {
            pslp_postsolve(presolve_info, result, original_problem);
        }
        cuhsdlp_presolve_info_free(presolve_info);
    }

    pdhg_final_log(result, params);
    pdhg_solver_state_free(state);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

static void sync_inner_count_to_gpu(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpyAsync(state->control.inner_count_d,
                               &state->control.inner_count,
                               sizeof(int),
                               cudaMemcpyHostToDevice,
                               state->stream_A));
}

static void sync_step_sizes_to_gpu(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpyAsync(state->control.step_size_primal_d,
                               &state->control.step_size_primal,
                               sizeof(double),
                               cudaMemcpyHostToDevice,
                               state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(state->control.step_size_dual_d,
                               &state->control.step_size_dual,
                               sizeof(double),
                               cudaMemcpyHostToDevice,
                               state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(state->control.step_size_homogeneous_d,
                               &state->control.step_size_homogeneous,
                               sizeof(double),
                               cudaMemcpyHostToDevice,
                               state->stream_A));
}

static void compute_step_sizes(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params)
{
    double omega  = state->control.primal_weight;
    double lambda = state->control.step_size_base;
    double factor = step_size_params->step_size_factor;

    state->control.step_size_primal = lambda / omega;
    state->control.step_size_dual   = omega * lambda;
    state->control.step_size_homogeneous = 0.99 * factor / state->control.step_size_dual;
}

static void initialize_step_size_and_primal_weight(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    double sigma = state->rescaled_constraint_norm;
    double factor = params->step_size_params.step_size_factor;

    state->control.step_size_base = factor / sigma;

    if (params->scaling_params.bound_objective_rescaling)
    {
        state->control.primal_weight = 1.0;
    }
    else
    {
        state->control.primal_weight =
            (state->rescaled_objective_norm + 1.0) / (state->rescaled_bound_norm + 1.0);
    }

    initialize_pid_state(&state->control.pid);

    compute_step_sizes(state, &params->step_size_params);
}

static void hsd_run_iteration_step(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    int k_offset,
    bool is_major)
{
    double reflection_coefficient = params->reflection_coefficient;
    hsd_update_dual_solution(state, params, k_offset, is_major);
    update_hsd_primal_block(
        state,
        params,
        k_offset,
        reflection_coefficient,
        is_major);
}

static void hsd_update_dual_solution(
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
        state->matA_spmv_workspace_d));

    update_dual_box_cone_projection(
        cone_params,
        &state->dual_box_cone_state,
        state->current_dual_solution_d,
        state->reflected_dual_solution_d,
        state->pdhg_dual_solution_d,
        state->initial_dual_solution_d,
        state->primal_product_d,
        state->control.step_size_dual_d,
        state->hsd_primal_scale_current_d,
        state->control.inner_count_d,
        k_offset,
        reflection_coefficient,
        state->hsd_dual_scale_initial_d,
        state->hsd_dual_scale_current_d,
        state->hsd_dual_scale_pdhg_d,
        state->hsd_dual_scale_reflected_d,
        is_major,
        state->stream_A);

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
}

static void update_hsd_primal_block(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    const int k_offset,
    const double reflection_coefficient,
    bool is_major)
{
    const cone_parameters_t *cone_params = &params->cone_params;

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
        state->matAt_spmv_workspace_d));

    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));

    update_primal_box_cone_resolvent(
        cone_params,
        &state->primal_box_cone_state,
        state->current_primal_solution_d,
        state->reflected_primal_solution_d,
        state->pdhg_primal_solution_d,
        state->initial_primal_solution_d,
        state->dual_product_d,
        state->objective_vector_d,
        state->control.step_size_primal_d,
        state->control.step_size_homogeneous_d,
        state->hsd_dual_scale_reflected_d,
        state->control.inner_count_d,
        k_offset,
        reflection_coefficient,
        state->hsd_primal_scale_initial_d,
        state->hsd_primal_scale_current_d,
        state->hsd_primal_scale_pdhg_d,
        state->hsd_primal_scale_reflected_d,
        is_major,
        state->stream_A);
}

static void compute_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_kernel<<<
        state->num_blocks_primal_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->pdhg_primal_solution_d,
        state->reflected_primal_solution_d,
        state->delta_primal_solution_d,
        state->pdhg_dual_solution_d,
        state->reflected_dual_solution_d,
        state->delta_dual_solution_d,
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
        state->matAt_spmv_workspace_d));

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
        state->fixed_point_operator_cross_d));

    compute_fixed_point_error_kernel<<<1, 1, 0, state->stream_A>>>(
        state->control.fixed_point_error_mp,
        state->pdhg_primal_norm_d,
        state->pdhg_dual_norm_d,
        state->fixed_point_operator_cross_d,
        state->hsd_primal_scale_pdhg_d,
        state->hsd_primal_scale_reflected_d,
        state->hsd_dual_scale_pdhg_d,
        state->hsd_dual_scale_reflected_d,
        state->control.step_size_primal,
        state->control.step_size_dual,
        state->control.step_size_homogeneous);

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));
}

static bool should_do_adaptive_restart(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    const restart_parameters_t *restart_params = &params->restart_params;
    int termination_evaluation_frequency = params->termination_evaluation_frequency;

    bool do_restart = false;
    if (state->control.total_count == termination_evaluation_frequency)
    {
        do_restart = true;
    }
    else if (state->control.total_count > termination_evaluation_frequency)
    {
        if (*state->control.fixed_point_error_mp <= restart_params->sufficient_reduction_for_restart * state->control.initial_fixed_point_error)
        {
            do_restart = true;
        }
        if (*state->control.fixed_point_error_mp <= restart_params->necessary_reduction_for_restart * state->control.initial_fixed_point_error)
        {
            if (*state->control.fixed_point_error_mp > state->control.last_trial_fixed_point_error)
            {
                do_restart = true;
            }
        }
        if (state->control.inner_count >= restart_params->artificial_restart_threshold * state->control.total_count)
        {
            do_restart = true;
        }
    }
    state->control.last_trial_fixed_point_error = *state->control.fixed_point_error_mp;
    return do_restart;
}

static void perform_restart(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params)
{
    compute_delta_kernel<<<
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

    update_primal_weight_via_restart_pid(state, step_size_params);

    compute_step_sizes(state, step_size_params);

    double cone_warm_start_scale = apply_gauge_control(state);

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
    state->metrics_h->current_primal_scale = state->metrics_h->pdhg_primal_scale;

    restart_hsd_scalars_kernel<<<1, 1, 0, state->stream_A>>>(
        state->hsd_primal_scale_initial_d,             state->hsd_primal_scale_current_d,             state->hsd_primal_scale_pdhg_d,             state->hsd_primal_scale_reflected_d,
        state->hsd_dual_scale_initial_d,   state->hsd_dual_scale_current_d,   state->hsd_dual_scale_pdhg_d,   state->hsd_dual_scale_reflected_d,
        state->dual_box_cone_state.scalar_d,
        cone_warm_start_scale);

    state->control.inner_count = 0;
    state->control.last_trial_fixed_point_error = INFINITY;
}

static void update_primal_weight_via_restart_pid(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params)
{
    double restart_scalars_h[2];
    double hsd_scalars_h[HSD_SCALARS_COUNT];

    CUDA_CHECK(cudaMemcpyAsync(
        restart_scalars_h,
        state->pdhg_primal_norm_d,
        2 * sizeof(double),
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
    double dy2     = restart_scalars_h[1] * restart_scalars_h[1];

    double primal_scale_delta =
        hsd_scalars_h[HSD_PRIMAL_SCALE_PDHG] -
        hsd_scalars_h[HSD_PRIMAL_SCALE_INITIAL];
    double d_ac    = hsd_scalars_h[HSD_DUAL_SCALE_PDHG]
                   - hsd_scalars_h[HSD_DUAL_SCALE_INITIAL];
    double base2   = state->control.step_size_base * state->control.step_size_base;
    double alpha_p = base2 / step_size_params->step_size_factor;
    double primal_scale_delta_squared =
        primal_scale_delta * primal_scale_delta;

    double primal_dist = sqrt(dx2 + alpha_p * primal_scale_delta_squared);
    double dual_dist   = sqrt(dy2 + d_ac * d_ac);

    apply_restart_pid(&state->control.pid, &state->control.primal_weight, step_size_params,
                      primal_dist, dual_dist,
                      state->metrics_h->relative_objective_gap,
                      state->metrics_h->relative_primal_residual,
                      hsd_scalars_h[HSD_PRIMAL_SCALE_PDHG],
                      state->metrics_h->normalized_homogeneous_ratio);
}


static double apply_gauge_control(pdhg_solver_state_t *state)
{
    double primal_scale =
        state->metrics_h->pdhg_primal_scale;
    double max_block_norm = fmax(fmax(state->metrics_h->primal_solution_norm,
                                      state->metrics_h->dual_solution_norm),
                            fmax(fabs(primal_scale),
                                 fabs(state->metrics_h->pdhg_dual_scale)));

    double inv_scale = compute_gauge_inverse_scale(primal_scale, max_block_norm);
    if (inv_scale == 1.0)
    {
        return 1.0;
    }

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
    scale_hsd_scalars_kernel<<<1, 1, 0, state->stream_A>>>(
        state->hsd_primal_scale_pdhg_d,
        state->hsd_dual_scale_pdhg_d,
        inv_scale);
    state->metrics_h->pdhg_primal_scale      *= inv_scale;
    state->metrics_h->pdhg_dual_scale *= inv_scale;
    return inv_scale;
}

__global__ void compute_fixed_point_error_kernel(
    double *fixed_point_error_d,
    const double *primal_norm,
    const double *dual_norm,
    const double *dual_cross,
    const double *primal_scale_pdhg,
    const double *primal_scale_reflected,
    const double *dual_scale_pdhg,
    const double *dual_scale_reflected,
    double step_size_primal,
    double step_size_dual,
    double step_size_homogeneous)
{
    double delta_primal_scale = (*primal_scale_reflected) - (*primal_scale_pdhg);
    double delta_constraint_offset = (*dual_scale_reflected) - (*dual_scale_pdhg);
    double primal_norm_v = *primal_norm;
    double dual_norm_v = *dual_norm;
    double cross_term = *dual_cross;

    double dual_movement =
        (dual_norm_v * dual_norm_v +
         delta_constraint_offset * delta_constraint_offset) / step_size_dual;
    double movement =
        primal_norm_v * primal_norm_v / step_size_primal +
        dual_movement +
        delta_primal_scale * delta_primal_scale / step_size_homogeneous;
    double interaction = -2.0 * (delta_primal_scale * delta_constraint_offset + cross_term);
    fixed_point_error_d[0] = sqrt(fmax(0.0, movement + interaction));
}

__global__ void restart_hsd_scalars_kernel(
    double *primal_scale_initial,           double *primal_scale_current,           double *primal_scale_pdhg,           double *primal_scale_reflected,
    double *dual_scale_initial, double *dual_scale_current, double *dual_scale_pdhg, double *dual_scale_reflected,
    double *constraint_cone_scale_d,
    double  cone_warm_start_scale)
{
    double s = *primal_scale_pdhg;
    *primal_scale_initial   = s;
    *primal_scale_current   = s;
    *primal_scale_reflected = s;

    double dc = *dual_scale_pdhg;
    *dual_scale_initial   = dc;
    *dual_scale_current   = dc;
    *dual_scale_reflected = dc;

    *constraint_cone_scale_d *= cone_warm_start_scale;
}

__global__ void scale_hsd_scalars_kernel(
    double *primal_scale_pdhg,
    double *dual_scale_pdhg,
    double  inv_scale)
{
    *primal_scale_pdhg *= inv_scale;
    *dual_scale_pdhg   *= inv_scale;
}
