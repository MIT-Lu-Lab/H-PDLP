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

static void hsd_run_iteration_step(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params);
static void initialize_step_size_and_primal_weight(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params);
static void hsd_update_dual_solution(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params);
static void update_hsd_primal_block(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params);
static void sync_step_sizes_to_gpu(pdhg_solver_state_t *state);
static lp_problem_t create_working_problem(
    const lp_problem_t *original_problem,
    const pdhg_parameters_t *params);
static void restore_original_objective_sense(
    cuhsdlp_result_t *result,
    objective_sense_t objective_sense);
static void working_problem_free(
    const lp_problem_t *working_problem,
    const lp_problem_t *original_problem);

cuhsdlp_result_t *optimize(
    const pdhg_parameters_t *params,
    const lp_problem_t *original_problem)
{
    check_params_validity(params);

    lp_problem_t working_problem = create_working_problem(original_problem, params);
    print_initial_info(params, &working_problem);

    cuhsdlp_presolve_info_t *presolve_info = NULL;
    if (params->presolve)
    {
        presolve_info = pslp_presolve(&working_problem, params->verbose);
        if (presolve_info->problem_solved_during_presolve)
        {
            cuhsdlp_result_t *result = create_result_from_presolve(presolve_info, &working_problem);
            cuhsdlp_presolve_info_free(presolve_info);
            restore_original_objective_sense(result, original_problem->objective_sense);
            pdhg_final_log(result, params);
            working_problem_free(&working_problem, original_problem);
            return result;
        }
    }

    pdhg_solver_state_t *state = initialize_solver_state(
        presolve_info ? presolve_info->reduced_problem : &working_problem,
        original_problem,
        params);
    display_iteration_stats(state, params->verbose);

    initialize_step_size_and_primal_weight(state, params);
    sync_step_sizes_to_gpu(state);

    state->control.start_time = clock();

    while (state->control.total_count < params->termination_criteria.iteration_limit)
    {
        if (!state->graph_created)
        {
            cudaGraph_t graph = NULL;
            CUDA_CHECK(cudaStreamBeginCapture(state->stream_A,
                                              cudaStreamCaptureModeGlobal));

            for (int i = 0; i < params->termination_evaluation_frequency; i++)
            {
                hsd_run_iteration_step(state, params);
            }

            CUDA_CHECK(cudaStreamEndCapture(state->stream_A, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&state->graphExec, graph,
                                            NULL, NULL, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            state->graph_created = true;
        }
        CUDA_CHECK(cudaGraphLaunch(state->graphExec, state->stream_A));

        compute_termination_metrics(state, params);
        state->control.total_count += params->termination_evaluation_frequency;

        check_termination_criteria(state, &params->termination_criteria);
        display_iteration_stats(state, params->verbose);
        if (state->control.termination_reason != TERMINATION_REASON_UNSPECIFIED)
        {
            break;
        }
    }

    if (state->control.termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->control.termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_termination_metrics(state, params);
        display_iteration_stats(state, params->verbose);
    }

    cuhsdlp_result_t *result = create_result_from_state(state, &working_problem);

    if (params->presolve && presolve_info)
    {
        if (termination_has_solution(result->termination_reason))
        {
            pslp_postsolve(presolve_info, result, &working_problem);
        }
        else if (result->termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE)
        {
            pslp_postsolve_primal_infeasibility_ray(presolve_info, result, &working_problem);
        }
        else if (result->termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE)
        {
            pslp_postsolve_dual_infeasibility_ray(presolve_info, result, &working_problem);
        }
        cuhsdlp_presolve_info_free(presolve_info);
    }

    restore_original_objective_sense(result, original_problem->objective_sense);
    pdhg_final_log(result, params);
    pdhg_solver_state_free(state);
    CUDA_CHECK(cudaGetLastError());
    working_problem_free(&working_problem, original_problem);
    return result;
}

static lp_problem_t create_working_problem(
    const lp_problem_t *original_problem,
    const pdhg_parameters_t *params)
{
    lp_problem_t working_problem = *original_problem;

    if (original_problem->objective_sense == OBJECTIVE_SENSE_MAXIMIZE)
    {
        int num_variables = original_problem->num_variables;
        double *negated_objective = (double *)safe_malloc((size_t)num_variables * sizeof(double));
        for (int j = 0; j < num_variables; ++j)
        {
            negated_objective[j] = -original_problem->objective_vector[j];
        }
        working_problem.objective_vector = negated_objective;
        working_problem.objective_constant = -original_problem->objective_constant;
        working_problem.objective_sense = OBJECTIVE_SENSE_MINIMIZE;

        if (original_problem->dual_start != NULL)
        {
            int num_constraints = original_problem->num_constraints;
            double *negated_dual_start = (double *)safe_malloc((size_t)num_constraints * sizeof(double));
            for (int i = 0; i < num_constraints; ++i)
            {
                negated_dual_start[i] = -original_problem->dual_start[i];
            }
            working_problem.dual_start = negated_dual_start;
        }
    }

    filter_constraint_matrix_entries(&working_problem, original_problem, params);
    return working_problem;
}

static void restore_original_objective_sense(
    cuhsdlp_result_t *result,
    objective_sense_t objective_sense)
{
    if (objective_sense != OBJECTIVE_SENSE_MAXIMIZE)
    {
        return;
    }

    result->primal_objective_value = -result->primal_objective_value;
    result->dual_objective_value = -result->dual_objective_value;

    if (!termination_has_solution(result->termination_reason))
    {
        return;
    }

    if (result->dual_solution != NULL)
    {
        for (int i = 0; i < result->num_constraints; ++i)
        {
            result->dual_solution[i] = -result->dual_solution[i];
        }
    }
    if (result->reduced_cost != NULL)
    {
        for (int j = 0; j < result->num_variables; ++j)
        {
            result->reduced_cost[j] = -result->reduced_cost[j];
        }
    }
}

static void working_problem_free(
    const lp_problem_t *working_problem,
    const lp_problem_t *original_problem)
{
    if (working_problem->objective_vector != original_problem->objective_vector)
    {
        free(working_problem->objective_vector);
    }
    if (working_problem->dual_start != original_problem->dual_start)
    {
        free(working_problem->dual_start);
    }
    if (working_problem->constraint_matrix_values != original_problem->constraint_matrix_values)
    {
        free(working_problem->constraint_matrix_row_pointers);
        free(working_problem->constraint_matrix_col_indices);
        free(working_problem->constraint_matrix_values);
    }
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

    compute_step_sizes(state, &params->step_size_params);
}

static void hsd_run_iteration_step(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    hsd_update_dual_solution(state, params);
    update_hsd_primal_block(state, params);
}

static void hsd_update_dual_solution(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    const cone_parameters_t *cone_params = &params->cone_params;

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
        state->extrapolated_dual_solution_d,
        state->primal_product_d,
        state->control.step_size_dual_d,
        state->hsd_primal_scale_d,
        state->hsd_dual_scale_d,
        state->hsd_dual_scale_extrapolated_d,
        state->stream_A);

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
}

static void update_hsd_primal_block(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    const cone_parameters_t *cone_params = &params->cone_params;

    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle_At,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE,
        state->matAt,
        state->spmv_extrapolated_dual_solution,
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
        state->dual_product_d,
        state->objective_vector_d,
        state->control.step_size_primal_d,
        state->control.step_size_homogeneous_d,
        state->hsd_dual_scale_extrapolated_d,
        state->hsd_primal_scale_d,
        state->stream_A);
}
