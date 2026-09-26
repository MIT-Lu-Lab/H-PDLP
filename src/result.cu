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

#include "result.h"
#include "cuda_macros.h"
#include "termination_checker.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void rescale_solution_kernel(
    double *__restrict__ primal_sol,
    double *__restrict__ dual_sol,
    double *__restrict__ reduced_cost,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ cons_rescaling,
    const double obj_vec_rescaling,
    const double cons_bound_rescaling,
    double divisor,
    int n_vars,
    int n_cons);
static void rescale_solution(pdhg_solver_state_t *state);
static double compute_original_primal_infeasibility_margin(
    const pdhg_solver_state_t *state,
    const termination_reductions_t *reductions_h);
static double compute_original_dual_infeasibility_margin(
    const pdhg_solver_state_t *state,
    const termination_reductions_t *reductions_h);

bool termination_has_solution(termination_reason_t reason)
{
    return reason == TERMINATION_REASON_OPTIMAL ||
           reason == TERMINATION_REASON_TIME_LIMIT ||
           reason == TERMINATION_REASON_ITERATION_LIMIT;
}

cuhsdlp_result_t *create_result_from_state(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem)
{
    cuhsdlp_result_t *results = (cuhsdlp_result_t *)safe_calloc(1, sizeof(cuhsdlp_result_t));

    rescale_solution(state);

    if (termination_has_solution(state->control.termination_reason))
    {
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
    }
    else if (state->control.termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE)
    {
        results->dual_solution = (double *)safe_malloc(state->con_bytes);
        results->reduced_cost = (double *)safe_malloc(state->var_bytes);

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
    }
    else if (state->control.termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE)
    {
        results->primal_solution = (double *)safe_malloc(state->var_bytes);

        CUDA_CHECK(cudaMemcpyAsync(
            results->primal_solution,
            state->pdhg_primal_solution_d,
            state->var_bytes,
            cudaMemcpyDeviceToHost,
            state->stream_A));
        CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    }

    results->num_variables = working_problem->num_variables;
    results->num_constraints = working_problem->num_constraints;
    results->num_nonzeros = working_problem->constraint_matrix_num_nonzeros;
    results->total_count = state->control.total_count;
    results->rescaling_time_sec = state->rescaling_time_sec;
    results->cumulative_time_sec = state->control.cumulative_time_sec;
    results->absolute_primal_residual = state->metrics_h->absolute_primal_residual;
    results->relative_primal_residual = state->metrics_h->relative_primal_residual;
    results->absolute_dual_residual = state->metrics_h->absolute_dual_residual;
    results->relative_dual_residual = state->metrics_h->relative_dual_residual;
    results->primal_objective_value = state->metrics_h->primal_objective;
    results->dual_objective_value = state->metrics_h->dual_objective;
    results->objective_gap = state->metrics_h->objective_gap;
    results->relative_objective_gap = state->metrics_h->relative_objective_gap;
    results->farkas_margin = state->control.certificate_margin;
    results->farkas_residual = state->control.certificate_residual;
    results->termination_reason = state->control.termination_reason;

    results->restart_count = state->control.restart_count;
    results->active_set_raise_count = state->active_set.raise_count;
    results->active_set_revert_count = state->active_set.revert_count;
    results->active_set_power_iteration_count = state->active_set.power_iteration_count;
    results->homogeneous_scale = state->metrics_h->pdhg_primal_scale;
    results->final_step_size_primal = state->control.step_size_primal;
    results->final_step_size_dual = state->control.step_size_dual;
    results->final_step_size_homogeneous = state->control.step_size_homogeneous;

    return results;
}

static void rescale_solution(pdhg_solver_state_t *state)
{
    termination_reductions_t reductions_h;
    CUDA_CHECK(cudaMemcpy(&reductions_h, state->termination_reductions_d,
                          sizeof(reductions_h), cudaMemcpyDeviceToHost));

    double divisor;
    switch (state->control.termination_reason)
    {
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
        divisor = compute_original_primal_infeasibility_margin(state, &reductions_h);
        break;
    case TERMINATION_REASON_DUAL_INFEASIBLE:
        divisor = compute_original_dual_infeasibility_margin(state, &reductions_h);
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
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        state->variable_rescaling_d,
        state->constraint_rescaling_d,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
        divisor,
        state->num_variables,
        state->num_constraints);
}

static double compute_original_primal_infeasibility_margin(
    const pdhg_solver_state_t *state,
    const termination_reductions_t *reductions_h)
{
    double support_total =
        state->control.certificate_from_homogeneous
            ? reductions_h->constraint_support +
                  reductions_h->homogeneous_variable_support
            : reductions_h->primal_infeasibility_constraint_support +
                  reductions_h->primal_infeasibility_variable_support;
    return support_total /
           (state->constraint_bound_rescaling * state->objective_vector_rescaling);
}

static double compute_original_dual_infeasibility_margin(
    const pdhg_solver_state_t *state,
    const termination_reductions_t *reductions_h)
{
    double objective_dot_ray = state->control.certificate_from_homogeneous
                                   ? reductions_h->objective_dot_primal
                                   : reductions_h->dual_infeasibility_objective;
    return -objective_dot_ray /
           (state->constraint_bound_rescaling * state->objective_vector_rescaling);
}

__global__ void rescale_solution_kernel(
    double *__restrict__ primal_sol,
    double *__restrict__ dual_sol,
    double *__restrict__ reduced_cost,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
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
        double rc = reduced_cost[idx] * inv_divisor * var_rescaling[idx] / obj_vec_rescaling;
        if (!isfinite(var_lb[idx]))
            rc = fmin(rc, 0.0);
        if (!isfinite(var_ub[idx]))
            rc = fmax(rc, 0.0);
        reduced_cost[idx] = rc;
    }
}
