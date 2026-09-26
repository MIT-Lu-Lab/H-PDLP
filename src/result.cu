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

__global__ void rescale_solution_component_kernel(
    const double *__restrict__ input,
    double *__restrict__ output,
    const double *__restrict__ element_rescaling,
    double scalar_rescaling,
    double divisor,
    int n);
__global__ void rescale_reduced_cost_kernel(
    const double *__restrict__ input,
    double *__restrict__ output,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    const double *__restrict__ var_rescaling,
    double obj_vec_rescaling,
    double divisor,
    double input_sign,
    int n_vars);
static double compute_result_divisor(const pdhg_solver_state_t *state);

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
    cuhsdlp_result_t *result = (cuhsdlp_result_t *)safe_calloc(1, sizeof(cuhsdlp_result_t));
    const termination_reason_t reason = state->control.termination_reason;
    const bool has_solution = termination_has_solution(reason);
    const bool primal_infeasible = reason == TERMINATION_REASON_PRIMAL_INFEASIBLE;
    const bool dual_infeasible = reason == TERMINATION_REASON_DUAL_INFEASIBLE;
    const bool from_displacement =
        (primal_infeasible || dual_infeasible) &&
        !state->control.certificate_from_homogeneous;

    if (has_solution || primal_infeasible || dual_infeasible)
    {
        const double divisor = compute_result_divisor(state);

        if (has_solution || dual_infeasible)
        {
            const double *primal_input_d =
                from_displacement
                    ? state->delta_primal_solution_d
                    : state->pdhg_primal_solution_d;
            result->primal_solution = (double *)safe_malloc(state->var_bytes);
            rescale_solution_component_kernel<<<
                state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
                primal_input_d, state->variable_reduction_entries_d,
                state->variable_rescaling_d, state->constraint_bound_rescaling,
                divisor, state->num_variables);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(
                result->primal_solution, state->variable_reduction_entries_d,
                state->var_bytes, cudaMemcpyDeviceToHost, state->stream_A));
        }

        if (has_solution || primal_infeasible)
        {
            const double *dual_input_d =
                from_displacement
                    ? state->delta_dual_solution_d
                    : state->pdhg_dual_solution_d;
            result->dual_solution = (double *)safe_malloc(state->con_bytes);
            rescale_solution_component_kernel<<<
                state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
                dual_input_d, state->constraint_reduction_entries_d,
                state->constraint_rescaling_d, state->objective_vector_rescaling,
                divisor, state->num_constraints);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(
                result->dual_solution, state->constraint_reduction_entries_d,
                state->con_bytes, cudaMemcpyDeviceToHost, state->stream_A));

            const double *reduced_cost_input_d =
                from_displacement
                    ? state->dual_product_d
                    : (primal_infeasible
                           ? state->primal_infeasibility_reduced_cost_d
                           : state->pdhg_reduced_cost_d);
            result->reduced_cost = (double *)safe_malloc(state->var_bytes);
            rescale_reduced_cost_kernel<<<
                state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
                reduced_cost_input_d, state->variable_reduction_entries_d,
                state->variable_lower_bound_d, state->variable_upper_bound_d,
                state->variable_rescaling_d, state->objective_vector_rescaling,
                divisor, from_displacement ? -1.0 : 1.0, state->num_variables);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(
                result->reduced_cost, state->variable_reduction_entries_d,
                state->var_bytes, cudaMemcpyDeviceToHost, state->stream_A));
        }

        CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    }

    result->num_variables = working_problem->num_variables;
    result->num_constraints = working_problem->num_constraints;
    result->num_nonzeros = working_problem->constraint_matrix_num_nonzeros;
    result->total_count = state->control.total_count;
    result->rescaling_time_sec = state->rescaling_time_sec;
    result->cumulative_time_sec = state->control.cumulative_time_sec;
    result->absolute_primal_residual = state->metrics_h->absolute_primal_residual;
    result->relative_primal_residual = state->metrics_h->relative_primal_residual;
    result->absolute_dual_residual = state->metrics_h->absolute_dual_residual;
    result->relative_dual_residual = state->metrics_h->relative_dual_residual;
    result->primal_objective_value = state->metrics_h->primal_objective;
    result->dual_objective_value = state->metrics_h->dual_objective;
    result->objective_gap = state->metrics_h->objective_gap;
    result->relative_objective_gap = state->metrics_h->relative_objective_gap;
    result->farkas_margin = state->control.certificate_margin;
    result->farkas_residual = state->control.certificate_residual;
    result->termination_reason = state->control.termination_reason;

    result->restart_count = state->control.restart_count;
    result->active_set_raise_count = state->active_set.raise_count;
    result->active_set_revert_count = state->active_set.revert_count;
    result->active_set_power_iteration_count = state->active_set.power_iteration_count;
    result->homogeneous_scale = state->metrics_h->pdhg_primal_scale;
    result->final_step_size_primal = state->control.step_size_primal;
    result->final_step_size_dual = state->control.step_size_dual;
    result->final_step_size_homogeneous = state->control.step_size_homogeneous;

    return result;
}

static double compute_result_divisor(const pdhg_solver_state_t *state)
{
    if (state->control.termination_reason != TERMINATION_REASON_PRIMAL_INFEASIBLE &&
        state->control.termination_reason != TERMINATION_REASON_DUAL_INFEASIBLE)
        return state->metrics_h->pdhg_primal_scale;

    termination_reductions_t reductions_h;
    CUDA_CHECK(cudaMemcpy(&reductions_h, state->termination_reductions_d,
                          sizeof(reductions_h), cudaMemcpyDeviceToHost));

    double margin;
    if (state->control.termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE)
    {
        margin = state->control.certificate_from_homogeneous
                     ? reductions_h.constraint_support +
                           reductions_h.homogeneous_variable_support
                     : reductions_h.primal_infeasibility_constraint_support +
                           reductions_h.primal_infeasibility_variable_support;
    }
    else
    {
        margin = -(state->control.certificate_from_homogeneous
                       ? reductions_h.objective_dot_primal
                       : reductions_h.dual_infeasibility_objective);
    }
    return margin /
           (state->constraint_bound_rescaling * state->objective_vector_rescaling);
}

__global__ void rescale_solution_component_kernel(
    const double *__restrict__ input,
    double *__restrict__ output,
    const double *__restrict__ element_rescaling,
    double scalar_rescaling,
    double divisor,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double inv_divisor = (fabs(divisor) > 1e-50) ? 1.0 / divisor : 0.0;
    output[i] = input[i] * inv_divisor / element_rescaling[i] / scalar_rescaling;
}

__global__ void rescale_reduced_cost_kernel(
    const double *__restrict__ input,
    double *__restrict__ output,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    const double *__restrict__ var_rescaling,
    double obj_vec_rescaling,
    double divisor,
    double input_sign,
    int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vars) return;

    double reduced_cost = input_sign * input[i];
    double inv_divisor = (fabs(divisor) > 1e-50) ? 1.0 / divisor : 0.0;
    reduced_cost = reduced_cost * inv_divisor * var_rescaling[i] / obj_vec_rescaling;
    if (!isfinite(var_lb[i]))
        reduced_cost = fmin(reduced_cost, 0.0);
    if (!isfinite(var_ub[i]))
        reduced_cost = fmax(reduced_cost, 0.0);
    output[i] = reduced_cost;
}
