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

#include "cuhsdlp.h"
#include "matrix_formats.h"
#include "solver.h"
#include "utils.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool validate_bounds(const double *lower, const double *upper,
                            int n, const char *bound_type);
static bool validate_start_vector(const double *values, int n,
                                  const char *vector_type);
static void fill_or_copy(
    double **dst, int n,
    const double *src,
    double fill_val);
static void copy_start_vector(double **dst, const double *src, int n);

lp_problem_t *create_lp_problem(
    const double *objective_c,
    const matrix_desc_t *A_desc,
    const double *con_lb, const double *con_ub,
    const double *var_lb, const double *var_ub,
    const double *objective_constant)
{
    int *row_ptr = NULL;
    int *col_ind = NULL;
    double *vals = NULL;
    int nnz = 0;
    matrix_format_status_t matrix_status = matrix_desc_to_csr(
        A_desc,
        &row_ptr,
        &col_ind,
        &vals,
        &nnz);
    if (matrix_status != MATRIX_FORMAT_OK)
    {
        fprintf(stderr, "invalid input matrix: %s.\n",
                matrix_format_status_message(matrix_status));
        return NULL;
    }

    lp_problem_t *prob = (lp_problem_t *)safe_calloc(1, sizeof(lp_problem_t));
    prob->num_variables = A_desc->n;
    prob->num_constraints = A_desc->m;
    prob->constraint_matrix_row_pointers = row_ptr;
    prob->constraint_matrix_col_indices = col_ind;
    prob->constraint_matrix_values = vals;
    prob->constraint_matrix_num_nonzeros = nnz;

    prob->objective_constant = objective_constant ? *objective_constant : 0.0;
    prob->objective_sense = OBJECTIVE_SENSE_MINIMIZE;
    fill_or_copy(&prob->objective_vector, prob->num_variables, objective_c, 0.0);
    fill_or_copy(&prob->variable_lower_bound, prob->num_variables, var_lb, -INFINITY);
    fill_or_copy(&prob->variable_upper_bound, prob->num_variables, var_ub, INFINITY);
    fill_or_copy(&prob->constraint_lower_bound, prob->num_constraints, con_lb, -INFINITY);
    fill_or_copy(&prob->constraint_upper_bound, prob->num_constraints, con_ub, INFINITY);

    if (!validate_lp_problem_data(prob))
    {
        lp_problem_free(prob);
        return NULL;
    }
    return prob;
}

bool validate_lp_problem_data(const lp_problem_t *prob)
{
    if (!prob || prob->num_variables < 0 || prob->num_constraints < 0 ||
        prob->constraint_matrix_num_nonzeros < 0 ||
        (prob->num_variables > 0 &&
         (!prob->objective_vector || !prob->variable_lower_bound || !prob->variable_upper_bound)) ||
        (prob->num_constraints > 0 &&
         (!prob->constraint_lower_bound || !prob->constraint_upper_bound)) ||
        (prob->constraint_matrix_num_nonzeros > 0 && !prob->constraint_matrix_values))
    {
        fprintf(stderr, "invalid model: missing data or invalid dimensions.\n");
        return false;
    }
    if (!isfinite(prob->objective_constant))
    {
        fprintf(stderr, "invalid model: objective constant must be finite.\n");
        return false;
    }
    for (int j = 0; j < prob->num_variables; ++j)
    {
        if (!isfinite(prob->objective_vector[j]))
        {
            fprintf(stderr, "invalid model: objective coefficient at index %d must be finite.\n", j);
            return false;
        }
    }
    for (int k = 0; k < prob->constraint_matrix_num_nonzeros; ++k)
    {
        if (!isfinite(prob->constraint_matrix_values[k]))
        {
            fprintf(stderr, "invalid input matrix: coefficients must be finite.\n");
            return false;
        }
    }
    return validate_bounds(prob->variable_lower_bound, prob->variable_upper_bound,
                           prob->num_variables, "variable") &&
           validate_bounds(prob->constraint_lower_bound, prob->constraint_upper_bound,
                           prob->num_constraints, "constraint");
}

static bool validate_bounds(const double *lower, const double *upper,
                            int n, const char *bound_type)
{
    for (int i = 0; i < n; ++i)
    {
        if (isnan(lower[i]) || isnan(upper[i]) ||
            lower[i] == INFINITY || upper[i] == -INFINITY ||
            lower[i] > upper[i])
        {
            fprintf(stderr, "invalid %s bounds at index %d: [%g, %g].\n",
                    bound_type, i, lower[i], upper[i]);
            return false;
        }
    }
    return true;
}

static void fill_or_copy(
    double **dst, int n,
    const double *src,
    double fill_val)
{
    size_t bytes = (size_t)n * sizeof(double);
    *dst = (double *)safe_malloc(bytes);
    if (src)
    {
        memcpy(*dst, src, bytes);
        return;
    }

    if (fill_val == 0.0)
    {
        memset(*dst, 0, bytes);
        return;
    }

    for (int i = 0; i < n; ++i)
        (*dst)[i] = fill_val;
}

void cuhsdlp_result_free(cuhsdlp_result_t *results)
{
    if (results == NULL)
        return;

    free(results->primal_solution);
    free(results->dual_solution);
    free(results->reduced_cost);
    free(results);
}

void lp_problem_free(lp_problem_t *prob)
{
    if (!prob)
        return;
    free(prob->constraint_matrix_row_pointers);
    free(prob->constraint_matrix_col_indices);
    free(prob->constraint_matrix_values);
    free(prob->variable_lower_bound);
    free(prob->variable_upper_bound);
    free(prob->objective_vector);
    free(prob->constraint_lower_bound);
    free(prob->constraint_upper_bound);
    free(prob->primal_start);
    free(prob->dual_start);
    free(prob);
}

void set_objective_sense(lp_problem_t *prob, objective_sense_t sense)
{
    if (!prob)
        return;
    prob->objective_sense = sense;
}

bool set_start_values(
    lp_problem_t *prob,
    const double *primal,
    const double *dual)
{
    if (!prob)
        return false;

    int n = prob->num_variables;
    int m = prob->num_constraints;

    if (!validate_start_vector(primal, n, "primal") ||
        !validate_start_vector(dual, m, "dual"))
        return false;

    double *new_primal_start = NULL;
    double *new_dual_start = NULL;
    if (primal)
    {
        copy_start_vector(&new_primal_start, primal, n);
    }
    if (dual)
    {
        copy_start_vector(&new_dual_start, dual, m);
    }
    free(prob->primal_start);
    free(prob->dual_start);
    prob->primal_start = new_primal_start;
    prob->dual_start = new_dual_start;
    return true;
}

static bool validate_start_vector(const double *values, int n,
                                  const char *vector_type)
{
    if (!values)
        return true;
    for (int i = 0; i < n; ++i)
    {
        if (!isfinite(values[i]))
        {
            fprintf(stderr, "invalid %s start value at index %d: %g.\n",
                    vector_type, i, values[i]);
            return false;
        }
    }
    return true;
}

static void copy_start_vector(double **dst, const double *src, int n)
{
    size_t bytes = (size_t)n * sizeof(double);
    *dst = (double *)safe_malloc(bytes);
    memcpy(*dst, src, bytes);
}

void set_default_parameters(pdhg_parameters_t *params)
{
    params->presolve = true;
    params->verbose = true;
    params->diagnose = false;
    params->optimality_norm = NORM_TYPE_L2;
    params->reflection_coefficient = 1.0;
    params->termination_evaluation_frequency = 200;

    params->scaling_params.geometric_mean_iterations = 12;
    params->scaling_params.l_inf_ruiz_iterations = 10;
    params->scaling_params.has_pock_chambolle_alpha = true;
    params->scaling_params.pock_chambolle_alpha = 1.0;
    params->scaling_params.bound_objective_rescaling = true;
    params->scaling_params.matrix_zero_tol = 1e-9;
    params->scaling_params.infinite_bound = 1e20;

    params->power_iter_params.max_iter = 5000;
    params->power_iter_params.tol = 1e-4;

    params->step_size_params.step_size_factor = 0.998;
    params->step_size_params.k_p = 0.97;
    params->step_size_params.k_i = 0.05;
    params->step_size_params.k_d = 0.0;
    params->step_size_params.i_smooth = 0.3;

    params->cone_params.max_iter = 100;
    params->cone_params.tol = 1e-6;

    params->termination_criteria.eps_optimal_relative = 1e-4;
    params->termination_criteria.eps_feasible_relative = 1e-4;
    params->termination_criteria.eps_farkas_homogeneous = 1e-9;
    params->termination_criteria.eps_farkas_displacement = 1e-10;
    params->termination_criteria.time_sec_limit = 3600.0;
    params->termination_criteria.iteration_limit = INT32_MAX;

    params->restart_params.artificial_restart_threshold = 0.36;
    params->restart_params.sufficient_reduction_for_restart = 0.2;
    params->restart_params.necessary_reduction_for_restart = 0.5;

    params->active_set_params.step_size_boost = true;
    params->active_set_params.window_iterations = 10000;
    params->active_set_params.relative_tol = 1e-8;
    params->active_set_params.safety_factor = 0.9;
    params->active_set_params.min_raise_ratio = 1.1;
    params->active_set_params.reestimate_change_ratio = 0.01;
    params->active_set_params.divergence_ceiling_ratio = 0.7;
    params->active_set_params.divergence_margin = 0.05;
    params->active_set_params.max_reverts = 2;

    params->tau_trace_path = NULL;
}

cuhsdlp_result_t *solve_lp_problem(
    const lp_problem_t *prob,
    const pdhg_parameters_t *params)
{
    if (!prob)
    {
        fprintf(stderr, "[interface] solve_lp_problem: invalid arguments.\n");
        return NULL;
    }

    pdhg_parameters_t local_params;
    if (params)
        local_params = *params;
    else
        set_default_parameters(&local_params);

    return optimize(&local_params, prob);
}
