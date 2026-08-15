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

    return prob;
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

static void copy_start_vector(double **dst, const double *src, int n)
{
    size_t bytes = (size_t)n * sizeof(double);
    *dst = (double *)safe_malloc(bytes);
    memcpy(*dst, src, bytes);
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
    memset(prob, 0, sizeof(*prob));
    free(prob);
}

void set_objective_sense(lp_problem_t *prob, objective_sense_t sense)
{
    if (!prob)
        return;
    prob->objective_sense = sense;
}

void set_start_values(
    lp_problem_t *prob,
    const double *primal,
    const double *dual)
{
    if (!prob)
        return;

    int n = prob->num_variables;
    int m = prob->num_constraints;

    if (prob->primal_start)
    {
        free(prob->primal_start);
        prob->primal_start = NULL;
    }
    if (prob->dual_start)
    {
        free(prob->dual_start);
        prob->dual_start = NULL;
    }

    if (primal)
    {
        copy_start_vector(&prob->primal_start, primal, n);
    }
    if (dual)
    {
        copy_start_vector(&prob->dual_start, dual, m);
    }
}

void set_default_parameters(pdhg_parameters_t *params)
{
    params->presolve = true;
    params->verbose = true;
    params->optimality_norm = NORM_TYPE_L2;
    params->termination_evaluation_frequency = 200;

    params->scaling_params.l_inf_ruiz_iterations = 10;
    params->scaling_params.has_pock_chambolle_alpha = true;
    params->scaling_params.pock_chambolle_alpha = 1.0;
    params->scaling_params.bound_objective_rescaling = true;
    params->scaling_params.matrix_zero_tol = 1e-9;

    params->power_iter_params.max_iter = 5000;
    params->power_iter_params.tol = 1e-4;

    params->step_size_params.step_size_factor = 0.998;

    params->cone_params.max_iter = 100;
    params->cone_params.tol = 1e-6;

    params->termination_criteria.eps_optimal_relative = 1e-4;
    params->termination_criteria.eps_feasible_relative = 1e-4;
    params->termination_criteria.eps_farkas_homogeneous = 1e-7;
    params->termination_criteria.eps_farkas_displacement = 1e-10;
    params->termination_criteria.time_sec_limit = 3600.0;
    params->termination_criteria.iteration_limit = INT32_MAX;
}

cuhsdlp_result_t *solve_lp_problem(
    lp_problem_t *prob,
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

    cuhsdlp_result_t *res = optimize(&local_params, prob);
    if (!res)
    {
        fprintf(stderr, "[interface] optimize returned NULL.\n");
        return NULL;
    }

    return res;
}
