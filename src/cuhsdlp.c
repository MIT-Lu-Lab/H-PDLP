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
    lp_problem_t *prob = (lp_problem_t *)safe_malloc(sizeof(lp_problem_t));
    prob->primal_start = NULL;
    prob->dual_start = NULL;

    prob->num_variables = A_desc->n;
    prob->num_constraints = A_desc->m;

    matrix_format_status_t matrix_status = matrix_desc_to_csr(
        A_desc,
        &prob->constraint_matrix_row_pointers,
        &prob->constraint_matrix_col_indices,
        &prob->constraint_matrix_values,
        &prob->constraint_matrix_num_nonzeros);
    if (matrix_status != MATRIX_FORMAT_OK)
    {
        fprintf(stderr, "invalid input matrix: %s.\n",
                matrix_format_status_message(matrix_status));
        free(prob);
        return NULL;
    }

    prob->objective_constant = objective_constant ? *objective_constant : 0.0;
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

    cuhsdlp_result_t *res = optimize(&local_params, (lp_problem_t *)prob);
    if (!res)
    {
        fprintf(stderr, "[interface] optimize returned NULL.\n");
        return NULL;
    }

    return res;
}
