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

#include "hpdlp.h"
#include "error_handling.h"
#include "matrix_formats.h"
#include "solver.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static lp_problem_t *build_lp_problem(
    const double *objective_c,
    const matrix_desc_t *A_desc,
    const double *con_lb, const double *con_ub,
    const double *var_lb, const double *var_ub,
    const double *objective_constant);
static void fill_or_copy(
    double **dst, int n,
    const double *src,
    double fill_val);
static bool validate_start_vector(const double *values, int n,
                                  const char *vector_type);

lp_problem_t *create_lp_problem(
    const double *objective_c,
    const matrix_desc_t *A_desc,
    const double *con_lb, const double *con_ub,
    const double *var_lb, const double *var_ub,
    const double *objective_constant)
{
    error_scope_t scope;
    enter_error_scope(&scope);
    if (setjmp(scope.env) != 0)
    {
        leave_error_scope();
        return NULL;
    }
    lp_problem_t *prob = build_lp_problem(objective_c, A_desc, con_lb, con_ub,
                                             var_lb, var_ub, objective_constant);
    leave_error_scope();
    return prob;
}

static lp_problem_t *build_lp_problem(
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
        report_error("invalid input matrix: %s",
                     matrix_format_status_message(matrix_status));
        return NULL;
    }

    lp_problem_t *prob = (lp_problem_t *)safe_calloc(1, sizeof(lp_problem_t));
    prob->num_variables = A_desc->n;
    prob->num_constraints = A_desc->m;
    fill_or_copy(&prob->variable_lower_bound, prob->num_variables, var_lb, -INFINITY);
    fill_or_copy(&prob->variable_upper_bound, prob->num_variables, var_ub, INFINITY);
    fill_or_copy(&prob->objective_vector, prob->num_variables, objective_c, 0.0);
    prob->objective_constant = objective_constant ? *objective_constant : 0.0;
    prob->objective_sense = OBJECTIVE_SENSE_MINIMIZE;

    prob->constraint_matrix_row_pointers = row_ptr;
    prob->constraint_matrix_col_indices = col_ind;
    prob->constraint_matrix_values = vals;
    prob->constraint_matrix_num_nonzeros = nnz;

    fill_or_copy(&prob->constraint_lower_bound, prob->num_constraints, con_lb, -INFINITY);
    fill_or_copy(&prob->constraint_upper_bound, prob->num_constraints, con_ub, INFINITY);

    if (!validate_lp_problem_data(prob))
    {
        lp_problem_free(prob);
        return NULL;
    }
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
    for (int i = 0; i < n; ++i)
        (*dst)[i] = fill_val;
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

    error_scope_t scope;
    enter_error_scope(&scope);
    if (setjmp(scope.env) != 0)
    {
        leave_error_scope();
        return false;
    }
    double *new_primal_start = NULL;
    double *new_dual_start = NULL;
    if (primal)
        fill_or_copy(&new_primal_start, n, primal, 0.0);
    if (dual)
        fill_or_copy(&new_dual_start, m, dual, 0.0);
    leave_error_scope();

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
            report_error("invalid %s start value at index %d: %g",
                         vector_type, i, values[i]);
            return false;
        }
    }
    return true;
}

hpdlp_result_t *solve_lp_problem(
    const lp_problem_t *prob,
    const pdhg_parameters_t *params)
{
    if (!prob)
    {
        report_error("solve_lp_problem: problem must not be NULL");
        return NULL;
    }

    pdhg_parameters_t local_params;
    if (params)
        local_params = *params;
    else
        set_default_parameters(&local_params);

    return optimize(&local_params, prob);
}

void hpdlp_result_free(hpdlp_result_t *results)
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
    free(prob->variable_lower_bound);
    free(prob->variable_upper_bound);
    free(prob->objective_vector);
    free(prob->constraint_matrix_row_pointers);
    free(prob->constraint_matrix_col_indices);
    free(prob->constraint_matrix_values);
    free(prob->constraint_lower_bound);
    free(prob->constraint_upper_bound);
    free(prob->primal_start);
    free(prob->dual_start);
    free(prob);
}
