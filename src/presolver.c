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

#include "presolver.h"
#include "PSLP_sol.h"
#include "cuhsdlp.h"
#include "utils.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef PSLP_VERSION
#define PSLP_VERSION "unknown"
#endif

const char *get_presolve_status_str(int status)
{
    if ((status & UNBNDORINFEAS) == UNBNDORINFEAS)
        return "INFEASIBLE_OR_UNBOUNDED";
    if (status & INFEASIBLE)
        return "INFEASIBLE";
    if (status & REDUCED)
        return "REDUCED";
    if (status == UNCHANGED)
        return "UNCHANGED";
    return "UNKNOWN_STATUS";
}

static lp_problem_t *convert_pslp_to_cuhsdlp(PresolvedProblem *reduced_prob, const lp_problem_t *working_problem)
{
    lp_problem_t *prob = (lp_problem_t *)safe_malloc(sizeof(lp_problem_t));
    prob->primal_start = NULL;
    prob->dual_start = NULL;

    prob->objective_constant = working_problem->objective_constant + reduced_prob->obj_offset;
    prob->objective_sense = OBJECTIVE_SENSE_MINIMIZE;
    prob->objective_vector = reduced_prob->c;
    prob->constraint_lower_bound = reduced_prob->lhs;
    prob->constraint_upper_bound = reduced_prob->rhs;
    prob->variable_lower_bound = reduced_prob->lbs;
    prob->variable_upper_bound = reduced_prob->ubs;

    prob->constraint_matrix_num_nonzeros = reduced_prob->nnz;
    prob->constraint_matrix_row_pointers = reduced_prob->Ap;
    prob->constraint_matrix_col_indices = reduced_prob->Ai;
    prob->constraint_matrix_values = reduced_prob->Ax;

    prob->num_variables = reduced_prob->n;
    prob->num_constraints = reduced_prob->m;

    return prob;
}

cuhsdlp_presolve_info_t *pslp_presolve(const lp_problem_t *working_problem, bool verbose)
{
    if (working_problem->primal_start || working_problem->dual_start)
    {
        printf("Warning: Warm-starting is currently not supported when presolve is enabled.\n"
               "The provided initial solutions will be ignored.\n");
    }
    if (verbose)
    {
        printf("\nPresolving (PSLP %s)...\n", PSLP_VERSION);
    }
    clock_t start_time = clock();

    cuhsdlp_presolve_info_t *info = (cuhsdlp_presolve_info_t *)safe_calloc(1, sizeof(cuhsdlp_presolve_info_t));

    info->settings = default_settings();
    info->settings->verbose = false;

    info->presolver = new_presolver(
        working_problem->constraint_matrix_values,
        working_problem->constraint_matrix_col_indices,
        working_problem->constraint_matrix_row_pointers,
        working_problem->num_constraints,
        working_problem->num_variables,
        working_problem->constraint_matrix_num_nonzeros,
        working_problem->constraint_lower_bound,
        working_problem->constraint_upper_bound,
        working_problem->variable_lower_bound,
        working_problem->variable_upper_bound,
        working_problem->objective_vector,
        info->settings);

    PresolveStatus status = run_presolver(info->presolver);
    info->presolve_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    info->presolve_status = status;
    if (verbose)
    {
        printf("  %-15s  : %s\n", "status", get_presolve_status_str(status));
        printf("  %-15s  : %.3g sec\n", "presolve time", info->presolve_time);
        printf("  %-15s  : %zu rows, %zu columns, %zu nonzeros\n",
                "reduced problem",
                info->presolver->reduced_prob->m,
                info->presolver->reduced_prob->n,
                info->presolver->reduced_prob->nnz);
    }

    if (status & INFEASIBLE || status & UNBNDORINFEAS || info->presolver->reduced_prob->n == 0)
    {
        info->problem_solved_during_presolve = true;
        info->reduced_problem = NULL;
    }
    else
    {
        info->problem_solved_during_presolve = false;
        info->reduced_problem = convert_pslp_to_cuhsdlp(info->presolver->reduced_prob, working_problem);
    }
    return info;
}

static void copy_presolve_stats(const cuhsdlp_presolve_info_t *info,
                                cuhsdlp_result_t *result)
{
    result->num_reduced_variables = info->presolver->reduced_prob->n;
    result->num_reduced_constraints = info->presolver->reduced_prob->m;
    result->num_reduced_nonzeros = info->presolver->reduced_prob->nnz;
    result->presolve_status = info->presolve_status;
    result->presolve_time = info->presolve_time;
}

cuhsdlp_result_t *create_result_from_presolve(const cuhsdlp_presolve_info_t *info, const lp_problem_t *working_problem)
{
    cuhsdlp_result_t *result = (cuhsdlp_result_t *)safe_calloc(1, sizeof(cuhsdlp_result_t));
    result->num_variables = working_problem->num_variables;
    result->num_constraints = working_problem->num_constraints;
    result->num_nonzeros = working_problem->constraint_matrix_num_nonzeros;
    copy_presolve_stats(info, result);

    if ((info->presolve_status & UNBNDORINFEAS) == UNBNDORINFEAS)
    {
        result->termination_reason = TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED;
        result->absolute_primal_residual = INFINITY;
        result->relative_primal_residual = INFINITY;
        result->absolute_dual_residual = INFINITY;
        result->relative_dual_residual = INFINITY;
        result->primal_objective_value = INFINITY;
        result->dual_objective_value = -INFINITY;
        result->objective_gap = INFINITY;
        result->relative_objective_gap = INFINITY;
        return result;
    }
    else if (info->presolve_status & INFEASIBLE)
    {
        result->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
        result->absolute_primal_residual = INFINITY;
        result->relative_primal_residual = INFINITY;
        result->absolute_dual_residual = INFINITY;
        result->relative_dual_residual = INFINITY;
        result->primal_objective_value = INFINITY;
        result->dual_objective_value = -INFINITY;
        result->objective_gap = INFINITY;
        result->relative_objective_gap = INFINITY;
        return result;
    }
    else if (info->presolver->reduced_prob->n == 0)
    {
        result->termination_reason = TERMINATION_REASON_OPTIMAL;
        pslp_postsolve(info, result, working_problem);
        return result;
    }
    result->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    return result;
}

void pslp_postsolve(const cuhsdlp_presolve_info_t *info,
                    cuhsdlp_result_t *result,
                    const lp_problem_t *working_problem)
{
    postsolve(info->presolver,
              result->primal_solution,
              result->dual_solution,
              result->reduced_cost);

    copy_presolve_stats(info, result);

    free(result->primal_solution);
    free(result->dual_solution);
    free(result->reduced_cost);

    result->primal_solution = (double *)safe_malloc(working_problem->num_variables * sizeof(double));
    result->dual_solution = (double *)safe_malloc(working_problem->num_constraints * sizeof(double));
    result->reduced_cost = (double *)safe_malloc(working_problem->num_variables * sizeof(double));

    memcpy(result->primal_solution, info->presolver->sol->x, working_problem->num_variables * sizeof(double));
    memcpy(result->dual_solution, info->presolver->sol->y, working_problem->num_constraints * sizeof(double));
    memcpy(result->reduced_cost, info->presolver->sol->z, working_problem->num_variables * sizeof(double));
    if (info->presolver->reduced_prob->n == 0)
    {
        double obj = 0.0;
        for (int i = 0; i < working_problem->num_variables; i++)
        {
            obj += working_problem->objective_vector[i] * result->primal_solution[i];
        }
        obj += working_problem->objective_constant;
        result->primal_objective_value = obj;
        result->dual_objective_value = obj;
    }
    return;
}

static double compute_box_support(const double *direction,
                                  const double *lower_bound,
                                  const double *upper_bound,
                                  int n)
{
    double support = 0.0;
    for (int i = 0; i < n; i++)
    {
        if (direction[i] > 0.0 && isfinite(lower_bound[i]))
        {
            support += lower_bound[i] * direction[i];
        }
        else if (direction[i] < 0.0 && isfinite(upper_bound[i]))
        {
            support += upper_bound[i] * direction[i];
        }
    }
    return support;
}

void pslp_postsolve_primal_infeasibility_ray(const cuhsdlp_presolve_info_t *info,
                                             cuhsdlp_result_t *result,
                                             const lp_problem_t *working_problem)
{
    int num_constraints = working_problem->num_constraints;
    int num_variables = working_problem->num_variables;
    size_t num_reduced_constraints = info->presolver->reduced_prob->m;

    double *farkas_dual = (double *)safe_malloc(num_reduced_constraints * sizeof(double));
    for (size_t i = 0; i < num_reduced_constraints; i++)
    {
        farkas_dual[i] = -result->dual_solution[i];
    }
    double *dual_ray = (double *)safe_malloc(num_constraints * sizeof(double));
    postsolve_primal_infeas_ray(info->presolver, farkas_dual, dual_ray);
    free(farkas_dual);
    for (int i = 0; i < num_constraints; i++)
    {
        dual_ray[i] = -dual_ray[i];
    }

    const int *row_pointers = working_problem->constraint_matrix_row_pointers;
    const int *col_indices = working_problem->constraint_matrix_col_indices;
    const double *values = working_problem->constraint_matrix_values;
    double *reduced_cost_ray = (double *)safe_calloc(num_variables, sizeof(double));
    for (int i = 0; i < num_constraints; i++)
    {
        for (int k = row_pointers[i]; k < row_pointers[i + 1]; k++)
        {
            reduced_cost_ray[col_indices[k]] += values[k] * dual_ray[i];
        }
    }
    for (int j = 0; j < num_variables; j++)
    {
        double farkas_lower_bound = !isfinite(working_problem->variable_upper_bound[j]) ? 0.0 : -INFINITY;
        double farkas_upper_bound = !isfinite(working_problem->variable_lower_bound[j]) ? 0.0 : INFINITY;
        reduced_cost_ray[j] = fmax(farkas_lower_bound,
                                   fmin(-reduced_cost_ray[j], farkas_upper_bound));
    }

    double margin = compute_box_support(dual_ray,
                                        working_problem->constraint_lower_bound,
                                        working_problem->constraint_upper_bound,
                                        num_constraints)
                  + compute_box_support(reduced_cost_ray,
                                        working_problem->variable_lower_bound,
                                        working_problem->variable_upper_bound,
                                        num_variables);
    if (margin > 0.0)
    {
        for (int i = 0; i < num_constraints; i++)
        {
            dual_ray[i] /= margin;
        }
        for (int j = 0; j < num_variables; j++)
        {
            reduced_cost_ray[j] /= margin;
        }
    }

    free(result->dual_solution);
    free(result->reduced_cost);
    result->dual_solution = dual_ray;
    result->reduced_cost = reduced_cost_ray;

    copy_presolve_stats(info, result);
}

void pslp_postsolve_dual_infeasibility_ray(const cuhsdlp_presolve_info_t *info,
                                           cuhsdlp_result_t *result,
                                           const lp_problem_t *working_problem)
{
    int num_variables = working_problem->num_variables;

    double *primal_ray = (double *)safe_malloc(num_variables * sizeof(double));
    postsolve_dual_infeas_ray(info->presolver, result->primal_solution, primal_ray);

    double objective_dot_ray = 0.0;
    for (int j = 0; j < num_variables; j++)
    {
        objective_dot_ray += working_problem->objective_vector[j] * primal_ray[j];
    }

    double margin = -objective_dot_ray;
    if (margin > 0.0)
    {
        for (int j = 0; j < num_variables; j++)
        {
            primal_ray[j] /= margin;
        }
    }

    free(result->primal_solution);
    result->primal_solution = primal_ray;

    copy_presolve_stats(info, result);
}

void cuhsdlp_presolve_info_free(cuhsdlp_presolve_info_t *info)
{
    if (!info)
        return;
    if (info->presolver)
        free_presolver(info->presolver);
    if (info->settings)
        free_settings(info->settings);
    if (info->reduced_problem)
        free(info->reduced_problem);
    free(info);
}
