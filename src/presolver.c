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

const char *get_presolve_status_str(enum PresolveStatus_ status)
{
    switch (status)
    {
    case UNCHANGED:
        return "UNCHANGED";
    case REDUCED:
        return "REDUCED";
    case INFEASIBLE:
        return "INFEASIBLE";
    case UNBNDORINFEAS:
        return "INFEASIBLE_OR_UNBOUNDED";
    default:
        return "UNKNOWN_STATUS";
    }
}

static lp_problem_t *convert_pslp_to_cuhsdlp(PresolvedProblem *reduced_prob)
{
    lp_problem_t *prob = (lp_problem_t *)safe_malloc(sizeof(lp_problem_t));
    prob->primal_start = NULL;
    prob->dual_start = NULL;

    prob->objective_constant = reduced_prob->obj_offset;
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

cuhsdlp_presolve_info_t *pslp_presolve(const lp_problem_t *original_prob, const pdhg_parameters_t *params)
{
    if (original_prob->primal_start || original_prob->dual_start)
    {
        printf("Warning: Warm-starting is currently not supported when presolve is enabled.\n"
               "The provided initial solutions will be ignored.\n");
    }
    if (params->verbose)
    {
        printf("\nPresolving (PSLP %s)...\n", PSLP_VERSION);
    }
    clock_t start_time = clock();

    cuhsdlp_presolve_info_t *info = (cuhsdlp_presolve_info_t *)safe_calloc(1, sizeof(cuhsdlp_presolve_info_t));

    info->settings = default_settings();
    info->settings->verbose = false;

    info->presolver = new_presolver(
        original_prob->constraint_matrix_values,
        original_prob->constraint_matrix_col_indices,
        original_prob->constraint_matrix_row_pointers,
        original_prob->num_constraints,
        original_prob->num_variables,
        original_prob->constraint_matrix_num_nonzeros,
        original_prob->constraint_lower_bound,
        original_prob->constraint_upper_bound,
        original_prob->variable_lower_bound,
        original_prob->variable_upper_bound,
        original_prob->objective_vector,
        info->settings);

    PresolveStatus status = run_presolver(info->presolver);
    info->presolve_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    info->presolve_status = status;
    if (params->verbose)
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
        info->reduced_problem = convert_pslp_to_cuhsdlp(info->presolver->reduced_prob);
    }
    return info;
}

cuhsdlp_result_t *create_result_from_presolve(const cuhsdlp_presolve_info_t *info, const lp_problem_t *original_prob)
{
    cuhsdlp_result_t *result = (cuhsdlp_result_t *)safe_calloc(1, sizeof(cuhsdlp_result_t));
    result->num_variables = original_prob->num_variables;
    result->num_constraints = original_prob->num_constraints;
    result->num_nonzeros = original_prob->constraint_matrix_num_nonzeros;
    result->num_reduced_variables = info->presolver->reduced_prob->n;
    result->num_reduced_constraints = info->presolver->reduced_prob->m;
    result->num_reduced_nonzeros = info->presolver->reduced_prob->nnz;
    result->presolve_status = info->presolve_status;
    result->presolve_time = info->presolve_time;

    if (info->presolve_status == INFEASIBLE)
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
    }
    else if (info->presolve_status == UNBNDORINFEAS)
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
    }
    else if (info->presolver->reduced_prob->n == 0)
    {
        result->termination_reason = TERMINATION_REASON_OPTIMAL;
        pslp_postsolve(info, result, original_prob);
        return result;
    }
    else
    {
        result->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    }

    if (result->num_variables > 0)
    {
        result->primal_solution = (double *)safe_calloc(result->num_variables, sizeof(double));
        result->reduced_cost = (double *)safe_calloc(result->num_variables, sizeof(double));
    }
    if (result->num_constraints > 0)
    {
        result->dual_solution = (double *)safe_calloc(result->num_constraints, sizeof(double));
    }
    return result;
}

void pslp_postsolve(const cuhsdlp_presolve_info_t *info,
                    cuhsdlp_result_t *result,
                    const lp_problem_t *original_prob)
{
    postsolve(info->presolver,
              result->primal_solution,
              result->dual_solution,
              result->reduced_cost);

    result->num_reduced_variables = info->presolver->reduced_prob->n;
    result->num_reduced_constraints = info->presolver->reduced_prob->m;
    result->num_reduced_nonzeros = info->presolver->reduced_prob->nnz;
    result->presolve_status = info->presolve_status;

    free(result->primal_solution);
    free(result->dual_solution);
    free(result->reduced_cost);

    result->primal_solution = (double *)safe_malloc(original_prob->num_variables * sizeof(double));
    result->dual_solution = (double *)safe_malloc(original_prob->num_constraints * sizeof(double));
    result->reduced_cost = (double *)safe_malloc(original_prob->num_variables * sizeof(double));

    memcpy(result->primal_solution, info->presolver->sol->x, original_prob->num_variables * sizeof(double));
    memcpy(result->dual_solution, info->presolver->sol->y, original_prob->num_constraints * sizeof(double));
    memcpy(result->reduced_cost, info->presolver->sol->z, original_prob->num_variables * sizeof(double));
    result->presolve_time = info->presolve_time;
    if (info->presolver->reduced_prob->n == 0)
    {
        double obj = 0.0;
        for (int i = 0; i < original_prob->num_variables; i++)
        {
            obj += original_prob->objective_vector[i] * result->primal_solution[i];
        }
        obj += original_prob->objective_constant;
        result->primal_objective_value = obj;
        result->dual_objective_value = obj;
    }
    return;
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
