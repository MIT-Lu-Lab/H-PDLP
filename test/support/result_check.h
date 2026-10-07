#pragma once

#include "hopdlp.h"
#include <math.h>
#include <stdlib.h>

typedef struct
{
    double primal_residual;
    double dual_residual;
    double primal_objective;
    double dual_objective;
    double relative_primal_residual;
    double relative_dual_residual;
    double relative_gap;
} test_solution_metrics_t;

static double test_norm_add(double norm, double value, norm_type_t type)
{
    if (!isfinite(value)) return INFINITY;
    return type == NORM_TYPE_L_INF ? fmax(norm, fabs(value)) : hypot(norm, value);
}

static double test_bound_support(double multiplier, double lower, double upper)
{
    if (!isfinite(multiplier)) return -INFINITY;
    if (multiplier > 0) return multiplier * lower;
    if (multiplier < 0) return multiplier * upper;
    return 0;
}

static test_solution_metrics_t test_evaluate_solution(
    const lp_problem_t *problem, const pdhg_parameters_t *params,
    const hopdlp_result_t *result)
{
    const int n = problem->num_variables;
    const int m = problem->num_constraints;
    const double sign = problem->objective_sense == OBJECTIVE_SENSE_MAXIMIZE ? -1 : 1;
    const norm_type_t norm = params->optimality_norm;
    test_solution_metrics_t metrics = {0};
    double *stationarity = (double *)calloc(n > 0 ? (size_t)n : 1, sizeof(double));
    double objective_norm = 0, bound_norm = 0;
    metrics.primal_objective = problem->objective_constant;
    metrics.dual_objective = sign * problem->objective_constant;
    for (int j = 0; j < n; ++j)
    {
        const double x = result->primal_solution[j];
        const double lower = problem->variable_lower_bound[j];
        const double upper = problem->variable_upper_bound[j];
        metrics.primal_residual = test_norm_add(metrics.primal_residual,
            x - fmin(fmax(x, lower), upper), norm);
        metrics.primal_objective += problem->objective_vector[j] * x;
        objective_norm = test_norm_add(objective_norm, problem->objective_vector[j], norm);
        stationarity[j] = problem->objective_vector[j] - result->reduced_cost[j];
        metrics.dual_objective += test_bound_support(sign * result->reduced_cost[j], lower, upper);
    }
    for (int i = 0; i < m; ++i)
    {
        double activity = 0;
        for (int k = problem->constraint_matrix_row_pointers[i];
             k < problem->constraint_matrix_row_pointers[i + 1]; ++k)
        {
            const int j = problem->constraint_matrix_col_indices[k];
            const double value = problem->constraint_matrix_values[k];
            activity += value * result->primal_solution[j];
            stationarity[j] -= value * result->dual_solution[i];
        }
        const double lower = problem->constraint_lower_bound[i];
        const double upper = problem->constraint_upper_bound[i];
        metrics.primal_residual = test_norm_add(metrics.primal_residual,
            activity - fmin(fmax(activity, lower), upper), norm);
        metrics.dual_objective += test_bound_support(sign * result->dual_solution[i], lower, upper);
        if (isfinite(lower) && lower != upper)
            bound_norm = test_norm_add(bound_norm, lower, norm);
        if (isfinite(upper))
            bound_norm = test_norm_add(bound_norm, upper, norm);
    }
    for (int j = 0; j < n; ++j)
        metrics.dual_residual = test_norm_add(metrics.dual_residual, stationarity[j], norm);
    free(stationarity);
    metrics.dual_objective *= sign;
    if (!isfinite(metrics.dual_objective)) metrics.dual_residual = INFINITY;
    metrics.relative_primal_residual = metrics.primal_residual / (1 + bound_norm);
    metrics.relative_dual_residual = metrics.dual_residual / (1 + objective_norm);
    metrics.relative_gap = fabs(metrics.primal_objective - metrics.dual_objective) /
        (1 + fabs(metrics.primal_objective) + fabs(metrics.dual_objective));
    return metrics;
}

static int test_solution_is_optimal(const test_solution_metrics_t *metrics,
                                  const termination_criteria_t *criteria)
{
    return isfinite(metrics->relative_primal_residual) &&
        isfinite(metrics->relative_dual_residual) && isfinite(metrics->relative_gap) &&
        metrics->relative_primal_residual <= criteria->eps_feasible_relative &&
        metrics->relative_dual_residual <= criteria->eps_feasible_relative &&
        metrics->relative_gap <= criteria->eps_optimal_relative;
}
