#include "cuda_macros.h"
#include "gpu_support.h"
#include "hpdlp.h"
#include "result.h"
#include "setup.h"
#include "termination_checker.h"
#include "test_checks.h"
#include <cuda_runtime.h>
#include <float.h>
#include <initializer_list>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void check_small_box_certificate(bool scaled, norm_type_t norm)
{
    const double objective[] = {2e-5};
    const double lower[] = {0.0}, upper[] = {1e-4};
    const double constant = 0.0;
    const int row_ptr[] = {0, 1}, col_ind[] = {0};
    const double values[] = {1.0};
    matrix_desc_t matrix = {};
    matrix.m = matrix.n = 1;
    matrix.fmt = matrix_csr;
    matrix.data.csr.nnz = 1;
    matrix.data.csr.row_ptr = row_ptr;
    matrix.data.csr.col_ind = col_ind;
    matrix.data.csr.vals = values;
    lp_problem_t *problem = create_lp_problem(
        objective, &matrix, lower, upper, lower, upper, &constant);
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = params.presolve = false;
    params.optimality_norm = norm;
    params.termination_criteria.eps_feasible_relative = 1e-8;
    params.termination_criteria.eps_optimal_relative = 1e-8;
    if (!scaled)
    {
        disable_rescaling(&params);
    }
    pdhg_solver_state_t *state = create_solver_state(problem, &params);
    double column_scale, row_scale;
    CUDA_CHECK(cudaMemcpy(&column_scale, state->variable_rescaling_d,
                          sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&row_scale, state->constraint_rescaling_d,
                          sizeof(double), cudaMemcpyDeviceToHost));
    const double tau = 2.0;
    const double x = 5e-5 * column_scale * state->constraint_bound_rescaling * tau;
    const double y = 1e-5 * row_scale * state->objective_vector_rescaling * tau;
    CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution_d, &x, sizeof(x), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution_d, &y, sizeof(y), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution_d, &y, sizeof(y), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->hsd_primal_scale_pdhg_d, &tau, sizeof(tau), cudaMemcpyHostToDevice));
    compute_termination_metrics(state, &params);
    state->control.start_time = clock();
    check_termination_criteria(state, &params.termination_criteria);
    if (state->control.termination_reason != TERMINATION_REASON_OPTIMAL)
    {
        fprintf(stderr, "small box certificate rejected (scaled=%d norm=%d)\n", scaled, norm);
        ++failures;
    }
    else
    {
        hpdlp_result_t *result = create_result_from_state(state, problem);
        if (!(fabs(result->reduced_cost[0] - 1e-5) <= 1e-15 &&
              fabs(result->primal_solution[0] - 5e-5) <= 1e-15 &&
              result->absolute_dual_residual <= 1e-15 &&
              fabs(result->dual_solution[0] - 1e-5) <= 1e-15 &&
              fabs(result->objective_gap - 1e-9) <= 1e-15 &&
              result->dual_objective_value == 0.0))
        {
            fprintf(stderr, "small box exported certificate disagrees with its metrics\n");
            ++failures;
        }
        hpdlp_result_free(result);
    }
    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}

static void check_bound_candidates(norm_type_t norm)
{
    const int n = 7;
    const double c[n] = {3, -4, 5, -6, 7, -8, 9};
    const double lb[n] = {0, 0, 0, -INFINITY, -INFINITY, 2, 0};
    const double ub[n] = {1, 1, INFINITY, 1, INFINITY, 2, INFINITY};
    const double x[n] = {0.5, 0.5, 0.5, 0.5, 0, 2, 0.5};
    const double expected_bound_z[n] = {3, -4, 5, -6, 0, -8, 9};
    const double constant = 0, row_lb = -INFINITY, row_ub = INFINITY;
    const int row_ptr[] = {0, 0};
    matrix_desc_t matrix = {};
    matrix.m = 1; matrix.n = n;
    matrix.fmt = matrix_csr;
    matrix.data.csr.row_ptr = row_ptr;
    lp_problem_t *problem = create_lp_problem(c, &matrix, &row_lb, &row_ub, lb, ub, &constant);
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = params.presolve = false;
    params.optimality_norm = norm;
    disable_rescaling(&params);
    pdhg_solver_state_t *state = create_solver_state(problem, &params);
    const double tau = 1;
    CUDA_CHECK(cudaMemcpy(state->hsd_primal_scale_pdhg_d, &tau, sizeof(tau), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution_d, x, sizeof(x), cudaMemcpyHostToDevice));
    compute_termination_metrics(state, &params);
    double z[n];
    CUDA_CHECK(cudaMemcpy(z, state->pdhg_reduced_cost_d, sizeof(z), cudaMemcpyDeviceToHost));
    for (int j = 0; j < n; ++j)
        if (z[j] != 0) { fprintf(stderr, "unobserved projection column %d\n", j); ++failures; }
    if (!(fabs(state->metrics_h->absolute_dual_residual - (norm == NORM_TYPE_L_INF ? 9 : sqrt(280.0))) <= 1e-12) ||
        state->metrics_h->dual_objective != 0)
    {
        fprintf(stderr, "primary candidate residual/support mismatch\n");
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(z, state->bound_multiplier_d, sizeof(z), cudaMemcpyDeviceToHost));
    for (int j = 0; j < n; ++j)
        check("bound candidate respects finite bound signs", z[j] == expected_bound_z[j]);
    check("bound candidate retains the free-variable residual",
          state->metrics_h->bound_multiplier_absolute_dual_residual == 7);
    check("bound candidate support", state->metrics_h->bound_multiplier_dual_objective == -26);

    const double opposite_c[n] = {3, -4, -5, 6, 7, -8, -9};
    CUDA_CHECK(cudaMemcpy(state->objective_vector_d, opposite_c, sizeof(opposite_c), cudaMemcpyHostToDevice));
    compute_termination_metrics(state, &params);
    const double expected_norm = norm == NORM_TYPE_L_INF ? 9 : sqrt(280.0);
    if (!(fabs(state->metrics_h->absolute_dual_residual - expected_norm) <= 1e-12))
    { fprintf(stderr, "one-sided stationarity residual mismatch\n"); ++failures; }
    const double expected_bound_norm = norm == NORM_TYPE_L_INF ? 9 : sqrt(191.0);
    check("invalid one-sided signs remain in the bound candidate residual",
          fabs(state->metrics_h->bound_multiplier_absolute_dual_residual - expected_bound_norm) < 1e-12);
    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}



static double *copy_to_device(const double *values, size_t count)
{
    double *device_values;
    CUDA_CHECK(cudaMalloc(&device_values, count * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(device_values, values, count * sizeof(double), cudaMemcpyHostToDevice));
    return device_values;
}

static void check_values(const char *name, const double *actual,
                         const double *expected, int count)
{
    if (!actual)
    {
        fprintf(stderr, "%s is missing\n", name);
        ++failures;
        return;
    }
    for (int i = 0; i < count; ++i)
    {
        if (!isfinite(actual[i]) || fabs(actual[i] - expected[i]) > 1e-12)
        {
            fprintf(stderr, "%s[%d]: %.17g expected %.17g\n",
                    name, i, actual[i], expected[i]);
            ++failures;
        }
    }
}

static void check_device_values(const char *name, const double *device_values,
                                const double *expected, int count)
{
    double actual[2];
    CUDA_CHECK(cudaMemcpy(actual, device_values, count * sizeof(double), cudaMemcpyDeviceToHost));
    check_values(name, actual, expected, count);
}

static void check_result(pdhg_solver_state_t *state, lp_problem_t *problem,
                         termination_criteria_t *criteria,
                         termination_reason_t reason, bool homogeneous,
                         const double *primal, const double *dual,
                         const double *reduced_cost)
{
    state->control.termination_reason = TERMINATION_REASON_UNSPECIFIED;
    state->control.start_time = clock();
    check_termination_criteria(state, criteria);
    if (state->control.termination_reason != reason ||
        ((reason == TERMINATION_REASON_PRIMAL_INFEASIBLE ||
          reason == TERMINATION_REASON_DUAL_INFEASIBLE) &&
         state->control.certificate_from_homogeneous != homogeneous))
    {
        fprintf(stderr, "wrong termination decision: %d expected %d\n",
                state->control.termination_reason, reason);
        ++failures;
        return;
    }

    for (int repeat = 0; repeat < 2; ++repeat)
    {
        hpdlp_result_t *result = create_result_from_state(state, problem);
        if (primal)
            check_values("primal", result->primal_solution, primal, 2);
        else if (result->primal_solution)
            ++failures;
        if (dual)
            check_values("dual", result->dual_solution, dual, 1);
        else if (result->dual_solution)
            ++failures;
        if (reduced_cost)
            check_values("reduced cost", result->reduced_cost, reduced_cost, 2);
        else if (result->reduced_cost)
            ++failures;
        hpdlp_result_free(result);
    }
}



static void check_result_export(void)
{
    check_bound_candidates(NORM_TYPE_L_INF);
    check_bound_candidates(NORM_TYPE_L2);
    check_small_box_certificate(false, NORM_TYPE_L_INF);
    check_small_box_certificate(false, NORM_TYPE_L2);
    check_small_box_certificate(true, NORM_TYPE_L_INF);
    check_small_box_certificate(true, NORM_TYPE_L2);
    const double primal_source[] = {8, 12};
    const double dual_source[] = {30};
    const double reduced_cost_source[] = {-40, 20};
    const double delta_primal[] = {16, 20};
    const double delta_dual[] = {50};
    const double dual_product[] = {3, -7};
    const double primal_infeasibility_reduced_cost[] = {-6, 8};
    const double variable_rescaling[] = {2, 4};
    const double constraint_rescaling[] = {5};
    const double variable_lower_bound[] = {-INFINITY, 0};
    const double variable_upper_bound[] = {2, INFINITY};
    const double expected_primal[] = {1, 0.75};
    const double expected_homogeneous_primal[] = {0.4, 0.3};
    const double expected_delta_primal[] = {0.5, 0.3125};
    const double expected_dual[] = {0.3};
    const double expected_delta_dual[] = {0.25};
    const double expected_reduced_cost[] = {-4, 4};
    const double expected_homogeneous_reduced_cost[] = {-0.6, 1.6};
    const double expected_displacement_reduced_cost[] = {-0.15, 0.7};

    pdhg_solver_state_t state = {};
    pdhg_metrics_t metrics = {};
    lp_problem_t problem = {};
    pdhg_parameters_t params;
    set_default_parameters(&params);
    problem.num_variables = state.num_variables = 2;
    problem.num_constraints = state.num_constraints = 1;
    state.var_bytes = 2 * sizeof(double);
    state.con_bytes = sizeof(double);
    state.num_blocks_primal = 1;
    state.num_blocks_dual = 1;
    state.metrics_h = &metrics;
    state.constraint_bound_rescaling = 2;
    state.objective_vector_rescaling = 10;
    CUDA_CHECK(cudaStreamCreate(&state.stream_A));

    state.pdhg_primal_solution_d = copy_to_device(primal_source, 2);
    state.pdhg_dual_solution_d = copy_to_device(dual_source, 1);
    state.pdhg_reduced_cost_d = copy_to_device(reduced_cost_source, 2);
    state.delta_primal_solution_d = copy_to_device(delta_primal, 2);
    state.delta_dual_solution_d = copy_to_device(delta_dual, 1);
    state.dual_product_d = copy_to_device(dual_product, 2);
    state.primal_infeasibility_reduced_cost_d =
        copy_to_device(primal_infeasibility_reduced_cost, 2);
    state.variable_rescaling_d = copy_to_device(variable_rescaling, 2);
    state.constraint_rescaling_d = copy_to_device(constraint_rescaling, 1);
    state.variable_lower_bound_d = copy_to_device(variable_lower_bound, 2);
    state.variable_upper_bound_d = copy_to_device(variable_upper_bound, 2);
    CUDA_CHECK(cudaMalloc(&state.variable_reduction_entries_d, state.var_bytes));
    CUDA_CHECK(cudaMalloc(&state.constraint_reduction_entries_d, state.con_bytes));

    termination_reductions_t reductions = {};
    reductions.constraint_support = 20;
    reductions.homogeneous_variable_support = 20;
    reductions.primal_infeasibility_constraint_support = 20;
    reductions.primal_infeasibility_variable_support = 60;
    reductions.objective_dot_primal = -100;
    reductions.dual_infeasibility_objective = -160;
    CUDA_CHECK(cudaMalloc(&state.termination_reductions_d, sizeof(reductions)));
    CUDA_CHECK(cudaMemcpy(state.termination_reductions_d, &reductions,
                          sizeof(reductions), cudaMemcpyHostToDevice));

    metrics.pdhg_primal_scale = 2;
    check_result(&state, &problem, &params.termination_criteria,
                 TERMINATION_REASON_OPTIMAL, false,
                 expected_primal, expected_dual, expected_reduced_cost);

    metrics.normalized_homogeneous_ratio = 1;
    metrics.primal_infeasibility_homogeneous_margin = 1;
    check_result(&state, &problem, &params.termination_criteria,
                 TERMINATION_REASON_PRIMAL_INFEASIBLE, true,
                 NULL, expected_dual, expected_homogeneous_reduced_cost);

    metrics.primal_infeasibility_homogeneous_margin = 0;
    metrics.dual_infeasibility_homogeneous_margin = 1;
    check_result(&state, &problem, &params.termination_criteria,
                 TERMINATION_REASON_DUAL_INFEASIBLE, true,
                 expected_homogeneous_primal, NULL, NULL);

    metrics.normalized_homogeneous_ratio = 0.5;
    metrics.dual_infeasibility_homogeneous_margin = 0;
    metrics.primal_infeasibility_displacement_margin = 1;
    check_result(&state, &problem, &params.termination_criteria,
                 TERMINATION_REASON_PRIMAL_INFEASIBLE, false,
                 NULL, expected_delta_dual, expected_displacement_reduced_cost);

    metrics.primal_infeasibility_displacement_margin = 0;
    metrics.dual_infeasibility_displacement_margin = 1;
    check_result(&state, &problem, &params.termination_criteria,
                 TERMINATION_REASON_DUAL_INFEASIBLE, false,
                 expected_delta_primal, NULL, NULL);

    metrics.pdhg_primal_scale = NAN;
    check_result(&state, &problem, &params.termination_criteria,
                 TERMINATION_REASON_NUMERICAL_FAILURE, false,
                 NULL, NULL, NULL);

    state.num_variables = problem.num_variables = 0;
    state.num_constraints = problem.num_constraints = 0;
    state.var_bytes = state.con_bytes = 0;
    state.control.termination_reason = TERMINATION_REASON_OPTIMAL;
    metrics.pdhg_primal_scale = 2;
    hpdlp_result_t *empty_result = create_result_from_state(&state, &problem);
    if (!empty_result->primal_solution || !empty_result->dual_solution ||
        !empty_result->reduced_cost)
        ++failures;
    hpdlp_result_free(empty_result);

    check_device_values("state primal", state.pdhg_primal_solution_d, primal_source, 2);
    check_device_values("state dual", state.pdhg_dual_solution_d, dual_source, 1);
    check_device_values("state reduced cost", state.pdhg_reduced_cost_d,
                        reduced_cost_source, 2);
    check_device_values("state delta primal", state.delta_primal_solution_d,
                        delta_primal, 2);
    check_device_values("state delta dual", state.delta_dual_solution_d,
                        delta_dual, 1);
    check_device_values("state dual product", state.dual_product_d,
                        dual_product, 2);
    check_device_values("state infeasibility reduced cost",
                        state.primal_infeasibility_reduced_cost_d,
                        primal_infeasibility_reduced_cost, 2);

    CUDA_CHECK(cudaFree(state.pdhg_primal_solution_d));
    CUDA_CHECK(cudaFree(state.pdhg_dual_solution_d));
    CUDA_CHECK(cudaFree(state.pdhg_reduced_cost_d));
    CUDA_CHECK(cudaFree(state.delta_primal_solution_d));
    CUDA_CHECK(cudaFree(state.delta_dual_solution_d));
    CUDA_CHECK(cudaFree(state.dual_product_d));
    CUDA_CHECK(cudaFree(state.primal_infeasibility_reduced_cost_d));
    CUDA_CHECK(cudaFree(state.variable_rescaling_d));
    CUDA_CHECK(cudaFree(state.constraint_rescaling_d));
    CUDA_CHECK(cudaFree(state.variable_lower_bound_d));
    CUDA_CHECK(cudaFree(state.variable_upper_bound_d));
    CUDA_CHECK(cudaFree(state.variable_reduction_entries_d));
    CUDA_CHECK(cudaFree(state.constraint_reduction_entries_d));
    CUDA_CHECK(cudaFree(state.termination_reductions_d));
    CUDA_CHECK(cudaStreamDestroy(state.stream_A));
}

static void check_projection_multipliers(bool scaled, norm_type_t norm, double tau)
{
    const double c[] = {3.7e-5, -3.7e-5, -1e-5, -2, -3, 4, 5, -6, -7, 8, 9, -10, 11, -12, -12, 13, 13};
    const double lb[] = {0, 0, 0, 0, -10, -10, 0, 0, -INFINITY, -INFINITY, -INFINITY, 2, 0, 0, 0, -10, -10};
    const double ub[] = {.04, .04, 1e11, .04, 0, 0, INFINITY, INFINITY, 0, 0, INFINITY, 2, 0, 10, 10, 0, 0};
    const double x[] = {.003, .003, .003, .03, -.003, -.003, 1, 1, -1, -1, 0, 2, 0, 5, 4.9, -5, -4.9};
    const double expected_z[] = {3.7e-5, 0, 0, -2, -3, 0, 5, 0, -7, 0, 0, -10, 11, -12, 0, 13, 0};
    const int n = sizeof(c) / sizeof(c[0]);
    int rows[n + 1], columns[n];
    double values[n], row_lb[n], row_ub[n];
    for (int j = 0; j < n; ++j)
    {
        rows[j] = columns[j] = j;
        values[j] = 1;
        row_lb[j] = -INFINITY;
        row_ub[j] = INFINITY;
    }
    rows[n] = n;
    matrix_desc_t matrix = {};
    matrix.m = matrix.n = n;
    matrix.fmt = matrix_csr;
    matrix.data.csr.nnz = n;
    matrix.data.csr.row_ptr = rows;
    matrix.data.csr.col_ind = columns;
    matrix.data.csr.vals = values;
    lp_problem_t *problem = create_lp_problem(c, &matrix, row_lb, row_ub, lb, ub, NULL);
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = params.presolve = false;
    params.optimality_norm = norm;
    if (!scaled)
    {
        disable_rescaling(&params);
    }
    pdhg_solver_state_t *state = create_solver_state(problem, &params);
    double scales[n], homogeneous_x[n];
    CUDA_CHECK(cudaMemcpy(scales, state->variable_rescaling_d, sizeof(scales), cudaMemcpyDeviceToHost));
    for (int j = 0; j < n; ++j)
        homogeneous_x[j] = x[j] * scales[j] * state->constraint_bound_rescaling * tau;
    CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution_d, homogeneous_x, sizeof(homogeneous_x), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->hsd_primal_scale_pdhg_d, &tau, sizeof(tau), cudaMemcpyHostToDevice));
    double observed[n];
    for (int j = 0; j < n; ++j)
        observed[j] = expected_z[j] * state->objective_vector_rescaling * tau / scales[j];
    CUDA_CHECK(cudaMemcpy(state->pdhg_primal_projection_multiplier_d, observed,
                          sizeof(observed), cudaMemcpyHostToDevice));
    compute_termination_metrics(state, &params);

    state->control.termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
    hpdlp_result_t *result = create_result_from_state(state, problem);
    double expected_norm = 0, expected_primal = 0, expected_dual = 0;
    for (int j = 0; j < n; ++j)
    {
        char label[100];
        snprintf(label, sizeof(label), "multiplier %d (scaled=%d norm=%d tau=%g)", j, scaled, norm, tau);
        check_close(label, result->reduced_cost[j], expected_z[j], 1e-11);
        check_close("exported primal", result->primal_solution[j], x[j], 1e-11);
        const double residual = c[j] - expected_z[j];
        expected_norm = norm == NORM_TYPE_L_INF ? fmax(expected_norm, fabs(residual)) : hypot(expected_norm, residual);
        expected_primal += c[j] * x[j];
        if (expected_z[j] > 0) expected_dual += lb[j] * expected_z[j];
        if (expected_z[j] < 0) expected_dual += ub[j] * expected_z[j];
    }
    check_close("dual residual", result->absolute_dual_residual, expected_norm, 1e-11);
    check_close("primal objective", result->primal_objective_value, expected_primal, 1e-11);
    check_close("dual objective", result->dual_objective_value, expected_dual, 1e-11);
    check_close("gap", result->relative_objective_gap,
        fabs(expected_primal - expected_dual) / (1 + fabs(expected_primal) + fabs(expected_dual)), 1e-11);
    hpdlp_result_free(result);
    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}



static void check_multiplier_export(void)
{
    for (bool scaled : {false, true})
        for (norm_type_t norm : {NORM_TYPE_L2, NORM_TYPE_L_INF})
            for (double tau : {0.25, 1.0, 2.0})
                check_projection_multipliers(scaled, norm, tau);
}

int main(void)
{
    const int status = test_gpu_status();
    if (status) return status;
    check_result_export();
    check_multiplier_export();
    return failures != 0;
}
