#include "gpu_support.h"
#include "hpdlp.h"
#include "result_check.h"
#include "test_checks.h"

static void check_optimal_model(int m, int n, const double *c, const double *a,
    const double *row_lower, const double *row_upper,
    const double *lower, const double *upper, double offset,
    int maximize, int configuration, double expected_objective)
{
    matrix_desc_t matrix = {0};
    matrix.m = m;
    matrix.n = n;
    matrix.fmt = matrix_dense;
    matrix.data.dense.A = a;
    lp_problem_t *problem = create_lp_problem(c, &matrix, row_lower, row_upper,
                                            lower, upper, &offset);
    CHECK(problem != NULL);
    if (!problem) return;
    set_objective_sense(problem, maximize ? OBJECTIVE_SENSE_MAXIMIZE : OBJECTIVE_SENSE_MINIMIZE);
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = false;
    params.presolve = configuration == 1;
    params.optimality_norm = configuration == 2 ? NORM_TYPE_L_INF : NORM_TYPE_L2;
    params.active_set_params.step_size_boost = configuration != 3;
    params.termination_criteria.eps_optimal_relative = 1e-7;
    params.termination_criteria.eps_feasible_relative = 1e-7;
    params.termination_criteria.iteration_limit = 20000;
    if (configuration == 2)
        params.scaling_params.bound_objective_rescaling = false;
    hpdlp_result_t *result = solve_lp_problem(problem, &params);
    CHECK(result != NULL);
    if (result)
    {
        CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL);
        if (result->termination_reason == TERMINATION_REASON_OPTIMAL)
        {
            test_solution_metrics_t metrics = test_evaluate_solution(problem, &params, result);
            CHECK(test_solution_is_optimal(&metrics, &params.termination_criteria));
            CHECK(fabs(metrics.primal_objective - expected_objective) < 1e-5);
            CHECK(fabs(result->primal_objective_value - metrics.primal_objective) < 1e-9);
            CHECK(fabs(result->dual_objective_value - metrics.dual_objective) < 1e-9);
        }
    }
    hpdlp_result_free(result);
    lp_problem_free(problem);
}

static void check_certificates(void)
{
    const double objective = 1;
    const double matrix_values[] = {1, 1};
    const double lower[] = {2, -INFINITY}, upper[] = {INFINITY, 1};
    matrix_desc_t matrix = {0};
    matrix.m = 2; matrix.n = 1; matrix.fmt = matrix_dense;
    matrix.data.dense.A = matrix_values;
    lp_problem_t *problem = create_lp_problem(&objective, &matrix, lower, upper, NULL, NULL, NULL);
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = params.presolve = false;
    params.termination_criteria.eps_farkas_homogeneous = 1e-7;
    params.termination_criteria.eps_farkas_displacement = 1e-7;
    hpdlp_result_t *result = solve_lp_problem(problem, &params);
    CHECK(result && result->termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE);
    if (result && result->dual_solution && result->reduced_cost)
    {
        const double *y = result->dual_solution;
        CHECK(result->primal_solution == NULL);
        CHECK(y[0] >= 0 && y[1] <= 0);
        CHECK(fabs(y[0] + y[1] + result->reduced_cost[0]) < 1e-6);
        CHECK(fabs(result->reduced_cost[0]) < 1e-6);
        CHECK(2 * y[0] + y[1] > 0);
    }
    else CHECK(0);
    hpdlp_result_free(result);
    lp_problem_free(problem);
    const double decreasing = -1, zero = 0;
    matrix.m = matrix.n = 1;
    problem = create_lp_problem(&decreasing, &matrix, &zero, NULL, NULL, NULL, NULL);
    result = solve_lp_problem(problem, &params);
    CHECK(result && result->termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE);
    if (result && result->primal_solution)
    {
        CHECK(result->primal_solution[0] > 0);
        CHECK(result->dual_solution == NULL && result->reduced_cost == NULL);
    }
    else CHECK(0);
    hpdlp_result_free(result);
    lp_problem_free(problem);
}

static void check_null_arguments(void)
{
    const double objective = 1, matrix_value = 1, lower = 1;
    matrix_desc_t matrix = {0};
    matrix.m = matrix.n = 1; matrix.fmt = matrix_dense;
    matrix.data.dense.A = &matrix_value;
    lp_problem_t *problem = create_lp_problem(&objective, &matrix, &lower, NULL, NULL, NULL, NULL);
    CHECK(problem != NULL);
    check("a NULL problem is rejected", solve_lp_problem(NULL, NULL) == NULL);
    hpdlp_result_t *result = solve_lp_problem(problem, NULL);
    check("NULL parameters solve with the defaults",
          result && result->termination_reason == TERMINATION_REASON_OPTIMAL);
    hpdlp_result_free(result);
    lp_problem_free(problem);
}

int main(void)
{
    const int status = test_gpu_status();
    if (status) return status;
    const double c[] = {1, 1}, a[] = {1, 2, 0, 1, 3, 2};
    const double row_lower[] = {5, -INFINITY, -INFINITY}, row_upper[] = {5, 2, 8};
    const double lower[] = {0, 0};
    for (int configuration = 0; configuration < 4; ++configuration)
        check_optimal_model(3, 2, c, a, row_lower, row_upper, lower, NULL,
                            7, 0, configuration, 10);
    check_optimal_model(3, 2, c, a, row_lower, row_upper, lower, NULL, 2, 1, 0, 5.25);
    const double box_c[] = {1, -2}, box_lower[] = {-3, 2}, box_upper[] = {1, 4};
    check_optimal_model(0, 2, box_c, NULL, NULL, NULL, box_lower, box_upper, 0, 0, 2, -11);
    const double fixed_c[] = {1, 2}, equality[] = {1, 1}, rhs = 3;
    const double fixed_lower[] = {1, -INFINITY}, fixed_upper[] = {1, INFINITY};
    check_optimal_model(1, 2, fixed_c, equality, &rhs, &rhs, fixed_lower, fixed_upper, 0, 0, 1, 5);
    const double one = 1;
    check_optimal_model(1, 2, c, equality, &one, NULL, lower, NULL, 0, 0, 3, 1);
    const double zero = 0;
    check_optimal_model(1, 0, NULL, NULL, &zero, &one, NULL, NULL, 3, 0, 0, 3);
    check_optimal_model(1, 0, NULL, NULL, &zero, &one, NULL, NULL, 3, 0, 2, 3);
    check_certificates();
    check_null_arguments();
    return failures != 0;
}
