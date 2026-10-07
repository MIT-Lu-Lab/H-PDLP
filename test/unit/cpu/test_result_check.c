#include "result_check.h"
#include "test_checks.h"
#include <string.h>

int main(void)
{
    double c[] = {3, -4}, lower[] = {0, 0}, upper[] = {1, 1};
    double x[] = {0, 1}, y[] = {0}, z[] = {3, -4};
    double values[] = {1, 2}, row_lower[] = {0}, row_upper[] = {2};
    int pointers[] = {0, 2}, indices[] = {0, 1};
    lp_problem_t problem = {0};
    problem.num_variables = 2;
    problem.num_constraints = 1;
    problem.objective_vector = c;
    problem.objective_constant = 7;
    problem.variable_lower_bound = lower;
    problem.variable_upper_bound = upper;
    problem.constraint_lower_bound = row_lower;
    problem.constraint_upper_bound = row_upper;
    problem.constraint_matrix_row_pointers = pointers;
    problem.constraint_matrix_col_indices = indices;
    problem.constraint_matrix_values = values;
    hopdlp_result_t result = {0};
    result.primal_solution = x;
    result.dual_solution = y;
    result.reduced_cost = z;
    result.primal_objective_value = 123;
    result.termination_reason = TERMINATION_REASON_OPTIMAL;
    hopdlp_result_t snapshot = result;
    pdhg_parameters_t params;
    set_default_parameters(&params);
    for (int maximize = 0; maximize < 2; ++maximize)
    {
        problem.objective_sense = maximize ? OBJECTIVE_SENSE_MAXIMIZE : OBJECTIVE_SENSE_MINIMIZE;
        c[0] = z[0] = maximize ? -3 : 3;
        c[1] = z[1] = maximize ? 4 : -4;
        test_solution_metrics_t metrics = test_evaluate_solution(&problem, &params, &result);
        const double expected = maximize ? 11 : 3;
        CHECK(metrics.primal_objective == expected);
        CHECK(metrics.dual_objective == expected);
        CHECK(test_solution_is_optimal(&metrics, &params.termination_criteria));
        CHECK(memcmp(&result, &snapshot, sizeof(result)) == 0);
    }
    problem.objective_sense = OBJECTIVE_SENSE_MINIMIZE;
    c[0] = 3; c[1] = -4;
    x[0] = -3; x[1] = -4;
    z[0] = z[1] = 0;
    const norm_type_t norms[] = {NORM_TYPE_L2, NORM_TYPE_L_INF};
    for (int i = 0; i < 2; ++i)
    {
        params.optimality_norm = norms[i];
        test_solution_metrics_t metrics = test_evaluate_solution(&problem, &params, &result);
        CHECK(fabs(metrics.primal_residual - (i == 0 ? sqrt(146.0) : 11)) < 1e-12);
        CHECK(metrics.dual_residual == (i == 0 ? 5 : 4));
        CHECK(!test_solution_is_optimal(&metrics, &params.termination_criteria));
        CHECK(memcmp(&result, &snapshot, sizeof(result)) == 0);
        CHECK(z[0] == 0 && z[1] == 0);
    }
    x[0] = 0; x[1] = 1; z[0] = 3; z[1] = -4;
    lower[0] = -INFINITY;
    test_solution_metrics_t metrics = test_evaluate_solution(&problem, &params, &result);
    CHECK(!test_solution_is_optimal(&metrics, &params.termination_criteria));
    CHECK(isinf(metrics.dual_residual));
    CHECK(result.termination_reason == TERMINATION_REASON_OPTIMAL);
    return failures != 0;
}
