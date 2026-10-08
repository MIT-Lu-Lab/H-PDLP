#include "gpu_support.h"
#include "hpdlp.h"
#include "test_checks.h"
#include <math.h>
#include <stdio.h>

static int check_empty_solve(const matrix_desc_t *matrix)
{
    const double offsets[] = {0.0, -3.5, 4.5};
    int failures = 0;
    for (int i = 0; i < 3; ++i)
    {
        lp_problem_t *problem = create_lp_problem(
            NULL, matrix, NULL, NULL, NULL, NULL, &offsets[i]);
        if (!problem)
        {
            fprintf(stderr, "empty problem creation failed\n");
            return 1;
        }
        for (int maximize = 0; maximize < 2; ++maximize)
        {
            set_objective_sense(problem, maximize ? OBJECTIVE_SENSE_MAXIMIZE
                                                 : OBJECTIVE_SENSE_MINIMIZE);
            for (int presolve = 0; presolve < 2; ++presolve)
            {
                pdhg_parameters_t params;
                set_default_parameters(&params);
                params.verbose = false;
                params.presolve = presolve;
                hpdlp_result_t *result = solve_lp_problem(problem, &params);
                if (!result ||
                    result->termination_reason != TERMINATION_REASON_OPTIMAL ||
                    result->primal_objective_value != offsets[i] ||
                    result->dual_objective_value != offsets[i] ||
                    result->total_count != 0 ||
                    result->num_variables != 0 || result->num_constraints != 0 ||
                    result->num_nonzeros != 0 ||
                    result->primal_solution != NULL || result->dual_solution != NULL ||
                    result->reduced_cost != NULL ||
                    result->absolute_primal_residual != 0.0 ||
                    result->relative_primal_residual != 0.0 ||
                    result->absolute_dual_residual != 0.0 ||
                    result->relative_dual_residual != 0.0 ||
                    result->objective_gap != 0.0 ||
                    result->relative_objective_gap != 0.0)
                {
                    fprintf(stderr, "empty solve failed: format=%d offset=%g maximize=%d presolve=%d\n",
                            matrix->fmt, offsets[i], maximize, presolve);
                    ++failures;
                }
                hpdlp_result_free(result);
            }
        }
        lp_problem_free(problem);
    }
    return failures;
}
static int check_empty_problem(void)
{
    matrix_desc_t matrix = {0};
    matrix.fmt = matrix_dense;
    return check_empty_solve(&matrix) != 0;
}

static void solve_and_check(const char *label, int num_constraints, norm_type_t norm)
{
    const double objective[2] = {1.0, 2.0};
    const int row_ptr[2] = {0, 0};
    const double constraint_lower[1] = {0.0};
    const double constraint_upper[1] = {0.0};
    const double variable_lower[2] = {0.0, 0.0};
    const double variable_upper[2] = {5.0, 3.0};
    const double objective_constant = 0.0;

    matrix_desc_t A_desc;
    A_desc.m = num_constraints;
    A_desc.n = 2;
    A_desc.fmt = matrix_csr;
    A_desc.data.csr.nnz = 0;
    A_desc.data.csr.row_ptr = row_ptr;
    A_desc.data.csr.col_ind = NULL;
    A_desc.data.csr.vals = NULL;

    lp_problem_t *problem = create_lp_problem(
        objective, &A_desc, constraint_lower, constraint_upper,
        variable_lower, variable_upper, &objective_constant);
    if (problem == NULL)
    {
        printf("  FAIL: %s could not build the problem\n", label);
        ++failures;
        return;
    }

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    params.optimality_norm = norm;
    params.scaling_params.bound_objective_rescaling = false;
    params.termination_criteria.iteration_limit = 400;

    hpdlp_result_t *result = solve_lp_problem(problem, &params);
    if (result == NULL || result->primal_solution == NULL)
    {
        printf("  FAIL: %s produced no solution\n", label);
        ++failures;
        lp_problem_free(problem);
        return;
    }

    printf("%-26s objective %.17g  x = [%.17g, %.17g]\n", label,
           result->primal_objective_value, result->primal_solution[0], result->primal_solution[1]);
    check("termination is OPTIMAL", result->termination_reason == TERMINATION_REASON_OPTIMAL);
    check("objective is the closed-form optimum 0", fabs(result->primal_objective_value) < 1e-6);
    check("x0 sits at its lower bound", fabs(result->primal_solution[0]) < 1e-6);
    check("x1 sits at its lower bound", fabs(result->primal_solution[1]) < 1e-6);

    hpdlp_result_free(result);
    lp_problem_free(problem);
}

static int check_zero_constraint_norm(void)
{
    solve_and_check("no constraints, L2", 0, NORM_TYPE_L2);
    solve_and_check("no constraints, Linf", 0, NORM_TYPE_L_INF);
    solve_and_check("one all-zero constraint", 1, NORM_TYPE_L_INF);
    printf("%s (%d failure(s))\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures != 0;
}

static void check_signs(const char *label, bool presolve)
{
    const double objective[3] = {-1.0, 2.0, 0.0};
    const int row_ptr[2] = {0, 3};
    const int col_ind[3] = {0, 1, 2};
    const double values[3] = {1.0, 1.0, 1.0};
    const double constraint_lower[1] = {3.0};
    const double constraint_upper[1] = {3.0};
    const double variable_lower[3] = {-INFINITY, 0.0, -INFINITY};
    const double variable_upper[3] = {5.0, INFINITY, INFINITY};
    const double objective_constant = 0.0;

    matrix_desc_t A_desc;
    A_desc.m = 1;
    A_desc.n = 3;
    A_desc.fmt = matrix_csr;
    A_desc.data.csr.nnz = 3;
    A_desc.data.csr.row_ptr = row_ptr;
    A_desc.data.csr.col_ind = col_ind;
    A_desc.data.csr.vals = values;

    lp_problem_t *problem = create_lp_problem(
        objective, &A_desc, constraint_lower, constraint_upper,
        variable_lower, variable_upper, &objective_constant);
    if (problem == NULL)
    {
        printf("  FAIL: %s could not build the problem\n", label);
        ++failures;
        return;
    }

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = false;
    params.presolve = presolve;

    hpdlp_result_t *result = solve_lp_problem(problem, &params);
    if (result == NULL || result->reduced_cost == NULL)
    {
        printf("  FAIL: %s produced no reduced cost\n", label);
        ++failures;
        lp_problem_free(problem);
        return;
    }

    printf("%-14s rc = [%.17g, %.17g, %.17g]\n", label,
           result->reduced_cost[0], result->reduced_cost[1], result->reduced_cost[2]);

    check("termination is OPTIMAL", result->termination_reason == TERMINATION_REASON_OPTIMAL);
    check("objective is -5", fabs(result->primal_objective_value + 5.0) < 1e-3);
    check("no lower bound implies rc <= 0", result->reduced_cost[0] <= 0.0);
    check("no upper bound implies rc >= 0", result->reduced_cost[1] >= 0.0);
    check("free variable implies rc == 0", result->reduced_cost[2] == 0.0);

    const double tolerance = 1e-3;
    const double dual = result->dual_solution[0];
    for (int j = 0; j < 3; ++j)
    {
        const double rc = result->reduced_cost[j];
        const double x = result->primal_solution[j];
        char message[128];
        snprintf(message, sizeof(message), "c - A^T*y - rc vanishes on column %d", j);
        check(message, fabs(objective[j] - values[j] * dual - rc) < tolerance);
        snprintf(message, sizeof(message), "a positive rc sits at the lower bound on column %d", j);
        check(message, rc <= tolerance || fabs(x - variable_lower[j]) < tolerance);
        snprintf(message, sizeof(message), "a negative rc sits at the upper bound on column %d", j);
        check(message, rc >= -tolerance || fabs(x - variable_upper[j]) < tolerance);
    }

    hpdlp_result_free(result);
    lp_problem_free(problem);
}

static int check_reduced_cost_signs(void)
{
    check_signs("presolve on", true);
    check_signs("presolve off", false);
    printf("%s (%d failure(s))\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures != 0;
}

int main(void)
{
    const int status = test_gpu_status();
    if (status) return status;
    int failed = 0;
    failed |= check_empty_problem();
    failed |= check_zero_constraint_norm();
    failed |= check_reduced_cost_signs();
    return failed || failures != 0;
}
