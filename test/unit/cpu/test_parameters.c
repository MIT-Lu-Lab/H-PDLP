#include "hopdlp.h"
#include "test_checks.h"
#include <math.h>
#include <stdio.h>

static int check_active_set_parameters(void)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    check("step_size_boost defaults on",
          params.active_set_params.step_size_boost);
    check("window_iterations defaults to 10000", params.active_set_params.window_iterations == 10000);
    check("relative_tol defaults to 1e-8",
          params.active_set_params.relative_tol == 1e-8);
    check("safety_factor defaults to 0.9", params.active_set_params.safety_factor == 0.9);
    check("min_raise_ratio defaults to 1.1", params.active_set_params.min_raise_ratio == 1.1);
    check("reestimate_change_ratio defaults to 0.01",
          params.active_set_params.reestimate_change_ratio == 0.01);
    check("divergence_ceiling_ratio defaults to 0.7",
          params.active_set_params.divergence_ceiling_ratio == 0.7);
    check("divergence_margin defaults to 0.05", params.active_set_params.divergence_margin == 0.05);
    check("max_reverts defaults to 2", params.active_set_params.max_reverts == 2);
    printf("test_active_set_params: %d failure(s)\n", failures);
    return failures != 0;
}

static int expect_rejected(lp_problem_t *problem, pdhg_parameters_t *params,
                           const char *field)
{
    hopdlp_result_t *result = solve_lp_problem(problem, params);
    if (result)
    {
        fprintf(stderr, "invalid %s accepted (presolve=%d)\n", field, params->presolve);
        hopdlp_result_free(result);
        return 1;
    }
    return 0;
}



#define REJECT(field, value) \
    do \
    { \
        set_default_parameters(&params); \
        params.verbose = false; \
        params.presolve = presolve; \
        params.field = (value); \
        failures += expect_rejected(problem, &params, #field); \
    } while (0)

static int check_parameter_validation(void)
{
    const double one = 1.0;
    const double zero = 0.0;
    matrix_desc_t matrix = {0};
    matrix.fmt = matrix_dense;
    matrix.m = 1;
    matrix.n = 1;
    matrix.data.dense.A = &one;
    lp_problem_t *problem = create_lp_problem(
        &one, &matrix, &one, &one, &zero, &one, NULL);
    if (!problem)
        return 1;

    int failures = 0;
    pdhg_parameters_t params;
    for (int presolve = 0; presolve < 2; ++presolve)
    {
        REJECT(termination_evaluation_frequency, 2);
        REJECT(reflection_coefficient, NAN);
        REJECT(scaling_params.geometric_mean_iterations, -1);
        REJECT(scaling_params.l_inf_ruiz_iterations, -1);
        REJECT(scaling_params.pock_chambolle_alpha, 3.0);
        REJECT(scaling_params.matrix_zero_tol, INFINITY);
        REJECT(scaling_params.infinite_bound, 0.0);
        REJECT(power_iter_params.max_iter, 0);
        REJECT(power_iter_params.tol, NAN);
        REJECT(step_size_params.step_size_factor, 1.0);
        REJECT(step_size_params.k_p, NAN);
        REJECT(step_size_params.k_i, INFINITY);
        REJECT(step_size_params.k_d, -INFINITY);
        REJECT(step_size_params.i_smooth, 1.0);
        REJECT(cone_params.max_iter, 0);
        REJECT(cone_params.tol, 0.0);
        REJECT(active_set_params.window_iterations, 0);
        REJECT(active_set_params.relative_tol, -1.0);
        REJECT(active_set_params.safety_factor, 1.0);
        REJECT(active_set_params.min_raise_ratio, 0.5);
        REJECT(active_set_params.reestimate_change_ratio, -1.0);
        REJECT(active_set_params.divergence_ceiling_ratio, 2.0);
        REJECT(active_set_params.divergence_margin, -1.0);
        REJECT(active_set_params.max_reverts, -1);
        REJECT(termination_criteria.eps_optimal_relative, 0.0);
        REJECT(termination_criteria.eps_feasible_relative, INFINITY);
        REJECT(termination_criteria.eps_farkas_homogeneous, NAN);
        REJECT(termination_criteria.eps_farkas_displacement, -1.0);
        REJECT(termination_criteria.time_sec_limit, 0.0);
        REJECT(termination_criteria.iteration_limit, 0);
        REJECT(restart_params.artificial_restart_threshold, 0.0);
        REJECT(restart_params.sufficient_reduction_for_restart, 0.9);
        REJECT(restart_params.necessary_reduction_for_restart, 0.1);
    }
    set_default_parameters(&params);
    params.verbose = false;
    char error[256];
    failures += hopdlp_validate_parameters(&params, error, sizeof(error)) != 0;
    lp_problem_free(problem);
    printf("test_invalid_parameters: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}

#undef REJECT

int main(void)
{
    int failed = 0;
    failed |= check_active_set_parameters();
    failed |= check_parameter_validation();
    return failed || failures != 0;
}
