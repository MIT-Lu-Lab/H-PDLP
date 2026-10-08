#include "hpdlp.h"
#include "test_checks.h"
#include "utils.h"
#include <math.h>

static int check_bound_cleaning(void)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    CHECK(params.scaling_params.infinite_bound == 1e20);

    const double infinite_bound = params.scaling_params.infinite_bound;

    double lower[5] = {-1e30, -1e20, -3.0, 1e30, -INFINITY};
    double upper[5] = {1e30, 1e20, 4.0, 1e31, 7.0};
    double *working_lower = lower;
    double *working_upper = upper;

    replace_large_bounds_with_infinity(lower, upper, &working_lower, &working_upper, 5, infinite_bound);

    CHECK(working_lower != lower && working_upper != upper);
    CHECK(isinf(working_lower[0]) && working_lower[0] < 0.0);
    CHECK(isinf(working_upper[0]) && working_upper[0] > 0.0);
    CHECK(isinf(working_lower[1]) && working_lower[1] < 0.0);
    CHECK(isinf(working_upper[1]) && working_upper[1] > 0.0);
    CHECK(working_lower[2] == -3.0 && working_upper[2] == 4.0);
    CHECK(working_lower[3] == 1e30);
    CHECK(isinf(working_upper[3]) && working_upper[3] > 0.0);
    CHECK(isinf(working_lower[4]) && working_lower[4] < 0.0);
    CHECK(working_upper[4] == 7.0);
    CHECK(lower[0] == -1e30 && upper[0] == 1e30);
    free(working_lower);
    free(working_upper);

    double small_lower[2] = {-1.0, 0.0};
    double small_upper[2] = {2.0, 3.0};
    working_lower = small_lower;
    working_upper = small_upper;
    replace_large_bounds_with_infinity(
        small_lower, small_upper, &working_lower, &working_upper, 2, infinite_bound);
    CHECK(working_lower == small_lower && working_upper == small_upper);

    double objective[2] = {3.0, 4.0};
    double constraint_lower[3] = {-1e30, 2.0, 5.0};
    double constraint_upper[3] = {1e30, 6.0, 5.0};
    lp_problem_t problem;
    problem.num_variables = 2;
    problem.num_constraints = 3;
    problem.objective_vector = objective;
    problem.constraint_lower_bound = constraint_lower;
    problem.constraint_upper_bound = constraint_upper;

    double objective_norm = 0.0;
    double bound_norm = 0.0;

    compute_lp_problem_norms(&problem, NORM_TYPE_L2, infinite_bound, &objective_norm, &bound_norm);
    CHECK(fabs(objective_norm - 5.0) < 1e-12);
    CHECK(fabs(bound_norm - sqrt(4.0 + 36.0 + 25.0)) < 1e-12);

    compute_lp_problem_norms(&problem, NORM_TYPE_L_INF, infinite_bound, &objective_norm, &bound_norm);
    CHECK(fabs(objective_norm - 4.0) < 1e-12);
    CHECK(fabs(bound_norm - 6.0) < 1e-12);

    return 0;
}

int main(void)
{
    int failed = 0;
    failed |= check_bound_cleaning();
    return failed || failures != 0;
}
