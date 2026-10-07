#include "hopdlp.h"
#include "mps_parser.h"
#include "test_checks.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int check_public_defaults(void)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);

    CHECK(params.presolve);
    CHECK(params.termination_evaluation_frequency > 0);
    CHECK(params.termination_criteria.iteration_limit > 0);
    CHECK(params.termination_criteria.time_sec_limit > 0.0);
    return 0;
}

static void check_api_numbers(void)
{
    const double a = 1;
    matrix_desc_t matrix = {0};
    matrix.m = matrix.n = 1;
    matrix.fmt = matrix_dense;
    matrix.data.dense.A = &a;
    const double cases[][7] = {
        {NAN, 0, 0, 2, 0, 2, 0}, {INFINITY, 0, 0, 2, 0, 2, 0},
        {-INFINITY, 0, 0, 2, 0, 2, 0}, {1, NAN, 0, 2, 0, 2, 0},
        {1, INFINITY, 0, 2, 0, 2, 0}, {1, -INFINITY, 0, 2, 0, 2, 0},
        {1, 0, NAN, 2, 0, 2, 0}, {1, 0, 0, NAN, 0, 2, 0},
        {1, 0, 0, 2, NAN, 2, 0}, {1, 0, 0, 2, 0, NAN, 0},
        {1, 0, 3, 2, 0, 2, 0}, {1, 0, 0, 2, 3, 2, 0},
        {1, 0, INFINITY, INFINITY, 0, 2, 0},
        {1, 0, -INFINITY, -INFINITY, 0, 2, 0},
        {1, 0, 0, 2, INFINITY, INFINITY, 0},
        {1, 0, 0, 2, -INFINITY, -INFINITY, 0},
        {1, 7, -INFINITY, INFINITY, -INFINITY, INFINITY, 1},
        {1, 0, 2, 2, 2, 2, 1}
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        const double *v = cases[i];
        lp_problem_t *p = create_lp_problem(v, &matrix, v + 4, v + 5, v + 2, v + 3, v + 1);
        if ((p != NULL) != (v[6] != 0))
        {
            fprintf(stderr, "FAIL: API numeric case %zu\n", i);
            ++failures;
        }
        lp_problem_free(p);
    }
}



static int check_model_numbers(void)
{
    check_api_numbers();
    return failures != 0;
}

static lp_problem_t *make_problem(void)
{
    static const double a = 1.0;
    static const double c = 1.0;
    static const double bound = 1.0;
    matrix_desc_t matrix;
    memset(&matrix, 0, sizeof(matrix));
    matrix.m = 1;
    matrix.n = 1;
    matrix.fmt = matrix_dense;
    matrix.data.dense.A = &a;
    return create_lp_problem(&c, &matrix, &bound, &bound, NULL, NULL, NULL);
}

static int check_start_values(void)
{
    lp_problem_t *prob = make_problem();
    check("fixture builds", prob != NULL);
    if (prob == NULL)
    {
        printf("FAILED (%d failure(s))\n", failures + 1);
        return 1;
    }

    const double good_primal[] = {0.5};
    const double good_dual[] = {0.25};
    check("finite start values are accepted",
          set_start_values(prob, good_primal, good_dual));
    check("accepted primal start is stored",
          prob->primal_start != NULL && prob->primal_start[0] == 0.5);
    check("accepted dual start is stored",
          prob->dual_start != NULL && prob->dual_start[0] == 0.25);

    const double nan_primal[] = {NAN};
    check("NaN primal start is rejected",
          !set_start_values(prob, nan_primal, good_dual));
    check("rejected call leaves the primal start untouched",
          prob->primal_start != NULL && prob->primal_start[0] == 0.5);

    const double inf_dual[] = {INFINITY};
    check("infinite dual start is rejected",
          !set_start_values(prob, good_primal, inf_dual));
    check("rejected call leaves the dual start untouched",
          prob->dual_start != NULL && prob->dual_start[0] == 0.25);

    check("NULL clears both vectors", set_start_values(prob, NULL, NULL));
    check("cleared primal start is NULL", prob->primal_start == NULL);
    check("cleared dual start is NULL", prob->dual_start == NULL);

    check("NULL problem is rejected", !set_start_values(NULL, good_primal, good_dual));

    lp_problem_free(prob);
    printf("%s (%d failure(s))\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures != 0;
}

static int check_start_aliasing(void)
{
    const double matrix_values[] = {1, 0, 0, 1};
    const double x[] = {2, 3}, y[] = {4, 5};
    matrix_desc_t matrix = {0};
    matrix.m = matrix.n = 2;
    matrix.fmt = matrix_dense;
    matrix.data.dense.A = matrix_values;
    lp_problem_t *p = create_lp_problem(NULL, &matrix, NULL, NULL, NULL, NULL, NULL);
    if (!p) return 1;
    int failures = 0;

    set_start_values(p, x, y);
    set_start_values(p, p->primal_start, p->dual_start);
    if (p->primal_start[0] != 2 || p->primal_start[1] != 3 ||
        p->dual_start[0] != 4 || p->dual_start[1] != 5)
    {
        fprintf(stderr, "self-referenced start values were corrupted\n");
        ++failures;
    }

    set_start_values(p, x, y);
    set_start_values(p, p->dual_start, p->primal_start);
    if (p->primal_start[0] != 4 || p->primal_start[1] != 5 ||
        p->dual_start[0] != 2 || p->dual_start[1] != 3)
    {
        fprintf(stderr, "cross-referenced start values were corrupted\n");
        ++failures;
    }

    set_start_values(p, x, y);
    set_start_values(p, p->dual_start, NULL);
    if (p->primal_start[0] != 4 || p->primal_start[1] != 5 || p->dual_start != NULL)
        ++failures;
    set_start_values(p, NULL, NULL);
    if (p->primal_start || p->dual_start) ++failures;
    lp_problem_free(p);
    return failures != 0;
}

int main(void)
{
    int failed = 0;
    failed |= check_public_defaults();
    failed |= check_model_numbers();
    failed |= check_start_values();
    failed |= check_start_aliasing();
    return failed || failures != 0;
}
