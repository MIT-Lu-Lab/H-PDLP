#include "hopdlp.h"
#include "mps_parser.h"
#include "test_checks.h"
#include <math.h>
#include <stdio.h>

static const char *INT_DEFAULT_MPS =
    "NAME          TEST\n"
    "ROWS\n"
    " N  COST\n"
    " L  LIM1\n"
    "COLUMNS\n"
    "    MARKER                 'MARKER'                 'INTORG'\n"
    "    IX1       COST         1.0        LIM1         1.0\n"
    "    IX2       COST         1.0        LIM1         1.0\n"
    "    IX3       COST         1.0        LIM1         1.0\n"
    "    IX4       COST         1.0        LIM1         1.0\n"
    "    MARKER                 'MARKER'                 'INTEND'\n"
    "    CX1       COST         1.0        LIM1         1.0\n"
    "RHS\n"
    "    RHS       LIM1         10.0\n"
    "BOUNDS\n"
    " UP BND       IX2          10.0\n"
    " LO BND       IX3          2.0\n"
    " MI BND       IX4\n"
    "ENDATA\n";

static const char *NEGATIVE_UPPER_MPS =
    "NAME          TEST\n"
    "ROWS\n"
    " N  COST\n"
    " L  LIM1\n"
    "COLUMNS\n"
    "    XA        COST         1.0        LIM1         1.0\n"
    "    XB        COST         1.0        LIM1         1.0\n"
    "RHS\n"
    "    RHS       LIM1         10.0\n"
    "BOUNDS\n"
    " UP BND       XA           -5.0\n"
    " LO BND       XB           -10.0\n"
    " UP BND       XB           -5.0\n"
    "ENDATA\n";

static const char *QUADOBJ_MPS =
    "NAME          TEST\n"
    "ROWS\n"
    " N  COST\n"
    " L  LIM1\n"
    "COLUMNS\n"
    "    XA        COST         1.0        LIM1         1.0\n"
    "RHS\n"
    "    RHS       LIM1         10.0\n"
    "QUADOBJ\n"
    "    XA        XA           2.0\n"
    "ENDATA\n";

static const char *SOS_MPS =
    "NAME          TEST\n"
    "ROWS\n"
    " N  COST\n"
    " L  LIM1\n"
    "COLUMNS\n"
    "    XA        COST         1.0        LIM1         1.0\n"
    "    XB        COST         1.0        LIM1         1.0\n"
    "RHS\n"
    "    RHS       LIM1         10.0\n"
    "BOUNDS\n"
    " UP BND       XA           4.0\n"
    "SOS\n"
    " S1 SOS       SET1         1\n"
    "    XA        1.0\n"
    "    XB        2.0\n"
    "ENDATA\n";

static lp_problem_t *parse_text(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    if (f == NULL)
        return NULL;
    fputs(text, f);
    fclose(f);

    lp_problem_t *prob = read_mps_file(path);
    remove(path);
    return prob;
}

static int check_mps_bounds(void)
{
    lp_problem_t *prob = parse_text("test_mps_bounds_int.mps", INT_DEFAULT_MPS);
    check("integer fixture parses", prob != NULL);
    if (prob != NULL)
    {
        check("integer fixture has 5 columns", prob->num_variables == 5);
        check("integer column with no bounds is [0, 1]",
              prob->variable_lower_bound[0] == 0.0 && prob->variable_upper_bound[0] == 1.0);
        check("UP cancels the binary default",
              prob->variable_lower_bound[1] == 0.0 && prob->variable_upper_bound[1] == 10.0);
        check("LO cancels the binary default",
              prob->variable_lower_bound[2] == 2.0 && isinf(prob->variable_upper_bound[2]));
        check("MI cancels the binary default",
              isinf(prob->variable_lower_bound[3]) && prob->variable_lower_bound[3] < 0.0 &&
                  isinf(prob->variable_upper_bound[3]));
        check("INTEND ends the integer block",
              prob->variable_lower_bound[4] == 0.0 && isinf(prob->variable_upper_bound[4]));
        lp_problem_free(prob);
    }

    prob = parse_text("test_mps_bounds_neg.mps", NEGATIVE_UPPER_MPS);
    check("negative-upper fixture parses", prob != NULL);
    if (prob != NULL)
    {
        check("negative UP with no lower bound implies -infinity",
              isinf(prob->variable_lower_bound[0]) && prob->variable_lower_bound[0] < 0.0 &&
                  prob->variable_upper_bound[0] == -5.0);
        check("explicit LO survives a negative UP",
              prob->variable_lower_bound[1] == -10.0 && prob->variable_upper_bound[1] == -5.0);
        lp_problem_free(prob);
    }

    prob = parse_text("test_mps_bounds_quad.mps", QUADOBJ_MPS);
    check("QUADOBJ is rejected", prob == NULL);

    prob = parse_text("test_mps_bounds_sos.mps", SOS_MPS);
    check("SOS fixture parses", prob != NULL);
    if (prob != NULL)
    {
        check("SOS section does not corrupt BOUNDS",
              prob->variable_lower_bound[0] == 0.0 && prob->variable_upper_bound[0] == 4.0 &&
                  prob->variable_lower_bound[1] == 0.0 && isinf(prob->variable_upper_bound[1]));
        lp_problem_free(prob);
    }

    printf("%s (%d failure(s))\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures != 0;
}

static const char *FINITE_MPS =
    "NAME          FINITE\n"
    "ROWS\n"
    " N  COST\n"
    " L  LIM1\n"
    "COLUMNS\n"
    "    X1        COST         1.0        LIM1         1.0\n"
    "RHS\n"
    "    RHS       LIM1         10.0\n"
    "BOUNDS\n"
    " UP BND       X1           4.0\n"
    "ENDATA\n";

static const char *NAN_COLUMN_MPS =
    "NAME          NANCOL\n"
    "ROWS\n"
    " N  COST\n"
    " L  LIM1\n"
    "COLUMNS\n"
    "    X1        COST         nan        LIM1         1.0\n"
    "RHS\n"
    "    RHS       LIM1         10.0\n"
    "ENDATA\n";

static const char *INF_RHS_MPS =
    "NAME          INFRHS\n"
    "ROWS\n"
    " N  COST\n"
    " L  LIM1\n"
    "COLUMNS\n"
    "    X1        COST         1.0        LIM1         1.0\n"
    "RHS\n"
    "    RHS       LIM1         inf\n"
    "ENDATA\n";

static const char *OVERFLOW_BOUND_MPS =
    "NAME          OVRBND\n"
    "ROWS\n"
    " N  COST\n"
    " L  LIM1\n"
    "COLUMNS\n"
    "    X1        COST         1.0        LIM1         1.0\n"
    "RHS\n"
    "    RHS       LIM1         10.0\n"
    "BOUNDS\n"
    " UP BND       X1           1e400\n"
    "ENDATA\n";

static int check_mps_numbers(void)
{
    lp_problem_t *prob = parse_text("test_mps_non_finite_ok.mps", FINITE_MPS);
    check("finite fixture parses", prob != NULL);
    if (prob != NULL)
    {
        check("finite fixture keeps its coefficient",
              prob->constraint_matrix_num_nonzeros == 1);
        lp_problem_free(prob);
    }

    prob = parse_text("test_mps_non_finite_nan.mps", NAN_COLUMN_MPS);
    check("nan in COLUMNS is rejected", prob == NULL);
    if (prob != NULL)
        lp_problem_free(prob);

    prob = parse_text("test_mps_non_finite_inf.mps", INF_RHS_MPS);
    check("inf in RHS is rejected", prob == NULL);
    if (prob != NULL)
        lp_problem_free(prob);

    prob = parse_text("test_mps_non_finite_ovr.mps", OVERFLOW_BOUND_MPS);
    check("overflowing literal in BOUNDS is rejected", prob == NULL);
    if (prob != NULL)
        lp_problem_free(prob);

    printf("%s (%d failure(s))\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures != 0;
}

static void check_objective(const char *selector, int after_rows,
                            double coefficient, double constant, int accepted)
{
    const char *path = "test_mps_objective.mps";
    FILE *file = fopen(path, "w");
    if (!file)
    {
        check("create fixture", 0);
        return;
    }
    fputs("NAME OBJECTIVE\n", file);
    if (!after_rows) fputs(selector, file);
    fputs("ROWS\n N FIRST\n N SECOND\n L LIMIT\n", file);
    if (after_rows) fputs(selector, file);
    fputs("COLUMNS\n X FIRST 1 SECOND -2\n X LIMIT 3\n"
          "RHS\n R LIMIT 4 FIRST 5\n R SECOND 7\nENDATA\n", file);
    fclose(file);
    lp_problem_t *problem = read_mps_file(path);
    remove(path);
    check("selection accepted", (problem != NULL) == accepted);
    if (problem)
    {
        check("constraint count", problem->num_constraints == 1);
        check("objective coefficient", problem->objective_vector[0] == coefficient);
        check("objective constant", problem->objective_constant == constant);
        check("constraint coefficient", problem->constraint_matrix_values[0] == 3);
        check("constraint bound", problem->constraint_upper_bound[0] == 4);
        lp_problem_free(problem);
    }
}

static int check_mps_objective_selection(void)
{
    check_objective("", 0, 1, -5, 1);
    check_objective("OBJNAME\n SECOND\n", 0, -2, -7, 1);
    check_objective("OBJNAME SECOND\n", 0, -2, -7, 1);
    check_objective("OBJNAME\n SECOND\n", 1, -2, -7, 1);
    check_objective("OBJNAME\n MISSING\n", 0, 0, 0, 0);
    check_objective("OBJNAME\n LIMIT\n", 0, 0, 0, 0);
    return failures != 0;
}

static void check_mps(const char *name, const char *text, int accepted)
{
    const char *path = "test_model_validation_fixture.mps";
    FILE *f = fopen(path, "w");
    if (!f) { check("fixture file opened", 0); return; }
    fputs(text, f);
    fclose(f);
    lp_problem_t *p = read_mps_file(path);
    remove(path);
    check(name, (p != NULL) == accepted);
    lp_problem_free(p);
}
static int check_mps_ranges(void)
{
    check_mps("finite ranges accepted for all row types",
        "NAME T\nROWS\n N COST\n G GROW\n L LROW\n E EP\n E EN\n"
        "COLUMNS\n X COST 1 GROW 1\n X LROW 1 EP 1\n X EN 1\n"
        "RHS\n RHS GROW 3 LROW 3\n RHS EP 3 EN 3\n"
        "RANGES\n RNG GROW -2 LROW -2\n RNG EP 2 EN -2\nENDATA\n", 1);
    check_mps("inverted bounds rejected",
        "NAME T\nROWS\n N COST\n L R\nCOLUMNS\n X COST 1 R 1\nRHS\n RHS R 1\nBOUNDS\n LO B X 2\n UP B X 1\nENDATA\n", 0);
    check_mps("explicit zero lower bound with negative upper bound rejected",
        "NAME T\nROWS\n N COST\n L R\nCOLUMNS\n X COST 1 R 1\nRHS\n RHS R 1\nBOUNDS\n LO B X 0\n UP B X -5\nENDATA\n", 0);
    check_mps("free variable and finite D exponent accepted",
        "NAME T\nROWS\n N COST\n L R\nCOLUMNS\n X COST 1D0 R 1\nRHS\n RHS R 1\nBOUNDS\n FR B X\nENDATA\n", 1);
    return failures != 0;
}

int main(void)
{
    int failed = 0;
    failed |= check_mps_bounds();
    failed |= check_mps_numbers();
    failed |= check_mps_objective_selection();
    failed |= check_mps_ranges();
    return failed || failures != 0;
}
