#include "hpdlp.h"
#include "test_checks.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static lp_problem_t *create_small_problem(void)
{
    double objective[1] = {1.0};
    double constraint_lower[1] = {0.5};
    double variable_lower[1] = {0.0};
    double variable_upper[1] = {1.0};
    int row_ptr[2] = {0, 1};
    int col_ind[1] = {0};
    double values[1] = {1.0};
    matrix_desc_t desc = {0};
    desc.m = 1;
    desc.n = 1;
    desc.fmt = matrix_csr;
    desc.data.csr.nnz = 1;
    desc.data.csr.row_ptr = row_ptr;
    desc.data.csr.col_ind = col_ind;
    desc.data.csr.vals = values;
    return create_lp_problem(objective, &desc, constraint_lower, NULL,
                             variable_lower, variable_upper, NULL);
}

int main(void)
{
    setenv("CUDA_VISIBLE_DEVICES", "", 1);

    lp_problem_t *problem = create_small_problem();
    check("problem is created without a GPU", problem != NULL);
    if (!problem)
        return 1;

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.presolve = false;
    params.verbose = false;

    hpdlp_result_t *result = solve_lp_problem(problem, &params);
    check("solve without a visible GPU returns NULL instead of exiting", result == NULL);
    check("the failure is recorded", strstr(hpdlp_last_error(), " error at ") != NULL);
    hpdlp_result_free(result);

    lp_problem_t *next_problem = create_small_problem();
    check("the next call succeeds and clears the recorded error",
          next_problem != NULL && hpdlp_last_error()[0] == '\0');
    lp_problem_free(next_problem);
    lp_problem_free(problem);
    return failures != 0;
}
