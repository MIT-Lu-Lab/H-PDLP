/*
Copyright 2026 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "display.h"
#include "cuhsdlp.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>

#define PRINT_DIFF_INT(name, current, default_val) \
    do { \
        if ((current) != (default_val)) { \
            printf("  %-18s : %d\n", name, current); \
        } \
    } while(0)

#define PRINT_DIFF_DBL(name, current, default_val) \
    do { \
        if (fabs((current) - (default_val)) > 1e-9) { \
            printf("  %-18s : %.1e\n", name, (double)(current)); \
        } \
    } while(0)

#define PRINT_DIFF_BOOL(name, current, default_val) \
    do { \
        if ((current) != (default_val)) { \
            printf("  %-18s : %s\n", name, (current) ? "on" : "off"); \
        } \
    } while (0)

const char *termination_reason_to_string(termination_reason_t reason)
{
    switch (reason)
    {
    case TERMINATION_REASON_OPTIMAL:
        return "OPTIMAL";
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
        return "PRIMAL_INFEASIBLE";
    case TERMINATION_REASON_DUAL_INFEASIBLE:
        return "DUAL_INFEASIBLE";
    case TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED:
        return "INFEASIBLE_OR_UNBOUNDED";
    case TERMINATION_REASON_TIME_LIMIT:
        return "TIME_LIMIT";
    case TERMINATION_REASON_ITERATION_LIMIT:
        return "ITERATION_LIMIT";
    case TERMINATION_REASON_NUMERICAL_FAILURE:
        return "NUMERICAL_FAILURE";
    case TERMINATION_REASON_UNSPECIFIED:
        return "UNSPECIFIED";
    default:
        return "UNKNOWN";
    }
}

void print_initial_info(
    const pdhg_parameters_t *params,
    lp_problem_t *problem)
{
    pdhg_parameters_t default_params;
    set_default_parameters(&default_params);

    if (!params->verbose)
    {
        return;
    }
    printf("-----------------------------------------------------------"
           "-----------------------------------------------------------------\n");
    printf("%*scuHSDLP v%s\n", (124 - 14) / 2, "", CUHSDLP_VERSION);
    printf("%*sA GPU-Accelerated First-Order LP Solver\n", (124 - 39) / 2, "");
    printf(
        "%*s(c) Haihao Lu, Massachusetts Institute of Technology, 2026\n",
        (124 - 58) / 2,
        "");
    printf("-----------------------------------------------------------"
           "-----------------------------------------------------------------\n");
    printf("\n");

    printf(
        "Problem: %d rows, %d columns, %d nonzeros\n",
        problem->num_constraints,
        problem->num_variables,
        problem->constraint_matrix_num_nonzeros);

    printf("Settings:\n");
    printf("  iter_limit         : %d\n",
           params->termination_criteria.iteration_limit);
    printf("  time_limit         : %.2f sec\n",
           params->termination_criteria.time_sec_limit);
    printf("  eps_opt            : %.1e\n",
           params->termination_criteria.eps_optimal_relative);
    printf("  eps_feas           : %.1e\n",
           params->termination_criteria.eps_feasible_relative);
    if (params->optimality_norm != default_params.optimality_norm)
    {
        printf("  optimality_norm    : %s\n",
               params->optimality_norm == NORM_TYPE_L_INF ? "L_inf" : "L2");
    }

    PRINT_DIFF_INT("l_inf_ruiz_iter",
                   params->scaling_params.l_inf_ruiz_iterations,
                   default_params.scaling_params.l_inf_ruiz_iterations);
    PRINT_DIFF_DBL("pock_chambolle_alpha",
                   params->scaling_params.pock_chambolle_alpha,
                   default_params.scaling_params.pock_chambolle_alpha);
    PRINT_DIFF_BOOL("has_pock_chambolle_alpha",
                    params->scaling_params.has_pock_chambolle_alpha,
                    default_params.scaling_params.has_pock_chambolle_alpha);
    PRINT_DIFF_BOOL("bound_obj_rescaling",
                    params->scaling_params.bound_objective_rescaling,
                    default_params.scaling_params.bound_objective_rescaling);
    PRINT_DIFF_INT("power_iter_max_iter",
                   params->power_iter_params.max_iter,
                   default_params.power_iter_params.max_iter);
    PRINT_DIFF_DBL("power_iter_tol",
                   params->power_iter_params.tol,
                   default_params.power_iter_params.tol);
    PRINT_DIFF_INT("evaluation_freq",
                   params->termination_evaluation_frequency,
                   default_params.termination_evaluation_frequency);
    PRINT_DIFF_BOOL("presolve",
                    params->presolve,
                    default_params.presolve);
    PRINT_DIFF_DBL("matrix_zero_tol",
                   params->scaling_params.matrix_zero_tol,
                   default_params.scaling_params.matrix_zero_tol);
    PRINT_DIFF_INT("cone_newton_iter",
                   params->cone_params.max_iter,
                   default_params.cone_params.max_iter);
    PRINT_DIFF_DBL("cone_newton_tol",
                   params->cone_params.tol,
                   default_params.cone_params.tol);
}

#undef PRINT_DIFF_INT
#undef PRINT_DIFF_DBL
#undef PRINT_DIFF_BOOL

void print_iteration_header(bool verbose)
{
    if (!verbose)
        return;

    printf("-----------------------------------------------------------"
           "-----------------------------------------------------------------\n");
    printf(" %s | %s | %s | %s | %s | %s\n",
           "   runtime    ",
           "    objective     ",
           "  absolute residuals   ",
           "  relative residuals   ",
           "      step sizes       ",
           "  hsd  ");
    printf(" %s %s | %s %s | %s %s %s | %s %s %s | %s %s %s | %s\n",
           "  iter",
           "  time ",
           " pr obj ",
           "  du obj ",
           " pr res",
           " du res",
           "  gap  ",
           " pr res",
           " du res",
           "  gap  ",
           " pr ss ",
           " du ss ",
           " sc ss ",
           " scale ");
    printf("-----------------------------------------------------------"
           "-----------------------------------------------------------------\n");
}

int get_print_frequency(int iter)
{
    int print_interval = 10;
    long long iteration_threshold = 1000;

    while (iter >= iteration_threshold)
    {
        print_interval *= 10;
        iteration_threshold *= 10;
    }
    return print_interval;
}

void display_iteration_stats(
    const pdhg_solver_state_t *state,
    bool verbose,
    bool is_restart)
{
    if (!verbose)
    {
        return;
    }
    if (is_restart || state->total_count % get_print_frequency(state->total_count) == 0)
    {
        printf(
            "%c%6d %.1e | %8.1e  %8.1e | %.1e %.1e %.1e | %.1e %.1e %.1e | %.1e %.1e %.1e | %7.1e\n",
            is_restart ? 'R' : ' ',
            state->total_count,
            state->cumulative_time_sec,
            state->metrics_h->primal_objective,
            state->metrics_h->dual_objective,
            state->metrics_h->absolute_primal_residual,
            state->metrics_h->absolute_dual_residual,
            state->metrics_h->objective_gap,
            state->metrics_h->relative_primal_residual,
            state->metrics_h->relative_dual_residual,
            state->metrics_h->relative_objective_gap,
            state->step_size_primal,
            state->step_size_dual,
            state->step_size_homogeneous,
            state->metrics_h->current_homogeneous_scale);
    }
}

void pdhg_final_log(
    const cuhsdlp_result_t *result,
    const pdhg_parameters_t *params)
{
    if (params->verbose)
    {
        printf("-----------------------------------------------------------"
           "-----------------------------------------------------------------\n");
        printf("Solution Summary\n");
        printf(
            "  Status                 : %s\n",
            termination_reason_to_string(result->termination_reason));
        if (params->presolve)
        {
            printf("  Presolve time          : %.3g sec\n", result->presolve_time);
        }
        printf("  Precondition time      : %.5g sec\n", result->rescaling_time_sec);
        printf("  Solve time             : %.3g sec\n", result->cumulative_time_sec);
        printf("  Iterations             : %d\n", result->total_count);
        printf("  Primal objective       : %.10g\n", result->primal_objective_value);
        printf("  Dual objective         : %.10g\n", result->dual_objective_value);
        printf("  Objective gap          : %.3e\n", result->relative_objective_gap);
        printf("  Primal infeas          : %.3e\n", result->relative_primal_residual);
        printf("  Dual infeas            : %.3e\n", result->relative_dual_residual);
    }
}
