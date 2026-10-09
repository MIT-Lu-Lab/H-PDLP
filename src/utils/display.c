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
#include "active_set_boost.h"
#include "hpdlp.h"
#include "spmv_backend.h"
#include "termination_checker.h"
#include <math.h>
#include <stdio.h>

#define SETTING_LABEL_WIDTH 32

#define PRINT_DIFF_INT(name, current, default_val) \
    do { \
        if ((current) != (default_val)) { \
            printf("  %-*s : %d\n", SETTING_LABEL_WIDTH, name, current); \
        } \
    } while(0)

#define PRINT_DIFF_DBL(name, current, default_val) \
    do { \
        if ((current) != (default_val)) { \
            printf("  %-*s : %g\n", SETTING_LABEL_WIDTH, name, (double)(current)); \
        } \
    } while(0)

#define PRINT_DIFF_BOOL(name, current, default_val) \
    do { \
        if ((current) != (default_val)) { \
            printf("  %-*s : %s\n", SETTING_LABEL_WIDTH, name, (current) ? "on" : "off"); \
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
    const lp_problem_t *problem)
{
    pdhg_parameters_t default_params;
    set_default_parameters(&default_params);

    if (!params->verbose)
    {
        return;
    }
    printf("-----------------------------------------------------------"
           "-----------------------------------------------------------------\n");
    printf("%*sH-PDLP v%s\n", (124 - (int)(sizeof("H-PDLP v" HPDLP_VERSION) - 1)) / 2,
           "", HPDLP_VERSION);
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
    printf("  %-*s : %g\n", SETTING_LABEL_WIDTH, "eps_optimal_relative",
           params->termination_criteria.eps_optimal_relative);
    printf("  %-*s : %g\n", SETTING_LABEL_WIDTH, "eps_feasible_relative",
           params->termination_criteria.eps_feasible_relative);
    printf("  %-*s : %g\n", SETTING_LABEL_WIDTH, "eps_farkas_homogeneous",
           params->termination_criteria.eps_farkas_homogeneous);
    printf("  %-*s : %g\n", SETTING_LABEL_WIDTH, "eps_farkas_displacement",
           params->termination_criteria.eps_farkas_displacement);
    printf("  %-*s : %g\n", SETTING_LABEL_WIDTH, "time_sec_limit",
           params->termination_criteria.time_sec_limit);
    printf("  %-*s : %d\n", SETTING_LABEL_WIDTH, "iteration_limit",
           params->termination_criteria.iteration_limit);
    printf("  %-*s : %s\n", SETTING_LABEL_WIDTH, "spmv_backend", spmv_backend_name());

    PRINT_DIFF_BOOL("presolve",
                    params->presolve,
                    default_params.presolve);
    if (params->optimality_norm != default_params.optimality_norm)
    {
        printf("  %-*s : %s\n", SETTING_LABEL_WIDTH, "optimality_norm",
               params->optimality_norm == NORM_TYPE_L_INF ? "linf" : "l2");
    }
    PRINT_DIFF_DBL("reflection_coefficient",
                   params->reflection_coefficient,
                   default_params.reflection_coefficient);
    PRINT_DIFF_INT("termination_evaluation_frequency",
                   params->termination_evaluation_frequency,
                   default_params.termination_evaluation_frequency);
    PRINT_DIFF_INT("geometric_mean_iterations",
                   params->scaling_params.geometric_mean_iterations,
                   default_params.scaling_params.geometric_mean_iterations);
    PRINT_DIFF_INT("l_inf_ruiz_iterations",
                   params->scaling_params.l_inf_ruiz_iterations,
                   default_params.scaling_params.l_inf_ruiz_iterations);
    PRINT_DIFF_BOOL("has_pock_chambolle_alpha",
                    params->scaling_params.has_pock_chambolle_alpha,
                    default_params.scaling_params.has_pock_chambolle_alpha);
    PRINT_DIFF_DBL("pock_chambolle_alpha",
                   params->scaling_params.pock_chambolle_alpha,
                   default_params.scaling_params.pock_chambolle_alpha);
    PRINT_DIFF_BOOL("bound_objective_rescaling",
                    params->scaling_params.bound_objective_rescaling,
                    default_params.scaling_params.bound_objective_rescaling);
    PRINT_DIFF_DBL("matrix_zero_tol",
                   params->scaling_params.matrix_zero_tol,
                   default_params.scaling_params.matrix_zero_tol);
    PRINT_DIFF_DBL("infinite_bound",
                   params->scaling_params.infinite_bound,
                   default_params.scaling_params.infinite_bound);
    PRINT_DIFF_INT("sv_max_iter",
                   params->power_iter_params.max_iter,
                   default_params.power_iter_params.max_iter);
    PRINT_DIFF_DBL("sv_tol",
                   params->power_iter_params.tol,
                   default_params.power_iter_params.tol);
    PRINT_DIFF_DBL("step_size_factor",
                   params->step_size_params.step_size_factor,
                   default_params.step_size_params.step_size_factor);
    PRINT_DIFF_DBL("k_p",
                   params->step_size_params.k_p,
                   default_params.step_size_params.k_p);
    PRINT_DIFF_DBL("k_i",
                   params->step_size_params.k_i,
                   default_params.step_size_params.k_i);
    PRINT_DIFF_DBL("k_d",
                   params->step_size_params.k_d,
                   default_params.step_size_params.k_d);
    PRINT_DIFF_DBL("i_smooth",
                   params->step_size_params.i_smooth,
                   default_params.step_size_params.i_smooth);
    PRINT_DIFF_BOOL("primal_weight_safeguard",
                    params->step_size_params.primal_weight_safeguard,
                    default_params.step_size_params.primal_weight_safeguard);
    PRINT_DIFF_INT("cone_newton_max_iter",
                   params->cone_params.max_iter,
                   default_params.cone_params.max_iter);
    PRINT_DIFF_DBL("cone_newton_tol",
                   params->cone_params.tol,
                   default_params.cone_params.tol);
    PRINT_DIFF_DBL("artificial_restart_threshold",
                   params->restart_params.artificial_restart_threshold,
                   default_params.restart_params.artificial_restart_threshold);
    PRINT_DIFF_DBL("sufficient_reduction_for_restart",
                   params->restart_params.sufficient_reduction_for_restart,
                   default_params.restart_params.sufficient_reduction_for_restart);
    PRINT_DIFF_DBL("necessary_reduction_for_restart",
                   params->restart_params.necessary_reduction_for_restart,
                   default_params.restart_params.necessary_reduction_for_restart);
    PRINT_DIFF_BOOL("active_set_boost",
                    params->active_set_params.step_size_boost,
                    default_params.active_set_params.step_size_boost);
    PRINT_DIFF_INT("asb_window_iter",
                   params->active_set_params.window_iterations,
                   default_params.active_set_params.window_iterations);
    PRINT_DIFF_DBL("asb_relative_tol",
                   params->active_set_params.relative_tol,
                   default_params.active_set_params.relative_tol);
    PRINT_DIFF_DBL("asb_safety_factor",
                   params->active_set_params.safety_factor,
                   default_params.active_set_params.safety_factor);
    PRINT_DIFF_DBL("asb_min_raise_ratio",
                   params->active_set_params.min_raise_ratio,
                   default_params.active_set_params.min_raise_ratio);
    PRINT_DIFF_DBL("asb_reestimate_change_ratio",
                   params->active_set_params.reestimate_change_ratio,
                   default_params.active_set_params.reestimate_change_ratio);
    PRINT_DIFF_DBL("asb_divergence_ceiling_ratio",
                   params->active_set_params.divergence_ceiling_ratio,
                   default_params.active_set_params.divergence_ceiling_ratio);
    PRINT_DIFF_DBL("asb_divergence_margin",
                   params->active_set_params.divergence_margin,
                   default_params.active_set_params.divergence_margin);
    PRINT_DIFF_INT("asb_max_reverts",
                   params->active_set_params.max_reverts,
                   default_params.active_set_params.max_reverts);
}

#undef PRINT_DIFF_INT
#undef PRINT_DIFF_DBL
#undef PRINT_DIFF_BOOL

static const char iteration_header[] =
    "   iter   time  |  pr obj    du obj "
    " |  pr res  du res   gap   |  pr res  du res   gap  ";

static const char diagnostic_header[] =
    " |  scale     gap     ratio  "
    " |  pr ss    du ss    sc ss  "
    " |  omega     tau     alpha  "
    " |   |x|      |y|   ";

static void print_iteration_divider(bool diagnose)
{
    size_t width = sizeof(iteration_header) - 1;
    if (diagnose)
        width += sizeof(diagnostic_header) - 1;
    for (size_t i = 0; i < width; ++i)
        putchar('-');
    putchar('\n');
}

void print_iteration_header(const pdhg_parameters_t *params)
{
    if (!params->verbose)
        return;

    print_iteration_divider(params->diagnose);
    printf(" %s | %s | %s | %s",
           "   runtime    ",
           "    objective     ",
           "  absolute residuals   ",
           "  relative residuals   ");
    if (params->diagnose)
        printf(" |            hsd             |        step sizes         "
               " |     weight and scales      |   block norms    ");
    printf("\n%s%s\n", iteration_header, params->diagnose ? diagnostic_header : "");
    print_iteration_divider(params->diagnose);
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
    const pdhg_parameters_t *params,
    bool is_restart)
{
    if (!params->verbose)
    {
        return;
    }
    if (is_restart || state->control.total_count % get_print_frequency(state->control.total_count) == 0)
    {
        printf(
            "%c%6d %.1e | %8.1e  %8.1e | %.1e %.1e %.1e | %.1e %.1e %.1e",
            is_restart ? 'R' : ' ',
            state->control.total_count,
            state->control.cumulative_time_sec,
            state->objective_sign * state->metrics_h->primal_objective,
            state->objective_sign * state->metrics_h->dual_objective,
            state->metrics_h->absolute_primal_residual,
            state->metrics_h->absolute_dual_residual,
            state->metrics_h->objective_gap,
            state->metrics_h->relative_primal_residual,
            state->metrics_h->relative_dual_residual,
            state->metrics_h->relative_objective_gap);
        if (params->diagnose)
        {
            printf(" | %8.1e %8.1e %8.1e | %8.1e %8.1e %8.1e"
                   " | %8.1e %8.1e %8.1e | %8.1e %8.1e",
                   state->metrics_h->current_primal_scale,
                   state->metrics_h->normalized_homogeneous_gap,
                   state->metrics_h->normalized_homogeneous_ratio,
                   state->control.step_size_primal,
                   state->control.step_size_dual,
                   state->control.step_size_homogeneous,
                   state->control.primal_weight,
                   state->metrics_h->pdhg_primal_scale,
                   state->metrics_h->pdhg_dual_scale,
                   state->metrics_h->primal_solution_norm,
                   state->metrics_h->dual_solution_norm);
        }
        printf("\n");
    }
}

void display_active_set_boost_raise(
    const pdhg_parameters_t *params,
    double previous_step_size,
    double updated_step_size,
    double masked_constraint_norm)
{
    if (!params->verbose)
    {
        return;
    }
    printf("  active-set boost: step size %.3e -> %.3e (masked norm %.3e)\n",
           previous_step_size,
           updated_step_size,
           masked_constraint_norm);
}

void display_active_set_boost_revert(
    const pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    if (!params->verbose)
    {
        return;
    }
    printf("  active-set boost: boosted epoch diverged, step size reverted "
           "(%d of %d)\n",
           state->active_set.revert_count,
           params->active_set_params.max_reverts);
}

void pdhg_final_log(
    const hpdlp_result_t *result,
    const pdhg_parameters_t *params)
{
    if (params->verbose)
    {
        print_iteration_divider(params->diagnose);
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
        if (is_active_set_boost_enabled(params) && result->total_count > 0)
        {
            printf("  Active-set boost       : %d raises, %d reverts, "
                   "%d power iterations\n",
                   result->active_set_raise_count,
                   result->active_set_revert_count,
                   result->active_set_power_iteration_count);
        }
        if (result->termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE ||
            result->termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE)
        {
            printf("  Farkas margin          : %.3e\n", result->farkas_margin);
            printf("  Farkas residual        : %.3e\n", result->farkas_residual);
        }
    }
}
