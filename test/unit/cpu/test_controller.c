#include "active_set_boost.h"
#include "hpdlp.h"
#include "pid_controller.h"
#include "termination_checker.h"
#include "test_checks.h"
#include <math.h>
#include <stdio.h>

static void restart(pid_state_t *pid, double *weight, bool guard,
                    double primal, double dual, double gap, double ratio)
{
    const step_size_parameters_t params = {
        .k_p = 0.2, .k_i = 0.05, .k_d = 0.1, .i_smooth = 0.3,
        .primal_weight_safeguard = guard};
    apply_restart_pid(pid, weight, &params, 1.0, 1.0,
                      primal, dual, gap, 1.0, ratio);
}

static void check_pid_control(void)
{
    pid_state_t pid;
    double weight;

    initialize_pid_state(&pid);
    weight = 4.0;
    restart(&pid, &weight, true, 1e-2, 1e-2, 0.0, 0.0);
    weight = 9.0;
    restart(&pid, &weight, true, 1.0, 1.0, 0.0, 0.0);
    check("no rollback before reaching the tail", weight != 4.0);

    initialize_pid_state(&pid);
    weight = 4.0;
    restart(&pid, &weight, true, 2e-2, 2e-2, 0.0, 0.0);
    weight = 6.0;
    restart(&pid, &weight, true, 1e-2, 1e-2, 0.0, 0.0);
    weight = 9.0;
    restart(&pid, &weight, true, 1e-3, 2e-3, 0.0, 0.0);
    weight = 16.0;
    restart(&pid, &weight, true, 1e-6, 4e-6, 0.0, 0.0);
    weight = 25.0;
    restart(&pid, &weight, true, 1e-6, 4e-6, 4e-4, 0.0);
    check("rollback uses a weight recorded below 1e-2 but before the tail", weight == 9.0);
    check("rollback clears PID history", pid.error_sum == 0.0 && !pid.has_last_error);
    weight = 36.0;
    restart(&pid, &weight, true, 1e-6, 1e-6, 1e-3, 0.0);
    check("rollback retains guard history on repeated growth", weight == 9.0);

    pid_state_t fresh;
    initialize_pid_state(&fresh);
    double fresh_weight = 9.0;
    restart(&fresh, &fresh_weight, false, 1e-6, 2e-6, 0.0, 0.0);
    restart(&pid, &weight, true, 1e-6, 2e-6, 0.0, 0.0);
    check("PID resumes with cleared history", fabs(weight - fresh_weight) < 1e-14);

    initialize_pid_state(&pid);
    weight = 4.0;
    restart(&pid, &weight, true, 1e-3, 2e-3, 0.0, 0.0);
    weight = 9.0;
    restart(&pid, &weight, true, 1e-3, 1e-3, 2e-2, 0.0);
    weight = 16.0;
    restart(&pid, &weight, true, 1e-6, 4e-6, 0.0, 0.0);
    weight = 25.0;
    restart(&pid, &weight, true, 1e-6, 1e-6, 1e-3, 0.0);
    check("recording stays enabled after the objective gap rebounds", weight == 9.0);

    initialize_pid_state(&pid);
    weight = 4.0;
    restart(&pid, &weight, true, 2e-2, 2e-2, 0.0, 0.0);
    weight = 9.0;
    restart(&pid, &weight, true, 1e-3, 2e-3, 0.0, 0.0);
    weight = 16.0;
    restart(&pid, &weight, true, 1.5e-2, 2e-2, 0.0, 0.0);
    weight = 25.0;
    restart(&pid, &weight, true, 1e-6, 4e-6, 0.0, 0.0);
    weight = 36.0;
    restart(&pid, &weight, true, 1e-6, 1e-6, 1e-3, 0.0);
    check("rollback uses the better balance recorded during a residual rebound", weight == 16.0);

    initialize_pid_state(&pid);
    weight = 4.0;
    restart(&pid, &weight, true, 1e-6, 8e-6, 0.0, 0.0);
    weight = 9.0;
    restart(&pid, &weight, true, 4e-6, 1e-6, 0.0, 0.0);
    weight = 16.0;
    restart(&pid, &weight, true, 1e-6, 1e-6, 1e-3, 0.0);
    check("balance ranking is symmetric in primal and dual", weight == 9.0);
    weight = 25.0;
    restart(&pid, &weight, true, 1e-6, 1e-6, 1e-3, 0.95);
    check("high homogeneous ratio leaves the weight unchanged", weight == 25.0);
    restart(&pid, &weight, true, 1e-6, 1e-6, 1e-3, 0.0);
    check("high homogeneous ratio preserves guard history", weight == 9.0);

    initialize_pid_state(&pid);
    weight = 4.0;
    restart(&pid, &weight, false, 1e-6, 1e-6, 0.0, 0.0);
    weight = 9.0;
    restart(&pid, &weight, false, 1e-6, 1e-6, 1e-3, 0.0);
    check("disabled guard does not roll back", weight != 4.0);
}

static void check_boost_revert_policy(void)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    pdhg_metrics_t metrics = {0};
    pdhg_solver_state_t state = {0};
    double fixed_point_error = 1.0;
    state.metrics_h = &metrics;
    state.control.termination_reason = TERMINATION_REASON_UNSPECIFIED;
    state.control.step_size_base = 1.0;
    state.control.step_size = 2.0;
    state.control.fixed_point_error_mp = &fixed_point_error;
    state.control.initial_fixed_point_error = 1.0;

    double *residuals[] = {
        &metrics.relative_primal_residual,
        &metrics.relative_dual_residual,
        &metrics.relative_objective_gap
    };
    const char *names[] = {"primal residual", "dual residual", "objective gap"};
    const double invalid_values[] = {NAN, INFINITY, -INFINITY};
    metrics.relative_primal_residual = 1e-5;
    metrics.relative_dual_residual = 1e-5;
    metrics.relative_objective_gap = 1e-5;
    check("finite residuals do not revert a stable boosted epoch",
          !should_revert_active_set_boost(&state, &params));
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            *residuals[i] = invalid_values[j];
            char message[128];
            snprintf(message, sizeof(message), "a single nonfinite %s (%g) triggers revert",
                     names[i], invalid_values[j]);
            check(message, should_revert_active_set_boost(&state, &params));
        }
        *residuals[i] = 1e-5;
    }

    fixed_point_error = 1.05;
    check("the divergence threshold itself does not revert",
          !should_revert_active_set_boost(&state, &params));
    fixed_point_error = 1.050001;
    check("a fixed-point error beyond the threshold reverts",
          should_revert_active_set_boost(&state, &params));

    state.control.step_size = state.control.step_size_base;
    check("the base step never reverts",
          !should_revert_active_set_boost(&state, &params));
    state.control.step_size = 2.0;

    state.control.termination_reason = TERMINATION_REASON_OPTIMAL;
    check("an optimal result does not revert",
          !should_revert_active_set_boost(&state, &params));
    state.control.termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
    check("an infeasibility certificate does not revert",
          !should_revert_active_set_boost(&state, &params));
    state.control.termination_reason = TERMINATION_REASON_TIME_LIMIT;
    check("a time limit may restore a diverging boosted epoch",
          should_revert_active_set_boost(&state, &params));
    state.control.termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
    check("an iteration limit may restore a diverging boosted epoch",
          should_revert_active_set_boost(&state, &params));
    state.control.termination_reason = TERMINATION_REASON_NUMERICAL_FAILURE;
    fixed_point_error = 1.0;
    check("a numerical failure reverts a boosted epoch",
          should_revert_active_set_boost(&state, &params));
}

int main(void)
{
    check_pid_control();
    check_boost_revert_policy();
    return failures != 0;
}
