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

#include "pid_controller.h"
#include <math.h>

static void update_reversal_streak(pid_state_t *pid,
                                   double primal_weight,
                                   double primal_dist,
                                   double dual_dist);
static void reset_pid_integrator(pid_state_t *pid);
static void apply_pid_update(pid_state_t *pid,
                             double *primal_weight,
                             const step_size_parameters_t *params,
                             double primal_dist,
                             double dual_dist,
                             bool clamp_step);

void apply_restart_pid(pid_state_t *pid,
                       double *primal_weight,
                       const step_size_parameters_t *params,
                       double primal_dist,
                       double dual_dist,
                       bool dist_ok,
                       bool ratio_ok,
                       double gap_to_primal_ratio)
{
    const int oscillation_streak_threshold = 8;
    const int stall_streak_threshold = 8;

    bool gate_ok = dist_ok && ratio_ok;
    pid->stall_streak = gate_ok ? 0 : pid->stall_streak + 1;

    update_reversal_streak(pid, *primal_weight, primal_dist, dual_dist);

    if (pid->reversal_streak >= oscillation_streak_threshold)
    {
        *primal_weight = pid->best_primal_weight;
        reset_pid_integrator(pid);
    }
    else if (gate_ok)
    {
        apply_pid_update(pid, primal_weight, params, primal_dist, dual_dist, false);
        double balance = fabs(log10(gap_to_primal_ratio));
        if (balance < pid->best_balance)
        {
            pid->best_balance = balance;
            pid->best_primal_weight = *primal_weight;
        }
    }
    else if (pid->stall_streak >= stall_streak_threshold)
    {
        apply_pid_update(pid, primal_weight, params, primal_dist, dual_dist, true);
    }
    else if (!isinf(pid->best_balance))
    {
        *primal_weight = pid->best_primal_weight;
        reset_pid_integrator(pid);
    }
    else
    {
        apply_pid_update(pid, primal_weight, params, primal_dist, dual_dist, true);
    }
}

void initialize_pid_state(pid_state_t *pid, double primal_weight)
{
    pid->error_sum = 0.0;
    pid->last_error = 0.0;
    pid->best_primal_weight = primal_weight;
    pid->best_balance = INFINITY;
    pid->reversal_streak = 0;
    pid->last_step_sign = 0;
    pid->stall_streak = 0;
}

static void update_reversal_streak(pid_state_t *pid,
                                   double primal_weight,
                                   double primal_dist,
                                   double dual_dist)
{
    double error = log(dual_dist) - log(primal_dist) - log(primal_weight);
    int step_sign = (error > 1e-12) ? 1 : (error < -1e-12 ? -1 : 0);
    if (step_sign == 0)
        return;
    if (pid->last_step_sign != 0 && step_sign != pid->last_step_sign)
        pid->reversal_streak++;
    else if (step_sign == pid->last_step_sign)
        pid->reversal_streak = 0;
    pid->last_step_sign = step_sign;
}

static void reset_pid_integrator(pid_state_t *pid)
{
    pid->error_sum = 0.0;
    pid->last_error = 0.0;
}

static void apply_pid_update(pid_state_t *pid,
                             double *primal_weight,
                             const step_size_parameters_t *params,
                             double primal_dist,
                             double dual_dist,
                             bool clamp_step)
{
    double error = log(dual_dist) - log(primal_dist) - log(*primal_weight);
    if (!isfinite(error))
        return;
    pid->error_sum *= params->i_smooth;
    pid->error_sum += error;
    double delta_error = error - pid->last_error;
    double step = params->k_p * error +
                  params->k_i * pid->error_sum +
                  params->k_d * delta_error;
    if (clamp_step && step * error < 0.0)
    {
        const double max_log_step = log(10.0);
        step = fmin(fmax(step, -max_log_step), max_log_step);
    }
    *primal_weight *= exp(step);
    pid->last_error = error;
}
