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
                       bool ratio_ok)
{
    const int oscillation_streak_threshold = 8;

    update_reversal_streak(pid, *primal_weight, primal_dist, dual_dist);

    if (pid->reversal_streak >= oscillation_streak_threshold)
    {
        *primal_weight = pid->best_primal_weight;
        pid->error_sum = 0.0;
        pid->last_error = 0.0;
    }
    else
    {
        bool clamp_step = !(dist_ok && ratio_ok);
        apply_pid_update(pid, primal_weight, params, primal_dist, dual_dist, clamp_step);
    }
}

void initialize_pid_state(pid_state_t *pid, double primal_weight)
{
    pid->error_sum = 0.0;
    pid->last_error = 0.0;
    pid->best_primal_weight = primal_weight;
    pid->reversal_streak = 0;
    pid->last_step_sign = 0;
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
    {
        pid->reversal_streak = 0;
        pid->best_primal_weight = primal_weight;
    }
    pid->last_step_sign = step_sign;
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
    if (clamp_step)
    {
        const double max_log_step = log(10.0);
        step = fmin(fmax(step, -max_log_step), max_log_step);
    }
    *primal_weight *= exp(step);
    pid->last_error = error;
}
