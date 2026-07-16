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

static void reset_pid_integrator(pid_state_t *pid);
static void align_pid_last_error(pid_state_t *pid,
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
                       double relative_objective_gap,
                       double relative_primal_residual,
                       double homogeneous_scale,
                       double homogeneous_ratio)
{
    double gap_to_primal_ratio = relative_objective_gap / relative_primal_residual;
    double scale_reference = fmax(homogeneous_scale, 1e-300);
    bool dist_ok = primal_dist > 1e-16 * scale_reference &&
                   dual_dist > 1e-16 * scale_reference &&
                   primal_dist < 1e12 * scale_reference &&
                   dual_dist < 1e12 * scale_reference;
    bool ratio_high = gap_to_primal_ratio >= 1e7 ||
                      isinf(gap_to_primal_ratio);
    bool ratio_low = gap_to_primal_ratio <= 1e-7 &&
                     homogeneous_ratio < 0.5;

    if (ratio_high)
    {
        *primal_weight /= 10.0;
        reset_pid_integrator(pid);
        if (dist_ok)
            align_pid_last_error(pid, *primal_weight, primal_dist, dual_dist);
        return;
    }

    if (ratio_low)
    {
        *primal_weight *= 10.0;
        reset_pid_integrator(pid);
        if (dist_ok)
            align_pid_last_error(pid, *primal_weight, primal_dist, dual_dist);
        return;
    }

    if (dist_ok)
    {
        apply_pid_update(pid, primal_weight, params, primal_dist, dual_dist, false);
    }
    else
    {
        apply_pid_update(pid, primal_weight, params, primal_dist, dual_dist, true);
    }
}

static void reset_pid_integrator(pid_state_t *pid)
{
    pid->error_sum = 0.0;
}

static void align_pid_last_error(pid_state_t *pid,
                                 double primal_weight,
                                 double primal_dist,
                                 double dual_dist)
{
    double error = log(dual_dist) - log(primal_dist) - log(primal_weight);
    if (isfinite(error))
        pid->last_error = error;
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
    if (clamp_step && step * error > 0.0)
    {
        const double max_log_step = log(10.0);
        step = fmin(fmax(step, -max_log_step), max_log_step);
    }
    *primal_weight *= exp(step);
    pid->last_error = error;
}

void initialize_pid_state(pid_state_t *pid)
{
    pid->error_sum = 0.0;
    pid->last_error = 0.0;
}
