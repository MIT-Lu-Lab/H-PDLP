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

static bool compute_pid_error(double primal_weight,
                              double primal_dist,
                              double dual_dist,
                              double *error);
static void apply_pid_rebase(pid_state_t *pid,
                             double *primal_weight,
                             double primal_dist,
                             double dual_dist,
                             bool dist_ok);
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
static double clamp_primal_weight(double primal_weight);


void initialize_pid_state(pid_state_t *pid)
{
    pid->error_sum = 0.0;
    pid->last_error = 0.0;
    pid->suspended = false;
    pid->has_last_error = false;
}


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
    if (homogeneous_ratio > 0.9)
    {
        pid->suspended = true;
        return;
    }

    if (pid->suspended)
    {
        pid->suspended = false;
        pid->has_last_error = false;
    }

    double gap_to_primal_ratio = relative_objective_gap / relative_primal_residual;
    double scale_reference = fmax(homogeneous_scale, 1e-300);
    bool dist_ok = primal_dist > 1e-16 * scale_reference &&
                   dual_dist > 1e-16 * scale_reference &&
                   primal_dist < 1e12 * scale_reference &&
                   dual_dist < 1e12 * scale_reference;
    bool ratio_high = gap_to_primal_ratio >= 1e7 || isinf(gap_to_primal_ratio);
    bool ratio_low = gap_to_primal_ratio <= 1e-7;

    if (ratio_high)
    {
        *primal_weight /= 10.0;
        apply_pid_rebase(pid, primal_weight, primal_dist, dual_dist, dist_ok);
        return;
    }

    if (ratio_low)
    {
        *primal_weight *= 10.0;
        apply_pid_rebase(pid, primal_weight, primal_dist, dual_dist, dist_ok);
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

static void align_pid_last_error(pid_state_t *pid,
                                 double primal_weight,
                                 double primal_dist,
                                 double dual_dist)
{
    double error = 0.0;
    if (compute_pid_error(primal_weight, primal_dist, dual_dist, &error))
    {
        pid->last_error = error;
        pid->has_last_error = true;
    }
    else
    {
        pid->has_last_error = false;
    }
}

static void apply_pid_rebase(pid_state_t *pid,
                             double *primal_weight,
                             double primal_dist,
                             double dual_dist,
                             bool dist_ok)
{
    *primal_weight = clamp_primal_weight(*primal_weight);
    pid->error_sum = 0.0;
    pid->has_last_error = false;
    if (dist_ok)
        align_pid_last_error(pid, *primal_weight, primal_dist, dual_dist);
}

static void apply_pid_update(pid_state_t *pid,
                             double *primal_weight,
                             const step_size_parameters_t *params,
                             double primal_dist,
                             double dual_dist,
                             bool clamp_step)
{
    double error = 0.0;
    if (!compute_pid_error(*primal_weight, primal_dist, dual_dist, &error))
        return;
    double next_error_sum = params->i_smooth * pid->error_sum + error;
    double delta_error = pid->has_last_error ? error - pid->last_error : 0.0;
    double step = params->k_p * error +
                  params->k_i * next_error_sum +
                  params->k_d * delta_error;
    if (clamp_step && step * error > 0.0)
    {
        const double max_log_step = log(10.0);
        step = fmin(fmax(step, -max_log_step), max_log_step);
    }

    double candidate_weight = *primal_weight * exp(step);
    bool pushes_outward = (candidate_weight >= 1e8 && error > 0.0) ||
                          (candidate_weight <= 1e-8 && error < 0.0);
    if (!pushes_outward)
        pid->error_sum = next_error_sum;

    *primal_weight = clamp_primal_weight(candidate_weight);
    pid->last_error = error;
    pid->has_last_error = true;
}

static bool compute_pid_error(double primal_weight,
                              double primal_dist,
                              double dual_dist,
                              double *error)
{
    if (!(primal_weight > 0.0) || !(primal_dist > 0.0) || !(dual_dist > 0.0) ||
        !isfinite(primal_weight) || !isfinite(primal_dist) || !isfinite(dual_dist))
        return false;

    double value = log(dual_dist) - log(primal_dist) - log(primal_weight);
    if (!isfinite(value))
        return false;

    *error = value;
    return true;
}

static double clamp_primal_weight(double primal_weight)
{
    return fmin(fmax(primal_weight, 1e-8), 1e8);
}
