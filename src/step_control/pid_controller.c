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

static void reset_pid_errors(pid_state_t *pid);
static bool apply_primal_weight_safeguard(pid_state_t *pid,
                                          double *primal_weight,
                                          double relative_objective_gap,
                                          double relative_primal_residual,
                                          double relative_dual_residual);
static void apply_pid_update(pid_state_t *pid,
                             double *primal_weight,
                             const step_size_parameters_t *step_size_params,
                             double primal_dist,
                             double dual_dist,
                             bool clamp_step);

void initialize_pid_state(pid_state_t *pid)
{
    reset_pid_errors(pid);
    pid->min_kkt_error = INFINITY;
    pid->max_residual_balance = 0.0;
    pid->balanced_primal_weight = 0.0;
}

static void reset_pid_errors(pid_state_t *pid)
{
    pid->error_sum = 0.0;
    pid->last_error = 0.0;
    pid->has_last_error = false;
}

void apply_restart_pid(pid_state_t *pid,
                       double *primal_weight,
                       const step_size_parameters_t *step_size_params,
                       double primal_dist,
                       double dual_dist,
                       double relative_objective_gap,
                       double relative_primal_residual,
                       double relative_dual_residual,
                       double homogeneous_scale,
                       double homogeneous_ratio)
{
    if (homogeneous_ratio > 0.9)
    {
        reset_pid_errors(pid);
        return;
    }

    if (step_size_params->primal_weight_safeguard &&
        apply_primal_weight_safeguard(pid, primal_weight, relative_objective_gap,
                                      relative_primal_residual, relative_dual_residual))
        return;

    double scale_reference = fmax(homogeneous_scale, 1e-300);
    bool dist_ok = primal_dist > 1e-16 * scale_reference &&
                   dual_dist > 1e-16 * scale_reference &&
                   primal_dist < 1e16 * scale_reference &&
                   dual_dist < 1e16 * scale_reference;

    double primal_error = fmax(relative_primal_residual, relative_objective_gap);
    double dual_error = fmax(relative_dual_residual, relative_objective_gap);
    double error_ratio = primal_error / dual_error;
    bool ratio_ok = error_ratio > 1e-7 && error_ratio < 1e7;

    apply_pid_update(pid, primal_weight, step_size_params, primal_dist, dual_dist,
                     !(dist_ok && ratio_ok));
}

static bool apply_primal_weight_safeguard(pid_state_t *pid,
                                          double *primal_weight,
                                          double relative_objective_gap,
                                          double relative_primal_residual,
                                          double relative_dual_residual)
{
    const double kkt_recording_threshold = 1e-2;
    const double kkt_activation_threshold = 1e-5;
    const double kkt_growth_factor = 100.0;
    double kkt_error = fmax(relative_objective_gap,
                            fmax(relative_primal_residual, relative_dual_residual));

    if (pid->min_kkt_error < kkt_activation_threshold &&
        pid->balanced_primal_weight > 0.0 &&
        kkt_error >= kkt_growth_factor * pid->min_kkt_error)
    {
        *primal_weight = pid->balanced_primal_weight;
        reset_pid_errors(pid);
        return true;
    }
    if (kkt_error < pid->min_kkt_error)
        pid->min_kkt_error = kkt_error;

    if (pid->min_kkt_error < kkt_recording_threshold &&
        relative_primal_residual > 0.0 && relative_dual_residual > 0.0)
    {
        double residual_balance = fmin(relative_primal_residual, relative_dual_residual) /
                                  fmax(relative_primal_residual, relative_dual_residual);
        if (residual_balance > pid->max_residual_balance)
        {
            pid->max_residual_balance = residual_balance;
            pid->balanced_primal_weight = *primal_weight;
        }
    }
    return false;
}

static void apply_pid_update(pid_state_t *pid,
                             double *primal_weight,
                             const step_size_parameters_t *step_size_params,
                             double primal_dist,
                             double dual_dist,
                             bool clamp_step)
{
    double error = log(dual_dist) - log(primal_dist) - log(*primal_weight);
    if (!isfinite(error))
        return;

    pid->error_sum = step_size_params->i_smooth * pid->error_sum + error;
    double delta_error = pid->has_last_error ? error - pid->last_error : 0.0;
    double step = step_size_params->k_p * error +
                  step_size_params->k_i * pid->error_sum +
                  step_size_params->k_d * delta_error;
    if (clamp_step && step * error > 0.0)
    {
        const double max_log_step = log(10.0);
        step = fmin(fmax(step, -max_log_step), max_log_step);
    }

    *primal_weight *= exp(step);
    pid->last_error = error;
    pid->has_last_error = true;
}
