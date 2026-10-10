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

#include "hpdlp.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int parameter_error(char *error, size_t error_size,
                           const char *requirement, double value);

void set_default_parameters(pdhg_parameters_t *params)
{
    params->presolve = true;
    params->verbose = true;
    params->diagnose = false;
    params->optimality_norm = NORM_TYPE_L2;
    params->reflection_coefficient = 1.0;
    params->termination_evaluation_frequency = 200;

    params->scaling_params.geometric_mean_iterations = 12;
    params->scaling_params.l_inf_ruiz_iterations = 10;
    params->scaling_params.has_pock_chambolle_alpha = true;
    params->scaling_params.pock_chambolle_alpha = 1.0;
    params->scaling_params.bound_objective_rescaling = true;
    params->scaling_params.matrix_zero_tol = 1e-9;
    params->scaling_params.infinite_bound = 1e20;

    params->power_iter_params.max_iter = 5000;
    params->power_iter_params.tol = 1e-4;

    params->step_size_params.step_size_factor = 0.998;
    params->step_size_params.k_p = 0.97;
    params->step_size_params.k_i = 0.05;
    params->step_size_params.k_d = 0.0;
    params->step_size_params.i_smooth = 0.3;
    params->step_size_params.primal_weight_safeguard = true;

    params->cone_params.max_iter = 100;
    params->cone_params.tol = 1e-6;

    params->termination_criteria.eps_optimal_relative = 1e-4;
    params->termination_criteria.eps_feasible_relative = 1e-4;
    params->termination_criteria.eps_farkas_homogeneous = 1e-9;
    params->termination_criteria.eps_farkas_displacement = 1e-10;
    params->termination_criteria.time_sec_limit = 3600.0;
    params->termination_criteria.iteration_limit = INT32_MAX;

    params->restart_params.artificial_restart_threshold = 0.36;
    params->restart_params.sufficient_reduction_for_restart = 0.2;
    params->restart_params.necessary_reduction_for_restart = 0.5;

    params->active_set_params.step_size_boost = true;
    params->active_set_params.window_iterations = 10000;
    params->active_set_params.relative_tol = 1e-8;
    params->active_set_params.safety_factor = 0.9;
    params->active_set_params.min_raise_ratio = 1.1;
    params->active_set_params.reestimate_change_ratio = 0.01;
    params->active_set_params.divergence_ceiling_ratio = 0.7;
    params->active_set_params.divergence_margin = 0.05;
    params->active_set_params.max_reverts = 2;
}

int hpdlp_validate_parameters(const pdhg_parameters_t *params, char *error, size_t error_size)
{
    if (error && error_size)
        error[0] = '\0';
    if (!params)
    {
        if (error && error_size)
            snprintf(error, error_size, "parameters must not be NULL");
        return -1;
    }
    if (params->optimality_norm != NORM_TYPE_L2 && params->optimality_norm != NORM_TYPE_L_INF)
        return parameter_error(error, error_size, "'optimality_norm' must be L2 or L_INF", params->optimality_norm);
    if (!(params->reflection_coefficient > 0.0 &&
          params->reflection_coefficient <= 1.0))
    {
        return parameter_error(error, error_size, "'reflection_coefficient' must be in (0, 1]",
                           params->reflection_coefficient);
    }
    if (params->termination_evaluation_frequency < 3)
    {
        return parameter_error(error, error_size, "'termination_evaluation_frequency' must be >= 3",
                           params->termination_evaluation_frequency);
    }
    if (params->scaling_params.geometric_mean_iterations < 0)
    {
        return parameter_error(error, error_size, "'geometric_mean_iterations' must be >= 0",
                           params->scaling_params.geometric_mean_iterations);
    }
    if (params->scaling_params.l_inf_ruiz_iterations < 0)
    {
        return parameter_error(error, error_size, "'l_inf_ruiz_iterations' must be >= 0",
                           params->scaling_params.l_inf_ruiz_iterations);
    }
    if (params->scaling_params.has_pock_chambolle_alpha &&
        !(params->scaling_params.pock_chambolle_alpha >= 0.0 &&
          params->scaling_params.pock_chambolle_alpha <= 2.0))
    {
        return parameter_error(error, error_size, "'pock_chambolle_alpha' must be in [0, 2]",
                           params->scaling_params.pock_chambolle_alpha);
    }
    if (!(params->scaling_params.matrix_zero_tol >= 0.0 &&
          isfinite(params->scaling_params.matrix_zero_tol)))
    {
        return parameter_error(error, error_size, "'matrix_zero_tol' must be finite and >= 0",
                           params->scaling_params.matrix_zero_tol);
    }
    if (!(params->scaling_params.infinite_bound > 0.0))
    {
        return parameter_error(error, error_size, "'infinite_bound' must be > 0",
                           params->scaling_params.infinite_bound);
    }
    if (params->power_iter_params.max_iter < 1)
    {
        return parameter_error(error, error_size, "'sv_max_iter' must be >= 1",
                           params->power_iter_params.max_iter);
    }
    if (!(params->power_iter_params.tol > 0.0 &&
          isfinite(params->power_iter_params.tol)))
    {
        return parameter_error(error, error_size, "'sv_tol' must be finite and > 0",
                           params->power_iter_params.tol);
    }
    if (!(params->step_size_params.step_size_factor > 0.0 &&
          params->step_size_params.step_size_factor < 1.0))
    {
        return parameter_error(error, error_size, "'step_size_factor' must be in (0, 1)",
                           params->step_size_params.step_size_factor);
    }
    if (!isfinite(params->step_size_params.k_p))
    {
        return parameter_error(error, error_size, "'k_p' must be finite",
                           params->step_size_params.k_p);
    }
    if (!isfinite(params->step_size_params.k_i))
    {
        return parameter_error(error, error_size, "'k_i' must be finite",
                           params->step_size_params.k_i);
    }
    if (!isfinite(params->step_size_params.k_d))
    {
        return parameter_error(error, error_size, "'k_d' must be finite",
                           params->step_size_params.k_d);
    }
    if (!(params->step_size_params.i_smooth >= 0.0 &&
          params->step_size_params.i_smooth < 1.0))
    {
        return parameter_error(error, error_size, "'i_smooth' must be in [0, 1)",
                           params->step_size_params.i_smooth);
    }
    if (params->cone_params.max_iter < 1)
    {
        return parameter_error(error, error_size, "'cone_newton_max_iter' must be >= 1",
                           params->cone_params.max_iter);
    }
    if (!(params->cone_params.tol > 0.0 && isfinite(params->cone_params.tol)))
    {
        return parameter_error(error, error_size, "'cone_newton_tol' must be finite and > 0",
                           params->cone_params.tol);
    }
    if (!(params->termination_criteria.eps_optimal_relative > 0.0 &&
          isfinite(params->termination_criteria.eps_optimal_relative)))
    {
        return parameter_error(error, error_size, "'eps_optimal_relative' must be finite and > 0",
                           params->termination_criteria.eps_optimal_relative);
    }
    if (!(params->termination_criteria.eps_feasible_relative > 0.0 &&
          isfinite(params->termination_criteria.eps_feasible_relative)))
    {
        return parameter_error(error, error_size, "'eps_feasible_relative' must be finite and > 0",
                           params->termination_criteria.eps_feasible_relative);
    }
    if (!(params->termination_criteria.eps_farkas_homogeneous > 0.0 &&
          isfinite(params->termination_criteria.eps_farkas_homogeneous)))
    {
        return parameter_error(error, error_size, "'eps_farkas_homogeneous' must be finite and > 0",
                           params->termination_criteria.eps_farkas_homogeneous);
    }
    if (!(params->termination_criteria.eps_farkas_displacement > 0.0 &&
          isfinite(params->termination_criteria.eps_farkas_displacement)))
    {
        return parameter_error(error, error_size, "'eps_farkas_displacement' must be finite and > 0",
                           params->termination_criteria.eps_farkas_displacement);
    }
    if (!(params->termination_criteria.time_sec_limit > 0.0))
    {
        return parameter_error(error, error_size, "'time_sec_limit' must be > 0",
                           params->termination_criteria.time_sec_limit);
    }
    if (params->termination_criteria.iteration_limit < 1)
    {
        return parameter_error(error, error_size, "'iteration_limit' must be >= 1",
                           params->termination_criteria.iteration_limit);
    }
    if (!(params->restart_params.artificial_restart_threshold > 0.0 &&
          params->restart_params.artificial_restart_threshold <= 1.0))
    {
        return parameter_error(error, error_size, "'artificial_restart_threshold' must be in (0, 1]",
                           params->restart_params.artificial_restart_threshold);
    }
    if (!(params->restart_params.sufficient_reduction_for_restart > 0.0))
    {
        return parameter_error(error, error_size, "'sufficient_reduction_for_restart' must be > 0",
                           params->restart_params.sufficient_reduction_for_restart);
    }
    if (!(params->restart_params.necessary_reduction_for_restart < 1.0))
    {
        return parameter_error(error, error_size, "'necessary_reduction_for_restart' must be < 1",
                           params->restart_params.necessary_reduction_for_restart);
    }
    if (!(params->restart_params.sufficient_reduction_for_restart <=
          params->restart_params.necessary_reduction_for_restart))
    {
        return parameter_error(error, error_size, "'sufficient_reduction_for_restart' must be <= 'necessary_reduction_for_restart'",
                           params->restart_params.sufficient_reduction_for_restart);
    }
    if (params->active_set_params.window_iterations < 1)
    {
        return parameter_error(error, error_size, "'asb_window_iter' must be >= 1",
                           params->active_set_params.window_iterations);
    }
    if (!(params->active_set_params.relative_tol >= 0.0 &&
          isfinite(params->active_set_params.relative_tol)))
    {
        return parameter_error(error, error_size, "'asb_relative_tol' must be finite and >= 0",
                           params->active_set_params.relative_tol);
    }
    if (!(params->active_set_params.safety_factor > 0.0 &&
          params->active_set_params.safety_factor < 1.0))
    {
        return parameter_error(error, error_size, "'asb_safety_factor' must be in (0, 1)",
                           params->active_set_params.safety_factor);
    }
    if (!(params->active_set_params.min_raise_ratio >= 1.0 &&
          isfinite(params->active_set_params.min_raise_ratio)))
    {
        return parameter_error(error, error_size, "'asb_min_raise_ratio' must be finite and >= 1",
                           params->active_set_params.min_raise_ratio);
    }
    if (!(params->active_set_params.reestimate_change_ratio >= 0.0 &&
          isfinite(params->active_set_params.reestimate_change_ratio)))
    {
        return parameter_error(error, error_size, "'asb_reestimate_change_ratio' must be finite and >= 0",
                           params->active_set_params.reestimate_change_ratio);
    }
    if (!(params->active_set_params.divergence_ceiling_ratio > 0.0 &&
          params->active_set_params.divergence_ceiling_ratio <= 1.0))
    {
        return parameter_error(error, error_size, "'asb_divergence_ceiling_ratio' must be in (0, 1]",
                           params->active_set_params.divergence_ceiling_ratio);
    }
    if (!(params->active_set_params.divergence_margin >= 0.0 &&
          isfinite(params->active_set_params.divergence_margin)))
    {
        return parameter_error(error, error_size, "'asb_divergence_margin' must be finite and >= 0",
                           params->active_set_params.divergence_margin);
    }
    if (params->active_set_params.max_reverts < 0)
    {
        return parameter_error(error, error_size, "'asb_max_reverts' must be >= 0",
                           params->active_set_params.max_reverts);
    }
    return 0;
}

static int parameter_error(char *error, size_t error_size,
                           const char *requirement, double value)
{
    if (error && error_size)
        snprintf(error, error_size, "%s (got %g)", requirement, value);
    return -1;
}
