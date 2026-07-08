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

#pragma once

#include "cuhsdlp_types.h"
#include "internal_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct pdhg_metrics_t {
        double absolute_primal_residual;
        double relative_primal_residual;
        double absolute_dual_residual;
        double relative_dual_residual;

        double primal_objective;
        double dual_objective;
        double objective_gap;
        double relative_objective_gap;

        double current_primal_scale;
        double pdhg_primal_scale;
        double pdhg_dual_scale;

        double primal_solution_norm;
        double dual_solution_norm;

        double primal_infeasibility_ratio;
        double dual_infeasibility_ratio;
        double max_primal_ray_infeasibility;
        double max_dual_ray_infeasibility;
        double primal_ray_linear_objective;
        double dual_ray_objective;

        double normalized_homogeneous_gap;
        double normalized_homogeneous_ratio;
    } pdhg_metrics_t;

    void compute_termination_metrics(
        pdhg_solver_state_t *state,
        const pdhg_parameters_t *params);

    void check_termination_criteria(
        pdhg_solver_state_t *solver_state,
        const termination_criteria_t *criteria);

    bool should_do_adaptive_restart(
        pdhg_solver_state_t *solver_state,
        const pdhg_parameters_t *params);

    bool termination_has_solution(termination_reason_t reason);

    cuhsdlp_result_t *create_result_from_state(
        pdhg_solver_state_t *state,
        const lp_problem_t *original_problem);

#ifdef __cplusplus
}
#endif
