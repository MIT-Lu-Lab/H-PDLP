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

        double initial_homogeneous_scale;
        double current_homogeneous_scale;
        double pdhg_homogeneous_scale;
        double pdhg_dual_constraint_offset;
        double pdhg_dual_bound_offset;

        double primal_solution_norm;
        double dual_solution_norm;

        double relative_primal_infeasibility_residual;
        double relative_primal_infeasibility_objective;
        double relative_dual_infeasibility_residual;
        double relative_dual_infeasibility_objective;
    } pdhg_metrics_t;

    void compute_residual(pdhg_solver_state_t *state, norm_type_t optimality_norm);

    void check_termination_criteria(
        pdhg_solver_state_t *solver_state,
        const termination_criteria_t *criteria);

    bool should_do_adaptive_restart(
        pdhg_solver_state_t *solver_state,
        const pdhg_parameters_t *params);

#ifdef __cplusplus
}
#endif
