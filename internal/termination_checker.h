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

#include "hpdlp_types.h"
#include "internal_types.h"

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

        double bound_multiplier_absolute_dual_residual;
        double bound_multiplier_relative_dual_residual;
        double bound_multiplier_dual_objective;
        double bound_multiplier_objective_gap;
        double bound_multiplier_relative_objective_gap;

        double primal_current_scale;
        double primal_pdhg_scale;
        double dual_pdhg_scale;

        double primal_solution_norm;
        double dual_solution_norm;

        certificate_metrics_t primal_infeasibility_iterate;
        certificate_metrics_t dual_infeasibility_iterate;
        certificate_metrics_t primal_infeasibility_step;
        certificate_metrics_t dual_infeasibility_step;
        certificate_metrics_t primal_infeasibility_restart;
        certificate_metrics_t dual_infeasibility_restart;

        double normalized_homogeneous_gap;
        double normalized_homogeneous_ratio;
    } pdhg_metrics_t;

    void compute_termination_metrics(
        pdhg_solver_state_t *state,
        const pdhg_parameters_t *params);

    void compute_device_termination_metrics(
        pdhg_solver_state_t *state,
        const pdhg_parameters_t *params);

    void copy_termination_metrics_to_host(pdhg_solver_state_t *state);

    void compute_certificate_residuals(pdhg_solver_state_t *state, double objective_margin);

    void check_termination_criteria(
        pdhg_solver_state_t *state,
        const termination_criteria_t *criteria);

    void compute_certificate_dual_product(pdhg_solver_state_t *state);

    candidate_vectors_t select_candidate_vectors(
        pdhg_solver_state_t *state,
        certificate_source_t source);

    candidate_reductions_t *select_candidate_reductions(
        termination_reductions_t *reductions,
        certificate_source_t source);

#ifdef __cplusplus
}
#endif
