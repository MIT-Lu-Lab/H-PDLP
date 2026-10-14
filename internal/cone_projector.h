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
#include <cuda_runtime.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
    double negative_derivative;
    double function_value;
} newton_terms_t;

enum {
    BOX_CONE_SCALE_LATEST   = 0,
    BOX_CONE_SCALE_PREVIOUS = 1,
    BOX_CONE_SCALE_OLDEST   = 2,
    BOX_CONE_SCALE_COUNT    = 3,
};

typedef struct {
    const double *lower_bound_d;
    const double *upper_bound_d;
    int num_elements;
    bool newton_terms_are_zero;
    int num_blocks_newton;
    double *scale_history_d;
    newton_terms_t *newton_partials_d;
} box_newton_state_t;

void update_dual_box_cone_projection(
    const cone_parameters_t *cone_params,
    const box_newton_state_t *dual_state,
    double *dual_current,
    double *dual_reflected,
    double *dual_evaluation,
    double *constraint_dual_projection_input,
    const double *dual_initial,
    const double *gradient,
    const double *step_size_d,
    const double *primal_current_scale,
    const int *base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *dual_initial_scale,
    double *dual_current_scale,
    double *dual_evaluation_scale,
    double *dual_reflected_scale,
    bool is_major,
    cudaStream_t stream);

void update_primal_box_cone_resolvent(
    const cone_parameters_t *cone_params,
    const box_newton_state_t *primal_state,
    double *primal_current,
    double *primal_reflected,
    double *primal_evaluation,
    double *variable_primal_projection_multiplier,
    const double *primal_initial,
    const double *gradient,
    const double *objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    const double *homogeneous_dual_gradient,
    const int *base_count_d,
    int k_offset,
    double reflection_coefficient,
    double normalization_strength,
    const double *primal_initial_scale,
    double *primal_current_scale,
    double *primal_evaluation_scale,
    double *primal_reflected_scale,
    bool is_major,
    cudaStream_t stream);

int compute_max_cooperative_dual_box_cone_blocks();

int compute_max_cooperative_primal_box_cone_blocks();

#ifdef __cplusplus
}
#endif
