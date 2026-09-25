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

#include "cuda_macros.h"
#include "cuhsdlp_types.h"
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

typedef struct {
    const double *lower_bound_d;
    const double *upper_bound_d;
    int num_elements;
    bool newton_terms_are_zero;
    int num_blocks_newton;
    double *scalar_d;
    newton_terms_t *newton_partials_d;
} box_newton_state_t;

void update_dual_box_cone_projection(
    const cone_parameters_t *cone_params,
    box_newton_state_t *dual_state,
    double *current_dual,
    double *reflected_dual,
    double *evaluation_dual,
    double *constraint_dual_projection_input,
    const double *initial_dual,
    const double *gradient,
    const double *step_size_d,
    const double *primal_scale_current,
    const int *base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *dual_scale_initial,
    double *dual_scale_current,
    double *dual_scale_evaluation,
    double *dual_scale_reflected,
    bool is_major,
    cudaStream_t stream);

void update_primal_box_cone_resolvent(
    const cone_parameters_t *cone_params,
    box_newton_state_t *primal_state,
    double *current_primal,
    double *reflected_primal,
    double *evaluation_primal,
    double *variable_primal_projection_multiplier,
    const double *initial_primal,
    const double *gradient,
    const double *objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    const double *homogeneous_dual_gradient,
    const int *base_count_d,
    int k_offset,
    double reflection_coefficient,
    double *initial_primal_scale,
    double *current_primal_scale,
    double *evaluation_primal_scale,
    double *reflected_primal_scale,
    bool is_major,
    cudaStream_t stream);

int compute_max_cooperative_dual_box_cone_blocks();

int compute_max_cooperative_primal_box_cone_blocks();

#ifdef __cplusplus
}
#endif
