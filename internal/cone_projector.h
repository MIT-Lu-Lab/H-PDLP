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
    double derivative;
    double function_value;
} newton_terms_t;

typedef struct {
    const double *lower_bound_d;
    const double *upper_bound_d;
    int num_elements;
    int num_blocks_newton;
    double *scalar_d;
    double *unprojected_scalar_d;
    newton_terms_t *newton_partials_d;
} box_newton_state_t;

void update_hsd_cone_dual(
    const cone_parameters_t *cone_params,
    box_newton_state_t *cone_state,
    double *current_dual,
    double *reflected_dual,
    double *evaluation_dual,
    const double *initial_dual,
    const double *gradient,
    const double *step_size_d,
    const double *homogeneous_scale_current,
    const int *base_count_d,
    int k_offset,
    double reflection_coefficient,
    const double *offset_initial,
    double *offset_current,
    double *offset_evaluation,
    double *offset_reflected,
    bool save_evaluation_iterate,
    cudaStream_t stream);

void update_primal_box_resolvent(
    const cone_parameters_t *cone_params,
    box_newton_state_t *newton_state,
    double *current_primal,
    double *reflected_primal,
    double *evaluation_primal,
    const double *initial_primal,
    const double *unprojected_vector,
    const double *objective_d,
    const double *step_size_d,
    const double *step_size_homogeneous_d,
    const int *base_count_d,
    int k_offset,
    double reflection_coefficient,
    double *initial_homogeneous_scale,
    double *current_homogeneous_scale,
    double *evaluation_homogeneous_scale,
    double *reflected_homogeneous_scale,
    bool save_evaluation_iterate,
    cudaStream_t stream);

int max_cooperative_dual_cone_blocks();

int max_cooperative_primal_box_blocks();

#ifdef __cplusplus
}
#endif
