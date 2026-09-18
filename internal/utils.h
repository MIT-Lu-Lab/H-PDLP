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
#include "internal_types.h"
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
struct absolute_value_transform_t
{
    __host__ __device__ double operator()(const double &value) const
    {
        return value < 0.0 ? -value : value;
    }
};
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    void *safe_malloc(size_t size);
    void *safe_calloc(size_t num, size_t size);
    void *safe_realloc(void *ptr, size_t new_size);

    void allocate_zeroed_device_buffer(void **dst, size_t bytes);

    void allocate_and_copy_to_device(
        void **dst,
        const void *src,
        size_t bytes);

    void allocate_cub_reduce_workspace(pdhg_solver_state_t *state);

    void reduce_device_sum(
        pdhg_solver_state_t *state,
        const double *input,
        int length,
        double *output);

    void reduce_device_max(
        pdhg_solver_state_t *state,
        const double *input,
        int length,
        double *output);

    void reduce_device_abs_max(
        pdhg_solver_state_t *state,
        const double *input,
        int length,
        double *output);

    void filter_constraint_matrix_entries(
        lp_problem_t *working_problem,
        const lp_problem_t *original_problem,
        const pdhg_parameters_t *params);

    void warn_on_large_objective_coefficients(const lp_problem_t *problem);

    void replace_large_bounds_with_infinity(
        const double *original_lower,
        const double *original_upper,
        double **working_lower,
        double **working_upper,
        int length,
        double infinite_bound);

    void compute_lp_problem_norms(
        const lp_problem_t *problem,
        norm_type_t optimality_norm,
        double infinite_bound,
        double *out_objective_norm,
        double *out_bound_norm);

    void compute_lp_problem_norms_device(
        pdhg_solver_state_t *state,
        norm_type_t optimality_norm,
        double *out_objective_norm,
        double *out_bound_norm);

#ifdef __cplusplus
}
#endif
