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

#include "internal_types.h"
#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif

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

    void reduce_device_abs_max(
        pdhg_solver_state_t *state,
        const double *input,
        int length,
        double *output);

    void reduce_device_box_support(
        pdhg_solver_state_t *state,
        const double *direction,
        const double *lb,
        const double *ub,
        double sign,
        int length,
        double *output);

    void reduce_device_box_support_magnitude(
        pdhg_solver_state_t *state,
        const double *direction,
        const double *lb,
        const double *ub,
        double sign,
        int length,
        double *output);

    void reduce_device_absolute_product(
        pdhg_solver_state_t *state,
        const double *left,
        const double *right,
        int length,
        double *output);

    void reduce_device_recession_violation(
        pdhg_solver_state_t *state,
        const double *value,
        const double *lb,
        const double *ub,
        const double *rescaling,
        int length,
        double *output);

    void reduce_device_reduced_cost_violation(
        pdhg_solver_state_t *state,
        const double *dual_product,
        const double *lb,
        const double *ub,
        const double *rescaling,
        int length,
        double *output);

    void compute_device_abs(
        pdhg_solver_state_t *state,
        const double *input,
        int length,
        double *output);

#ifdef __cplusplus
}
#endif

__host__ __device__ __forceinline__ double project_box_recession(
    double value,
    double lb,
    double ub)
{
    const double lower = isfinite(lb) ? 0.0 : -INFINITY;
    const double upper = isfinite(ub) ? 0.0 : INFINITY;
    return fmax(lower, fmin(value, upper));
}

__host__ __device__ __forceinline__ double project_box_multiplier(
    double value,
    double lb,
    double ub)
{
    const double lower = isfinite(ub) ? -INFINITY : 0.0;
    const double upper = isfinite(lb) ? INFINITY : 0.0;
    return fmax(lower, fmin(value, upper));
}
