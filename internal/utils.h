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

#include "hopdlp_types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void *safe_malloc(size_t size);
    void *safe_calloc(size_t num, size_t size);
    void *safe_realloc(void *ptr, size_t new_size);

    bool validate_lp_problem_data(const lp_problem_t *prob);

    void filter_constraint_matrix_entries(
        const lp_problem_t *original_problem,
        const pdhg_parameters_t *params,
        int **owned_row_pointers,
        int **owned_col_indices,
        double **owned_values);

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

#ifdef __cplusplus
}
#endif
