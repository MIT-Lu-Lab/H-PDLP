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
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    HPDLP_API lp_problem_t *create_lp_problem(
        const double *objective_c,
        const matrix_desc_t *A_desc,
        const double *con_lb,
        const double *con_ub,
        const double *var_lb,
        const double *var_ub,
        const double *objective_constant);

    HPDLP_API void set_objective_sense(lp_problem_t *prob, objective_sense_t sense);

    HPDLP_API bool set_start_values(lp_problem_t *prob, const double *primal, const double *dual);

    HPDLP_API hpdlp_result_t *solve_lp_problem(
        const lp_problem_t *prob,
        const pdhg_parameters_t *params);

    HPDLP_API void set_default_parameters(pdhg_parameters_t *params);

    /* Returns 0 for valid parameters, -1 otherwise. No GPU work is performed.
       When provided, error receives a NUL-terminated diagnostic. */
    HPDLP_API int hpdlp_validate_parameters(const pdhg_parameters_t *params,
                                             char *error, size_t error_size);

    HPDLP_API void hpdlp_result_free(hpdlp_result_t *result);

    HPDLP_API void lp_problem_free(lp_problem_t *prob);

    HPDLP_API const char *hpdlp_last_error(void);

#ifdef __cplusplus
}
#endif
