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

    const char *termination_reason_to_string(termination_reason_t reason);

    void print_initial_info(const pdhg_parameters_t *params, const lp_problem_t *problem);

    void print_iteration_header(bool verbose);

    void display_step_sizes(const pdhg_solver_state_t *state, bool verbose);

    void display_iteration_stats(const pdhg_solver_state_t *state, bool verbose);

    void pdhg_final_log(
        const cuhsdlp_result_t *result,
        const pdhg_parameters_t *params);

    int get_print_frequency(int iter);

#ifdef __cplusplus
}
#endif
