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
#include "internal_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    const char *termination_reason_to_string(termination_reason_t reason);

    void print_initial_info(const pdhg_parameters_t *params, const lp_problem_t *problem);

    void print_iteration_header(const pdhg_parameters_t *params);

    void display_iteration_stats(const pdhg_solver_state_t *state,
                                 const pdhg_parameters_t *params,
                                 bool is_restart);

    void display_active_set_boost_raise(
        const pdhg_parameters_t *params,
        double previous_step_size,
        double updated_step_size,
        double masked_constraint_norm);

    void display_active_set_boost_revert(
        const pdhg_solver_state_t *state,
        const pdhg_parameters_t *params);

    void pdhg_final_log(
        const hopdlp_result_t *result,
        const pdhg_parameters_t *params);

    int get_print_frequency(int iter);

    typedef struct tau_trace_t tau_trace_t;

    tau_trace_t *tau_trace_open(const char *path);

    void tau_trace_write_row(tau_trace_t *trace,
                             const pdhg_solver_state_t *state,
                             const char *event);

    void tau_trace_close(tau_trace_t *trace);

#ifdef __cplusplus
}
#endif
