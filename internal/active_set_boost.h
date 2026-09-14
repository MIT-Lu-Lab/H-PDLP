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
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    static inline bool is_active_set_boost_enabled(
        const pdhg_parameters_t *params)
    {
        return params->active_set_params.step_size_boost &&
               params->active_set_params.max_reverts > 0;
    }

    void initialize_active_set(
        pdhg_solver_state_t *state,
        const pdhg_parameters_t *params);

    void active_set_free(active_set_state_t *active_set);

    void update_active_set_masks(
        pdhg_solver_state_t *state,
        const pdhg_parameters_t *params);

    bool should_revert_active_set_boost(
        const pdhg_solver_state_t *state,
        const pdhg_parameters_t *params);

    void try_raise_active_set_step_size(
        pdhg_solver_state_t *state,
        const pdhg_parameters_t *params);

    void revert_active_set_step_size(
        pdhg_solver_state_t *state,
        const pdhg_parameters_t *params);

#ifdef __cplusplus
}
#endif
