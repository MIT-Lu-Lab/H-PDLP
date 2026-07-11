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
#include <stdbool.h>

typedef struct
{
    double error_sum;
    double last_error;
} pid_state_t;

#ifdef __cplusplus
extern "C"
{
#endif

void apply_restart_pid(pid_state_t *pid,
                       double *primal_weight,
                       const step_size_parameters_t *params,
                       double primal_dist,
                       double dual_dist,
                       double relative_objective_gap,
                       double relative_primal_residual,
                       double homogeneous_scale,
                       double homogeneous_ratio);

void initialize_pid_state(pid_state_t *pid);

#ifdef __cplusplus
}
#endif
