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
#include <cuda_runtime.h>

void solve_cone_newton(
    const cone_parameters_t *cone_params,
    cone_projection_state_t *cone_state,
    const double *unprojected_vector,
    cudaStream_t stream);

int max_cooperative_cone_grid();
