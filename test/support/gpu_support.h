#pragma once

#include "setup.h"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

static int test_gpu_status(void)
{
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && count == 0))
    {
        fprintf(stderr, "SKIP: no usable CUDA device\n");
        return 77;
    }
    if (status != cudaSuccess)
    {
        fprintf(stderr, "CUDA device query failed: %s\n", cudaGetErrorString(status));
        return 1;
    }
    return 0;
}

static inline void disable_rescaling(pdhg_parameters_t *params)
{
    params->scaling_params.geometric_mean_iterations = 0;
    params->scaling_params.l_inf_ruiz_iterations = 0;
    params->scaling_params.has_pock_chambolle_alpha = false;
    params->scaling_params.bound_objective_rescaling = false;
}

static inline pdhg_solver_state_t *create_solver_state(const lp_problem_t *problem,
                                                       const pdhg_parameters_t *params)
{
    pdhg_solver_state_t *state =
        (pdhg_solver_state_t *)calloc(1, sizeof(pdhg_solver_state_t));
    initialize_solver_state(state, problem, problem, params);
    return state;
}
