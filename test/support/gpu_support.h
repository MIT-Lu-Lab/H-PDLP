#pragma once

#include "hpdlp.h"
#include "setup.h"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static inline lp_problem_t *create_diagonal_problem(const double values[2])
{
    const double objective[] = {1.0, 1.0};
    const int row_ptr[] = {0, 1, 2};
    const int col_ind[] = {0, 1};
    const double lower[] = {0.0, 0.0};
    const double upper[] = {1.0, 1.0};
    const double constant = 0.0;
    matrix_desc_t matrix;
    memset(&matrix, 0, sizeof(matrix));
    matrix.m = matrix.n = 2;
    matrix.fmt = matrix_csr;
    matrix.data.csr.nnz = 2;
    matrix.data.csr.row_ptr = row_ptr;
    matrix.data.csr.col_ind = col_ind;
    matrix.data.csr.vals = values;
    return create_lp_problem(objective, &matrix, lower, upper, lower, upper, &constant);
}
