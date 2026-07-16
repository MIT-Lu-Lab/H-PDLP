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

#include "conjugate_gradient.h"
#include "utils.h"
#include <math.h>

static constexpr double CG_TOLERANCE = 1.0e-12;
static constexpr int CG_MAX_ITER = 1000;

void solve_initial_primal_system(
    pdhg_solver_state_t *state,
    double *solution_d,
    double *residual_d)
{
    const int n_vars = state->num_variables;
    const double lambda = state->control.step_size_base;
    const double inv_lambda = 1.0 / lambda;
    const double neg_lambda = -lambda;

    double residual_norm_squared = 0.0;
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle_A,
        n_vars,
        residual_d, 1,
        residual_d, 1,
        &residual_norm_squared));
    if (!(residual_norm_squared > 0.0))
    {
        return;
    }

    const size_t allocation_bytes =
        state->var_bytes > 0 ? state->var_bytes : sizeof(double);
    double *direction_d = NULL;
    double *operator_product_d = NULL;
    CUDA_CHECK(cudaMalloc(&direction_d, allocation_bytes));
    CUDA_CHECK(cudaMalloc(&operator_product_d, allocation_bytes));

    cusparseDnVecDescr_t spmv_direction;
    cusparseDnVecDescr_t spmv_operator_product;
    CUSPARSE_CHECK(cusparseCreateDnVec(
        &spmv_direction, n_vars, direction_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(
        &spmv_operator_product, n_vars, operator_product_d, CUDA_R_64F));

    const double residual_target = CG_TOLERANCE * sqrt(residual_norm_squared);
    CUDA_CHECK(cudaMemcpyAsync(
        direction_d,
        residual_d,
        state->var_bytes,
        cudaMemcpyDeviceToDevice,
        state->stream_A));

    for (int iteration = 0; iteration < CG_MAX_ITER; ++iteration)
    {
        CUSPARSE_CHECK(cusparseSpMV(
            state->sparse_handle_A,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &HOST_ONE,
            state->matA,
            spmv_direction,
            &HOST_ZERO,
            state->spmv_primal_product,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG2,
            state->matA_spmv_workspace_d));
        CUSPARSE_CHECK(cusparseSpMV(
            state->sparse_handle_A,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &HOST_ONE,
            state->matAt,
            state->spmv_primal_product,
            &HOST_ZERO,
            spmv_operator_product,
            CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG2,
            state->matAt_spmv_workspace_d));
        CUBLAS_CHECK(cublasDscal(
            state->blas_handle_A,
            n_vars,
            &neg_lambda,
            operator_product_d, 1));
        CUBLAS_CHECK(cublasDaxpy(
            state->blas_handle_A,
            n_vars,
            &inv_lambda,
            direction_d, 1,
            operator_product_d, 1));

        double direction_curvature = 0.0;
        CUBLAS_CHECK(cublasDdot(
            state->blas_handle_A,
            n_vars,
            direction_d, 1,
            operator_product_d, 1,
            &direction_curvature));
        if (!(direction_curvature > 0.0))
        {
            break;
        }
        const double step = residual_norm_squared / direction_curvature;
        const double neg_step = -step;
        CUBLAS_CHECK(cublasDaxpy(
            state->blas_handle_A,
            n_vars,
            &step,
            direction_d, 1,
            solution_d, 1));
        CUBLAS_CHECK(cublasDaxpy(
            state->blas_handle_A,
            n_vars,
            &neg_step,
            operator_product_d, 1,
            residual_d, 1));

        double next_residual_norm_squared = 0.0;
        CUBLAS_CHECK(cublasDdot(
            state->blas_handle_A, n_vars, residual_d, 1, residual_d, 1,
            &next_residual_norm_squared));
        if (sqrt(next_residual_norm_squared) <= residual_target)
        {
            break;
        }
        const double direction_mix =
            next_residual_norm_squared / residual_norm_squared;
        CUBLAS_CHECK(cublasDscal(
            state->blas_handle_A,
            n_vars,
            &direction_mix,
            direction_d, 1));
        CUBLAS_CHECK(cublasDaxpy(
            state->blas_handle_A,
            n_vars,
            &HOST_ONE,
            residual_d, 1,
            direction_d, 1));
        residual_norm_squared = next_residual_norm_squared;
    }

    CUSPARSE_CHECK(cusparseDestroyDnVec(spmv_direction));
    CUSPARSE_CHECK(cusparseDestroyDnVec(spmv_operator_product));
    CUDA_CHECK(cudaFree(direction_d));
    CUDA_CHECK(cudaFree(operator_product_d));
}
