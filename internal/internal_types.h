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
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

typedef struct
{
    int num_rows;
    int num_cols;
    int num_nonzeros;
    int *row_ptr;
    int *col_ind;
    int *row_ind;
    double *val;
    int *transpose_map;
} cu_sparse_matrix_csr_t;

typedef struct
{
    int num_variables;
    int num_constraints;
    double *variable_lower_bound;
    double *variable_upper_bound;
    double *objective_vector;
    double objective_constant;
    cu_sparse_matrix_csr_t *constraint_matrix;
    cu_sparse_matrix_csr_t *constraint_matrix_t;
    double *constraint_lower_bound;
    double *constraint_upper_bound;

    int num_blocks_primal;
    int num_blocks_dual;
    int num_blocks_primal_dual;
    int num_blocks_nnz;

    cusparseHandle_t sparse_handle_A;
    cusparseHandle_t sparse_handle_At;
    cudaStream_t stream_A;
    cudaStream_t stream_At;
    cudaEvent_t event_A;
    cudaEvent_t event_At;
    cublasHandle_t blas_handle_A;
    cublasHandle_t blas_handle_At;
    curandGenerator_t curand_gen;
    void *primal_spmv_buffer;
    void *dual_spmv_buffer;

    double objective_vector_norm;
    double constraint_bound_norm;

    double *initial_primal_solution;
    double *current_primal_solution;
    double *pdhg_primal_solution;
    double *reflected_primal_solution;
    double *dual_product;
    double *primal_slack;
    double *dual_slack;

    double *initial_dual_solution;
    double *current_dual_solution;
    double *pdhg_dual_solution;
    double *reflected_dual_solution;
    double *primal_product;

    double step_size;
    double primal_weight;

    int total_count;
    int inner_count;
    clock_t start_time;
    double cumulative_time_sec;

    double primal_weight_error_sum;
    double primal_weight_last_error;
    double best_primal_weight;
    double best_primal_dual_residual_gap;

    double *constraint_rescaling;
    double *variable_rescaling;
    double constraint_bound_rescaling;
    double objective_vector_rescaling;
    double rescaling_time_sec;

    double *constraint_lower_bound_finite_val;
    double *constraint_upper_bound_finite_val;
    double *variable_lower_bound_finite_val;
    double *variable_upper_bound_finite_val;

    double *primal_residual;
    double absolute_primal_residual;
    double relative_primal_residual;
    double *dual_residual;
    double absolute_dual_residual;
    double relative_dual_residual;
    double primal_objective_value;
    double dual_objective_value;
    double objective_gap;
    double relative_objective_gap;
    termination_reason_t termination_reason;

    double *delta_primal_solution;
    double *delta_dual_solution;
    double fixed_point_error;
    double initial_fixed_point_error;
    double last_trial_fixed_point_error;

    cusparseSpMatDescr_t matA;
    cusparseSpMatDescr_t matAt;
    cusparseDnVecDescr_t vec_primal_sol;
    cusparseDnVecDescr_t vec_dual_sol;
    cusparseDnVecDescr_t vec_primal_prod;
    cusparseDnVecDescr_t vec_dual_prod;

    void *reduce_workspace_d;
    size_t reduce_workspace_bytes;
    double *scalar_pool_d;

    double *ones_primal_d;
    double *ones_dual_d;
} pdhg_solver_state_t;

