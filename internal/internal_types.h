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

enum {
    HSD_CURRENT   = 0,
    HSD_REFLECTED = 1,
    HSD_PDHG      = 2,
    HSD_INITIAL   = 3,
    HSD_PRIMAL_SCALE    = 0,
    HSD_DUAL_CONSTRAINT = 4,
    HSD_DUAL_BOUND      = 8,
    HSD_SCALARS_COUNT   = 12,
};

enum {
    CONE_DERIV_SUM  = 0,
    CONE_VALUE_SUM  = 1,
    CONE_SCALE      = 2,
    CONE_TAU_LO     = 3,
    CONE_TAU_HI     = 4,
    CONE_CONVERGED  = 5,
    CONE_NEG_SCALAR = 6,
    CONE_BUFFER_SIZE = 7,
};

enum {
    SCALAR_POOL_PRIMAL_NORM      = 0,
    SCALAR_POOL_DUAL_NORM        = 1,
    SCALAR_POOL_CROSS_TERM       = 2,
    SCALAR_POOL_BOUND_DUAL_NORM  = 3,
    SCALAR_POOL_BOUND_CROSS_TERM = 4,
    SCALAR_POOL_C_DOT_X          = 7,
};

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
    int num_blocks_primal_dual_slack;
    int num_blocks_nnz;

    size_t var_bytes;
    size_t con_bytes;
    cusparseHandle_t sparse_handle_A;
    cusparseHandle_t sparse_handle_At;
    cudaStream_t stream_A;
    cudaStream_t stream_At;
    cudaEvent_t event_A;
    cudaEvent_t event_At;
    cudaEvent_t event_s;
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
    double *delta_primal_solution;
    double *primal_product;
    double *primal_vec_buffer;

    double *initial_dual_solution;
    double *current_dual_solution;
    double *pdhg_dual_solution;
    double *reflected_dual_solution;
    double *delta_dual_solution;
    double *dual_product;
    double *dual_vec_buffer;

    double *initial_dual_slack;
    double *current_dual_slack;
    double *pdhg_dual_slack;
    double *reflected_dual_slack;
    double *delta_dual_slack;

    double initial_primal_scale;
    double current_primal_scale;
    double reflected_primal_scale;
    double pdhg_primal_scale;
    
    double initial_dual_constraint_offset;
    double current_dual_constraint_offset;
    double reflected_dual_constraint_offset;
    double pdhg_dual_constraint_offset;

    double initial_dual_bound_offset;
    double current_dual_bound_offset;
    double reflected_dual_bound_offset;
    double pdhg_dual_bound_offset;

    double *cone_projection_reduce_buffer_constraint;
    double *cone_projection_reduce_buffer_bound;
    double *hsd_scalars_d;
    double *cone_newton_constraint_d;
    double *cone_newton_bound_d;

    double step_size_primal;
    double step_size_dual;
    double step_size_homogeneous;

    double primal_weight;
    double sigma_A;
    double objective_coupling_budget;

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

    double fixed_point_error;
    double initial_fixed_point_error;
    double last_trial_fixed_point_error;

    cusparseSpMatDescr_t matA;
    cusparseSpMatDescr_t matAt;
    cusparseDnVecDescr_t vec_primal_sol;
    cusparseDnVecDescr_t vec_dual_sol;
    cusparseDnVecDescr_t vec_primal_prod;
    cusparseDnVecDescr_t vec_dual_prod;

    void *cub_reduce_workspace;
    void *cub_reduce_workspace_bound;
    size_t cub_reduce_workspace_bytes;
    double *scalar_pool_d;
} pdhg_solver_state_t;

