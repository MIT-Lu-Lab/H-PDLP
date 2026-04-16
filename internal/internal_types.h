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
    CONE_SCALE = 0,
    CONE_BUFFER_SIZE = 1,
};

enum {
    WORKSPACE_SCALAR_PRIMAL_NORM      = 0,
    WORKSPACE_SCALAR_DUAL_NORM        = 1,
    WORKSPACE_SCALAR_CROSS_TERM       = 2,
    WORKSPACE_SCALAR_BOUND_DUAL_NORM  = 3,
    WORKSPACE_SCALAR_BOUND_CROSS_TERM = 4,
    WORKSPACE_SCALAR_INV_TAU          = 5,
    WORKSPACE_SCALAR_C_DOT_X          = 6,
    WORKSPACE_SCALAR_COUNT            = 7,
};

enum {
    RESIDUAL_METRIC_ABSOLUTE_PRIMAL = 0,
    RESIDUAL_METRIC_RELATIVE_PRIMAL = 1,
    RESIDUAL_METRIC_ABSOLUTE_DUAL = 2,
    RESIDUAL_METRIC_RELATIVE_DUAL = 3,
    RESIDUAL_METRIC_PRIMAL_OBJECTIVE = 4,
    RESIDUAL_METRIC_DUAL_OBJECTIVE = 5,
    RESIDUAL_METRIC_OBJECTIVE_GAP = 6,
    RESIDUAL_METRIC_RELATIVE_GAP = 7,
    RESIDUAL_METRIC_COUNT = 8,
};

enum {
    NORM_RESULT_OBJECTIVE = 0,
    NORM_RESULT_BOUND = 1,
    NORM_RESULT_COUNT = 2,
};

enum {
    RESCALING_FACTOR_CONSTRAINT = 0,
    RESCALING_FACTOR_OBJECTIVE = 1,
    RESCALING_FACTOR_COUNT = 2,
};

enum {
    RESTART_DISTANCE_PRIMAL = 0,
    RESTART_DISTANCE_DUAL = 1,
    RESTART_DISTANCE_COUNT = 2,
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

    double original_objective_norm;
    double original_bound_norm;
    double original_constraint_norm;
    double rescaled_objective_norm;
    double rescaled_bound_norm;
    double rescaled_constraint_norm;

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

    double *power_iteration_vector_d;
    double *hsd_scalars_d;
    double *cone_newton_constraint_d;
    double *cone_newton_bound_d;

    double step_size_primal;
    double step_size_dual;
    double step_size_homogeneous;

    double primal_weight;
    double step_size_base;

    int total_count;
    int inner_count;
    int restart_count;
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
    cusparseDnVecDescr_t spmv_current_primal_solution;
    cusparseDnVecDescr_t spmv_reflected_dual_solution;
    cusparseDnVecDescr_t spmv_pdhg_primal_solution;
    cusparseDnVecDescr_t spmv_pdhg_dual_solution;
    cusparseDnVecDescr_t spmv_delta_dual_solution;
    cusparseDnVecDescr_t spmv_primal_product;
    cusparseDnVecDescr_t spmv_dual_product;
    cusparseDnVecDescr_t spmv_power_iteration_vector;
    cusparseDnVecDescr_t spmv_power_iteration_next_vector;
    cusparseDnVecDescr_t spmv_power_iteration_product;

    void *cub_reduce_workspace;
    void *cub_reduce_workspace_bound;
    size_t cub_reduce_workspace_bytes;
    double *workspace_scalars_d;
    double *norm_results_d;
    double *rescaling_factors_d;
    double *restart_distances_d;
    double *fixed_point_error_d;
    double *residual_metrics_d;
} pdhg_solver_state_t;

