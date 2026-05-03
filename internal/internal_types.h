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

enum HsdState {
    HSD_INITIAL     = 0,
    HSD_CURRENT     = 1,
    HSD_PDHG        = 2,
    HSD_REFLECTED   = 3,
    HSD_STATE_COUNT = 4,
};

enum HsdGroup {
    HSD_SCALE           = 0,
    HSD_DUAL_CONSTRAINT = 1,
    HSD_DUAL_BOUND      = 2,
    HSD_GROUP_COUNT     = 3,
};

#define HSD_IDX(group, state) ((group) * HSD_STATE_COUNT + (state))
#define HSD_SCALARS_COUNT     (HSD_GROUP_COUNT * HSD_STATE_COUNT)

enum PdhgReduction {
    PDHG_PRIMAL_NORM   = 0,
    PDHG_DUAL_NORM     = 1,
    PDHG_SLACK_NORM    = 2,
    PDHG_PRIMAL_CROSS  = 3,
    PDHG_DUAL_CROSS    = 4,
    PDHG_SLACK_CROSS   = 5,
    PDHG_COUNT         = 6,
};

enum FarkasReduction {
    FARKAS_DUAL_SOLUTION_NORM    = 0,
    FARKAS_DUAL_SLACK_NORM       = 1,
    FARKAS_HOMOGENEOUS_DUAL_NORM = 2,
    FARKAS_CONSTRAINT_RECESSION  = 3,
    FARKAS_VARIABLE_RECESSION    = 4,
    FARKAS_SUPPORT_CONSTRAINT    = 5,
    FARKAS_SUPPORT_VARIABLE      = 6,
    FARKAS_COUNT                 = 7,
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

typedef struct {
    double denom;
    double numer;
} newton_denom_numer_t;

typedef struct {
    const double *lower_bound;
    const double *upper_bound;
    int num;
    int blocks;
    double *scale_d;
    double *unprojected_scale_d;
    int *converged_d;
    newton_denom_numer_t *newton_partials_d;
} cone_projection_state_t;

typedef struct {
    double absolute_primal_residual;
    double relative_primal_residual;
    double absolute_dual_residual;
    double relative_dual_residual;

    double primal_objective;
    double dual_objective;
    double objective_gap;
    double relative_objective_gap;

    double initial_homogeneous_scale;
    double current_homogeneous_scale;
    double pdhg_homogeneous_scale;
    double pdhg_dual_constraint_offset;
    double pdhg_dual_bound_offset;

    double primal_solution_norm;
    double dual_solution_norm;

    double relative_primal_infeasibility_residual;
    double relative_primal_infeasibility_objective;
    double relative_dual_infeasibility_residual;
    double relative_dual_infeasibility_objective;
} pdhg_metrics_t;

typedef struct
{
    int num_variables;
    int num_constraints;
    cu_sparse_matrix_csr_t *constraint_matrix;
    cu_sparse_matrix_csr_t *constraint_matrix_t;
    double *variable_lower_bound;
    double *variable_upper_bound;
    double *constraint_lower_bound;
    double *constraint_upper_bound;
    double *objective_vector;
    double objective_constant;

    int num_blocks_primal;
    int num_blocks_dual;
    int num_blocks_primal_dual;
    int num_blocks_primal_dual_slack;
    int num_blocks_nnz;
    size_t var_bytes;
    size_t con_bytes;

    cusparseHandle_t  sparse_handle_A;
    cusparseHandle_t  sparse_handle_At;
    cublasHandle_t    blas_handle_A;
    cublasHandle_t    blas_handle_At;
    curandGenerator_t curand_gen;
    cudaStream_t      stream_A;
    cudaStream_t      stream_At;
    cudaEvent_t       event_A;
    cudaEvent_t       event_At;
    cudaEvent_t       event_s;

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
    cusparseDnVecDescr_t spmv_power_iteration_normalized_vector;
    cusparseDnVecDescr_t spmv_power_iteration_product;
    void   *primal_spmv_buffer;
    void   *dual_spmv_buffer;
    void   *cub_reduce_workspace;
    size_t  cub_reduce_workspace_bytes;

    double  original_objective_norm;
    double  original_bound_norm;
    double  original_constraint_norm;
    double  rescaled_objective_norm;
    double  rescaled_bound_norm;
    double  rescaled_constraint_norm;
    double *precond_bound_norm_mp;
    double *precond_objective_norm_mp;
    double *constraint_rescaling_d;
    double *variable_rescaling_d;
    double  constraint_bound_rescaling;
    double  objective_vector_rescaling;
    double *constraint_bound_rescaling_d;
    double *objective_vector_rescaling_d;
    double  rescaling_time_sec;

    double *initial_primal_solution_d;
    double *current_primal_solution_d;
    double *pdhg_primal_solution_d;
    double *reflected_primal_solution_d;
    double *delta_primal_solution_d;
    double *primal_product_d;
    double *primal_vec_buffer_d;

    double *initial_dual_solution_d;
    double *current_dual_solution_d;
    double *pdhg_dual_solution_d;
    double *reflected_dual_solution_d;
    double *delta_dual_solution_d;
    double *dual_product_d;
    double *dual_vec_buffer_d;

    double *initial_dual_slack_d;
    double *current_dual_slack_d;
    double *pdhg_dual_slack_d;
    double *reflected_dual_slack_d;
    double *delta_dual_slack_d;

    double *hsd_scalars_d;
    double *hsd_inverse_scale_d;
    double *hsd_scale_initial_d;
    double *hsd_scale_current_d;
    double *hsd_scale_pdhg_d;
    double *hsd_scale_reflected_d;
    double *hsd_dual_constraint_initial_d;
    double *hsd_dual_constraint_current_d;
    double *hsd_dual_constraint_pdhg_d;
    double *hsd_dual_constraint_reflected_d;
    double *hsd_dual_bound_initial_d;
    double *hsd_dual_bound_current_d;
    double *hsd_dual_bound_pdhg_d;
    double *hsd_dual_bound_reflected_d;

    double *power_iteration_vector_d;
    double *power_iteration_normalized_vector_d;
    double *power_iteration_product_d;
    double *power_iteration_vector_norm_scalar_d;
    double *power_iter_sigma_squared_mp;
    double *power_iter_residual_norm_mp;

    double *primal_residual_d;
    double *dual_residual_d;
    double *homogeneous_dual_residual_d;
    double *constraint_recession_residual_d;
    double *variable_recession_residual_d;

    cone_projection_state_t constraint_cone_state;
    cone_projection_state_t bound_cone_state;

    double step_size_primal;
    double step_size_dual;
    double step_size_homogeneous;
    double step_size_base;
    double primal_weight;
    double primal_weight_error_sum;
    double primal_weight_last_error;
    double best_primal_weight;
    double best_primal_dual_residual_gap;

    double *fixed_point_error_mp;
    double  initial_fixed_point_error;
    double  last_trial_fixed_point_error;

    double *pdhg_reduce_d;
    double *pdhg_primal_norm_d;
    double *pdhg_dual_norm_d;
    double *pdhg_slack_norm_d;
    double *pdhg_primal_cross_d;
    double *pdhg_dual_cross_d;
    double *pdhg_slack_cross_d;

    double *farkas_reduce_d;
    double *farkas_dual_solution_norm_d;
    double *farkas_dual_slack_norm_d;
    double *farkas_homogeneous_dual_norm_d;
    double *farkas_constraint_recession_d;
    double *farkas_variable_recession_d;
    double *farkas_support_constraint_d;
    double *farkas_support_variable_d;

    pdhg_metrics_t *metrics_d;
    pdhg_metrics_t *metrics_h;

    int total_count;
    int inner_count;
    int restart_count;
    clock_t start_time;
    double  cumulative_time_sec;
    termination_reason_t termination_reason;
} pdhg_solver_state_t;
