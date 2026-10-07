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

#include "cone_projector.h"
#include "hopdlp_types.h"
#include "pid_controller.h"
#include "spmv_backend.h"
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

typedef struct pdhg_metrics_t pdhg_metrics_t;

typedef struct {
    double delta_primal_norm;
    double delta_dual_norm;
    double operator_cross_term;
} fixed_point_reductions_t;

typedef struct {
    double delta_primal_norm;
    double delta_dual_norm;
} restart_reductions_t;

typedef struct {
    double primal_residual_norm;
    double dual_residual_norm;
    double bound_multiplier_dual_residual_norm;
    double primal_infeasibility_stationarity_residual_norm;
    double constraint_recession_residual_norm;
    double variable_recession_residual_norm;
    double original_dual_ray_norm;
    double original_reduced_cost_ray_norm;
    double original_primal_ray_norm;
    double objective_dot_primal;
    double constraint_support;
    double variable_support;
    double bound_multiplier_variable_support;
    double homogeneous_variable_support;
    double dual_infeasibility_objective;
    double primal_infeasibility_constraint_support;
    double primal_infeasibility_variable_support;
    double dual_infeasibility_residual;
    double primal_infeasibility_residual;
} termination_reductions_t;

typedef struct {
    double step_size_primal;
    double step_size_dual;
    double step_size_homogeneous;
    double step_size_base;
    double step_size;
    int total_count;
    int inner_count;
    int restart_count;
    clock_t start_time;
    double cumulative_time_sec;
    termination_reason_t termination_reason;
    double certificate_margin;
    double certificate_residual;
    bool certificate_from_homogeneous;

    int *inner_count_d;
    double *step_size_primal_d;
    double *step_size_dual_d;
    double *step_size_homogeneous_d;

    double primal_weight;
    pid_state_t pid;
    double *fixed_point_error_mp;
    double initial_fixed_point_error;
    double last_trial_fixed_point_error;
} solver_control_t;

enum HsdScalarSlot {
    HSD_PRIMAL_SCALE_INITIAL      = 0,
    HSD_PRIMAL_SCALE_CURRENT      = 1,
    HSD_PRIMAL_SCALE_PDHG         = 2,
    HSD_PRIMAL_SCALE_REFLECTED    = 3,
    HSD_DUAL_SCALE_INITIAL        = 4,
    HSD_DUAL_SCALE_CURRENT        = 5,
    HSD_DUAL_SCALE_PDHG           = 6,
    HSD_DUAL_SCALE_REFLECTED      = 7,
    HSD_PRIMAL_CONE_SCALE_HISTORY = 8,
    HSD_DUAL_CONE_SCALE_HISTORY   = 11,
    HSD_SCALARS_COUNT             = 14,
};

typedef struct {
    int num_nonzeros;
    int *row_ptr;
    int *col_ind;
    int *row_ind;
    double *val;
    int *transpose_map;
} cu_sparse_matrix_csr_t;

typedef struct sv_estimator_ctx sv_estimator_ctx_t;

typedef struct {
    double *primal_solution_d;
    double *dual_solution_d;
    double *primal_projection_multiplier_d;
    double *hsd_scalars_d;
    double primal_weight;
    pid_state_t pid;
} active_set_anchor_t;

typedef struct {
    int variables_added;
    int variables_removed;
    int constraints_added;
    int constraints_removed;
} active_set_mask_changes_t;

typedef struct {
    double *constraint_dual_projection_input_d;
    int *variable_last_free_iter_d;
    int *constraint_last_binding_iter_d;
    bool *free_variable_mask_d;
    bool *binding_constraint_mask_d;
    active_set_mask_changes_t *mask_changes_d;
    sv_estimator_ctx_t *sv_estimator;
    int free_variable_count;
    int binding_constraint_count;
    int mask_changes_since_estimate;
    bool raise_checked;
    double step_size_ceiling;
    active_set_anchor_t anchor;
    int raise_count;
    int revert_count;
    int power_iteration_count;
} active_set_state_t;

typedef struct {
    int num_variables;
    int num_constraints;
    size_t var_bytes;
    size_t con_bytes;
    int num_blocks_primal;
    int num_blocks_dual;
    int num_blocks_primal_dual;
    int num_blocks_nnz;

    cu_sparse_matrix_csr_t *constraint_matrix;
    cu_sparse_matrix_csr_t *constraint_matrix_t;
    double *variable_lower_bound_d;
    double *variable_upper_bound_d;
    double *constraint_lower_bound_d;
    double *constraint_upper_bound_d;
    double *objective_vector_d;
    double objective_constant;

    cusparseHandle_t  sparse_handle_A;
    cusparseHandle_t  sparse_handle_At;
    cublasHandle_t    blas_handle_A;
    curandGenerator_t curand_gen;
    cudaStream_t      stream_A;
    cudaStream_t      stream_At;
    cudaEvent_t       event_A;
    cudaEvent_t       event_At;
    cudaGraphExec_t   iteration_graph_exec;

    cusparseSpMatDescr_t matA;
    cusparseSpMatDescr_t matAt;
    cusparseDnVecDescr_t spmv_current_primal_solution;
    cusparseDnVecDescr_t spmv_reflected_dual_solution;
    cusparseDnVecDescr_t spmv_pdhg_primal_solution;
    cusparseDnVecDescr_t spmv_pdhg_dual_solution;
    cusparseDnVecDescr_t spmv_delta_primal_solution;
    cusparseDnVecDescr_t spmv_delta_dual_solution;
    cusparseDnVecDescr_t spmv_primal_product;
    cusparseDnVecDescr_t spmv_dual_product;
    spmv_plan_t constraint_matrix_spmv_plan;
    spmv_plan_t constraint_matrix_transpose_spmv_plan;
    void   *cub_reduce_workspace_d;
    size_t  cub_reduce_workspace_bytes;

    double  original_objective_norm;
    double  original_bound_norm;
    double  rescaled_objective_norm;
    double  rescaled_bound_norm;
    double  rescaled_constraint_norm;
    double  constraint_bound_rescaling;
    double  objective_vector_rescaling;
    double  objective_sign;
    double  rescaling_time_sec;
    double *bound_norm_workspace_mp;
    double *objective_norm_workspace_mp;
    double *constraint_rescaling_d;
    double *variable_rescaling_d;
    double *constraint_bound_rescaling_d;
    double *objective_vector_rescaling_d;

    double *initial_primal_solution_d;
    double *current_primal_solution_d;
    double *pdhg_primal_solution_d;
    double *reflected_primal_solution_d;
    double *delta_primal_solution_d;
    double *primal_product_d;
    double *variable_reduction_entries_d;

    double *initial_dual_solution_d;
    double *current_dual_solution_d;
    double *pdhg_dual_solution_d;
    double *reflected_dual_solution_d;
    double *delta_dual_solution_d;
    double *dual_product_d;
    double *constraint_reduction_entries_d;

    double *pdhg_reduced_cost_d;
    double *bound_multiplier_d;
    double *pdhg_primal_projection_multiplier_d;
    double *primal_infeasibility_reduced_cost_d;
    double *reduced_cost_ray_d;
    double *dual_ray_d;
    double *primal_ray_d;

    double *hsd_scalars_d;
    double *hsd_inverse_scale_d;
    double *hsd_primal_scale_initial_d;
    double *hsd_primal_scale_current_d;
    double *hsd_primal_scale_pdhg_d;
    double *hsd_primal_scale_reflected_d;
    double *hsd_dual_scale_initial_d;
    double *hsd_dual_scale_current_d;
    double *hsd_dual_scale_pdhg_d;
    double *hsd_dual_scale_reflected_d;
    double *primal_residual_d;
    double *dual_residual_d;
    double *bound_multiplier_dual_residual_d;
    double *variable_recession_residual_d;

    box_newton_state_t dual_box_cone_state;
    box_newton_state_t primal_box_cone_state;

    solver_control_t control;
    active_set_state_t active_set;

    fixed_point_reductions_t *fixed_point_reductions_d;
    restart_reductions_t *restart_reductions_mp;
    termination_reductions_t *termination_reductions_d;

    pdhg_metrics_t *metrics_d;
    pdhg_metrics_t *metrics_h;
} pdhg_solver_state_t;
