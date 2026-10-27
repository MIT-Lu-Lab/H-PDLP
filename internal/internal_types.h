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
#include "hpdlp_types.h"
#include "pid_controller.h"
#include "spmv_backend.h"
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse.h>
#include <stdbool.h>
#include <time.h>

typedef struct pdhg_metrics_t pdhg_metrics_t;

typedef struct {
    double primal_delta_norm;
    double dual_delta_norm;
    double operator_cross_term;
    double primal_delta_scale;
    double dual_delta_scale;
} fixed_point_reductions_t;

typedef struct {
    double primal_delta_norm;
    double dual_delta_norm;
} restart_reductions_t;

typedef struct {
    double objective_dot;
    double constraint_support;
    double variable_support;
    double constraint_recession_residual;
    double stationarity_residual;
    double sum_absolute_objective_terms;
    double sum_absolute_constraint_support_terms;
    double sum_absolute_variable_support_terms;
    double scaled_constraint_recession_residual;
    double scaled_stationarity_residual;
} candidate_reductions_t;

typedef struct {
    double margin;
    double residual;
    double relative_margin;
    double scaled_residual;
    double relative_bound_residual;
    double scaled_activity;
    double relative_residual;
} certificate_metrics_t;

typedef struct {
    double primal_residual_norm;
    double dual_residual_norm;
    double bound_multiplier_dual_residual_norm;
    double variable_recession_residual_norm;
    double scaled_variable_recession_residual;
    double scaled_primal_ray_norm;
    double original_dual_ray_norm;
    double original_reduced_cost_ray_norm;
    double original_primal_ray_norm;
    double variable_support;
    double bound_multiplier_variable_support;
    candidate_reductions_t iterate;
    candidate_reductions_t step;
    candidate_reductions_t restart;
} termination_reductions_t;

typedef enum {
    CERTIFICATE_SOURCE_ITERATE,
    CERTIFICATE_SOURCE_STEP,
    CERTIFICATE_SOURCE_RESTART,
    CERTIFICATE_SOURCE_COUNT
} certificate_source_t;

typedef struct {
    double *primal_d;
    double *dual_d;
    cusparseDnVecDescr_t spmv_primal;
    cusparseDnVecDescr_t spmv_dual;
} candidate_vectors_t;

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
    certificate_source_t certificate_source;

    int *inner_count_d;
    double *step_size_primal_d;
    double *step_size_dual_d;
    double *step_size_homogeneous_d;

    double primal_weight;
    pid_state_t pid;
    double fixed_point_error;
    double initial_fixed_point_error;
    double last_trial_fixed_point_error;
} solver_control_t;

enum {
    HSD_PRIMAL_SCALE_INITIAL      = 0,
    HSD_PRIMAL_SCALE_CURRENT      = 1,
    HSD_PRIMAL_SCALE_PDHG         = 2,
    HSD_PRIMAL_SCALE_REFLECTED    = 3,
    HSD_DUAL_SCALE_INITIAL        = 4,
    HSD_DUAL_SCALE_CURRENT        = 5,
    HSD_DUAL_SCALE_PDHG           = 6,
    HSD_DUAL_SCALE_REFLECTED      = 7,
    HSD_PRIMAL_CONE_SCALE_HISTORY = 8,
    HSD_DUAL_CONE_SCALE_HISTORY   = HSD_PRIMAL_CONE_SCALE_HISTORY + BOX_CONE_SCALE_COUNT,
    HSD_SCALARS_COUNT             = HSD_DUAL_CONE_SCALE_HISTORY + BOX_CONE_SCALE_COUNT,
};

typedef struct {
    int num_nonzeros;
    int *row_ptr;
    int *col_ind;
    int *row_ind;
    double *val;
    double *abs_val;
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

    cudaStream_t      stream_A;
    cudaStream_t      stream_At;
    cudaEvent_t       event_A;
    cudaEvent_t       event_At;
    cusparseHandle_t  sparse_handle_A;
    cusparseHandle_t  sparse_handle_At;
    cublasHandle_t    blas_handle_A;
    curandGenerator_t curand_gen;
    cudaGraphExec_t   iteration_graph_exec;

    cusparseSpMatDescr_t matA;
    cusparseSpMatDescr_t matAt;
    cusparseSpMatDescr_t matAbsA;
    cusparseSpMatDescr_t matAbsAt;
    cusparseDnVecDescr_t spmv_primal_current_solution;
    cusparseDnVecDescr_t spmv_dual_reflected_solution;
    cusparseDnVecDescr_t spmv_primal_pdhg_solution;
    cusparseDnVecDescr_t spmv_dual_pdhg_solution;
    cusparseDnVecDescr_t spmv_primal_step_delta;
    cusparseDnVecDescr_t spmv_dual_step_delta;
    cusparseDnVecDescr_t spmv_primal_restart_delta;
    cusparseDnVecDescr_t spmv_dual_restart_delta;
    cusparseDnVecDescr_t spmv_primal_product;
    cusparseDnVecDescr_t spmv_dual_product;
    cusparseDnVecDescr_t spmv_variable_reduction_entries;
    cusparseDnVecDescr_t spmv_constraint_reduction_entries;
    spmv_plan_t constraint_matrix_spmv_plan;
    spmv_plan_t constraint_matrix_transpose_spmv_plan;
    spmv_plan_t absolute_constraint_matrix_spmv_plan;
    spmv_plan_t absolute_constraint_matrix_transpose_spmv_plan;
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

    double *primal_initial_solution_d;
    double *primal_current_solution_d;
    double *primal_pdhg_solution_d;
    double *primal_reflected_solution_d;
    double *primal_step_delta_d;
    double *primal_restart_delta_d;
    double *primal_product_d;
    double *variable_reduction_entries_d;

    double *dual_initial_solution_d;
    double *dual_current_solution_d;
    double *dual_pdhg_solution_d;
    double *dual_reflected_solution_d;
    double *dual_step_delta_d;
    double *dual_restart_delta_d;
    double *dual_product_d;
    double *constraint_reduction_entries_d;

    double *pdhg_reduced_cost_d;
    double *bound_multiplier_d;
    double *primal_pdhg_projection_multiplier_d;
    double *primal_infeasibility_reduced_cost_d;
    double *reduced_cost_iterate_ray_d;
    double *dual_iterate_ray_d;
    double *primal_iterate_ray_d;

    double *hsd_scalars_d;
    double *hsd_inverse_scale_d;
    double *hsd_primal_initial_scale_d;
    double *hsd_primal_current_scale_d;
    double *hsd_primal_pdhg_scale_d;
    double *hsd_primal_reflected_scale_d;
    double *hsd_dual_initial_scale_d;
    double *hsd_dual_current_scale_d;
    double *hsd_dual_pdhg_scale_d;
    double *hsd_dual_reflected_scale_d;
    double *primal_residual_d;
    double *dual_residual_d;
    double *bound_multiplier_dual_residual_d;
    double *variable_recession_residual_d;

    box_newton_state_t dual_box_cone_state;
    box_newton_state_t primal_box_cone_state;

    solver_control_t control;
    active_set_state_t active_set;

    fixed_point_reductions_t *fixed_point_reductions_mp;
    restart_reductions_t *restart_reductions_mp;
    termination_reductions_t *termination_reductions_d;

    pdhg_metrics_t *metrics_d;
    pdhg_metrics_t *metrics_h;
} pdhg_solver_state_t;
