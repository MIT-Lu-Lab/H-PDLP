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
#include "cuhsdlp_types.h"
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

typedef struct pdhg_metrics_t pdhg_metrics_t;

enum PdhgReduction {
    PDHG_PRIMAL_NORM         = 0,
    PDHG_DUAL_NORM           = 1,
    PDHG_OBJECTIVE_DOT_X     = 2,
    PDHG_COUNT               = 3,
};

typedef struct {
    double dual_solution_norm;
    double primal_infeasibility_stationarity_residual_norm;
    double constraint_recession_residual_norm;
    double variable_recession_residual_norm;
    double original_dual_ray_norm;
    double original_reduced_cost_ray_norm;
    double original_primal_ray_norm;
    double constraint_support;
    double variable_support;
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
    int total_count;
    clock_t start_time;
    double cumulative_time_sec;
    termination_reason_t termination_reason;
    double certificate_margin;
    double certificate_residual;
    bool certificate_from_homogeneous;

    double *step_size_primal_d;
    double *step_size_dual_d;
    double *step_size_homogeneous_d;

    double primal_weight;

    int primal_basis_changes;
    int dual_basis_changes;
    int phase_changes;
    int last_phase_change;
    int last_phase_change_primal;
    int last_phase_change_dual;
} solver_control_t;

enum HsdScalarSlot {
    HSD_PRIMAL_SCALE              = 0,
    HSD_DUAL_SCALE                = 1,
    HSD_DUAL_SCALE_EXTRAPOLATED   = 2,
    HSD_SCALARS_COUNT             = 3,
};

typedef struct {
    int num_nonzeros;
    int *row_ptr;
    int *col_ind;
    int *row_ind;
    double *val;
    int *transpose_map;
} cu_sparse_matrix_csr_t;

typedef struct {
    int num_variables;
    int num_constraints;
    size_t var_bytes;
    size_t con_bytes;
    int num_blocks_primal;
    int num_blocks_dual;
    int num_blocks_primal_dual;
    int num_blocks_rescale;
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
    cublasHandle_t    blas_handle_At;
    curandGenerator_t curand_gen;
    cudaStream_t      stream_A;
    cudaStream_t      stream_At;
    cudaEvent_t       event_A;
    cudaEvent_t       event_At;
    cudaGraphExec_t   graphExec;
    bool              graph_created;

    cusparseSpMatDescr_t matA;
    cusparseSpMatDescr_t matAt;
    cusparseDnVecDescr_t spmv_current_primal_solution;
    cusparseDnVecDescr_t spmv_current_dual_solution;
    cusparseDnVecDescr_t spmv_extrapolated_dual_solution;
    cusparseDnVecDescr_t spmv_delta_primal_solution;
    cusparseDnVecDescr_t spmv_delta_dual_solution;
    cusparseDnVecDescr_t spmv_primal_product;
    cusparseDnVecDescr_t spmv_dual_product;
    cusparseDnVecDescr_t spmv_power_iteration_normalized_vector;
    void   *matA_spmv_workspace_d;
    void   *matAt_spmv_workspace_d;
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
    double *delta_primal_solution_d;
    double *primal_product_d;
    double *variable_reduction_entries_d;

    double *initial_dual_solution_d;
    double *current_dual_solution_d;
    double *extrapolated_dual_solution_d;
    double *delta_dual_solution_d;
    double *dual_product_d;
    double *constraint_reduction_entries_d;

    double *pdhg_reduced_cost_d;
    double *primal_infeasibility_reduced_cost_d;
    double *reduced_cost_ray_d;
    double *dual_ray_d;
    double *primal_ray_d;

    double *hsd_scalars_d;
    double *hsd_primal_scale_d;
    double *hsd_dual_scale_d;
    double *hsd_dual_scale_extrapolated_d;

    double *power_iteration_normalized_vector_d;
    double *power_iteration_vector_norm_scalar_d;
    double *power_iter_sigma_squared_mp;
    double *power_iter_residual_norm_mp;

    double *primal_residual_d;
    double *dual_residual_d;
    double *primal_infeasibility_stationarity_residual_d;
    double *constraint_recession_residual_d;
    double *variable_recession_residual_d;

    box_newton_state_t dual_box_cone_state;
    box_newton_state_t primal_box_cone_state;
    cone_step_stats_t *step_stats_h;
    cuhsdlp_basis_change_t *basis_trace;
    int basis_trace_capacity;

    solver_control_t control;

    double *pdhg_reduce_d;
    double *pdhg_primal_norm_d;
    double *pdhg_dual_norm_d;
    double *objective_dot_primal_d;

    termination_reductions_t *termination_reductions_d;

    pdhg_metrics_t *metrics_d;
    pdhg_metrics_t *metrics_h;
} pdhg_solver_state_t;
