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

#include "setup.h"
#include "active_set_boost.h"
#include "cone_projector.h"
#include "cuda_macros.h"
#include "cublas_checks.h"
#include "cusparse_checks.h"
#include "curand_checks.h"
#include "cuda_utils.h"
#include "hpdlp.h"
#include "display.h"
#include "matrix_formats.h"
#include "preconditioner.h"
#include "spmv_backend.h"
#include "termination_checker.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static constexpr int COOP_NEWTON_THRESHOLD = 4 * THREADS_PER_BLOCK;

__global__ void build_row_ind_kernel(
    const int *__restrict__ row_ptr,
    int num_rows,
    int *__restrict__ row_ind);
__global__ void init_hsd_scalars_kernel(double *hsd_scalars_d);
static void create_csr_descriptor(
    cusparseSpMatDescr_t *descriptor,
    int num_rows,
    int num_cols,
    const cu_sparse_matrix_csr_t *matrix);
static void create_dense_vector_descriptor(
    cusparseDnVecDescr_t *descriptor,
    int length,
    double *values);
static void setup_sparse_resources(pdhg_solver_state_t *state);
static void upload_problem_data(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem);
static void allocate_solver_work_vectors(pdhg_solver_state_t *state);
static void initialize_homogeneous_scalars(pdhg_solver_state_t *state);
static void allocate_host_staging_buffers(pdhg_solver_state_t *state);
static void allocate_device_scalars(pdhg_solver_state_t *state);
static void apply_warm_starts(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem);
static bool bound_is_zero_or_infinite(double bound);
static void initialize_newton_structure(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem);
static void initialize_newton_workspace(
    box_newton_state_t *newton_state,
    int cooperative_block_limit);
static void initialize_newton_workspaces(pdhg_solver_state_t *state);
static void release_preprocessing_resources(pdhg_solver_state_t *state);
static void release_setup_only_matrix_data(pdhg_solver_state_t *state);
static void initialize_iterates_from_starts(pdhg_solver_state_t *state);
static void initialize_convergence_tracking(pdhg_solver_state_t *state);
static void free_sparse_resources(pdhg_solver_state_t *state);
static void free_device_sparse_matrix(cu_sparse_matrix_csr_t *matrix);
static void free_solver_device_buffers(pdhg_solver_state_t *state);

bool check_params_validity(const pdhg_parameters_t *params)
{
    char error[256];
    if (hpdlp_validate_parameters(params, error, sizeof(error)) == 0)
        return true;
    report_error("%s", error);
    return false;
}

void initialize_solver_state(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem,
    const lp_problem_t *original_problem,
    const pdhg_parameters_t *params)
{
    int n_vars = working_problem->num_variables;
    int n_cons = working_problem->num_constraints;
    int nnz = working_problem->constraint_matrix_num_nonzeros;

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->var_bytes = (size_t)n_vars * sizeof(double);
    state->con_bytes = (size_t)n_cons * sizeof(double);
    state->objective_constant = working_problem->objective_constant;
    state->objective_sign = (original_problem->objective_sense == OBJECTIVE_SENSE_MAXIMIZE) ? -1.0 : 1.0;

    state->constraint_matrix = (cu_sparse_matrix_csr_t *)safe_calloc(1, sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix_t = (cu_sparse_matrix_csr_t *)safe_calloc(1, sizeof(cu_sparse_matrix_csr_t));

    state->constraint_matrix->num_nonzeros = nnz;

    state->constraint_matrix_t->num_nonzeros = nnz;

    state->control.termination_reason = TERMINATION_REASON_UNSPECIFIED;

    state->num_blocks_primal = (state->num_variables + 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_dual = (state->num_constraints + 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_primal_dual =
        (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_nnz = (state->constraint_matrix->num_nonzeros + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (state->num_blocks_nnz < 1)
        state->num_blocks_nnz = 1;

    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle_A));
    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle_At));
    CUDA_CHECK(cudaStreamCreate(&state->stream_A));
    CUDA_CHECK(cudaStreamCreate(&state->stream_At));
    CUDA_CHECK(cudaEventCreateWithFlags(&state->event_A, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&state->event_At, cudaEventDisableTiming));
    CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle_A, state->stream_A));
    CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle_At, state->stream_At));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle_A));
    CUBLAS_CHECK(cublasSetStream(state->blas_handle_A, state->stream_A));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle_A, CUBLAS_POINTER_MODE_DEVICE));

    CURAND_CHECK(curandCreateGenerator(&state->curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(state->curand_gen, 42));
    CURAND_CHECK(curandSetStream(state->curand_gen, state->stream_A));

    allocate_cub_reduce_workspace(state);
    upload_problem_data(state, working_problem);
    allocate_solver_work_vectors(state);
    allocate_host_staging_buffers(state);
    initialize_homogeneous_scalars(state);
    initialize_active_set(state, params);
    allocate_device_scalars(state);
    initialize_newton_structure(state, working_problem);
    initialize_newton_workspaces(state);
    apply_warm_starts(state, working_problem);

    init_hsd_scalars_kernel<<<1, 1, 0, state->stream_A>>>(state->hsd_scalars_d);

    setup_sparse_resources(state);

    allocate_zeroed_device_buffer(
        (void **)&state->constraint_rescaling_d,
        state->con_bytes);
    allocate_zeroed_device_buffer(
        (void **)&state->variable_rescaling_d,
        state->var_bytes);

    compute_lp_problem_norms(
        original_problem, params->optimality_norm,
        params->scaling_params.infinite_bound,
        &state->original_objective_norm,
        &state->original_bound_norm);

    rescale_problem(params, state);

    release_preprocessing_resources(state);
    release_setup_only_matrix_data(state);
    initialize_iterates_from_starts(state);
    initialize_convergence_tracking(state);

    print_iteration_header(params);
}

static void create_csr_descriptor(
    cusparseSpMatDescr_t *descriptor,
    int num_rows,
    int num_cols,
    const cu_sparse_matrix_csr_t *matrix)
{
    CUSPARSE_CHECK(cusparseCreateCsr(
        descriptor,
        num_rows,
        num_cols,
        matrix->num_nonzeros,
        matrix->row_ptr,
        matrix->col_ind,
        matrix->val,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));
    CUDA_CHECK(cudaGetLastError());
}

static void create_dense_vector_descriptor(
    cusparseDnVecDescr_t *descriptor,
    int length,
    double *values)
{
    CUSPARSE_CHECK(cusparseCreateDnVec(descriptor, length, values, CUDA_R_64F));
}

static void setup_sparse_resources(pdhg_solver_state_t *state)
{
    create_csr_descriptor(
        &state->matA,
        state->num_constraints,
        state->num_variables,
        state->constraint_matrix);
    create_csr_descriptor(
        &state->matAt,
        state->num_variables,
        state->num_constraints,
        state->constraint_matrix_t);

    create_dense_vector_descriptor(
        &state->spmv_current_primal_solution,
        state->num_variables,
        state->current_primal_solution_d);
    create_dense_vector_descriptor(
        &state->spmv_reflected_dual_solution,
        state->num_constraints,
        state->reflected_dual_solution_d);
    create_dense_vector_descriptor(
        &state->spmv_pdhg_primal_solution,
        state->num_variables,
        state->pdhg_primal_solution_d);
    create_dense_vector_descriptor(
        &state->spmv_pdhg_dual_solution,
        state->num_constraints,
        state->pdhg_dual_solution_d);
    create_dense_vector_descriptor(
        &state->spmv_delta_primal_solution,
        state->num_variables,
        state->delta_primal_solution_d);
    create_dense_vector_descriptor(
        &state->spmv_delta_dual_solution,
        state->num_constraints,
        state->delta_dual_solution_d);
    create_dense_vector_descriptor(
        &state->spmv_primal_product,
        state->num_constraints,
        state->primal_product_d);
    create_dense_vector_descriptor(
        &state->spmv_dual_product,
        state->num_variables,
        state->dual_product_d);
}

void initialize_spmv_plans(pdhg_solver_state_t *state)
{
    create_spmv_plan(
        state->sparse_handle_A,
        state->matA,
        state->spmv_current_primal_solution,
        state->spmv_primal_product,
        &state->constraint_matrix_spmv_plan);
    create_spmv_plan(
        state->sparse_handle_At,
        state->matAt,
        state->spmv_reflected_dual_solution,
        state->spmv_dual_product,
        &state->constraint_matrix_transpose_spmv_plan);
}

static void upload_problem_data(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem)
{
    const size_t var_bytes = state->var_bytes;
    const size_t con_bytes = state->con_bytes;
    const int n_vars = state->num_variables;
    const int n_cons = state->num_constraints;
    const int nnz = state->constraint_matrix->num_nonzeros;

    allocate_and_copy_to_device((void **)&state->constraint_matrix->row_ptr,
                                working_problem->constraint_matrix_row_pointers,
                                (size_t)(n_cons + 1) * sizeof(int));
    allocate_and_copy_to_device((void **)&state->constraint_matrix->col_ind,
                                working_problem->constraint_matrix_col_indices,
                                (size_t)nnz * sizeof(int));
    allocate_and_copy_to_device((void **)&state->constraint_matrix->val,
                                working_problem->constraint_matrix_values,
                                (size_t)nnz * sizeof(double));

    const size_t row_index_count =
        (nnz > 0) ? (size_t)nnz : 1;
    CUDA_CHECK(cudaMalloc(
        &state->constraint_matrix->row_ind,
        row_index_count * sizeof(int)));
    build_row_ind_kernel<<<
        state->num_blocks_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->constraint_matrix->row_ptr, n_cons, state->constraint_matrix->row_ind);
    CUDA_CHECK(cudaGetLastError());

    int *transpose_row_ptr_h = NULL;
    int *transpose_col_ind_h = NULL;
    int *transpose_map_h = NULL;
    double *transpose_val_h = NULL;
    transpose_csr_and_build_map(working_problem,
                                &transpose_row_ptr_h,
                                &transpose_col_ind_h,
                                &transpose_val_h,
                                &transpose_map_h);

    allocate_and_copy_to_device((void **)&state->constraint_matrix_t->row_ptr,
                                transpose_row_ptr_h,
                                (size_t)(n_vars + 1) * sizeof(int));
    allocate_and_copy_to_device((void **)&state->constraint_matrix_t->col_ind,
                                transpose_col_ind_h,
                                (size_t)nnz * sizeof(int));
    allocate_and_copy_to_device((void **)&state->constraint_matrix_t->val,
                                transpose_val_h,
                                (size_t)nnz * sizeof(double));

    state->constraint_matrix_t->row_ind = NULL;

    allocate_and_copy_to_device((void **)&state->constraint_matrix->transpose_map,
                                transpose_map_h,
                                (size_t)nnz * sizeof(int));
    state->constraint_matrix_t->transpose_map = NULL;

    free(transpose_row_ptr_h);
    free(transpose_col_ind_h);
    free(transpose_val_h);
    free(transpose_map_h);

    allocate_and_copy_to_device((void **)&state->variable_lower_bound_d,
                                working_problem->variable_lower_bound, var_bytes);
    allocate_and_copy_to_device((void **)&state->variable_upper_bound_d,
                                working_problem->variable_upper_bound, var_bytes);
    allocate_and_copy_to_device((void **)&state->objective_vector_d,
                                working_problem->objective_vector, var_bytes);
    allocate_and_copy_to_device((void **)&state->constraint_lower_bound_d,
                                working_problem->constraint_lower_bound, con_bytes);
    allocate_and_copy_to_device((void **)&state->constraint_upper_bound_d,
                                working_problem->constraint_upper_bound, con_bytes);
}

static void allocate_solver_work_vectors(pdhg_solver_state_t *state)
{
    const size_t var_bytes = state->var_bytes;
    const size_t con_bytes = state->con_bytes;

    allocate_zeroed_device_buffer((void **)&state->initial_primal_solution_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->current_primal_solution_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->pdhg_primal_solution_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->reflected_primal_solution_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->delta_primal_solution_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->primal_product_d, con_bytes);
    allocate_zeroed_device_buffer(
        (void **)&state->variable_reduction_entries_d,
        var_bytes);

    allocate_zeroed_device_buffer((void **)&state->initial_dual_solution_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->current_dual_solution_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->pdhg_dual_solution_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->reflected_dual_solution_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->delta_dual_solution_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->dual_product_d, var_bytes);
    allocate_zeroed_device_buffer(
        (void **)&state->constraint_reduction_entries_d,
        con_bytes);

    allocate_zeroed_device_buffer((void **)&state->pdhg_reduced_cost_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->bound_multiplier_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->pdhg_primal_projection_multiplier_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->primal_infeasibility_reduced_cost_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->reduced_cost_ray_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->dual_ray_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->primal_ray_d, var_bytes);

    allocate_zeroed_device_buffer((void **)&state->primal_residual_d, con_bytes + var_bytes);
    allocate_zeroed_device_buffer((void **)&state->dual_residual_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->bound_multiplier_dual_residual_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->variable_recession_residual_d, var_bytes);
}

static void initialize_homogeneous_scalars(pdhg_solver_state_t *state)
{
    state->metrics_h->current_primal_scale = 1.0;
    state->metrics_h->pdhg_primal_scale = 1.0;
    state->metrics_h->pdhg_dual_scale = 1.0;
}

static void allocate_host_staging_buffers(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMallocHost(&state->metrics_h, sizeof(pdhg_metrics_t)));
    memset(state->metrics_h, 0, sizeof(pdhg_metrics_t));
    CUDA_CHECK(cudaHostAlloc(
        &state->bound_norm_workspace_mp,
        sizeof(double),
        cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(
        &state->objective_norm_workspace_mp,
        sizeof(double),
        cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&state->control.fixed_point_error_mp, sizeof(double), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(
        &state->restart_reductions_mp,
        sizeof(restart_reductions_t),
        cudaHostAllocMapped));
}

static void allocate_device_scalars(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMalloc(&state->hsd_scalars_d, HSD_SCALARS_COUNT * sizeof(double)));
    state->hsd_primal_scale_initial_d   = state->hsd_scalars_d + HSD_PRIMAL_SCALE_INITIAL;
    state->hsd_primal_scale_current_d   = state->hsd_scalars_d + HSD_PRIMAL_SCALE_CURRENT;
    state->hsd_primal_scale_pdhg_d      = state->hsd_scalars_d + HSD_PRIMAL_SCALE_PDHG;
    state->hsd_primal_scale_reflected_d = state->hsd_scalars_d + HSD_PRIMAL_SCALE_REFLECTED;
    state->hsd_dual_scale_initial_d     = state->hsd_scalars_d + HSD_DUAL_SCALE_INITIAL;
    state->hsd_dual_scale_current_d     = state->hsd_scalars_d + HSD_DUAL_SCALE_CURRENT;
    state->hsd_dual_scale_pdhg_d        = state->hsd_scalars_d + HSD_DUAL_SCALE_PDHG;
    state->hsd_dual_scale_reflected_d   = state->hsd_scalars_d + HSD_DUAL_SCALE_REFLECTED;

    allocate_zeroed_device_buffer(
        (void **)&state->fixed_point_reductions_d,
        sizeof(fixed_point_reductions_t));

    allocate_zeroed_device_buffer(
        (void **)&state->termination_reductions_d,
        sizeof(termination_reductions_t));

    CUDA_CHECK(cudaMalloc(&state->metrics_d, sizeof(pdhg_metrics_t)));
    CUDA_CHECK(cudaMalloc(&state->hsd_inverse_scale_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->constraint_bound_rescaling_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->objective_vector_rescaling_d, sizeof(double)));
    allocate_zeroed_device_buffer((void **)&state->control.inner_count_d, sizeof(int));
    allocate_zeroed_device_buffer((void **)&state->control.step_size_primal_d, sizeof(double));
    allocate_zeroed_device_buffer((void **)&state->control.step_size_dual_d, sizeof(double));
    allocate_zeroed_device_buffer((void **)&state->control.step_size_homogeneous_d, sizeof(double));

    state->primal_box_cone_state.scale_history_d = state->hsd_scalars_d + HSD_PRIMAL_CONE_SCALE_HISTORY;
    state->primal_box_cone_state.lower_bound_d   = state->variable_lower_bound_d;
    state->primal_box_cone_state.upper_bound_d   = state->variable_upper_bound_d;
    state->primal_box_cone_state.num_elements    = state->num_variables;

    state->dual_box_cone_state.scale_history_d = state->hsd_scalars_d + HSD_DUAL_CONE_SCALE_HISTORY;
    state->dual_box_cone_state.lower_bound_d   = state->constraint_lower_bound_d;
    state->dual_box_cone_state.upper_bound_d   = state->constraint_upper_bound_d;
    state->dual_box_cone_state.num_elements    = state->num_constraints;
}

static void apply_warm_starts(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem)
{
    if (working_problem->primal_start)
    {
        CUDA_CHECK(cudaMemcpy(state->initial_primal_solution_d,
                              working_problem->primal_start,
                              state->var_bytes,
                              cudaMemcpyHostToDevice));
    }
    if (working_problem->dual_start)
    {
        CUDA_CHECK(cudaMemcpy(state->initial_dual_solution_d,
                              working_problem->dual_start,
                              state->con_bytes,
                              cudaMemcpyHostToDevice));
    }
}

static bool bound_is_zero_or_infinite(double bound)
{
    return bound == 0.0 || isinf(bound);
}

static void initialize_newton_structure(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem)
{
    state->dual_box_cone_state.newton_terms_are_zero = true;
    for (int i = 0; i < state->num_constraints; ++i)
    {
        if (!bound_is_zero_or_infinite(
                working_problem->constraint_lower_bound[i]) ||
            !bound_is_zero_or_infinite(
                working_problem->constraint_upper_bound[i]))
        {
            state->dual_box_cone_state.newton_terms_are_zero = false;
            break;
        }
    }

    state->primal_box_cone_state.newton_terms_are_zero = true;
    for (int i = 0; i < state->num_variables; ++i)
    {
        if (working_problem->objective_vector[i] != 0.0 ||
            !bound_is_zero_or_infinite(
                working_problem->variable_lower_bound[i]) ||
            !bound_is_zero_or_infinite(
                working_problem->variable_upper_bound[i]))
        {
            state->primal_box_cone_state.newton_terms_are_zero = false;
            break;
        }
    }
}

static void initialize_newton_workspace(
    box_newton_state_t *newton_state,
    int cooperative_block_limit)
{
    const int n = newton_state->num_elements;

    if (n <= COOP_NEWTON_THRESHOLD)
    {
        newton_state->num_blocks_newton = 1;
        newton_state->newton_partials_d = NULL;
        return;
    }

    int n_blocks_newton = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (n_blocks_newton > cooperative_block_limit)
    {
        n_blocks_newton = cooperative_block_limit;
    }
    if (n_blocks_newton < 1)
    {
        n_blocks_newton = 1;
    }
    newton_state->num_blocks_newton = n_blocks_newton;
    allocate_zeroed_device_buffer(
        (void **)&newton_state->newton_partials_d,
        2 * (size_t)n_blocks_newton * sizeof(newton_terms_t));
}

static void initialize_newton_workspaces(pdhg_solver_state_t *state)
{
    initialize_newton_workspace(
        &state->dual_box_cone_state,
        compute_max_cooperative_dual_box_cone_blocks());
    initialize_newton_workspace(
        &state->primal_box_cone_state,
        compute_max_cooperative_primal_box_cone_blocks());
}

static void release_preprocessing_resources(
    pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaFreeHost(state->bound_norm_workspace_mp));
    state->bound_norm_workspace_mp = NULL;
    CUDA_CHECK(cudaFreeHost(state->objective_norm_workspace_mp));
    state->objective_norm_workspace_mp = NULL;
}

static void release_setup_only_matrix_data(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaFree(state->constraint_matrix->row_ind));
    state->constraint_matrix->row_ind = NULL;
    CUDA_CHECK(cudaFree(state->constraint_matrix->transpose_map));
    state->constraint_matrix->transpose_map = NULL;
}

static void initialize_iterates_from_starts(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution_d,
                          state->initial_primal_solution_d,
                          state->var_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution_d,
                          state->initial_dual_solution_d,
                          state->con_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution_d,
                          state->initial_primal_solution_d,
                          state->var_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution_d,
                          state->initial_dual_solution_d,
                          state->con_bytes,
                          cudaMemcpyDeviceToDevice));
}

static void initialize_convergence_tracking(pdhg_solver_state_t *state)
{
    state->control.initial_fixed_point_error = INFINITY;
    state->control.last_trial_fixed_point_error = INFINITY;
}

void pdhg_solver_state_free(pdhg_solver_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

    if (state->stream_A)
        CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    if (state->stream_At)
        CUDA_CHECK(cudaStreamSynchronize(state->stream_At));

    if (state->iteration_graph_exec)
    {
        CUDA_CHECK(cudaGraphExecDestroy(state->iteration_graph_exec));
        state->iteration_graph_exec = NULL;
    }

    active_set_free(&state->active_set);
    free_sparse_resources(state);
    free_solver_device_buffers(state);
    free_device_sparse_matrix(state->constraint_matrix);
    free_device_sparse_matrix(state->constraint_matrix_t);

    if (state->curand_gen)
        CURAND_CHECK(curandDestroyGenerator(state->curand_gen));
    if (state->blas_handle_A)
        CUBLAS_CHECK(cublasDestroy(state->blas_handle_A));
    if (state->sparse_handle_A)
        CUSPARSE_CHECK(cusparseDestroy(state->sparse_handle_A));
    if (state->sparse_handle_At)
        CUSPARSE_CHECK(cusparseDestroy(state->sparse_handle_At));
    if (state->event_A)
        CUDA_CHECK(cudaEventDestroy(state->event_A));
    if (state->event_At)
        CUDA_CHECK(cudaEventDestroy(state->event_At));
    if (state->stream_A)
        CUDA_CHECK(cudaStreamDestroy(state->stream_A));
    if (state->stream_At)
        CUDA_CHECK(cudaStreamDestroy(state->stream_At));

    free(state);
}

static void free_sparse_resources(pdhg_solver_state_t *state)
{
    destroy_spmv_plan(&state->constraint_matrix_spmv_plan);
    destroy_spmv_plan(&state->constraint_matrix_transpose_spmv_plan);

    if (state->matA)
        CUSPARSE_CHECK(cusparseDestroySpMat(state->matA));
    if (state->matAt)
        CUSPARSE_CHECK(cusparseDestroySpMat(state->matAt));
    if (state->spmv_current_primal_solution)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_current_primal_solution));
    if (state->spmv_reflected_dual_solution)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_reflected_dual_solution));
    if (state->spmv_pdhg_primal_solution)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_pdhg_primal_solution));
    if (state->spmv_pdhg_dual_solution)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_pdhg_dual_solution));
    if (state->spmv_delta_primal_solution)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_delta_primal_solution));
    if (state->spmv_delta_dual_solution)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_delta_dual_solution));
    if (state->spmv_primal_product)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_primal_product));
    if (state->spmv_dual_product)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_dual_product));
}

static void free_device_sparse_matrix(cu_sparse_matrix_csr_t *matrix)
{
    if (!matrix)
        return;

    if (matrix->row_ptr)
        CUDA_CHECK(cudaFree(matrix->row_ptr));
    if (matrix->col_ind)
        CUDA_CHECK(cudaFree(matrix->col_ind));
    if (matrix->row_ind)
        CUDA_CHECK(cudaFree(matrix->row_ind));
    if (matrix->val)
        CUDA_CHECK(cudaFree(matrix->val));
    if (matrix->transpose_map)
        CUDA_CHECK(cudaFree(matrix->transpose_map));

    free(matrix);
}

static void free_solver_device_buffers(pdhg_solver_state_t *state)
{
#define FREE_DEVICE_BUFFER(ptr) \
    do                          \
    {                           \
        if ((ptr))              \
            CUDA_CHECK(cudaFree(ptr)); \
    } while (0)

    FREE_DEVICE_BUFFER(state->variable_lower_bound_d);
    FREE_DEVICE_BUFFER(state->variable_upper_bound_d);
    FREE_DEVICE_BUFFER(state->constraint_lower_bound_d);
    FREE_DEVICE_BUFFER(state->constraint_upper_bound_d);
    FREE_DEVICE_BUFFER(state->objective_vector_d);
    FREE_DEVICE_BUFFER(state->constraint_rescaling_d);
    FREE_DEVICE_BUFFER(state->variable_rescaling_d);
    FREE_DEVICE_BUFFER(state->constraint_bound_rescaling_d);
    FREE_DEVICE_BUFFER(state->objective_vector_rescaling_d);

    FREE_DEVICE_BUFFER(state->initial_primal_solution_d);
    FREE_DEVICE_BUFFER(state->current_primal_solution_d);
    FREE_DEVICE_BUFFER(state->pdhg_primal_solution_d);
    FREE_DEVICE_BUFFER(state->reflected_primal_solution_d);
    FREE_DEVICE_BUFFER(state->delta_primal_solution_d);
    FREE_DEVICE_BUFFER(state->primal_product_d);
    FREE_DEVICE_BUFFER(state->variable_reduction_entries_d);

    FREE_DEVICE_BUFFER(state->initial_dual_solution_d);
    FREE_DEVICE_BUFFER(state->current_dual_solution_d);
    FREE_DEVICE_BUFFER(state->pdhg_dual_solution_d);
    FREE_DEVICE_BUFFER(state->reflected_dual_solution_d);
    FREE_DEVICE_BUFFER(state->delta_dual_solution_d);
    FREE_DEVICE_BUFFER(state->dual_product_d);
    FREE_DEVICE_BUFFER(state->constraint_reduction_entries_d);

    FREE_DEVICE_BUFFER(state->pdhg_reduced_cost_d);
    FREE_DEVICE_BUFFER(state->bound_multiplier_d);
    FREE_DEVICE_BUFFER(state->pdhg_primal_projection_multiplier_d);
    FREE_DEVICE_BUFFER(state->primal_infeasibility_reduced_cost_d);
    FREE_DEVICE_BUFFER(state->reduced_cost_ray_d);
    FREE_DEVICE_BUFFER(state->dual_ray_d);
    FREE_DEVICE_BUFFER(state->primal_ray_d);

    FREE_DEVICE_BUFFER(state->hsd_scalars_d);
    FREE_DEVICE_BUFFER(state->hsd_inverse_scale_d);
    FREE_DEVICE_BUFFER(state->primal_residual_d);
    FREE_DEVICE_BUFFER(state->dual_residual_d);
    FREE_DEVICE_BUFFER(state->bound_multiplier_dual_residual_d);
    FREE_DEVICE_BUFFER(state->variable_recession_residual_d);

    FREE_DEVICE_BUFFER(state->dual_box_cone_state.newton_partials_d);
    FREE_DEVICE_BUFFER(state->primal_box_cone_state.newton_partials_d);

    FREE_DEVICE_BUFFER(state->cub_reduce_workspace_d);
    FREE_DEVICE_BUFFER(state->control.inner_count_d);
    FREE_DEVICE_BUFFER(state->control.step_size_primal_d);
    FREE_DEVICE_BUFFER(state->control.step_size_dual_d);
    FREE_DEVICE_BUFFER(state->control.step_size_homogeneous_d);
    FREE_DEVICE_BUFFER(state->fixed_point_reductions_d);
    FREE_DEVICE_BUFFER(state->termination_reductions_d);
    FREE_DEVICE_BUFFER(state->metrics_d);

#undef FREE_DEVICE_BUFFER

#define FREE_HOST_BUFFER(ptr) \
    do                        \
    {                         \
        if ((ptr))            \
            CUDA_CHECK(cudaFreeHost(ptr)); \
    } while (0)

    FREE_HOST_BUFFER(state->metrics_h);
    FREE_HOST_BUFFER(state->bound_norm_workspace_mp);
    FREE_HOST_BUFFER(state->objective_norm_workspace_mp);
    FREE_HOST_BUFFER(state->control.fixed_point_error_mp);
    FREE_HOST_BUFFER(state->restart_reductions_mp);

#undef FREE_HOST_BUFFER
}

__global__ void build_row_ind_kernel(
    const int *__restrict__ row_ptr,
    int num_rows,
    int *__restrict__ row_ind)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows)
        return;

    for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
        row_ind[k] = row;
}

__global__ void init_hsd_scalars_kernel(double *hsd_scalars_d)
{
    for (int i = 0; i < HSD_SCALARS_COUNT; ++i)
        hsd_scalars_d[i] = 1.0;
}
