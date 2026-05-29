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
#include "cone_projector.h"
#include "cuhsdlp.h"
#include "display.h"
#include "preconditioner.h"
#include "termination_checker.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static constexpr int COOP_NEWTON_THRESHOLD = 2 * THREADS_PER_BLOCK;

__global__ void build_row_ind_kernel(
    const int *__restrict__ row_ptr,
    int num_rows,
    int *__restrict__ row_ind);
__global__ void init_hsd_scalars_kernel(double *hsd_scalars_d);
static void transpose_csr_and_build_map(
    const lp_problem_t *problem,
    int **transpose_row_ptr,
    int **transpose_col_ind,
    double **transpose_val,
    int **transpose_map);
static void create_csr_descriptor(
    cusparseSpMatDescr_t *descriptor,
    int num_rows,
    int num_cols,
    const cu_sparse_matrix_csr_t *matrix);
static void create_dense_vector_descriptor(
    cusparseDnVecDescr_t *descriptor,
    int length,
    double *values);
static void initialize_spmv_buffer(
    cusparseHandle_t handle,
    cusparseSpMatDescr_t matrix,
    cusparseDnVecDescr_t input,
    cusparseDnVecDescr_t output,
    void **buffer);
static void setup_sparse_resources(pdhg_solver_state_t *state);
static void allocate_and_copy_to_device(
    void **dst,
    const void *src,
    size_t bytes);
static void allocate_zeroed_device_buffer(
    void **dst,
    size_t bytes);
static void allocate_cub_reduce_workspace(pdhg_solver_state_t *state);
static void upload_problem_data(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem);
static void allocate_solver_work_vectors(pdhg_solver_state_t *state);
static void initialize_homogeneous_scalars(pdhg_solver_state_t *state);
static void initialize_power_iteration_buffers(pdhg_solver_state_t *state);
static void allocate_host_staging_buffers(pdhg_solver_state_t *state);
static void allocate_device_scalars(pdhg_solver_state_t *state);
static void apply_warm_starts(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem);
static void initialize_cone_grids(pdhg_solver_state_t *state);
static void release_setup_only_matrix_data(pdhg_solver_state_t *state);
static void initialize_iterates_from_starts(pdhg_solver_state_t *state);
static void initialize_convergence_tracking(pdhg_solver_state_t *state);
static void free_sparse_resources(pdhg_solver_state_t *state);
static void free_device_sparse_matrix(cu_sparse_matrix_csr_t *matrix);
static void free_solver_device_buffers(pdhg_solver_state_t *state);

void check_params_validity(const pdhg_parameters_t *params)
{
    if (params->termination_evaluation_frequency < 3)
    {
        fprintf(stderr,
                "Error: termination_evaluation_frequency must be >= 3 (got %d).\n",
                params->termination_evaluation_frequency);
        exit(EXIT_FAILURE);
    }
}

pdhg_solver_state_t *initialize_solver_state(
    lp_problem_t *working_problem,
    const lp_problem_t *original_problem,
    const pdhg_parameters_t *params)
{
    pdhg_solver_state_t *state = (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    int n_vars = working_problem->num_variables;
    int n_cons = working_problem->num_constraints;
    int nnz = working_problem->constraint_matrix_num_nonzeros;

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->var_bytes = (size_t)n_vars * sizeof(double);
    state->con_bytes = (size_t)n_cons * sizeof(double);
    state->objective_constant = working_problem->objective_constant;

    state->constraint_matrix = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix_t = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));

    state->constraint_matrix->num_rows = n_cons;
    state->constraint_matrix->num_cols = n_vars;
    state->constraint_matrix->num_nonzeros = nnz;

    state->constraint_matrix_t->num_rows = n_vars;
    state->constraint_matrix_t->num_cols = n_cons;
    state->constraint_matrix_t->num_nonzeros = nnz;

    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;

    state->num_blocks_primal = (state->num_variables + 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_dual = (state->num_constraints + 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_primal_dual = state->num_blocks_dual + state->num_blocks_primal;
    state->num_blocks_primal_dual_slack =
        (2 * state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_nnz = (state->constraint_matrix->num_nonzeros + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle_A));
    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle_At));
    CUDA_CHECK(cudaStreamCreate(&state->stream_A));
    CUDA_CHECK(cudaStreamCreate(&state->stream_At));
    CUDA_CHECK(cudaEventCreateWithFlags(&state->event_A, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&state->event_At, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&state->event_s, cudaEventDisableTiming));
    CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle_A, state->stream_A));
    CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle_At, state->stream_At));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle_A));
    CUBLAS_CHECK(cublasSetStream(state->blas_handle_A, state->stream_A));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle_A, CUBLAS_POINTER_MODE_DEVICE));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle_At));
    CUBLAS_CHECK(cublasSetStream(state->blas_handle_At, state->stream_At));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle_At, CUBLAS_POINTER_MODE_DEVICE));

    CURAND_CHECK(curandCreateGenerator(&state->curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(state->curand_gen, 42));
    CURAND_CHECK(curandSetStream(state->curand_gen, state->stream_A));

    allocate_cub_reduce_workspace(state);
    upload_problem_data(state, working_problem);
    allocate_solver_work_vectors(state);
    allocate_host_staging_buffers(state);
    initialize_homogeneous_scalars(state);
    initialize_power_iteration_buffers(state);
    allocate_device_scalars(state);
    initialize_cone_grids(state);
    apply_warm_starts(state, working_problem);

    init_hsd_scalars_kernel<<<1, 1, 0, state->stream_A>>>(state->hsd_scalars_d);

    setup_sparse_resources(state);

    CUDA_CHECK(cudaMalloc(&state->constraint_rescaling_d, state->con_bytes));
    CUDA_CHECK(cudaMalloc(&state->variable_rescaling_d, state->var_bytes));

    compute_lp_problem_norms(
        original_problem, params->optimality_norm,
        &state->original_objective_norm,
        &state->original_bound_norm,
        &state->original_constraint_norm);

    rescale_problem(params, state);

    release_setup_only_matrix_data(state);
    initialize_iterates_from_starts(state);
    initialize_convergence_tracking(state);

    print_iteration_header(params->verbose);

    return state;
}

static void transpose_csr_and_build_map(
    const lp_problem_t *problem,
    int **transpose_row_ptr,
    int **transpose_col_ind,
    double **transpose_val,
    int **transpose_map)
{
    const int num_rows = problem->num_constraints;
    const int num_cols = problem->num_variables;
    const int nnz = problem->constraint_matrix_num_nonzeros;
    const int *row_ptr = problem->constraint_matrix_row_pointers;
    const int *col_ind = problem->constraint_matrix_col_indices;
    const double *val = problem->constraint_matrix_values;
    const size_t alloc_nnz = (nnz > 0) ? (size_t)nnz : 1;
    const size_t alloc_cols = (num_cols > 0) ? (size_t)num_cols : 1;

    *transpose_row_ptr = (int *)safe_calloc((size_t)(num_cols + 1), sizeof(int));
    *transpose_col_ind = (int *)safe_malloc(alloc_nnz * sizeof(int));
    *transpose_val = (double *)safe_malloc(alloc_nnz * sizeof(double));
    *transpose_map = (int *)safe_malloc(alloc_nnz * sizeof(int));

    for (int k = 0; k < nnz; ++k)
        (*transpose_row_ptr)[col_ind[k] + 1]++;

    for (int j = 0; j < num_cols; ++j)
        (*transpose_row_ptr)[j + 1] += (*transpose_row_ptr)[j];

    int *next = (int *)safe_malloc(alloc_cols * sizeof(int));
    if (num_cols > 0)
        memcpy(next, *transpose_row_ptr, (size_t)num_cols * sizeof(int));

    for (int i = 0; i < num_rows; ++i)
    {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        {
            int j = col_ind[k];
            int pos = next[j]++;
            (*transpose_col_ind)[pos] = i;
            (*transpose_val)[pos] = val[k];
            (*transpose_map)[k] = pos;
        }
    }

    free(next);
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

static void initialize_spmv_buffer(
    cusparseHandle_t handle,
    cusparseSpMatDescr_t matrix,
    cusparseDnVecDescr_t input,
    cusparseDnVecDescr_t output,
    void **buffer)
{
    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, matrix, input,
        &HOST_ZERO, output, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &buffer_size));

    CUDA_CHECK(cudaMalloc(buffer, buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, matrix, input,
        &HOST_ZERO, output, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, *buffer));
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
    create_dense_vector_descriptor(
        &state->spmv_power_iteration_vector,
        state->num_constraints,
        state->power_iteration_vector_d);
    create_dense_vector_descriptor(
        &state->spmv_power_iteration_normalized_vector,
        state->num_constraints,
        state->power_iteration_normalized_vector_d);
    create_dense_vector_descriptor(
        &state->spmv_power_iteration_product,
        state->num_variables,
        state->power_iteration_product_d);

    initialize_spmv_buffer(
        state->sparse_handle_A,
        state->matA,
        state->spmv_current_primal_solution,
        state->spmv_primal_product,
        &state->primal_spmv_buffer_d);
    initialize_spmv_buffer(
        state->sparse_handle_At,
        state->matAt,
        state->spmv_reflected_dual_solution,
        state->spmv_dual_product,
        &state->dual_spmv_buffer_d);
}

static void allocate_and_copy_to_device(
    void **dst,
    const void *src,
    size_t bytes)
{
    CUDA_CHECK(cudaMalloc(dst, bytes));
    CUDA_CHECK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
}

static void allocate_zeroed_device_buffer(
    void **dst,
    size_t bytes)
{
    CUDA_CHECK(cudaMalloc(dst, bytes));
    CUDA_CHECK(cudaMemset(*dst, 0, bytes));
}

static void allocate_cub_reduce_workspace(pdhg_solver_state_t *state)
{
    const int max_n = (state->num_variables > state->num_constraints)
                          ? state->num_variables
                          : state->num_constraints;
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum((void *)NULL, bytes,
                                      (double *)NULL, (double *)NULL, max_n));
    state->cub_reduce_workspace_bytes = bytes;
    CUDA_CHECK(cudaMalloc(&state->cub_reduce_workspace_d, bytes));
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

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix->row_ind, (size_t)nnz * sizeof(int)));
    build_row_ind_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
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

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ind, (size_t)nnz * sizeof(int)));
    build_row_ind_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->constraint_matrix_t->row_ptr, n_vars, state->constraint_matrix_t->row_ind);
    CUDA_CHECK(cudaGetLastError());

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
    allocate_zeroed_device_buffer((void **)&state->primal_vec_buffer_d, var_bytes);

    allocate_zeroed_device_buffer((void **)&state->initial_dual_solution_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->current_dual_solution_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->pdhg_dual_solution_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->reflected_dual_solution_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->delta_dual_solution_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->dual_product_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->dual_vec_buffer_d, con_bytes);

    allocate_zeroed_device_buffer((void **)&state->initial_dual_slack_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->current_dual_slack_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->pdhg_dual_slack_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->reflected_dual_slack_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->delta_dual_slack_d, var_bytes);

    allocate_zeroed_device_buffer((void **)&state->primal_residual_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->dual_residual_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->homogeneous_dual_residual_d, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->constraint_recession_residual_d, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->variable_recession_residual_d, var_bytes);
}

static void initialize_homogeneous_scalars(pdhg_solver_state_t *state)
{
    state->metrics_h->initial_homogeneous_scale = 1.0;
    state->metrics_h->current_homogeneous_scale = 1.0;
    state->metrics_h->pdhg_homogeneous_scale = 1.0;
    state->metrics_h->pdhg_dual_constraint_offset = 1.0;
    state->metrics_h->pdhg_dual_bound_offset = 1.0;
}

static void initialize_power_iteration_buffers(pdhg_solver_state_t *state)
{
    const size_t power_iteration_vector_count = ((size_t)state->num_constraints + 1) & ~(size_t)1;

    CUDA_CHECK(cudaMalloc(&state->power_iteration_vector_d, power_iteration_vector_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->power_iteration_normalized_vector_d, power_iteration_vector_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->power_iteration_product_d, state->var_bytes));
    CUDA_CHECK(cudaMalloc(&state->power_iteration_vector_norm_scalar_d, sizeof(double)));
}

static void allocate_host_staging_buffers(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMallocHost(&state->metrics_h, sizeof(pdhg_metrics_t)));
    CUDA_CHECK(cudaHostAlloc(&state->precond_bound_norm_mp, sizeof(double), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&state->precond_objective_norm_mp, sizeof(double), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&state->power_iter_sigma_squared_mp, sizeof(double), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&state->power_iter_residual_norm_mp, sizeof(double), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&state->fixed_point_error_mp, sizeof(double), cudaHostAllocMapped));
}

static void allocate_device_scalars(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMalloc(&state->hsd_scalars_d, HSD_SCALARS_COUNT * sizeof(double)));
    state->hsd_scale_initial_d             = state->hsd_scalars_d + HSD_SCALE_INITIAL;
    state->hsd_scale_current_d             = state->hsd_scalars_d + HSD_SCALE_CURRENT;
    state->hsd_scale_pdhg_d                = state->hsd_scalars_d + HSD_SCALE_PDHG;
    state->hsd_scale_reflected_d           = state->hsd_scalars_d + HSD_SCALE_REFLECTED;
    state->hsd_dual_constraint_initial_d   = state->hsd_scalars_d + HSD_DUAL_CONSTRAINT_INITIAL;
    state->hsd_dual_constraint_current_d   = state->hsd_scalars_d + HSD_DUAL_CONSTRAINT_CURRENT;
    state->hsd_dual_constraint_pdhg_d      = state->hsd_scalars_d + HSD_DUAL_CONSTRAINT_PDHG;
    state->hsd_dual_constraint_reflected_d = state->hsd_scalars_d + HSD_DUAL_CONSTRAINT_REFLECTED;
    state->hsd_dual_bound_initial_d        = state->hsd_scalars_d + HSD_DUAL_BOUND_INITIAL;
    state->hsd_dual_bound_current_d        = state->hsd_scalars_d + HSD_DUAL_BOUND_CURRENT;
    state->hsd_dual_bound_pdhg_d           = state->hsd_scalars_d + HSD_DUAL_BOUND_PDHG;
    state->hsd_dual_bound_reflected_d      = state->hsd_scalars_d + HSD_DUAL_BOUND_REFLECTED;

    CUDA_CHECK(cudaMalloc(&state->pdhg_reduce_d, PDHG_COUNT * sizeof(double)));
    state->pdhg_primal_norm_d  = state->pdhg_reduce_d + PDHG_PRIMAL_NORM;
    state->pdhg_dual_norm_d    = state->pdhg_reduce_d + PDHG_DUAL_NORM;
    state->pdhg_slack_norm_d   = state->pdhg_reduce_d + PDHG_SLACK_NORM;
    state->pdhg_primal_cross_d = state->pdhg_reduce_d + PDHG_PRIMAL_CROSS;
    state->pdhg_dual_cross_d   = state->pdhg_reduce_d + PDHG_DUAL_CROSS;
    state->pdhg_slack_cross_d  = state->pdhg_reduce_d + PDHG_SLACK_CROSS;

    CUDA_CHECK(cudaMalloc(&state->farkas_reduce_d, FARKAS_COUNT * sizeof(double)));
    state->farkas_dual_solution_norm_d    = state->farkas_reduce_d + FARKAS_DUAL_SOLUTION_NORM;
    state->farkas_dual_slack_norm_d       = state->farkas_reduce_d + FARKAS_DUAL_SLACK_NORM;
    state->farkas_homogeneous_dual_norm_d = state->farkas_reduce_d + FARKAS_HOMOGENEOUS_DUAL_NORM;
    state->farkas_constraint_recession_d  = state->farkas_reduce_d + FARKAS_CONSTRAINT_RECESSION;
    state->farkas_variable_recession_d    = state->farkas_reduce_d + FARKAS_VARIABLE_RECESSION;
    state->farkas_support_constraint_d    = state->farkas_reduce_d + FARKAS_SUPPORT_CONSTRAINT;
    state->farkas_support_variable_d      = state->farkas_reduce_d + FARKAS_SUPPORT_VARIABLE;

    CUDA_CHECK(cudaMalloc(&state->metrics_d, sizeof(pdhg_metrics_t)));
    CUDA_CHECK(cudaMalloc(&state->hsd_inverse_scale_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->constraint_bound_rescaling_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->objective_vector_rescaling_d, sizeof(double)));
    allocate_zeroed_device_buffer((void **)&state->inner_count_d, sizeof(int));
    allocate_zeroed_device_buffer((void **)&state->step_size_primal_d, sizeof(double));
    allocate_zeroed_device_buffer((void **)&state->step_size_dual_d, sizeof(double));
    allocate_zeroed_device_buffer((void **)&state->step_size_homogeneous_d, sizeof(double));
    state->graphExec = NULL;
    state->graph_created = false;

    allocate_zeroed_device_buffer((void **)&state->constraint_cone_state.scale_d, sizeof(double));
    allocate_zeroed_device_buffer((void **)&state->constraint_cone_state.unprojected_scale_d, sizeof(double));
    state->constraint_cone_state.lower_bound_d = state->constraint_lower_bound_d;
    state->constraint_cone_state.upper_bound_d = state->constraint_upper_bound_d;
    state->constraint_cone_state.num_elements = state->num_constraints;

    allocate_zeroed_device_buffer((void **)&state->bound_cone_state.scale_d, sizeof(double));
    allocate_zeroed_device_buffer((void **)&state->bound_cone_state.unprojected_scale_d, sizeof(double));
    state->bound_cone_state.lower_bound_d = state->variable_lower_bound_d;
    state->bound_cone_state.upper_bound_d = state->variable_upper_bound_d;
    state->bound_cone_state.num_elements = state->num_variables;
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

static void size_cone_grid(cone_projection_state_t *cone_state, int newton_budget)
{
    const int n = cone_state->num_elements;

    if (n <= COOP_NEWTON_THRESHOLD)
    {
        cone_state->num_blocks_newton = 1;
        cone_state->newton_partials_d = NULL;
        return;
    }

    int n_blocks_newton = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (n_blocks_newton > newton_budget)
    {
        n_blocks_newton = newton_budget;
    }
    if (n_blocks_newton < 1)
    {
        n_blocks_newton = 1;
    }
    cone_state->num_blocks_newton = n_blocks_newton;
    allocate_zeroed_device_buffer((void **)&cone_state->newton_partials_d, 2 * (size_t)n_blocks_newton * sizeof(newton_denom_numer_t));
}

static void initialize_cone_grids(pdhg_solver_state_t *state)
{
    const int coop_cap = max_cooperative_cone_grid();
    const int n_cons = state->constraint_cone_state.num_elements;
    const int n_vars = state->bound_cone_state.num_elements;
    const bool both_cooperative =
        n_cons > COOP_NEWTON_THRESHOLD && n_vars > COOP_NEWTON_THRESHOLD;

    int budget_constraint = coop_cap;
    int budget_bound = coop_cap;
    if (both_cooperative)
    {
        const long total = (long)n_cons + (long)n_vars;
        budget_constraint = (int)((long)coop_cap * n_cons / total);
        if (budget_constraint < 1)
            budget_constraint = 1;
        if (budget_constraint > coop_cap - 1)
            budget_constraint = coop_cap - 1;
        budget_bound = coop_cap - budget_constraint;
    }

    size_cone_grid(&state->constraint_cone_state, budget_constraint);
    size_cone_grid(&state->bound_cone_state, budget_bound);
}

static void release_setup_only_matrix_data(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaFree(state->constraint_matrix->row_ind));
    state->constraint_matrix->row_ind = NULL;
    CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ind));
    state->constraint_matrix_t->row_ind = NULL;
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
    state->best_primal_dual_residual_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->pid_has_succeeded = false;
}

void pdhg_solver_state_free(pdhg_solver_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

    if (state->sparse_handle_A)
        CUSPARSE_CHECK(cusparseDestroy(state->sparse_handle_A));
    if (state->sparse_handle_At)
        CUSPARSE_CHECK(cusparseDestroy(state->sparse_handle_At));
    if (state->stream_A)
        CUDA_CHECK(cudaStreamDestroy(state->stream_A));
    if (state->stream_At)
        CUDA_CHECK(cudaStreamDestroy(state->stream_At));
    if (state->event_A)
        CUDA_CHECK(cudaEventDestroy(state->event_A));
    if (state->event_At)
        CUDA_CHECK(cudaEventDestroy(state->event_At));
    if (state->event_s)
        CUDA_CHECK(cudaEventDestroy(state->event_s));
    if (state->blas_handle_A)
        CUBLAS_CHECK(cublasDestroy(state->blas_handle_A));
    if (state->blas_handle_At)
        CUBLAS_CHECK(cublasDestroy(state->blas_handle_At));
    if (state->curand_gen)
        CURAND_CHECK(curandDestroyGenerator(state->curand_gen));

    free_sparse_resources(state);
    free_solver_device_buffers(state);
    free_device_sparse_matrix(state->constraint_matrix);
    free_device_sparse_matrix(state->constraint_matrix_t);

    free(state);
}

static void free_sparse_resources(pdhg_solver_state_t *state)
{
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
    if (state->spmv_delta_dual_solution)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_delta_dual_solution));
    if (state->spmv_primal_product)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_primal_product));
    if (state->spmv_dual_product)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_dual_product));
    if (state->spmv_power_iteration_vector)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_power_iteration_vector));
    if (state->spmv_power_iteration_normalized_vector)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_power_iteration_normalized_vector));
    if (state->spmv_power_iteration_product)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_power_iteration_product));
    if (state->primal_spmv_buffer_d)
        CUDA_CHECK(cudaFree(state->primal_spmv_buffer_d));
    if (state->dual_spmv_buffer_d)
        CUDA_CHECK(cudaFree(state->dual_spmv_buffer_d));
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
    FREE_DEVICE_BUFFER(state->primal_vec_buffer_d);

    FREE_DEVICE_BUFFER(state->initial_dual_solution_d);
    FREE_DEVICE_BUFFER(state->current_dual_solution_d);
    FREE_DEVICE_BUFFER(state->pdhg_dual_solution_d);
    FREE_DEVICE_BUFFER(state->reflected_dual_solution_d);
    FREE_DEVICE_BUFFER(state->delta_dual_solution_d);
    FREE_DEVICE_BUFFER(state->dual_product_d);
    FREE_DEVICE_BUFFER(state->dual_vec_buffer_d);

    FREE_DEVICE_BUFFER(state->initial_dual_slack_d);
    FREE_DEVICE_BUFFER(state->current_dual_slack_d);
    FREE_DEVICE_BUFFER(state->pdhg_dual_slack_d);
    FREE_DEVICE_BUFFER(state->reflected_dual_slack_d);
    FREE_DEVICE_BUFFER(state->delta_dual_slack_d);

    FREE_DEVICE_BUFFER(state->hsd_scalars_d);
    FREE_DEVICE_BUFFER(state->hsd_inverse_scale_d);
    FREE_DEVICE_BUFFER(state->power_iteration_vector_d);
    FREE_DEVICE_BUFFER(state->power_iteration_normalized_vector_d);
    FREE_DEVICE_BUFFER(state->power_iteration_product_d);
    FREE_DEVICE_BUFFER(state->power_iteration_vector_norm_scalar_d);

    FREE_DEVICE_BUFFER(state->primal_residual_d);
    FREE_DEVICE_BUFFER(state->dual_residual_d);
    FREE_DEVICE_BUFFER(state->homogeneous_dual_residual_d);
    FREE_DEVICE_BUFFER(state->constraint_recession_residual_d);
    FREE_DEVICE_BUFFER(state->variable_recession_residual_d);

    FREE_DEVICE_BUFFER(state->constraint_cone_state.scale_d);
    FREE_DEVICE_BUFFER(state->constraint_cone_state.unprojected_scale_d);
    FREE_DEVICE_BUFFER(state->constraint_cone_state.newton_partials_d);
    FREE_DEVICE_BUFFER(state->bound_cone_state.scale_d);
    FREE_DEVICE_BUFFER(state->bound_cone_state.unprojected_scale_d);
    FREE_DEVICE_BUFFER(state->bound_cone_state.newton_partials_d);

    FREE_DEVICE_BUFFER(state->cub_reduce_workspace_d);
    FREE_DEVICE_BUFFER(state->inner_count_d);
    FREE_DEVICE_BUFFER(state->step_size_primal_d);
    FREE_DEVICE_BUFFER(state->step_size_dual_d);
    FREE_DEVICE_BUFFER(state->step_size_homogeneous_d);
    FREE_DEVICE_BUFFER(state->pdhg_reduce_d);
    FREE_DEVICE_BUFFER(state->farkas_reduce_d);
    FREE_DEVICE_BUFFER(state->metrics_d);

    if (state->graphExec)
    {
        CUDA_CHECK(cudaGraphExecDestroy(state->graphExec));
        state->graphExec = NULL;
    }

#undef FREE_DEVICE_BUFFER

#define FREE_HOST_BUFFER(ptr) \
    do                        \
    {                         \
        if ((ptr))            \
            CUDA_CHECK(cudaFreeHost(ptr)); \
    } while (0)

    FREE_HOST_BUFFER(state->metrics_h);
    FREE_HOST_BUFFER(state->precond_bound_norm_mp);
    FREE_HOST_BUFFER(state->precond_objective_norm_mp);
    FREE_HOST_BUFFER(state->power_iter_sigma_squared_mp);
    FREE_HOST_BUFFER(state->power_iter_residual_norm_mp);
    FREE_HOST_BUFFER(state->fixed_point_error_mp);

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
