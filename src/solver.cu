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

#include "cuhsdlp.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "presolver.h"
#include "solver.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__global__ void build_row_ind(
    const int *__restrict__ row_ptr,
    int num_rows,
    int *__restrict__ row_ind);
__global__ void hsd_gradient_step_kernel(
    double *__restrict__ out,
    const double *__restrict__ current,
    const double *__restrict__ product,
    double step_size,
    int n);
__global__ void cone_newton_solve_kernel(
    const double *__restrict__ w_bar,
    const double *__restrict__ lb,
    const double *__restrict__ ub,
    double *__restrict__ cone_state,
    const double *__restrict__ hsd_scalars,
    double step_size, int scalar_offset,
    int n, int max_iter, double tol);
__global__ void cone_box_project_kernel(
    double *__restrict__ projected,
    const double *__restrict__ w_bar,
    const double *__restrict__ lb,
    const double *__restrict__ ub,
    const double *__restrict__ cone_state,
    int n);
__global__ void hsd_halpern_kernel(
    double *__restrict__ cur,
    double *__restrict__ refl,
    const double *__restrict__ proj,
    const double *__restrict__ init,
    int base_count,
    int k_offset,
    double refl_coeff,
    int n,
    double *hsd_scalars_d,
    const double *cone_newton_buffer,
    double dual_step,
    int scalar_offset,
    bool is_major);
__global__ void hsd_primal_scale_update_kernel(
    double *hsd_scalars_d,
    const double *c_dot_x_d,
    double primal_step,
    int inner_count,
    int k_offset,
    double reflection_coefficient,
    bool is_major);
__global__ void hsd_primal_solution_update_kernel(
    double *__restrict__ cur,
    double *__restrict__ pdhg,
    double *__restrict__ refl,
    const double *__restrict__ init,
    const double *__restrict__ dual_prod,
    const double *__restrict__ refl_dual_slack,
    const double *__restrict__ obj_vec,
    const double *hsd_scalars_d,
    double step_size,
    int base_count,
    int k_offset,
    double refl_coeff,
    int n,
    bool is_major);
__global__ void compute_delta_kernel(
    const double *__restrict__ initial_primal,
    const double *__restrict__ pdhg_primal,
    double *__restrict__ delta_primal,
    const double *__restrict__ initial_dual,
    const double *__restrict__ pdhg_dual,
    double *__restrict__ delta_dual,
    const double *__restrict__ initial_dual_slack,
    const double *__restrict__ pdhg_dual_slack,
    double *__restrict__ delta_dual_slack,
    int n_vars,
    int n_cons);
__global__ void rescale_solution_kernel(
    double *__restrict__ primal_sol,
    double *__restrict__ dual_sol,
    double *__restrict__ dual_slack,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ cons_rescaling,
    const double obj_vec_rescaling,
    const double cons_bound_rescaling,
    double tau,
    int n_vars,
    int n_cons);
__global__ void compute_fixed_point_error_kernel(
    double *fixed_point_error_d,
    const double *workspace_scalars_d,
    const double *hsd_scalars_d,
    double step_size_primal,
    double step_size_dual,
    double step_size_homogeneous);
__global__ void init_hsd_scalars_kernel(double *hsd_scalars_d);
__global__ void restart_hsd_scalars_kernel(double *hsd_scalars_d);
__global__ void normalize_hsd_scalars_kernel(double *hsd_scalars_d, double inv_tau);
static void check_params_validity(const pdhg_parameters_t *params);
static void compute_fixed_point_error(pdhg_solver_state_t *state);
static void fetch_fixed_point_error_to_host(pdhg_solver_state_t *state);
static void run_hsd_iteration_step(
    pdhg_solver_state_t *state,
    const cone_parameters_t *cone_params,
    int k_offset,
    double reflection_coefficient,
    bool is_major);
static pdhg_solver_state_t *initialize_solver_state(
    lp_problem_t *working_problem,
    const pdhg_parameters_t *params);
static void transpose_csr_and_build_map(
    const lp_problem_t *problem,
    int **transpose_row_ptr,
    int **transpose_col_ind,
    double **transpose_val,
    int **transpose_map);
static void initialize_step_size_and_primal_weight(
    pdhg_solver_state_t *state, 
    const pdhg_parameters_t *params);
static void create_dense_vector_descriptor(
    cusparseDnVecDescr_t *descriptor,
    int length,
    double *values);
static void prepare_spmv_buffer(
    cusparseHandle_t handle,
    cusparseSpMatDescr_t matrix,
    cusparseDnVecDescr_t input,
    cusparseDnVecDescr_t output,
    void **buffer);
static void allocate_and_copy_to_device(
    void **dst,
    const void *src,
    size_t bytes);
static void allocate_zeroed_device_buffer(
    void **dst,
    size_t bytes);
static void upload_problem_data(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem,
    size_t var_bytes,
    size_t con_bytes,
    int n_vars,
    int n_cons,
    int nnz);
static void allocate_solver_work_vectors(
    pdhg_solver_state_t *state,
    size_t var_bytes,
    size_t con_bytes);
static void initialize_homogeneous_scalars(pdhg_solver_state_t *state);
static void initialize_power_iteration_buffers(
    pdhg_solver_state_t *state,
    size_t var_bytes);
static void allocate_host_staging_buffers(pdhg_solver_state_t *state);
static void apply_warm_starts(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem,
    size_t var_bytes,
    size_t con_bytes);
static void release_setup_only_matrix_data(pdhg_solver_state_t *state);
static void initialize_iterates_from_starts(
    pdhg_solver_state_t *state,
    size_t var_bytes,
    size_t con_bytes);
static void initialize_solver_progress(pdhg_solver_state_t *state);
void hsd_update_primal_scale(
    pdhg_solver_state_t *state, 
    const int k_offset,
    const double reflection_coefficient,
    bool is_major);
static void solve_box_cone_projection(
    const cone_parameters_t *cone_params,
    const double *unprojected_vector,
    const double *lower_bound,
    const double *upper_bound,
    int n,
    double *cone_newton_buffer,
    const double *hsd_scalars_d,
    double step_size,
    int scalar_offset,
    cudaStream_t stream);
static void project_onto_box_cone(
    double *unprojected_vector,
    const double *lower_bound,
    const double *upper_bound,
    int n,
    double *projected_vector,
    const double *cone_newton_buffer,
    cudaStream_t stream);
void hsd_prepare_dual_solution_update(
    pdhg_solver_state_t *state,
    const cone_parameters_t *cone_params);
void hsd_prepare_dual_slack_update(
    pdhg_solver_state_t *state,
    const cone_parameters_t *cone_params);
void hsd_project_and_update_dual_solution(
    pdhg_solver_state_t *state,
    const int k_offset,
    const double reflection_coefficient,
    bool is_major);
void hsd_project_and_update_dual_slack(
    pdhg_solver_state_t *state,
    const int k_offset,
    const double reflection_coefficient,
    bool is_major);
void hsd_update_primal_solution(
    pdhg_solver_state_t *state,
    const int k_offset,
    const double reflection_coefficient,
    bool is_major);
static void perform_restart(
    pdhg_solver_state_t *state,
    const step_size_parameters_t *step_size_params);
static void print_iteration_header(bool verbose);
static void destroy_solver_gpu_handles(pdhg_solver_state_t *state);
static void free_device_sparse_matrix(cu_sparse_matrix_csr_t *matrix);
static void free_solver_device_buffers(pdhg_solver_state_t *state);
static void normalize_hsd_variables(pdhg_solver_state_t *state);
static void rescale_solution(pdhg_solver_state_t *state);
static cuhsdlp_result_t *create_result_from_state(
    pdhg_solver_state_t *state, 
    const lp_problem_t *original_problem);
void pdhg_solver_state_free(pdhg_solver_state_t *state);

cuhsdlp_result_t *optimize(
    const pdhg_parameters_t *params,
    lp_problem_t *original_problem)
{
    check_params_validity(params);
    print_initial_info(params, original_problem);

    cuhsdlp_presolve_info_t *presolve_info = NULL;
    lp_problem_t *working_problem = original_problem;

    if (params->presolve)
    {
        presolve_info = pslp_presolve(original_problem, params->verbose);
        if (presolve_info->problem_solved_during_presolve)
        {
            cuhsdlp_result_t *result = create_result_from_presolve(presolve_info, original_problem);
            cuhsdlp_presolve_info_free(presolve_info);
            pdhg_final_log(result, params);
            return result;
        }
        working_problem = presolve_info->reduced_problem;
    }

    pdhg_solver_state_t *state = initialize_solver_state(working_problem, params);
    display_iteration_stats(state, params->verbose, false);

    initialize_step_size_and_primal_weight(state, params);

    state->start_time = clock();
    bool do_restart = false;

    while (state->total_count < params->termination_criteria.iteration_limit)
    {
        run_hsd_iteration_step(state, &params->cone_params, 1,
                               params->reflection_coefficient, true);

        if (do_restart)
        {
            compute_fixed_point_error(state);
            fetch_fixed_point_error_to_host(state);
            state->initial_fixed_point_error = state->fixed_point_error;
            do_restart = false;
        }

        for (int i = 2; i <= params->termination_evaluation_frequency - 1; i++)
        {
            run_hsd_iteration_step(state, &params->cone_params, i,
                                   params->reflection_coefficient, false);
        }

        run_hsd_iteration_step(state, &params->cone_params,
                               params->termination_evaluation_frequency,
                               params->reflection_coefficient, true);

        compute_fixed_point_error(state);
        compute_residual(state, params->optimality_norm);
        state->inner_count += params->termination_evaluation_frequency;
        state->total_count += params->termination_evaluation_frequency;


        check_termination_criteria(state, &params->termination_criteria);
        if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
        {
            display_iteration_stats(state, params->verbose, false);
            break;
        }

        do_restart = should_do_adaptive_restart(state, &params->restart_params, params->termination_evaluation_frequency);
        if (do_restart)
        {
            state->restart_count++;
            display_iteration_stats(state, params->verbose, true);
            perform_restart(state, &params->step_size_params);
        }
        else
        {
            display_iteration_stats(state, params->verbose, false);
        }
    }

    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_residual(state, params->optimality_norm);
        display_iteration_stats(state, params->verbose, false);
    }

    cuhsdlp_result_t *result = create_result_from_state(state, original_problem);

    if (params->presolve && presolve_info)
    {
        pslp_postsolve(presolve_info, result, original_problem);
        cuhsdlp_presolve_info_free(presolve_info);
    }

    pdhg_final_log(result, params);
    pdhg_solver_state_free(state);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

static void check_params_validity(const pdhg_parameters_t *params)
{
    if (params->termination_evaluation_frequency < 3)
    {
        fprintf(stderr,
                "Error: termination_evaluation_frequency must be >= 3 (got %d).\n",
                params->termination_evaluation_frequency);
        exit(EXIT_FAILURE);
    }
}

static pdhg_solver_state_t *initialize_solver_state(lp_problem_t *working_problem,
                                                    const pdhg_parameters_t *params)
{
    pdhg_solver_state_t *state = (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    int n_vars = working_problem->num_variables;
    int n_cons = working_problem->num_constraints;
    int nnz = working_problem->constraint_matrix_num_nonzeros;

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->var_bytes = n_vars * sizeof(double);
    state->con_bytes = n_cons * sizeof(double);

    size_t var_bytes = state->var_bytes;
    size_t con_bytes = state->con_bytes;
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
    state->num_blocks_primal_dual =
        (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
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

    {
        int max_n = (n_vars > n_cons) ? n_vars : n_cons;
        size_t tb_sum = 0, tb_max = 0;
        CUDA_CHECK(cub::DeviceReduce::Sum((void *)NULL, tb_sum, (double *)NULL, (double *)NULL, max_n));
        CUDA_CHECK(cub::DeviceReduce::Max((void *)NULL, tb_max, (double *)NULL, (double *)NULL, max_n));
        state->cub_reduce_workspace_bytes = (tb_sum > tb_max) ? tb_sum : tb_max;
        CUDA_CHECK(cudaMalloc(&state->cub_reduce_workspace, state->cub_reduce_workspace_bytes));
    }

    upload_problem_data(state, working_problem, var_bytes, con_bytes, n_vars, n_cons, nnz);
    allocate_solver_work_vectors(state, var_bytes, con_bytes);
    initialize_homogeneous_scalars(state);
    initialize_power_iteration_buffers(state, var_bytes);
    allocate_host_staging_buffers(state);
    CUDA_CHECK(cudaMalloc(&state->hsd_scalars_d, HSD_SCALARS_COUNT * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->cone_newton_constraint_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->cone_newton_bound_d, sizeof(double)));
    CUDA_CHECK(cudaMemset(state->cone_newton_constraint_d, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(state->cone_newton_bound_d, 0, sizeof(double)));
    apply_warm_starts(state, working_problem, var_bytes, con_bytes);

    CUDA_CHECK(cudaMalloc(&state->workspace_scalars_d, WORKSPACE_SCALAR_COUNT * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->constraint_bound_rescaling_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->objective_vector_rescaling_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->dscal_alpha_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->fixed_point_error_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->residual_metrics_d, RESIDUAL_METRIC_COUNT * sizeof(double)));

    init_hsd_scalars_kernel<<<1, 1, 0, state->stream_A>>>(state->hsd_scalars_d);

    CUSPARSE_CHECK(cusparseCreateCsr(&state->matA,
                                     state->num_constraints,
                                     state->num_variables,
                                     state->constraint_matrix->num_nonzeros,
                                     state->constraint_matrix->row_ptr,
                                     state->constraint_matrix->col_ind,
                                     state->constraint_matrix->val,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));
    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(cusparseCreateCsr(&state->matAt,
                                     state->num_variables,
                                     state->num_constraints,
                                     state->constraint_matrix_t->num_nonzeros,
                                     state->constraint_matrix_t->row_ptr,
                                     state->constraint_matrix_t->col_ind,
                                     state->constraint_matrix_t->val,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));
    CUDA_CHECK(cudaGetLastError());

    create_dense_vector_descriptor(&state->spmv_current_primal_solution, state->num_variables, state->current_primal_solution);
    create_dense_vector_descriptor(&state->spmv_reflected_dual_solution, state->num_constraints, state->reflected_dual_solution);
    create_dense_vector_descriptor(&state->spmv_pdhg_primal_solution, state->num_variables, state->pdhg_primal_solution);
    create_dense_vector_descriptor(&state->spmv_pdhg_dual_solution, state->num_constraints, state->pdhg_dual_solution);
    create_dense_vector_descriptor(&state->spmv_delta_dual_solution, state->num_constraints, state->delta_dual_solution);
    create_dense_vector_descriptor(&state->spmv_primal_product, state->num_constraints, state->primal_product);
    create_dense_vector_descriptor(&state->spmv_dual_product, state->num_variables, state->dual_product);
    create_dense_vector_descriptor(&state->spmv_power_iteration_vector, state->num_constraints, state->power_iteration_vector_d);
    create_dense_vector_descriptor(&state->spmv_power_iteration_normalized_vector, state->num_constraints, state->power_iteration_normalized_vector_d);
    create_dense_vector_descriptor(&state->spmv_power_iteration_product, state->num_variables, state->power_iteration_product_d);

    prepare_spmv_buffer(state->sparse_handle_A,
                        state->matA,
                        state->spmv_current_primal_solution,
                        state->spmv_primal_product,
                        &state->primal_spmv_buffer);
    prepare_spmv_buffer(state->sparse_handle_At,
                        state->matAt,
                        state->spmv_reflected_dual_solution,
                        state->spmv_dual_product,
                        &state->dual_spmv_buffer);

    CUDA_CHECK(cudaMalloc(&state->constraint_rescaling, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->variable_rescaling, var_bytes));

    rescale_problem(&params->scaling_params, &params->power_iter_params,
                    params->optimality_norm, params->verbose, state);

    release_setup_only_matrix_data(state);
    initialize_iterates_from_starts(state, var_bytes, con_bytes);
    initialize_solver_progress(state);

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

static void create_dense_vector_descriptor(
    cusparseDnVecDescr_t *descriptor,
    int length,
    double *values)
{
    CUSPARSE_CHECK(cusparseCreateDnVec(descriptor, length, values, CUDA_R_64F));
}

static void prepare_spmv_buffer(
    cusparseHandle_t handle,
    cusparseSpMatDescr_t matrix,
    cusparseDnVecDescr_t input,
    cusparseDnVecDescr_t output,
    void **buffer)
{
    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           matrix,
                                           input,
                                           &HOST_ZERO,
                                           output,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           &buffer_size));

    CUDA_CHECK(cudaMalloc(buffer, buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           matrix,
                                           input,
                                           &HOST_ZERO,
                                           output,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           *buffer));
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

static void upload_problem_data(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem,
    size_t var_bytes,
    size_t con_bytes,
    int n_vars,
    int n_cons,
    int nnz)
{
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
    build_row_ind<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
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
    build_row_ind<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
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

    allocate_and_copy_to_device((void **)&state->variable_lower_bound,
                                working_problem->variable_lower_bound, var_bytes);
    allocate_and_copy_to_device((void **)&state->variable_upper_bound,
                                working_problem->variable_upper_bound, var_bytes);
    allocate_and_copy_to_device((void **)&state->objective_vector,
                                working_problem->objective_vector, var_bytes);
    allocate_and_copy_to_device((void **)&state->constraint_lower_bound,
                                working_problem->constraint_lower_bound, con_bytes);
    allocate_and_copy_to_device((void **)&state->constraint_upper_bound,
                                working_problem->constraint_upper_bound, con_bytes);
}

static void allocate_solver_work_vectors(
    pdhg_solver_state_t *state,
    size_t var_bytes,
    size_t con_bytes)
{
    allocate_zeroed_device_buffer((void **)&state->initial_primal_solution, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->current_primal_solution, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->pdhg_primal_solution, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->reflected_primal_solution, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->primal_product, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->dual_residual, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->delta_primal_solution, var_bytes);

    allocate_zeroed_device_buffer((void **)&state->initial_dual_solution, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->current_dual_solution, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->pdhg_dual_solution, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->reflected_dual_solution, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->dual_product, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->primal_residual, con_bytes);
    allocate_zeroed_device_buffer((void **)&state->delta_dual_solution, con_bytes);

    allocate_zeroed_device_buffer((void **)&state->current_dual_slack, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->initial_dual_slack, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->pdhg_dual_slack, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->reflected_dual_slack, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->delta_dual_slack, var_bytes);

    allocate_zeroed_device_buffer((void **)&state->primal_vec_buffer, var_bytes);
    allocate_zeroed_device_buffer((void **)&state->dual_vec_buffer, con_bytes);
}

static void initialize_homogeneous_scalars(pdhg_solver_state_t *state)
{
    state->current_primal_scale = 1.0;
    state->pdhg_primal_scale = 1.0;
    state->pdhg_dual_constraint_offset = 1.0;
    state->pdhg_dual_bound_offset = 1.0;
}

static void initialize_power_iteration_buffers(
    pdhg_solver_state_t *state,
    size_t var_bytes)
{
    size_t power_iteration_vector_count = ((size_t)state->num_constraints + 1) & ~(size_t)1;
    CUDA_CHECK(cudaMalloc(&state->power_iteration_vector_d, power_iteration_vector_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->power_iteration_normalized_vector_d, power_iteration_vector_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->power_iteration_product_d, var_bytes));
    CUDA_CHECK(cudaMalloc(&state->power_iteration_vector_norm_scalar_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->power_iteration_sigma_squared_scalar_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->power_iteration_residual_norm_scalar_d, sizeof(double)));
}

static void allocate_host_staging_buffers(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMallocHost(&state->reduce_staging_h, 2 * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&state->power_iteration_sigma_sq_h, sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&state->power_iteration_residual_norm_h, sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&state->residual_metrics_h, RESIDUAL_METRIC_COUNT * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&state->fixed_point_error_h, sizeof(double)));
}

static void apply_warm_starts(
    pdhg_solver_state_t *state,
    const lp_problem_t *working_problem,
    size_t var_bytes,
    size_t con_bytes)
{
    if (working_problem->primal_start)
    {
        CUDA_CHECK(cudaMemcpy(state->initial_primal_solution,
                              working_problem->primal_start,
                              var_bytes,
                              cudaMemcpyHostToDevice));
    }
    if (working_problem->dual_start)
    {
        CUDA_CHECK(cudaMemcpy(state->initial_dual_solution,
                              working_problem->dual_start,
                              con_bytes,
                              cudaMemcpyHostToDevice));
    }
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

static void initialize_iterates_from_starts(
    pdhg_solver_state_t *state,
    size_t var_bytes,
    size_t con_bytes)
{
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->initial_dual_solution, con_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution, state->initial_dual_solution, con_bytes, cudaMemcpyDeviceToDevice));
}

static void initialize_solver_progress(pdhg_solver_state_t *state)
{
    state->best_primal_dual_residual_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size_primal = 0.0;
    state->step_size_dual = 0.0;
    state->step_size_homogeneous = 0.0;
    state->primal_weight_error_sum = 0.0;
    state->primal_weight_last_error = 0.0;
}

static void compute_step_sizes(pdhg_solver_state_t *state, const step_size_parameters_t *step_size_params)
{
    double omega = state->primal_weight;
    double base  = state->step_size_base;
    double c_norm = state->rescaled_objective_norm;
    double factor = step_size_params->step_size_factor;
    double xi = step_size_params->objective_coupling_budget;

    state->step_size_primal      = base / omega;
    state->step_size_dual        = base * omega;
    state->step_size_homogeneous = factor * xi
                                 / (c_norm * c_norm * base / omega + 2.0 * base * omega);
}

static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    double sigma = state->rescaled_constraint_norm;
    double factor = params->step_size_params.step_size_factor;
    double xi = params->step_size_params.objective_coupling_budget;

    state->step_size_base = factor * sqrt((1.0 - xi) / (sigma * sigma + 1.0));

    if (params->scaling_params.bound_objective_rescaling)
    {
        state->primal_weight = 1.0;
    }
    else
    {
        state->primal_weight = (state->rescaled_objective_norm + 1.0)
                             / (state->rescaled_bound_norm + 1.0);
    }

    compute_step_sizes(state, &params->step_size_params);

    state->best_primal_weight = state->primal_weight;
}

static void run_hsd_iteration_step(
    pdhg_solver_state_t *state,
    const cone_parameters_t *cone_params,
    int k_offset,
    double reflection_coefficient,
    bool is_major)
{
    hsd_update_primal_scale(state, k_offset, reflection_coefficient, is_major);
    CUDA_CHECK(cudaEventRecord(state->event_s, state->stream_A));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_s, 0));
    hsd_prepare_dual_solution_update(state, cone_params);
    hsd_prepare_dual_slack_update(state, cone_params);
    hsd_project_and_update_dual_solution(state, k_offset, reflection_coefficient, is_major);
    hsd_project_and_update_dual_slack(state, k_offset, reflection_coefficient, is_major);
    hsd_update_primal_solution(state, k_offset, reflection_coefficient, is_major);
}

void hsd_update_primal_scale(
    pdhg_solver_state_t *state,
    const int k_offset,
    const double reflection_coefficient,
    bool is_major)
{
    double homogeneous_step = state->step_size_homogeneous;

    CUBLAS_CHECK(cublasDdot(state->blas_handle_A,
                            state->num_variables,
                            state->objective_vector, 1,
                            state->current_primal_solution, 1,
                            state->workspace_scalars_d + WORKSPACE_SCALAR_C_DOT_X));

    hsd_primal_scale_update_kernel<<<1, 1, 0, state->stream_A>>>(
        state->hsd_scalars_d,
        state->workspace_scalars_d + WORKSPACE_SCALAR_C_DOT_X,
        homogeneous_step,
        state->inner_count,
        k_offset,
        reflection_coefficient,
        is_major);
}

void hsd_prepare_dual_solution_update(
    pdhg_solver_state_t *state,
    const cone_parameters_t *cone_params)
{
    double dual_step = state->step_size_dual;

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle_A,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matA,
                                state->spmv_current_primal_solution,
                                &HOST_ZERO,
                                state->spmv_primal_product,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->primal_spmv_buffer));

    hsd_gradient_step_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->dual_vec_buffer,
        state->current_dual_solution,
        state->primal_product,
        dual_step,
        state->num_constraints);

    solve_box_cone_projection(cone_params,
                              state->dual_vec_buffer,
                              state->constraint_lower_bound,
                              state->constraint_upper_bound,
                              state->num_constraints,
                              state->cone_newton_constraint_d,
                              state->hsd_scalars_d,
                              dual_step,
                              HSD_DUAL_CONSTRAINT,
                              state->stream_A);
}

void hsd_project_and_update_dual_solution(pdhg_solver_state_t *state,
                                          const int k_offset,
                                          const double reflection_coefficient,
                                          bool is_major)
{
    double dual_step = state->step_size_dual;
    project_onto_box_cone(state->dual_vec_buffer,
                          state->constraint_lower_bound,
                          state->constraint_upper_bound,
                          state->num_constraints,
                          state->pdhg_dual_solution,
                          state->cone_newton_constraint_d,
                          state->stream_A);

    hsd_halpern_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->current_dual_solution,
        state->reflected_dual_solution,
        state->pdhg_dual_solution,
        state->initial_dual_solution,
        state->inner_count,
        k_offset,
        reflection_coefficient,
        state->num_constraints,
        state->hsd_scalars_d,
        state->cone_newton_constraint_d,
        dual_step,
        HSD_DUAL_CONSTRAINT,
        is_major);

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
}

void hsd_prepare_dual_slack_update(pdhg_solver_state_t *state,
                                   const cone_parameters_t *cone_params)
{
    double dual_step = state->step_size_dual;

    hsd_gradient_step_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_At>>>(
        state->primal_vec_buffer,
        state->current_dual_slack,
        state->current_primal_solution,
        dual_step,
        state->num_variables);

    solve_box_cone_projection(cone_params,
                              state->primal_vec_buffer,
                              state->variable_lower_bound,
                              state->variable_upper_bound,
                              state->num_variables,
                              state->cone_newton_bound_d,
                              state->hsd_scalars_d,
                              dual_step,
                              HSD_DUAL_BOUND,
                              state->stream_At);
}

void hsd_project_and_update_dual_slack(pdhg_solver_state_t *state,
                                       const int k_offset,
                                       const double reflection_coefficient,
                                       bool is_major)
{
    double dual_step = state->step_size_dual;
    project_onto_box_cone(state->primal_vec_buffer,
                          state->variable_lower_bound,
                          state->variable_upper_bound,
                          state->num_variables,
                          state->pdhg_dual_slack,
                          state->cone_newton_bound_d,
                          state->stream_At);

    hsd_halpern_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_At>>>(
        state->current_dual_slack,
        state->reflected_dual_slack,
        state->pdhg_dual_slack,
        state->initial_dual_slack,
        state->inner_count,
        k_offset,
        reflection_coefficient,
        state->num_variables,
        state->hsd_scalars_d,
        state->cone_newton_bound_d,
        dual_step,
        HSD_DUAL_BOUND,
        is_major);
}

static void solve_box_cone_projection(
    const cone_parameters_t *cone_params,
    const double *unprojected_vector,
    const double *lower_bound,
    const double *upper_bound,
    int n,
    double *cone_newton_buffer,
    const double *hsd_scalars_d,
    double step_size,
    int scalar_offset,
    cudaStream_t stream)
{
    const int BLOCK = 256;
    size_t smem = 2 * BLOCK * sizeof(double);
    cone_newton_solve_kernel<<<1, BLOCK, smem, stream>>>(
        unprojected_vector, lower_bound, upper_bound,
        cone_newton_buffer, hsd_scalars_d,
        step_size, scalar_offset,
        n, cone_params->max_iter, cone_params->tol);
}

static void project_onto_box_cone(
    double *unprojected_vector,
    const double *lower_bound,
    const double *upper_bound,
    int n,
    double *projected_vector,
    const double *cone_newton_buffer,
    cudaStream_t stream)
{
    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cone_box_project_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        projected_vector, unprojected_vector,
        lower_bound, upper_bound,
        cone_newton_buffer, n);
}

void hsd_update_primal_solution(pdhg_solver_state_t *state,
                       const int k_offset,
                       const double reflection_coefficient,
                       bool is_major)
{
    double primal_step = state->step_size_primal;

    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle_At,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matAt,
                                state->spmv_reflected_dual_solution,
                                &HOST_ZERO,
                                state->spmv_dual_product,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->dual_spmv_buffer));

    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));

    hsd_primal_solution_update_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->current_primal_solution,
        state->pdhg_primal_solution,
        state->reflected_primal_solution,
        state->initial_primal_solution,
        state->dual_product,
        state->reflected_dual_slack,
        state->objective_vector,
        state->hsd_scalars_d,
        primal_step,
        state->inner_count,
        k_offset,
        reflection_coefficient,
        state->num_variables,
        is_major);
}

static void compute_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_kernel<<<state->num_blocks_primal_dual_slack, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->pdhg_primal_solution,
        state->reflected_primal_solution,
        state->delta_primal_solution,
        state->pdhg_dual_solution,
        state->reflected_dual_solution,
        state->delta_dual_solution,
        state->pdhg_dual_slack,
        state->reflected_dual_slack,
        state->delta_dual_slack,
        state->num_variables,
        state->num_constraints);

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle_At,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matAt,
                                state->spmv_delta_dual_solution,
                                &HOST_ZERO,
                                state->spmv_dual_product,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->dual_spmv_buffer));

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables, state->delta_primal_solution, 1, state->workspace_scalars_d + WORKSPACE_SCALAR_PRIMAL_NORM));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_constraints, state->delta_dual_solution, 1, state->workspace_scalars_d + WORKSPACE_SCALAR_DUAL_NORM));

    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));

    CUBLAS_CHECK(cublasDdot(state->blas_handle_A,
                            state->num_variables,
                            state->dual_product,
                            1,
                            state->delta_primal_solution,
                            1,
                            state->workspace_scalars_d + WORKSPACE_SCALAR_CROSS_TERM));

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables, state->delta_dual_slack, 1, state->workspace_scalars_d + WORKSPACE_SCALAR_BOUND_DUAL_NORM));

    CUBLAS_CHECK(cublasDdot(state->blas_handle_A,
                            state->num_variables,
                            state->delta_dual_slack,
                            1,
                            state->delta_primal_solution,
                            1,
                            state->workspace_scalars_d + WORKSPACE_SCALAR_BOUND_CROSS_TERM));

    compute_fixed_point_error_kernel<<<1, 1, 0, state->stream_A>>>(
        state->fixed_point_error_d,
        state->workspace_scalars_d,
        state->hsd_scalars_d,
        state->step_size_primal,
        state->step_size_dual,
        state->step_size_homogeneous);
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
}

static void fetch_fixed_point_error_to_host(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpyAsync(state->fixed_point_error_h,
                               state->fixed_point_error_d,
                               sizeof(double),
                               cudaMemcpyDeviceToHost,
                               state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    state->fixed_point_error = *state->fixed_point_error_h;
}

static void perform_restart(pdhg_solver_state_t *state, const step_size_parameters_t *step_size_params)
{
    compute_delta_kernel<<<state->num_blocks_primal_dual_slack, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->initial_primal_solution,
        state->pdhg_primal_solution,
        state->delta_primal_solution,
        state->initial_dual_solution,
        state->pdhg_dual_solution,
        state->delta_dual_solution,
        state->initial_dual_slack,
        state->pdhg_dual_slack,
        state->delta_dual_slack,
        state->num_variables,
        state->num_constraints);

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables, state->delta_primal_solution, 1, state->workspace_scalars_d + WORKSPACE_SCALAR_PRIMAL_NORM));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_constraints, state->delta_dual_solution, 1, state->workspace_scalars_d + WORKSPACE_SCALAR_DUAL_NORM));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables, state->delta_dual_slack, 1, state->workspace_scalars_d + WORKSPACE_SCALAR_CROSS_TERM));

    double restart_scalars_h[3];
    CUDA_CHECK(cudaMemcpyAsync(restart_scalars_h,
                               state->workspace_scalars_d + WORKSPACE_SCALAR_PRIMAL_NORM,
                               3 * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));

    double primal_dist = restart_scalars_h[0];
    double dual_dist = sqrt(restart_scalars_h[1] * restart_scalars_h[1] +
                            restart_scalars_h[2] * restart_scalars_h[2]);
    double ratio_infeas = state->relative_dual_residual / state->relative_primal_residual;

    if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 && ratio_infeas > 1e-8 &&
        ratio_infeas < 1e8)
    {
        double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);
        state->primal_weight_error_sum *= step_size_params->i_smooth;
        state->primal_weight_error_sum += error;
        double delta_error = error - state->primal_weight_last_error;
        state->primal_weight *=
            exp(step_size_params->k_p * error + step_size_params->k_i * state->primal_weight_error_sum +
                step_size_params->k_d * delta_error);
        state->primal_weight_last_error = error;
    }
    else
    {
        state->primal_weight = state->best_primal_weight;
        state->primal_weight_error_sum = 0.0;
        state->primal_weight_last_error = 0.0;
    }

    double primal_dual_residual_gap = fabs(log10(state->relative_dual_residual / state->relative_primal_residual));
    if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
    {
        state->best_primal_dual_residual_gap = primal_dual_residual_gap;
        state->best_primal_weight = state->primal_weight;
    }

    compute_step_sizes(state, step_size_params);

    normalize_hsd_variables(state);

    CUDA_CHECK(cudaMemcpyAsync(state->initial_primal_solution,
                               state->pdhg_primal_solution, state->var_bytes,
                               cudaMemcpyDeviceToDevice, state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(state->current_primal_solution,
                               state->pdhg_primal_solution, state->var_bytes,
                               cudaMemcpyDeviceToDevice, state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(state->initial_dual_solution,
                               state->pdhg_dual_solution, state->con_bytes,
                               cudaMemcpyDeviceToDevice, state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(state->current_dual_solution,
                               state->pdhg_dual_solution, state->con_bytes,
                               cudaMemcpyDeviceToDevice, state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(state->initial_dual_slack,
                               state->pdhg_dual_slack, state->var_bytes,
                               cudaMemcpyDeviceToDevice, state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(state->current_dual_slack,
                               state->pdhg_dual_slack, state->var_bytes,
                               cudaMemcpyDeviceToDevice, state->stream_A));

    state->current_primal_scale = state->pdhg_primal_scale;

    restart_hsd_scalars_kernel<<<1, 1, 0, state->stream_A>>>(state->hsd_scalars_d);
    CUDA_CHECK(cudaMemsetAsync(state->cone_newton_constraint_d, 0, sizeof(double), state->stream_A));
    CUDA_CHECK(cudaMemsetAsync(state->cone_newton_bound_d, 0, sizeof(double), state->stream_A));

    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

static void normalize_hsd_variables(pdhg_solver_state_t *state)
{
    double tau = state->pdhg_primal_scale;
    if (!isfinite(tau) || tau <= 1.0)
        return;
    double inv_tau = 1.0 / tau;
    CUDA_CHECK(cudaMemcpyAsync(state->dscal_alpha_d, &inv_tau, sizeof(double),
                                cudaMemcpyHostToDevice, state->stream_A));
    CUBLAS_CHECK(cublasDscal(state->blas_handle_A,
                             state->num_variables,
                             state->dscal_alpha_d,
                             state->pdhg_primal_solution, 1));
    CUBLAS_CHECK(cublasDscal(state->blas_handle_A,
                             state->num_constraints,
                             state->dscal_alpha_d,
                             state->pdhg_dual_solution, 1));
    CUBLAS_CHECK(cublasDscal(state->blas_handle_A,
                             state->num_variables,
                             state->dscal_alpha_d,
                             state->pdhg_dual_slack, 1));
    normalize_hsd_scalars_kernel<<<1, 1, 0, state->stream_A>>>(state->hsd_scalars_d, inv_tau);
    state->pdhg_primal_scale = 1.0;
    state->pdhg_dual_constraint_offset *= inv_tau;
    state->pdhg_dual_bound_offset *= inv_tau;
}

static cuhsdlp_result_t *create_result_from_state(pdhg_solver_state_t *state, const lp_problem_t *original_problem)
{
    cuhsdlp_result_t *results = (cuhsdlp_result_t *)safe_calloc(1, sizeof(cuhsdlp_result_t));

    rescale_solution(state);

    results->primal_solution = (double *)safe_malloc(state->var_bytes);
    results->dual_solution = (double *)safe_malloc(state->con_bytes);
    results->reduced_cost = (double *)safe_malloc(state->var_bytes);

    CUDA_CHECK(cudaMemcpyAsync(results->primal_solution,
                               state->pdhg_primal_solution,
                               state->var_bytes,
                               cudaMemcpyDeviceToHost,
                               state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(results->dual_solution,
                               state->pdhg_dual_solution,
                               state->con_bytes,
                               cudaMemcpyDeviceToHost,
                               state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(results->reduced_cost,
                               state->pdhg_dual_slack,
                               state->var_bytes,
                               cudaMemcpyDeviceToHost,
                               state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));

    results->num_variables = original_problem->num_variables;
    results->num_constraints = original_problem->num_constraints;
    results->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;
    results->total_count = state->total_count;
    results->rescaling_time_sec = state->rescaling_time_sec;
    results->cumulative_time_sec = state->cumulative_time_sec;
    results->absolute_primal_residual = state->absolute_primal_residual;
    results->relative_primal_residual = state->relative_primal_residual;
    results->absolute_dual_residual = state->absolute_dual_residual;
    results->relative_dual_residual = state->relative_dual_residual;
    results->primal_objective_value = state->primal_objective_value;
    results->dual_objective_value = state->dual_objective_value;
    results->objective_gap = state->objective_gap;
    results->relative_objective_gap = state->relative_objective_gap;
    results->termination_reason = state->termination_reason;

    results->restart_count = state->restart_count;
    results->homogeneous_scale = state->pdhg_primal_scale;
    results->final_step_size_primal = state->step_size_primal;
    results->final_step_size_dual = state->step_size_dual;
    results->final_step_size_homogeneous = state->step_size_homogeneous;

    return results;
}

static void rescale_solution(pdhg_solver_state_t *state)
{
    rescale_solution_kernel<<<state->num_blocks_primal_dual_slack, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->pdhg_primal_solution,
        state->pdhg_dual_solution,
        state->pdhg_dual_slack,
        state->variable_rescaling,
        state->constraint_rescaling,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
        state->pdhg_primal_scale,
        state->num_variables,
        state->num_constraints);
}

void pdhg_solver_state_free(pdhg_solver_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

    destroy_solver_gpu_handles(state);
    free_solver_device_buffers(state);
    free_device_sparse_matrix(state->constraint_matrix);
    free_device_sparse_matrix(state->constraint_matrix_t);

    free(state);
}

static void print_iteration_header(bool verbose)
{
    if (!verbose)
        return;

    printf("-----------------------------------------------------------"
           "-----------------------------------------------------------------\n");
    printf(" %s | %s | %s | %s | %s | %s\n",
           "   runtime    ",
           "    objective     ",
           "  absolute residuals   ",
           "  relative residuals   ",
           "      step sizes       ",
           "  hsd  ");
    printf(" %s %s | %s %s | %s %s %s | %s %s %s | %s %s %s | %s\n",
           "  iter",
           "  time ",
           " pr obj ",
           "  du obj ",
           " pr res",
           " du res",
           "  gap  ",
           " pr res",
           " du res",
           "  gap  ",
           " pr ss ",
           " du ss ",
           " sc ss ",
           " scale ");
    printf("-----------------------------------------------------------"
           "-----------------------------------------------------------------\n");
}

static void destroy_solver_gpu_handles(pdhg_solver_state_t *state)
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

    FREE_DEVICE_BUFFER(state->primal_spmv_buffer);
    FREE_DEVICE_BUFFER(state->dual_spmv_buffer);
    FREE_DEVICE_BUFFER(state->primal_vec_buffer);
    FREE_DEVICE_BUFFER(state->dual_vec_buffer);
    FREE_DEVICE_BUFFER(state->variable_lower_bound);
    FREE_DEVICE_BUFFER(state->variable_upper_bound);
    FREE_DEVICE_BUFFER(state->objective_vector);
    FREE_DEVICE_BUFFER(state->constraint_lower_bound);
    FREE_DEVICE_BUFFER(state->constraint_upper_bound);
    FREE_DEVICE_BUFFER(state->initial_primal_solution);
    FREE_DEVICE_BUFFER(state->current_primal_solution);
    FREE_DEVICE_BUFFER(state->pdhg_primal_solution);
    FREE_DEVICE_BUFFER(state->reflected_primal_solution);
    FREE_DEVICE_BUFFER(state->primal_product);
    FREE_DEVICE_BUFFER(state->initial_dual_solution);
    FREE_DEVICE_BUFFER(state->current_dual_solution);
    FREE_DEVICE_BUFFER(state->pdhg_dual_solution);
    FREE_DEVICE_BUFFER(state->reflected_dual_solution);
    FREE_DEVICE_BUFFER(state->dual_product);
    FREE_DEVICE_BUFFER(state->constraint_rescaling);
    FREE_DEVICE_BUFFER(state->variable_rescaling);
    FREE_DEVICE_BUFFER(state->primal_residual);
    FREE_DEVICE_BUFFER(state->dual_residual);
    FREE_DEVICE_BUFFER(state->delta_primal_solution);
    FREE_DEVICE_BUFFER(state->delta_dual_solution);
    FREE_DEVICE_BUFFER(state->current_dual_slack);
    FREE_DEVICE_BUFFER(state->initial_dual_slack);
    FREE_DEVICE_BUFFER(state->pdhg_dual_slack);
    FREE_DEVICE_BUFFER(state->reflected_dual_slack);
    FREE_DEVICE_BUFFER(state->delta_dual_slack);
    FREE_DEVICE_BUFFER(state->power_iteration_vector_d);
    FREE_DEVICE_BUFFER(state->power_iteration_normalized_vector_d);
    FREE_DEVICE_BUFFER(state->power_iteration_product_d);
    FREE_DEVICE_BUFFER(state->power_iteration_vector_norm_scalar_d);
    FREE_DEVICE_BUFFER(state->power_iteration_sigma_squared_scalar_d);
    FREE_DEVICE_BUFFER(state->power_iteration_residual_norm_scalar_d);
    FREE_DEVICE_BUFFER(state->hsd_scalars_d);
    FREE_DEVICE_BUFFER(state->cone_newton_constraint_d);
    FREE_DEVICE_BUFFER(state->cone_newton_bound_d);
    FREE_DEVICE_BUFFER(state->cub_reduce_workspace);
    FREE_DEVICE_BUFFER(state->workspace_scalars_d);
    FREE_DEVICE_BUFFER(state->constraint_bound_rescaling_d);
    FREE_DEVICE_BUFFER(state->objective_vector_rescaling_d);
    FREE_DEVICE_BUFFER(state->dscal_alpha_d);
    FREE_DEVICE_BUFFER(state->fixed_point_error_d);
    FREE_DEVICE_BUFFER(state->residual_metrics_d);

#undef FREE_DEVICE_BUFFER

#define FREE_HOST_BUFFER(ptr) \
    do                        \
    {                         \
        if ((ptr))            \
            CUDA_CHECK(cudaFreeHost(ptr)); \
    } while (0)

    FREE_HOST_BUFFER(state->reduce_staging_h);
    FREE_HOST_BUFFER(state->power_iteration_sigma_sq_h);
    FREE_HOST_BUFFER(state->power_iteration_residual_norm_h);
    FREE_HOST_BUFFER(state->residual_metrics_h);
    FREE_HOST_BUFFER(state->fixed_point_error_h);

#undef FREE_HOST_BUFFER
}

__global__ void build_row_ind(const int *__restrict__ row_ptr,
                              int num_rows,
                              int *__restrict__ row_ind)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows)
        return;

    for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
        row_ind[k] = row;
}

__global__ void hsd_gradient_step_kernel(
    double *__restrict__ out,
    const double *__restrict__ current,
    const double *__restrict__ product,
    double step_size,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = current[i] - step_size * product[i];
}

__global__ void cone_newton_solve_kernel(
    const double *__restrict__ w_bar,
    const double *__restrict__ lb,
    const double *__restrict__ ub,
    double *__restrict__ cone_state,
    const double *__restrict__ hsd_scalars,
    double step_size,
    int scalar_offset,
    int n,
    int max_iter,
    double tol)
{
    extern __shared__ double smem[];
    double *s_ci = smem;
    double *s_wi = smem + blockDim.x;

    __shared__ double scale, tau_lo, tau_hi, neg_scalar;
    __shared__ int converged;

    if (threadIdx.x == 0)
    {
        scale = *cone_state;
        tau_lo = 0.0;
        tau_hi = INFINITY;
        neg_scalar = step_size * hsd_scalars[HSD_SCALE + HSD_REFLECTED]
                   - hsd_scalars[scalar_offset + HSD_CURRENT];
        converged = 0;
    }
    __syncthreads();

    for (int it = 0; it < max_iter; it++)
    {
        if (converged) break;

        double my_ci = 0.0, my_wi = 0.0;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
        {
            double vi = -w_bar[i];
            double li = lb[i], ui = ub[i];
            if (isfinite(li) && vi < li * scale)
            {
                my_ci += li * li;
                my_wi += li * vi;
            }
            else if (isfinite(ui) && vi > ui * scale)
            {
                my_ci += ui * ui;
                my_wi += ui * vi;
            }
        }

        s_ci[threadIdx.x] = my_ci;
        s_wi[threadIdx.x] = my_wi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (threadIdx.x < s)
            {
                s_ci[threadIdx.x] += s_ci[threadIdx.x + s];
                s_wi[threadIdx.x] += s_wi[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            double ds = s_ci[0], vs = s_wi[0];
            double ns = fmax(0.0, (neg_scalar + vs) / (1.0 + ds));

            if (ns > scale)
                tau_lo = fmax(tau_lo, scale);
            else
                tau_hi = fmin(tau_hi, scale);

            double new_s;
            if (ns >= tau_lo && ns <= tau_hi)
                new_s = ns;
            else if (isfinite(tau_hi))
                new_s = 0.5 * (tau_lo + tau_hi);
            else
                new_s = fmax(ns, tau_lo);

            if (fabs(new_s - scale) < tol * (1.0 + fabs(new_s)))
                converged = 1;
            scale = new_s;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        *cone_state = scale;
}

__global__ void cone_box_project_kernel(
    double *__restrict__ projected,
    const double *__restrict__ w_bar,
    const double *__restrict__ lb,
    const double *__restrict__ ub,
    const double *__restrict__ cone_state,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double tau_star = *cone_state;
    double vi = -w_bar[i];
    double li = lb[i], ui = ub[i];
    double lo_c = isfinite(li) ? li * tau_star : -INFINITY;
    double hi_c = isfinite(ui) ? ui * tau_star : INFINITY;
    projected[i] = w_bar[i] + fmin(fmax(vi, lo_c), hi_c);
}

__global__ void hsd_halpern_kernel(
    double *__restrict__ cur,
    double *__restrict__ refl,
    const double *__restrict__ proj,
    const double *__restrict__ init,
    int base_count,
    int k_offset,
    double refl_coeff,
    int n,
    double *hsd_scalars_d,
    const double *cone_newton_buffer,
    double dual_step,
    int scalar_offset,
    bool is_major)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int k = base_count + k_offset;
        double weight = (double)(k) / (double)(k + 1);
        refl[i] = 2.0 * proj[i] - cur[i];
        double mixed = refl_coeff * refl[i] + (1.0 - refl_coeff) * cur[i];
        cur[i] = weight * mixed + (1.0 - weight) * init[i];
    }
    if (i == n)
    {
        int k = base_count + k_offset;
        double weight = (double)(k) / (double)(k + 1);
        double unprojected = hsd_scalars_d[scalar_offset + HSD_CURRENT]
                           - dual_step * hsd_scalars_d[HSD_SCALE + HSD_REFLECTED];
        double projected = unprojected + *cone_newton_buffer;
        if (is_major)
        {
            hsd_scalars_d[scalar_offset + HSD_PDHG] = projected;
        }
        double reflected = 2.0 * projected - hsd_scalars_d[scalar_offset + HSD_CURRENT];
        double mixed = refl_coeff * reflected + (1.0 - refl_coeff) * hsd_scalars_d[scalar_offset + HSD_CURRENT];
        hsd_scalars_d[scalar_offset + HSD_CURRENT] = weight * mixed + (1.0 - weight) * hsd_scalars_d[scalar_offset + HSD_INITIAL];
        hsd_scalars_d[scalar_offset + HSD_REFLECTED] = reflected;
    }
}


__global__ void hsd_primal_scale_update_kernel(
    double *hsd_scalars_d,
    const double *c_dot_x_d,
    double primal_step,
    int inner_count,
    int k_offset,
    double reflection_coefficient,
    bool is_major)
{
    double c_dot_x = c_dot_x_d[0];
    double gradient = c_dot_x
                    + hsd_scalars_d[HSD_DUAL_CONSTRAINT + HSD_CURRENT]
                    + hsd_scalars_d[HSD_DUAL_BOUND      + HSD_CURRENT];
    double projected = fmax(0.0, hsd_scalars_d[HSD_SCALE + HSD_CURRENT] + primal_step * gradient);

    if (is_major)
    {
        hsd_scalars_d[HSD_SCALE + HSD_PDHG] = projected;
    }

    int k = inner_count + k_offset;
    double weight = (double)(k) / (double)(k + 1);
    double reflected = 2.0 * projected - hsd_scalars_d[HSD_SCALE + HSD_CURRENT];
    double mixed = reflection_coefficient * reflected + (1.0 - reflection_coefficient) * hsd_scalars_d[HSD_SCALE + HSD_CURRENT];
    hsd_scalars_d[HSD_SCALE + HSD_CURRENT] = weight * mixed + (1.0 - weight) * hsd_scalars_d[HSD_SCALE + HSD_INITIAL];
    hsd_scalars_d[HSD_SCALE + HSD_REFLECTED] = reflected;
}

__global__ void hsd_primal_solution_update_kernel(
    double *__restrict__ cur,
    double *__restrict__ pdhg,
    double *__restrict__ refl,
    const double *__restrict__ init,
    const double *__restrict__ dual_prod,
    const double *__restrict__ refl_dual_slack,
    const double *__restrict__ obj_vec,
    const double *hsd_scalars_d,
    double step_size,
    int base_count,
    int k_offset,
    double refl_coeff,
    int n,
    bool is_major)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double tau_refl = hsd_scalars_d[HSD_SCALE + HSD_REFLECTED];
    int k = base_count + k_offset;
    double weight = (double)(k) / (double)(k + 1);

    double grad = obj_vec[i] * tau_refl - dual_prod[i] - refl_dual_slack[i];
    double next = cur[i] - step_size * grad;

    if (is_major)
    {
        pdhg[i] = next;
    }

    refl[i] = 2.0 * next - cur[i];
    double mixed = refl_coeff * refl[i] + (1.0 - refl_coeff) * cur[i];
    cur[i] = weight * mixed + (1.0 - weight) * init[i];
}

__global__ void rescale_solution_kernel(double *__restrict__ primal_sol,
                                        double *__restrict__ dual_sol,
                                        double *__restrict__ dual_slack,
                                        const double *__restrict__ var_rescaling,
                                        const double *__restrict__ cons_rescaling,
                                        const double obj_vec_rescaling,
                                        const double cons_bound_rescaling,
                                        double tau,
                                        int n_vars,
                                        int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double inv_tau = (fabs(tau) > 1e-30) ? 1.0 / tau : 0.0;
    if (i < n_vars)
    {
        primal_sol[i] = primal_sol[i] * inv_tau / var_rescaling[i] / cons_bound_rescaling;
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        dual_sol[idx] = dual_sol[idx] * inv_tau / cons_rescaling[idx] / obj_vec_rescaling;
    }
    else if (i < 2 * n_vars + n_cons)
    {
        int idx = i - n_vars - n_cons;
        dual_slack[idx] = dual_slack[idx] * inv_tau * var_rescaling[idx] / obj_vec_rescaling;
    }
}

__global__ void compute_delta_kernel(const double *__restrict__ initial_primal,
                                     const double *__restrict__ pdhg_primal,
                                     double *__restrict__ delta_primal,
                                     const double *__restrict__ initial_dual,
                                     const double *__restrict__ pdhg_dual,
                                     double *__restrict__ delta_dual,
                                     const double *__restrict__ initial_dual_slack,
                                     const double *__restrict__ pdhg_dual_slack,
                                     double *__restrict__ delta_dual_slack,
                                     int n_vars,
                                     int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = pdhg_primal[i] - initial_primal[i];
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        delta_dual[idx] = pdhg_dual[idx] - initial_dual[idx];
    }
    else if (i < 2 * n_vars + n_cons)
    {
        int idx = i - n_vars - n_cons;
        delta_dual_slack[idx] = pdhg_dual_slack[idx] - initial_dual_slack[idx];
    }
}

__global__ void compute_fixed_point_error_kernel(
    double *fixed_point_error_d,
    const double *workspace_scalars_d,
    const double *hsd_scalars_d,
    double step_size_primal,
    double step_size_dual,
    double step_size_homogeneous)
{
    double delta_primal_scale =
        hsd_scalars_d[HSD_SCALE + HSD_REFLECTED] -
        hsd_scalars_d[HSD_SCALE + HSD_PDHG];
    double delta_constraint_offset =
        hsd_scalars_d[HSD_DUAL_CONSTRAINT + HSD_REFLECTED] -
        hsd_scalars_d[HSD_DUAL_CONSTRAINT + HSD_PDHG];
    double delta_bound_offset =
        hsd_scalars_d[HSD_DUAL_BOUND + HSD_REFLECTED] -
        hsd_scalars_d[HSD_DUAL_BOUND + HSD_PDHG];
    double primal_norm = workspace_scalars_d[WORKSPACE_SCALAR_PRIMAL_NORM];
    double dual_norm = workspace_scalars_d[WORKSPACE_SCALAR_DUAL_NORM];
    double cross_term = workspace_scalars_d[WORKSPACE_SCALAR_CROSS_TERM];
    double bound_dual_norm = workspace_scalars_d[WORKSPACE_SCALAR_BOUND_DUAL_NORM];
    double bound_cross_term = workspace_scalars_d[WORKSPACE_SCALAR_BOUND_CROSS_TERM];

    double dual_movement = (dual_norm * dual_norm
                         + bound_dual_norm * bound_dual_norm
                         + delta_constraint_offset * delta_constraint_offset
                         + delta_bound_offset * delta_bound_offset)
                         / step_size_dual;
    double movement = primal_norm * primal_norm / step_size_primal
                    + dual_movement
                    + delta_primal_scale * delta_primal_scale / step_size_homogeneous;
    double interaction = 2.0 * (cross_term - bound_cross_term);

    fixed_point_error_d[0] = sqrt(fmax(0.0, movement + interaction));
}

__global__ void init_hsd_scalars_kernel(double *hsd_scalars_d)
{
    for (int i = 0; i < HSD_SCALARS_COUNT; ++i)
        hsd_scalars_d[i] = 1.0;
}

__global__ void restart_hsd_scalars_kernel(double *hsd_scalars_d)
{
    const int bases[3] = {HSD_SCALE, HSD_DUAL_CONSTRAINT, HSD_DUAL_BOUND};
    for (int i = 0; i < 3; ++i)
    {
        double pdhg = hsd_scalars_d[bases[i] + HSD_PDHG];
        hsd_scalars_d[bases[i] + HSD_CURRENT] = pdhg;
        hsd_scalars_d[bases[i] + HSD_REFLECTED] = pdhg;
        hsd_scalars_d[bases[i] + HSD_INITIAL] = pdhg;
    }
}

__global__ void normalize_hsd_scalars_kernel(double *hsd_scalars_d, double inv_tau)
{
    hsd_scalars_d[HSD_SCALE + HSD_PDHG] = 1.0;
    hsd_scalars_d[HSD_DUAL_CONSTRAINT + HSD_PDHG] *= inv_tau;
    hsd_scalars_d[HSD_DUAL_BOUND + HSD_PDHG] *= inv_tau;
}
