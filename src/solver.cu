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
#include <time.h>

__global__ void build_row_ind(
    const int *__restrict__ row_ptr,
    int num_rows,
    int nnz,
    int *__restrict__ row_ind);
__global__ void build_transpose_map(
    const int *__restrict__ A_row_ind,
    const int *__restrict__ A_col_ind,
    const int *__restrict__ At_row_ptr,
    const int *__restrict__ At_col_ind,
    int nnz,
    int *__restrict__ A_to_At);
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
    double step_size_primal,
    double step_size_dual,
    double step_size_homogeneous,
    double delta_primal_scale,
    double delta_constraint_offset,
    double delta_bound_offset);
__global__ void compute_restart_distances_kernel(
    double *restart_distances_d,
    const double *workspace_scalars_d);

static void check_params_validity(const pdhg_parameters_t *params);
static void compute_fixed_point_error(pdhg_solver_state_t *state);
static pdhg_solver_state_t *initialize_solver_state(
    lp_problem_t *working_problem,
    const pdhg_parameters_t *params);
static void sync_hsd_scalars_to_host(pdhg_solver_state_t *state);
static void sync_hsd_scalars_to_device(pdhg_solver_state_t *state);
static void initialize_step_size_and_primal_weight(
    pdhg_solver_state_t *state, 
    const pdhg_parameters_t *params);
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
        hsd_update_primal_scale(state, 1, params->reflection_coefficient, true);
        CUDA_CHECK(cudaEventRecord(state->event_s, state->stream_A));
        CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_s, 0));
        hsd_prepare_dual_solution_update(state, &params->cone_params);
        hsd_prepare_dual_slack_update(state, &params->cone_params);
        hsd_project_and_update_dual_solution(state, 1, params->reflection_coefficient, true);
        hsd_project_and_update_dual_slack(state, 1, params->reflection_coefficient, true);
        hsd_update_primal_solution(state, 1, params->reflection_coefficient, true);

        if (do_restart)
        {
            sync_hsd_scalars_to_host(state);
            compute_fixed_point_error(state);
            state->initial_fixed_point_error = state->fixed_point_error;
            do_restart = false;
        }

        for (int i = 2; i <= params->termination_evaluation_frequency - 1; i++)
        {
            hsd_update_primal_scale(state, i, params->reflection_coefficient, false);
            CUDA_CHECK(cudaEventRecord(state->event_s, state->stream_A));
            CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_s, 0));
            hsd_prepare_dual_solution_update(state, &params->cone_params);
            hsd_prepare_dual_slack_update(state, &params->cone_params);
            hsd_project_and_update_dual_solution(state, i, params->reflection_coefficient, false);
            hsd_project_and_update_dual_slack(state, i, params->reflection_coefficient, false);
            hsd_update_primal_solution(state, i, params->reflection_coefficient, false);
        }

        hsd_update_primal_scale(state, params->termination_evaluation_frequency, params->reflection_coefficient, true);
        CUDA_CHECK(cudaEventRecord(state->event_s, state->stream_A));
        CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_s, 0));
        hsd_prepare_dual_solution_update(state, &params->cone_params);
        hsd_prepare_dual_slack_update(state, &params->cone_params);
        hsd_project_and_update_dual_solution(state, params->termination_evaluation_frequency, params->reflection_coefficient, true);
        hsd_project_and_update_dual_slack(state, params->termination_evaluation_frequency, params->reflection_coefficient, true);
        hsd_update_primal_solution(state, params->termination_evaluation_frequency, params->reflection_coefficient, true);

        sync_hsd_scalars_to_host(state);
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
        CUDA_CHECK(cudaMalloc(&state->cub_reduce_workspace_bound, state->cub_reduce_workspace_bytes));
    }

#define ALLOC_AND_COPY(dest, src, bytes)  \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice));

    ALLOC_AND_COPY(state->constraint_matrix->row_ptr, working_problem->constraint_matrix_row_pointers, (n_cons + 1) * sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->col_ind, working_problem->constraint_matrix_col_indices, nnz * sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->val, working_problem->constraint_matrix_values, nnz * sizeof(double));

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix->row_ind, nnz * sizeof(int)));
    build_row_ind<<<state->num_blocks_nnz, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->row_ptr, n_cons, nnz, state->constraint_matrix->row_ind);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ptr, (n_vars + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->col_ind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->val, nnz * sizeof(double)));

    size_t buffer_size = 0;
    void *buffer = nullptr;
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        state->sparse_handle_A, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
        state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG_DEFAULT, &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        state->sparse_handle_A, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
        state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG_DEFAULT, buffer));

    CUDA_CHECK(cudaFree(buffer));

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ind, nnz * sizeof(int)));
    build_row_ind<<<state->num_blocks_nnz, THREADS_PER_BLOCK>>>(
        state->constraint_matrix_t->row_ptr, n_vars, nnz, state->constraint_matrix_t->row_ind);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix->transpose_map, nnz * sizeof(int)));
    CUDA_CHECK(cudaMemset(state->constraint_matrix->transpose_map, 0xFF, nnz * sizeof(int)));
    state->constraint_matrix_t->transpose_map = NULL;
    build_transpose_map<<<state->num_blocks_nnz, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->row_ind,
        state->constraint_matrix->col_ind,
        state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->col_ind,
        nnz,
        state->constraint_matrix->transpose_map);
    CUDA_CHECK(cudaGetLastError());

    ALLOC_AND_COPY(state->variable_lower_bound, working_problem->variable_lower_bound, var_bytes);
    ALLOC_AND_COPY(state->variable_upper_bound, working_problem->variable_upper_bound, var_bytes);
    ALLOC_AND_COPY(state->objective_vector, working_problem->objective_vector, var_bytes);
    ALLOC_AND_COPY(state->constraint_lower_bound, working_problem->constraint_lower_bound, con_bytes);
    ALLOC_AND_COPY(state->constraint_upper_bound, working_problem->constraint_upper_bound, con_bytes);

#define ALLOC_ZERO(dest, bytes)           \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    ALLOC_ZERO(state->initial_primal_solution, var_bytes);
    ALLOC_ZERO(state->current_primal_solution, var_bytes);
    ALLOC_ZERO(state->pdhg_primal_solution, var_bytes);
    ALLOC_ZERO(state->reflected_primal_solution, var_bytes);
    ALLOC_ZERO(state->primal_product, con_bytes);
    ALLOC_ZERO(state->dual_residual, var_bytes);
    ALLOC_ZERO(state->delta_primal_solution, var_bytes);

    ALLOC_ZERO(state->initial_dual_solution, con_bytes);
    ALLOC_ZERO(state->current_dual_solution, con_bytes);
    ALLOC_ZERO(state->pdhg_dual_solution, con_bytes);
    ALLOC_ZERO(state->reflected_dual_solution, con_bytes);
    ALLOC_ZERO(state->dual_product, var_bytes);
    ALLOC_ZERO(state->primal_residual, con_bytes);
    ALLOC_ZERO(state->delta_dual_solution, con_bytes);

    ALLOC_ZERO(state->current_dual_slack, var_bytes);
    ALLOC_ZERO(state->initial_dual_slack, var_bytes);
    ALLOC_ZERO(state->pdhg_dual_slack, var_bytes);
    ALLOC_ZERO(state->reflected_dual_slack, var_bytes);
    ALLOC_ZERO(state->delta_dual_slack, var_bytes);

    ALLOC_ZERO(state->primal_vec_buffer, var_bytes);
    ALLOC_ZERO(state->dual_vec_buffer, con_bytes);

    state->current_primal_scale = 1.0;
    state->reflected_primal_scale = 1.0;
    state->initial_primal_scale = 1.0;
    state->pdhg_primal_scale = 1.0;

    state->current_dual_constraint_offset = 1.0;
    state->reflected_dual_constraint_offset = 1.0;
    state->initial_dual_constraint_offset = 1.0;
    state->pdhg_dual_constraint_offset = 1.0;

    state->current_dual_bound_offset = 1.0;
    state->reflected_dual_bound_offset = 1.0;
    state->initial_dual_bound_offset = 1.0;
    state->pdhg_dual_bound_offset = 1.0;

    {
        size_t max_bytes = (var_bytes > con_bytes) ? var_bytes : con_bytes;
        CUDA_CHECK(cudaMalloc(&state->power_iteration_vector_d, 2 * max_bytes));
    }
    CUDA_CHECK(cudaMalloc(&state->hsd_scalars_d, HSD_SCALARS_COUNT * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->cone_newton_constraint_d, CONE_BUFFER_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->cone_newton_bound_d, CONE_BUFFER_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMemset(state->cone_newton_constraint_d, 0, CONE_BUFFER_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMemset(state->cone_newton_bound_d, 0, CONE_BUFFER_SIZE * sizeof(double)));

    if (working_problem->primal_start)
    {
        CUDA_CHECK(cudaMemcpy(
            state->initial_primal_solution, working_problem->primal_start, var_bytes, cudaMemcpyHostToDevice));
    }
    if (working_problem->dual_start)
    {
        CUDA_CHECK(
            cudaMemcpy(state->initial_dual_solution, working_problem->dual_start, con_bytes, cudaMemcpyHostToDevice));
    }

#undef ALLOC_AND_COPY
#undef ALLOC_ZERO

    CUDA_CHECK(cudaMalloc(&state->workspace_scalars_d, WORKSPACE_SCALAR_COUNT * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->norm_results_d, NORM_RESULT_COUNT * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->rescaling_factors_d, RESCALING_FACTOR_COUNT * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->restart_distances_d, RESTART_DISTANCE_COUNT * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->fixed_point_error_d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->residual_metrics_d, RESIDUAL_METRIC_COUNT * sizeof(double)));

    sync_hsd_scalars_to_device(state);

    size_t primal_spmv_buffer_size;
    size_t dual_spmv_buffer_size;

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

    CUSPARSE_CHECK(cusparseCreateDnVec(&state->spmv_current_primal_solution, state->num_variables, state->current_primal_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->spmv_reflected_dual_solution, state->num_constraints, state->reflected_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->spmv_pdhg_primal_solution, state->num_variables, state->pdhg_primal_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->spmv_pdhg_dual_solution, state->num_constraints, state->pdhg_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->spmv_delta_dual_solution, state->num_constraints, state->delta_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->spmv_primal_product, state->num_constraints, state->primal_product, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->spmv_dual_product, state->num_variables, state->dual_product, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->spmv_power_iteration_vector, state->num_constraints, state->power_iteration_vector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->spmv_power_iteration_next_vector, state->num_constraints, state->delta_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->spmv_power_iteration_product, state->num_variables, state->delta_primal_solution, CUDA_R_64F));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle_A,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matA,
                                           state->spmv_current_primal_solution,
                                           &HOST_ZERO,
                                           state->spmv_primal_product,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           &primal_spmv_buffer_size));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle_At,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matAt,
                                           state->spmv_reflected_dual_solution,
                                           &HOST_ZERO,
                                           state->spmv_dual_product,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           &dual_spmv_buffer_size));

    CUDA_CHECK(cudaMalloc(&state->primal_spmv_buffer, primal_spmv_buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(state->sparse_handle_A,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matA,
                                           state->spmv_current_primal_solution,
                                           &HOST_ZERO,
                                           state->spmv_primal_product,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           state->primal_spmv_buffer));

    CUDA_CHECK(cudaMalloc(&state->dual_spmv_buffer, dual_spmv_buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(state->sparse_handle_At,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matAt,
                                           state->spmv_reflected_dual_solution,
                                           &HOST_ZERO,
                                           state->spmv_dual_product,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           state->dual_spmv_buffer));

    CUDA_CHECK(cudaMalloc(&state->constraint_rescaling, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->variable_rescaling, var_bytes));

    rescale_problem(&params->scaling_params, &params->power_iter_params,
                    params->optimality_norm, params->verbose, state);

    CUDA_CHECK(cudaFree(state->constraint_matrix->row_ind));
    state->constraint_matrix->row_ind = NULL;
    CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ind));
    state->constraint_matrix_t->row_ind = NULL;
    CUDA_CHECK(cudaFree(state->constraint_matrix->transpose_map));
    state->constraint_matrix->transpose_map = NULL;

    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->initial_dual_solution, con_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution, state->initial_dual_solution, con_bytes, cudaMemcpyDeviceToDevice));


    state->best_primal_dual_residual_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size_primal = 0.0;
    state->step_size_dual = 0.0;
    state->step_size_homogeneous = 0.0;
    state->primal_weight_error_sum = 0.0;
    state->primal_weight_last_error = 0.0;

    if (params->verbose)
    {
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

    return state;
}

static void sync_hsd_scalars_to_host(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_At));
    double h[HSD_SCALARS_COUNT];
    CUDA_CHECK(cudaMemcpy(h, state->hsd_scalars_d, HSD_SCALARS_COUNT * sizeof(double), cudaMemcpyDeviceToHost));
    state->current_primal_scale              = h[HSD_PRIMAL_SCALE    + HSD_CURRENT];
    state->reflected_primal_scale            = h[HSD_PRIMAL_SCALE    + HSD_REFLECTED];
    state->pdhg_primal_scale                 = h[HSD_PRIMAL_SCALE    + HSD_PDHG];
    state->initial_primal_scale              = h[HSD_PRIMAL_SCALE    + HSD_INITIAL];
    state->current_dual_constraint_offset    = h[HSD_DUAL_CONSTRAINT + HSD_CURRENT];
    state->reflected_dual_constraint_offset  = h[HSD_DUAL_CONSTRAINT + HSD_REFLECTED];
    state->pdhg_dual_constraint_offset       = h[HSD_DUAL_CONSTRAINT + HSD_PDHG];
    state->initial_dual_constraint_offset    = h[HSD_DUAL_CONSTRAINT + HSD_INITIAL];
    state->current_dual_bound_offset         = h[HSD_DUAL_BOUND      + HSD_CURRENT];
    state->reflected_dual_bound_offset       = h[HSD_DUAL_BOUND      + HSD_REFLECTED];
    state->pdhg_dual_bound_offset            = h[HSD_DUAL_BOUND      + HSD_PDHG];
    state->initial_dual_bound_offset         = h[HSD_DUAL_BOUND      + HSD_INITIAL];
}

static void sync_hsd_scalars_to_device(pdhg_solver_state_t *state)
{
    double h[HSD_SCALARS_COUNT];
    h[HSD_PRIMAL_SCALE    + HSD_CURRENT]   = state->current_primal_scale;
    h[HSD_PRIMAL_SCALE    + HSD_REFLECTED]  = state->reflected_primal_scale;
    h[HSD_PRIMAL_SCALE    + HSD_PDHG]       = state->pdhg_primal_scale;
    h[HSD_PRIMAL_SCALE    + HSD_INITIAL]    = state->initial_primal_scale;
    h[HSD_DUAL_CONSTRAINT + HSD_CURRENT]    = state->current_dual_constraint_offset;
    h[HSD_DUAL_CONSTRAINT + HSD_REFLECTED]  = state->reflected_dual_constraint_offset;
    h[HSD_DUAL_CONSTRAINT + HSD_PDHG]       = state->pdhg_dual_constraint_offset;
    h[HSD_DUAL_CONSTRAINT + HSD_INITIAL]    = state->initial_dual_constraint_offset;
    h[HSD_DUAL_BOUND      + HSD_CURRENT]    = state->current_dual_bound_offset;
    h[HSD_DUAL_BOUND      + HSD_REFLECTED]  = state->reflected_dual_bound_offset;
    h[HSD_DUAL_BOUND      + HSD_PDHG]       = state->pdhg_dual_bound_offset;
    h[HSD_DUAL_BOUND      + HSD_INITIAL]    = state->initial_dual_bound_offset;
    CUDA_CHECK(cudaMemcpy(state->hsd_scalars_d, h, HSD_SCALARS_COUNT * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemsetAsync(state->cone_newton_constraint_d, 0, CONE_BUFFER_SIZE * sizeof(double), state->stream_A));
    CUDA_CHECK(cudaMemsetAsync(state->cone_newton_bound_d, 0, CONE_BUFFER_SIZE * sizeof(double), state->stream_A));
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

    double delta_primal_scale = state->reflected_primal_scale - state->pdhg_primal_scale;
    double delta_constraint_offset = state->reflected_dual_constraint_offset - state->pdhg_dual_constraint_offset;
    double delta_bound_offset = state->reflected_dual_bound_offset - state->pdhg_dual_bound_offset;

    compute_fixed_point_error_kernel<<<1, 1, 0, state->stream_A>>>(
        state->fixed_point_error_d,
        state->workspace_scalars_d,
        state->step_size_primal,
        state->step_size_dual,
        state->step_size_homogeneous,
        delta_primal_scale,
        delta_constraint_offset,
        delta_bound_offset);

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaEventSynchronize(state->event_A));
    CUDA_CHECK(cudaMemcpy(&state->fixed_point_error, state->fixed_point_error_d, sizeof(double), cudaMemcpyDeviceToHost));
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

    compute_restart_distances_kernel<<<1, 1, 0, state->stream_A>>>(
        state->restart_distances_d,
        state->workspace_scalars_d);

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaEventSynchronize(state->event_A));

    double restart_distances_h[RESTART_DISTANCE_COUNT];
    CUDA_CHECK(cudaMemcpy(restart_distances_h, state->restart_distances_d,
                          RESTART_DISTANCE_COUNT * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double primal_dist = restart_distances_h[RESTART_DISTANCE_PRIMAL];
    double dual_dist = restart_distances_h[RESTART_DISTANCE_DUAL];
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

    state->initial_primal_scale = state->pdhg_primal_scale;
    state->current_primal_scale = state->pdhg_primal_scale;
    state->reflected_primal_scale = state->pdhg_primal_scale;

    state->initial_dual_constraint_offset = state->pdhg_dual_constraint_offset;
    state->current_dual_constraint_offset = state->pdhg_dual_constraint_offset;
    state->reflected_dual_constraint_offset = state->pdhg_dual_constraint_offset;

    state->initial_dual_bound_offset = state->pdhg_dual_bound_offset;
    state->current_dual_bound_offset = state->pdhg_dual_bound_offset;
    state->reflected_dual_bound_offset = state->pdhg_dual_bound_offset;

    sync_hsd_scalars_to_device(state);

    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

static void normalize_hsd_variables(pdhg_solver_state_t *state)
{
    double tau = state->pdhg_primal_scale;
    if (!isfinite(tau) || tau <= 1.0)
        return;
    double inv_tau = 1.0 / tau;
    CUDA_CHECK(cudaMemcpyAsync(state->workspace_scalars_d + WORKSPACE_SCALAR_INV_TAU, &inv_tau, sizeof(double),
                                cudaMemcpyHostToDevice, state->stream_A));
    CUBLAS_CHECK(cublasDscal(state->blas_handle_A,
                             state->num_variables,
                             state->workspace_scalars_d + WORKSPACE_SCALAR_INV_TAU,
                             state->pdhg_primal_solution, 1));
    CUBLAS_CHECK(cublasDscal(state->blas_handle_A,
                             state->num_constraints,
                             state->workspace_scalars_d + WORKSPACE_SCALAR_INV_TAU,
                             state->pdhg_dual_solution, 1));
    CUBLAS_CHECK(cublasDscal(state->blas_handle_A,
                             state->num_variables,
                             state->workspace_scalars_d + WORKSPACE_SCALAR_INV_TAU,
                             state->pdhg_dual_slack, 1));
    state->pdhg_primal_scale = 1.0;
    state->pdhg_dual_constraint_offset *= inv_tau;
    state->pdhg_dual_bound_offset *= inv_tau;
}

static cuhsdlp_result_t *create_result_from_state(pdhg_solver_state_t *state, const lp_problem_t *original_problem)
{
    cuhsdlp_result_t *results = (cuhsdlp_result_t *)safe_calloc(1, sizeof(cuhsdlp_result_t));

    rescale_solution(state);
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));

    results->primal_solution = (double *)safe_malloc(state->var_bytes);
    results->dual_solution = (double *)safe_malloc(state->con_bytes);
    results->reduced_cost = (double *)safe_malloc(state->var_bytes);

    CUDA_CHECK(cudaMemcpy(results->primal_solution,
                          state->pdhg_primal_solution,
                          state->var_bytes, 
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->dual_solution,
                          state->pdhg_dual_solution,
                          state->con_bytes, 
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->reduced_cost,
                          state->pdhg_dual_slack,
                          state->var_bytes, 
                          cudaMemcpyDeviceToHost));

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
    if (state->spmv_power_iteration_next_vector)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_power_iteration_next_vector));
    if (state->spmv_power_iteration_product)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->spmv_power_iteration_product));
    if (state->primal_spmv_buffer)
        CUDA_CHECK(cudaFree(state->primal_spmv_buffer));
    if (state->dual_spmv_buffer)
        CUDA_CHECK(cudaFree(state->dual_spmv_buffer));
    if (state->primal_vec_buffer)
        CUDA_CHECK(cudaFree(state->primal_vec_buffer));
    if (state->dual_vec_buffer)
        CUDA_CHECK(cudaFree(state->dual_vec_buffer));
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

    if (state->variable_lower_bound)
        CUDA_CHECK(cudaFree(state->variable_lower_bound));
    if (state->variable_upper_bound)
        CUDA_CHECK(cudaFree(state->variable_upper_bound));
    if (state->objective_vector)
        CUDA_CHECK(cudaFree(state->objective_vector));
    if (state->constraint_lower_bound)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound));
    if (state->constraint_upper_bound)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound));
    if (state->constraint_matrix->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix->row_ptr));
    if (state->constraint_matrix->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix->col_ind));
    if (state->constraint_matrix->row_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix->row_ind));
    if (state->constraint_matrix->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix->val));
    if (state->constraint_matrix->transpose_map)
        CUDA_CHECK(cudaFree(state->constraint_matrix->transpose_map));
    if (state->constraint_matrix_t->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ptr));
    if (state->constraint_matrix_t->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->col_ind));
    if (state->constraint_matrix_t->row_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ind));
    if (state->constraint_matrix_t->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->val));
    if (state->constraint_matrix_t->transpose_map)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->transpose_map));
    if (state->initial_primal_solution)
        CUDA_CHECK(cudaFree(state->initial_primal_solution));
    if (state->current_primal_solution)
        CUDA_CHECK(cudaFree(state->current_primal_solution));
    if (state->pdhg_primal_solution)
        CUDA_CHECK(cudaFree(state->pdhg_primal_solution));
    if (state->reflected_primal_solution)
        CUDA_CHECK(cudaFree(state->reflected_primal_solution));
    if (state->primal_product)
        CUDA_CHECK(cudaFree(state->primal_product));
    if (state->initial_dual_solution)
        CUDA_CHECK(cudaFree(state->initial_dual_solution));
    if (state->current_dual_solution)
        CUDA_CHECK(cudaFree(state->current_dual_solution));
    if (state->pdhg_dual_solution)
        CUDA_CHECK(cudaFree(state->pdhg_dual_solution));
    if (state->reflected_dual_solution)
        CUDA_CHECK(cudaFree(state->reflected_dual_solution));
    if (state->dual_product)
        CUDA_CHECK(cudaFree(state->dual_product));
    if (state->constraint_rescaling)
        CUDA_CHECK(cudaFree(state->constraint_rescaling));
    if (state->variable_rescaling)
        CUDA_CHECK(cudaFree(state->variable_rescaling));
    if (state->primal_residual)
        CUDA_CHECK(cudaFree(state->primal_residual));
    if (state->dual_residual)
        CUDA_CHECK(cudaFree(state->dual_residual));
    if (state->delta_primal_solution)
        CUDA_CHECK(cudaFree(state->delta_primal_solution));
    if (state->delta_dual_solution)
        CUDA_CHECK(cudaFree(state->delta_dual_solution));
    if (state->current_dual_slack)
        CUDA_CHECK(cudaFree(state->current_dual_slack));
    if (state->initial_dual_slack)
        CUDA_CHECK(cudaFree(state->initial_dual_slack));
    if (state->pdhg_dual_slack)
        CUDA_CHECK(cudaFree(state->pdhg_dual_slack));
    if (state->reflected_dual_slack)
        CUDA_CHECK(cudaFree(state->reflected_dual_slack));
    if (state->delta_dual_slack)
        CUDA_CHECK(cudaFree(state->delta_dual_slack));
    if (state->power_iteration_vector_d)
        CUDA_CHECK(cudaFree(state->power_iteration_vector_d));
    if (state->hsd_scalars_d)
        CUDA_CHECK(cudaFree(state->hsd_scalars_d));
    if (state->cone_newton_constraint_d)
        CUDA_CHECK(cudaFree(state->cone_newton_constraint_d));
    if (state->cone_newton_bound_d)
        CUDA_CHECK(cudaFree(state->cone_newton_bound_d));
    if (state->cub_reduce_workspace)
        CUDA_CHECK(cudaFree(state->cub_reduce_workspace));
    if (state->cub_reduce_workspace_bound)
        CUDA_CHECK(cudaFree(state->cub_reduce_workspace_bound));
    if (state->workspace_scalars_d)
        CUDA_CHECK(cudaFree(state->workspace_scalars_d));
    if (state->norm_results_d)
        CUDA_CHECK(cudaFree(state->norm_results_d));
    if (state->rescaling_factors_d)
        CUDA_CHECK(cudaFree(state->rescaling_factors_d));
    if (state->restart_distances_d)
        CUDA_CHECK(cudaFree(state->restart_distances_d));
    if (state->fixed_point_error_d)
        CUDA_CHECK(cudaFree(state->fixed_point_error_d));
    if (state->residual_metrics_d)
        CUDA_CHECK(cudaFree(state->residual_metrics_d));

    if (state->constraint_matrix)
        free(state->constraint_matrix);
    if (state->constraint_matrix_t)
        free(state->constraint_matrix_t);

    free(state);
}

__global__ void build_row_ind(const int *__restrict__ row_ptr,
                              int num_rows,
                              int nnz,
                              int *__restrict__ row_ind)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz)
        return;

    int lo = 0, hi = num_rows;
    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if (row_ptr[mid + 1] <= k)
            lo = mid + 1;
        else
            hi = mid;
    }
    row_ind[k] = lo;
}

__global__ void build_transpose_map(
    const int *__restrict__ A_row_ind,
    const int *__restrict__ A_col_ind,
    const int *__restrict__ At_row_ptr,
    const int *__restrict__ At_col_ind,
    int nnz,
    int *__restrict__ A_to_At)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz)
        return;

    int i = A_row_ind[k];
    int j = A_col_ind[k];

    int lo = At_row_ptr[j];
    int hi = At_row_ptr[j + 1] - 1;

    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        int col = At_col_ind[mid];
        if (col == i)
        {
            A_to_At[k] = mid;
            return;
        }
        else if (col < i)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
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
        scale = cone_state[CONE_SCALE];
        tau_lo = 0.0;
        tau_hi = INFINITY;
        neg_scalar = step_size * hsd_scalars[HSD_PRIMAL_SCALE + HSD_REFLECTED]
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
        cone_state[CONE_SCALE] = scale;
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

    double tau_star = cone_state[CONE_SCALE];
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
                           - dual_step * hsd_scalars_d[HSD_PRIMAL_SCALE + HSD_REFLECTED];
        double projected = unprojected + cone_newton_buffer[CONE_SCALE];
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
    double projected = fmax(0.0, hsd_scalars_d[HSD_PRIMAL_SCALE + HSD_CURRENT] + primal_step * gradient);

    if (is_major)
    {
        hsd_scalars_d[HSD_PRIMAL_SCALE + HSD_PDHG] = projected;
    }

    int k = inner_count + k_offset;
    double weight = (double)(k) / (double)(k + 1);
    double reflected = 2.0 * projected - hsd_scalars_d[HSD_PRIMAL_SCALE + HSD_CURRENT];
    double mixed = reflection_coefficient * reflected + (1.0 - reflection_coefficient) * hsd_scalars_d[HSD_PRIMAL_SCALE + HSD_CURRENT];
    hsd_scalars_d[HSD_PRIMAL_SCALE + HSD_CURRENT] = weight * mixed + (1.0 - weight) * hsd_scalars_d[HSD_PRIMAL_SCALE + HSD_INITIAL];
    hsd_scalars_d[HSD_PRIMAL_SCALE + HSD_REFLECTED] = reflected;
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
    double tau_refl = hsd_scalars_d[HSD_PRIMAL_SCALE + HSD_REFLECTED];
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
    double step_size_primal,
    double step_size_dual,
    double step_size_homogeneous,
    double delta_primal_scale,
    double delta_constraint_offset,
    double delta_bound_offset)
{
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

__global__ void compute_restart_distances_kernel(
    double *restart_distances_d,
    const double *workspace_scalars_d)
{
    double primal_dist = workspace_scalars_d[WORKSPACE_SCALAR_PRIMAL_NORM];
    double dual_norm = workspace_scalars_d[WORKSPACE_SCALAR_DUAL_NORM];
    double slack_norm = workspace_scalars_d[WORKSPACE_SCALAR_CROSS_TERM];
    double dual_dist = sqrt(dual_norm * dual_norm + slack_norm * slack_norm);

    restart_distances_d[RESTART_DISTANCE_PRIMAL] = primal_dist;
    restart_distances_d[RESTART_DISTANCE_DUAL] = dual_dist;
}
