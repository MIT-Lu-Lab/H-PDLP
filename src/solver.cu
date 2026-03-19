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

__global__ void build_row_ind(const int *__restrict__ row_ptr,
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
__global__ void fill_finite_bounds_kernel(const double *__restrict__ lower_bound,
                                          const double *__restrict__ upper_bound,
                                          double *__restrict__ lower_bound_finite_val,
                                          double *__restrict__ upper_bound_finite_val,
                                          int num_elements);
__global__ void compute_next_primal_solution_kernel(double *__restrict__ current_primal,
                                                    double *__restrict__ reflected_primal,
                                                    const double *__restrict__ initial_primal,
                                                    const double *__restrict__ dual_product,
                                                    const double *__restrict__ objective,
                                                    const double *__restrict__ var_lb,
                                                    const double *__restrict__ var_ub,
                                                    int n,
                                                    double step_size,
                                                    int base_count,
                                                    int k_offset,
                                                    double reflection_coeff);
__global__ void compute_next_primal_solution_major_kernel(double *__restrict__ current_primal,
                                                          double *__restrict__ pdhg_primal,
                                                          double *__restrict__ reflected_primal,
                                                          const double *__restrict__ initial_primal,
                                                          const double *__restrict__ dual_product,
                                                          const double *__restrict__ objective,
                                                          const double *__restrict__ var_lb,
                                                          const double *__restrict__ var_ub,
                                                          int n,
                                                          double step_size,
                                                          double *__restrict__ dual_slack,
                                                          int base_count,
                                                          int k_offset,
                                                          double reflection_coeff);
__global__ void compute_next_dual_solution_kernel(double *__restrict__ current_dual,
                                                  const double *__restrict__ initial_dual,
                                                  const double *__restrict__ primal_product,
                                                  const double *__restrict__ const_lb,
                                                  const double *__restrict__ const_ub,
                                                  int n,
                                                  double step_size,
                                                  int base_count,
                                                  int k_offset,
                                                  double reflection_coeff);
__global__ void compute_next_dual_solution_major_kernel(double *__restrict__ current_dual,
                                                        double *__restrict__ pdhg_dual,
                                                        double *__restrict__ reflected_dual,
                                                        const double *__restrict__ initial_dual,
                                                        const double *__restrict__ primal_product,
                                                        const double *__restrict__ const_lb,
                                                        const double *__restrict__ const_ub,
                                                        int n,
                                                        double step_size,
                                                        int base_count,
                                                        int k_offset,
                                                        double reflection_coeff);
__global__ void compute_delta_solution_kernel(const double *__restrict__ initial_primal,
                                              const double *__restrict__ pdhg_primal,
                                              double *__restrict__ delta_primal,
                                              const double *__restrict__ initial_dual,
                                              const double *__restrict__ pdhg_dual,
                                              double *__restrict__ delta_dual,
                                              int n_vars,
                                              int n_cons);
__global__ void rescale_solution_kernel(double *__restrict__ primal_solution,
                                        double *__restrict__ dual_solution,
                                        const double *__restrict__ variable_rescaling,
                                        const double *__restrict__ constraint_rescaling,
                                        const double objective_vector_rescaling,
                                        const double constraint_bound_rescaling,
                                        int n_vars,
                                        int n_cons);
__global__ void compute_and_rescale_reduced_cost_kernel(double *__restrict__ reduced_cost,
                                                        const double *__restrict__ objective,
                                                        const double *__restrict__ dual_product,
                                                        const double *__restrict__ variable_rescaling,
                                                        const double objective_vector_rescaling,
                                                        const double constraint_bound_rescaling,
                                                        const double *__restrict__ variable_lower_bound,
                                                        const double *__restrict__ variable_upper_bound,
                                                        int n_vars);
__global__ void fill_ones_kernel(double *__restrict__ arr, int n);

static void check_params_validity(const pdhg_parameters_t *params);
static void compute_fixed_point_error(pdhg_solver_state_t *state);
static void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
static pdhg_solver_state_t *initialize_solver_state(lp_problem_t *working_problem,
                                                    const pdhg_parameters_t *params);
static cuhsdlp_result_t *create_result_from_state(pdhg_solver_state_t *state, const lp_problem_t *original_problem);
static void rescale_solution(pdhg_solver_state_t *state);
void compute_next_primal_solution(pdhg_solver_state_t *state,
                                  const int k_offset,
                                  const double reflection_coefficient,
                                  bool is_major);
void compute_next_dual_solution(pdhg_solver_state_t *state,
                                const int k_offset,
                                const double reflection_coefficient,
                                bool is_major);
void pdhg_solver_state_free(pdhg_solver_state_t *state);

cuhsdlp_result_t *optimize(const pdhg_parameters_t *params, lp_problem_t *original_problem)
{
    check_params_validity(params);
    print_initial_info(params, original_problem);

    cuhsdlp_presolve_info_t *presolve_info = NULL;
    lp_problem_t *working_problem = original_problem;

    if (params->presolve)
    {
        presolve_info = pslp_presolve(original_problem, params);
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
    display_iteration_stats(state, params->verbose);

    initialize_step_size_and_primal_weight(state, params);

    state->start_time = clock();
    bool do_restart = false;

    while (state->total_count < params->termination_criteria.iteration_limit)
    {
        compute_next_primal_solution(state, 1, params->reflection_coefficient, true);
        compute_next_dual_solution(state, 1, params->reflection_coefficient, true);

        if (do_restart)
        {
            compute_fixed_point_error(state);
            state->initial_fixed_point_error = state->fixed_point_error;
            do_restart = false;
        }

        for (int i = 2; i <= params->termination_evaluation_frequency - 1; i++)
        {
            compute_next_primal_solution(state, i, params->reflection_coefficient, false);
            compute_next_dual_solution(state, i, params->reflection_coefficient, false);
        }

        compute_next_primal_solution(
            state, params->termination_evaluation_frequency, params->reflection_coefficient, true);
        compute_next_dual_solution(
            state, params->termination_evaluation_frequency, params->reflection_coefficient, true);

        compute_fixed_point_error(state);
        compute_residual(state, params->optimality_norm);
        state->inner_count += params->termination_evaluation_frequency;
        state->total_count += params->termination_evaluation_frequency;

        if (state->total_count % get_print_frequency(state->total_count) == 0)
        {
            display_iteration_stats(state, params->verbose);
        }

        check_termination_criteria(state, &params->termination_criteria);
        if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
        {
            break;
        }

        do_restart =
            should_do_adaptive_restart(state, &params->restart_params, params->termination_evaluation_frequency);
        if (do_restart)
        {
            perform_restart(state, params);
        }
    }

    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_residual(state, params->optimality_norm);
        display_iteration_stats(state, params->verbose);
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

void compute_next_primal_solution(pdhg_solver_state_t *state,
                                  const int k_offset,
                                  const double reflection_coefficient,
                                  bool is_major)
{
    double primal_step = state->step_size / state->primal_weight;

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->current_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle_At,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matAt,
                                state->vec_dual_sol,
                                &HOST_ZERO,
                                state->vec_dual_prod,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->dual_spmv_buffer));

    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));

    if (is_major)
    {
        compute_next_primal_solution_major_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->current_primal_solution,
            state->pdhg_primal_solution,
            state->reflected_primal_solution,
            state->initial_primal_solution,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            primal_step,
            state->dual_slack,
            state->inner_count,
            k_offset,
            reflection_coefficient);
    }
    else
    {
        compute_next_primal_solution_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->current_primal_solution,
            state->reflected_primal_solution,
            state->initial_primal_solution,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            primal_step,
            state->inner_count,
            k_offset,
            reflection_coefficient);
    }
}

void compute_next_dual_solution(pdhg_solver_state_t *state,
                                const int k_offset,
                                const double reflection_coefficient,
                                bool is_major)
{
    double dual_step = state->step_size * state->primal_weight;

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, state->reflected_primal_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle_A,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matA,
                                state->vec_primal_sol,
                                &HOST_ZERO,
                                state->vec_primal_prod,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->primal_spmv_buffer));

    if (is_major)
    {
        compute_next_dual_solution_major_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->current_dual_solution,
            state->pdhg_dual_solution,
            state->reflected_dual_solution,
            state->initial_dual_solution,
            state->primal_product,
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->num_constraints,
            dual_step,
            state->inner_count,
            k_offset,
            reflection_coefficient);
    }
    else
    {
        compute_next_dual_solution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->current_dual_solution,
            state->initial_dual_solution,
            state->primal_product,
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->num_constraints,
            dual_step,
            state->inner_count,
            k_offset,
            reflection_coefficient);
    }

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
}

static void compute_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->pdhg_primal_solution,
        state->reflected_primal_solution,
        state->delta_primal_solution,
        state->pdhg_dual_solution,
        state->reflected_dual_solution,
        state->delta_dual_solution,
        state->num_variables,
        state->num_constraints);

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->delta_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle_At,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matAt,
                                state->vec_dual_sol,
                                &HOST_ZERO,
                                state->vec_dual_prod,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->dual_spmv_buffer));

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables, state->delta_primal_solution, 1, state->scalar_pool_d + 0));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_constraints, state->delta_dual_solution, 1, state->scalar_pool_d + 1));

    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));

    CUBLAS_CHECK(cublasDdot(state->blas_handle_A,
                            state->num_variables,
                            state->dual_product,
                            1,
                            state->delta_primal_solution,
                            1,
                            state->scalar_pool_d + 2));

    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));

    double h_buf[3];
    CUDA_CHECK(cudaMemcpy(h_buf, state->scalar_pool_d, 3 * sizeof(double), cudaMemcpyDeviceToHost));

    double primal_norm = h_buf[0];
    double dual_norm = h_buf[1];
    double cross_term = h_buf[2];

    double movement = primal_norm * primal_norm * state->primal_weight + dual_norm * dual_norm / state->primal_weight;
    double interaction = 2.0 * state->step_size * cross_term;

    state->fixed_point_error = sqrt(fmax(0.0, movement + interaction));
}

static void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->initial_primal_solution,
        state->pdhg_primal_solution,
        state->delta_primal_solution,
        state->initial_dual_solution,
        state->pdhg_dual_solution,
        state->delta_dual_solution,
        state->num_variables,
        state->num_constraints);

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_variables, state->delta_primal_solution, 1, state->scalar_pool_d + 0));
    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A, state->num_constraints, state->delta_dual_solution, 1, state->scalar_pool_d + 1));

    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));

    double h_buf[2];
    CUDA_CHECK(cudaMemcpy(h_buf, state->scalar_pool_d, 2 * sizeof(double), cudaMemcpyDeviceToHost));

    double primal_dist = h_buf[0];
    double dual_dist = h_buf[1];
    double ratio_infeas = state->relative_dual_residual / state->relative_primal_residual;

    if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 && ratio_infeas > 1e-8 &&
        ratio_infeas < 1e8)
    {
        double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);
        state->primal_weight_error_sum *= params->restart_params.i_smooth;
        state->primal_weight_error_sum += error;
        double delta_error = error - state->primal_weight_last_error;
        state->primal_weight *=
            exp(params->restart_params.k_p * error + params->restart_params.k_i * state->primal_weight_error_sum +
                params->restart_params.k_d * delta_error);
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

    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution,
                          state->pdhg_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution,
                          state->pdhg_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution,
                          state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution,
                          state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    if (state->constraint_matrix->num_nonzeros == 0)
    {
        state->step_size = 1.0;
    }
    else
    {
        double max_sv = estimate_maximum_singular_value(state->sparse_handle_A,
                                                        state->blas_handle_A,
                                                        state->curand_gen,
                                                        state->constraint_matrix,
                                                        state->constraint_matrix_t,
                                                        params->sv_max_iter,
                                                        params->sv_tol);
        state->step_size = 0.998 / max_sv;
    }

    if (params->bound_objective_rescaling)
    {
        state->primal_weight = 1.0;
    }
    else
    {
        state->primal_weight = (state->objective_vector_norm + 1.0) / (state->constraint_bound_norm + 1.0);
    }
    state->best_primal_weight = state->primal_weight;
}

static pdhg_solver_state_t *initialize_solver_state(lp_problem_t *working_problem,
                                                    const pdhg_parameters_t *params)
{
    pdhg_solver_state_t *state = (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    int n_vars = working_problem->num_variables;
    int n_cons = working_problem->num_constraints;
    int nnz = working_problem->constraint_matrix_num_nonzeros;
    size_t var_bytes = n_vars * sizeof(double);
    size_t con_bytes = n_cons * sizeof(double);

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
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

    state->num_blocks_primal = (state->num_variables + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_dual = (state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_primal_dual =
        (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_nnz = (state->constraint_matrix->num_nonzeros + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

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
        state->reduce_workspace_bytes = (tb_sum > tb_max) ? tb_sum : tb_max;
        CUDA_CHECK(cudaMalloc(&state->reduce_workspace_d, state->reduce_workspace_bytes));
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
    ALLOC_ZERO(state->dual_product, var_bytes);
    ALLOC_ZERO(state->dual_slack, var_bytes);
    ALLOC_ZERO(state->dual_residual, var_bytes);
    ALLOC_ZERO(state->delta_primal_solution, var_bytes);

    ALLOC_ZERO(state->initial_dual_solution, con_bytes);
    ALLOC_ZERO(state->current_dual_solution, con_bytes);
    ALLOC_ZERO(state->pdhg_dual_solution, con_bytes);
    ALLOC_ZERO(state->reflected_dual_solution, con_bytes);
    ALLOC_ZERO(state->primal_product, con_bytes);
    ALLOC_ZERO(state->primal_slack, con_bytes);
    ALLOC_ZERO(state->primal_residual, con_bytes);
    ALLOC_ZERO(state->delta_dual_solution, con_bytes);

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

    CUDA_CHECK(cudaMalloc(&state->scalar_pool_d, 8 * sizeof(double)));

    compute_objective_and_bound_norms(state, params);

    CUDA_CHECK(cudaMalloc(&state->constraint_rescaling, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->variable_rescaling, var_bytes));

    rescale_problem(params, state);

    CUDA_CHECK(cudaMemcpy(
        state->current_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(state->current_dual_solution, state->initial_dual_solution, con_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(state->pdhg_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(state->pdhg_dual_solution, state->initial_dual_solution, con_bytes, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMalloc(&state->constraint_lower_bound_finite_val, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->constraint_upper_bound_finite_val, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->variable_lower_bound_finite_val, var_bytes));
    CUDA_CHECK(cudaMalloc(&state->variable_upper_bound_finite_val, var_bytes));

    fill_finite_bounds_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(state->constraint_lower_bound,
                                                                             state->constraint_upper_bound,
                                                                             state->constraint_lower_bound_finite_val,
                                                                             state->constraint_upper_bound_finite_val,
                                                                             n_cons);

    fill_finite_bounds_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(state->variable_lower_bound,
                                                                               state->variable_upper_bound,
                                                                               state->variable_lower_bound_finite_val,
                                                                               state->variable_upper_bound_finite_val,
                                                                               n_vars);

    CUDA_CHECK(cudaFree(state->constraint_matrix->row_ind));
    state->constraint_matrix->row_ind = NULL;
    CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ind));
    state->constraint_matrix_t->row_ind = NULL;
    CUDA_CHECK(cudaFree(state->constraint_matrix->transpose_map));
    state->constraint_matrix->transpose_map = NULL;

    state->best_primal_dual_residual_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size = 0.0;
    state->primal_weight_error_sum = 0.0;
    state->primal_weight_last_error = 0.0;

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

    CUSPARSE_CHECK(
        cusparseCreateDnVec(&state->vec_primal_sol, state->num_variables, state->pdhg_primal_solution, CUDA_R_64F));
    CUSPARSE_CHECK(
        cusparseCreateDnVec(&state->vec_dual_sol, state->num_constraints, state->pdhg_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(
        cusparseCreateDnVec(&state->vec_primal_prod, state->num_constraints, state->primal_product, CUDA_R_64F));
    CUSPARSE_CHECK(
        cusparseCreateDnVec(&state->vec_dual_prod, state->num_variables, state->dual_product, CUDA_R_64F));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle_A,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matA,
                                           state->vec_primal_sol,
                                           &HOST_ZERO,
                                           state->vec_primal_prod,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           &primal_spmv_buffer_size));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle_At,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matAt,
                                           state->vec_dual_sol,
                                           &HOST_ZERO,
                                           state->vec_dual_prod,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           &dual_spmv_buffer_size));

    CUDA_CHECK(cudaMalloc(&state->primal_spmv_buffer, primal_spmv_buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(state->sparse_handle_A,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matA,
                                           state->vec_primal_sol,
                                           &HOST_ZERO,
                                           state->vec_primal_prod,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           state->primal_spmv_buffer));

    CUDA_CHECK(cudaMalloc(&state->dual_spmv_buffer, dual_spmv_buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(state->sparse_handle_At,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           state->matAt,
                                           state->vec_dual_sol,
                                           &HOST_ZERO,
                                           state->vec_dual_prod,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           state->dual_spmv_buffer));

    CUDA_CHECK(cudaMalloc(&state->ones_primal_d, state->num_variables * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->ones_dual_d, state->num_constraints * sizeof(double)));

    fill_ones_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(state->ones_primal_d, state->num_variables);
    CUDA_CHECK(cudaGetLastError());
    fill_ones_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(state->ones_dual_d, state->num_constraints);
    CUDA_CHECK(cudaGetLastError());

    if (params->verbose)
    {
        printf("---------------------------------------------------------------------"
               "------------------\n");
        printf("%s | %s | %s | %s \n",
               "   runtime    ",
               "    objective     ",
               "  absolute residuals   ",
               "  relative residuals   ");
        printf("%s %s | %s %s | %s %s %s | %s %s %s \n",
               "  iter",
               "  time ",
               " pr obj ",
               "  du obj ",
               " pr res",
               " du res",
               "  gap  ",
               " pr res",
               " du res",
               "  gap  ");
        printf("---------------------------------------------------------------------"
               "------------------\n");
    }

    return state;
}

static cuhsdlp_result_t *create_result_from_state(pdhg_solver_state_t *state, const lp_problem_t *original_problem)
{
    cuhsdlp_result_t *results = (cuhsdlp_result_t *)safe_calloc(1, sizeof(cuhsdlp_result_t));

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->pdhg_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle_At,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                state->matAt,
                                state->vec_dual_sol,
                                &HOST_ZERO,
                                state->vec_dual_prod,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                state->dual_spmv_buffer));

    CUDA_CHECK(cudaStreamSynchronize(state->stream_At));

    compute_and_rescale_reduced_cost_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->dual_slack,
        state->objective_vector,
        state->dual_product,
        state->variable_rescaling,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->num_variables);

    rescale_solution(state);

    results->primal_solution = (double *)safe_malloc(state->num_variables * sizeof(double));
    results->dual_solution = (double *)safe_malloc(state->num_constraints * sizeof(double));
    results->reduced_cost = (double *)safe_malloc(state->num_variables * sizeof(double));

    CUDA_CHECK(cudaMemcpy(results->primal_solution,
                          state->pdhg_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->dual_solution,
                          state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        results->reduced_cost, state->dual_slack, state->num_variables * sizeof(double), cudaMemcpyDeviceToHost));

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

    return results;
}

static void rescale_solution(pdhg_solver_state_t *state)
{
    rescale_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->pdhg_primal_solution,
        state->pdhg_dual_solution,
        state->variable_rescaling,
        state->constraint_rescaling,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
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
    if (state->vec_primal_sol)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_primal_sol));
    if (state->vec_dual_sol)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_dual_sol));
    if (state->vec_primal_prod)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_primal_prod));
    if (state->vec_dual_prod)
        CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_dual_prod));
    if (state->primal_spmv_buffer)
        CUDA_CHECK(cudaFree(state->primal_spmv_buffer));
    if (state->dual_spmv_buffer)
        CUDA_CHECK(cudaFree(state->dual_spmv_buffer));
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
    if (state->constraint_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound_finite_val));
    if (state->constraint_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound_finite_val));
    if (state->variable_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_lower_bound_finite_val));
    if (state->variable_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_upper_bound_finite_val));
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
    if (state->dual_product)
        CUDA_CHECK(cudaFree(state->dual_product));
    if (state->initial_dual_solution)
        CUDA_CHECK(cudaFree(state->initial_dual_solution));
    if (state->current_dual_solution)
        CUDA_CHECK(cudaFree(state->current_dual_solution));
    if (state->pdhg_dual_solution)
        CUDA_CHECK(cudaFree(state->pdhg_dual_solution));
    if (state->reflected_dual_solution)
        CUDA_CHECK(cudaFree(state->reflected_dual_solution));
    if (state->primal_product)
        CUDA_CHECK(cudaFree(state->primal_product));
    if (state->constraint_rescaling)
        CUDA_CHECK(cudaFree(state->constraint_rescaling));
    if (state->variable_rescaling)
        CUDA_CHECK(cudaFree(state->variable_rescaling));
    if (state->primal_slack)
        CUDA_CHECK(cudaFree(state->primal_slack));
    if (state->dual_slack)
        CUDA_CHECK(cudaFree(state->dual_slack));
    if (state->primal_residual)
        CUDA_CHECK(cudaFree(state->primal_residual));
    if (state->dual_residual)
        CUDA_CHECK(cudaFree(state->dual_residual));
    if (state->delta_primal_solution)
        CUDA_CHECK(cudaFree(state->delta_primal_solution));
    if (state->delta_dual_solution)
        CUDA_CHECK(cudaFree(state->delta_dual_solution));
    if (state->ones_primal_d)
        CUDA_CHECK(cudaFree(state->ones_primal_d));
    if (state->ones_dual_d)
        CUDA_CHECK(cudaFree(state->ones_dual_d));
    if (state->reduce_workspace_d)
        CUDA_CHECK(cudaFree(state->reduce_workspace_d));
    if (state->scalar_pool_d)
        CUDA_CHECK(cudaFree(state->scalar_pool_d));

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

__global__ void fill_finite_bounds_kernel(const double *__restrict__ lb,
                                          const double *__restrict__ ub,
                                          double *__restrict__ lb_finite,
                                          double *__restrict__ ub_finite,
                                          int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements)
        return;

    double Li = lb[i];
    double Ui = ub[i];

    lb_finite[i] = isfinite(Li) ? Li : 0.0;
    ub_finite[i] = isfinite(Ui) ? Ui : 0.0;
}

__global__ void compute_next_primal_solution_kernel(double *__restrict__ current_primal,
                                                    double *__restrict__ reflected_primal,
                                                    const double *__restrict__ initial_primal,
                                                    const double *__restrict__ dual_product,
                                                    const double *__restrict__ objective,
                                                    const double *__restrict__ var_lb,
                                                    const double *__restrict__ var_ub,
                                                    int n,
                                                    double step_size,
                                                    int base_count,
                                                    int k_offset,
                                                    double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int current_k = base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
        reflected_primal[i] = 2.0 * temp_proj - current_primal[i];
        double reflected = reflection_coeff * reflected_primal[i] + (1.0 - reflection_coeff) * current_primal[i];
        current_primal[i] = weight * reflected + (1.0 - weight) * initial_primal[i];
    }
}

__global__ void compute_next_primal_solution_major_kernel(double *__restrict__ current_primal,
                                                          double *__restrict__ pdhg_primal,
                                                          double *__restrict__ reflected_primal,
                                                          const double *__restrict__ initial_primal,
                                                          const double *__restrict__ dual_product,
                                                          const double *__restrict__ objective,
                                                          const double *__restrict__ var_lb,
                                                          const double *__restrict__ var_ub,
                                                          int n,
                                                          double step_size,
                                                          double *__restrict__ dual_slack,
                                                          int base_count,
                                                          int k_offset,
                                                          double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int current_k = base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        pdhg_primal[i] = fmax(var_lb[i], fmin(temp, var_ub[i]));
        dual_slack[i] = (pdhg_primal[i] - temp) / step_size;
        reflected_primal[i] = 2.0 * pdhg_primal[i] - current_primal[i];
        double reflected = reflection_coeff * reflected_primal[i] + (1.0 - reflection_coeff) * current_primal[i];
        current_primal[i] = weight * reflected + (1.0 - weight) * initial_primal[i];
    }
}

__global__ void compute_next_dual_solution_kernel(double *__restrict__ current_dual,
                                                  const double *__restrict__ initial_dual,
                                                  const double *__restrict__ primal_product,
                                                  const double *__restrict__ const_lb,
                                                  const double *__restrict__ const_ub,
                                                  int n,
                                                  double step_size,
                                                  int base_count,
                                                  int k_offset,
                                                  double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int current_k = base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        double reflected = reflection_coeff * (2.0 * (temp - temp_proj) * step_size - current_dual[i]) +
            (1.0 - reflection_coeff) * current_dual[i];
        current_dual[i] = weight * reflected + (1.0 - weight) * initial_dual[i];
    }
}

__global__ void compute_next_dual_solution_major_kernel(double *__restrict__ current_dual,
                                                        double *__restrict__ pdhg_dual,
                                                        double *__restrict__ reflected_dual,
                                                        const double *__restrict__ initial_dual,
                                                        const double *__restrict__ primal_product,
                                                        const double *__restrict__ const_lb,
                                                        const double *__restrict__ const_ub,
                                                        int n,
                                                        double step_size,
                                                        int base_count,
                                                        int k_offset,
                                                        double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int current_k = base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        pdhg_dual[i] = (temp - temp_proj) * step_size;
        reflected_dual[i] = 2.0 * pdhg_dual[i] - current_dual[i];
        double reflected = reflection_coeff * reflected_dual[i] + (1.0 - reflection_coeff) * current_dual[i];
        current_dual[i] = weight * reflected + (1.0 - weight) * initial_dual[i];
    }
}

__global__ void rescale_solution_kernel(double *__restrict__ primal_solution,
                                        double *__restrict__ dual_solution,
                                        const double *__restrict__ variable_rescaling,
                                        const double *__restrict__ constraint_rescaling,
                                        const double objective_vector_rescaling,
                                        const double constraint_bound_rescaling,
                                        int n_vars,
                                        int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        primal_solution[i] = primal_solution[i] / variable_rescaling[i] / constraint_bound_rescaling;
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        dual_solution[idx] = dual_solution[idx] / constraint_rescaling[idx] / objective_vector_rescaling;
    }
}

__global__ void compute_delta_solution_kernel(const double *__restrict__ initial_primal,
                                              const double *__restrict__ pdhg_primal,
                                              double *__restrict__ delta_primal,
                                              const double *__restrict__ initial_dual,
                                              const double *__restrict__ pdhg_dual,
                                              double *__restrict__ delta_dual,
                                              int n_vars,
                                              int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = (pdhg_primal[i] - initial_primal[i]);
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        delta_dual[idx] = (pdhg_dual[idx] - initial_dual[idx]);
    }
}

__global__ void compute_and_rescale_reduced_cost_kernel(double *__restrict__ reduced_cost,
                                                        const double *__restrict__ objective,
                                                        const double *__restrict__ dual_product,
                                                        const double *__restrict__ variable_rescaling,
                                                        const double objective_vector_rescaling,
                                                        const double constraint_bound_rescaling,
                                                        const double *__restrict__ variable_lower_bound,
                                                        const double *__restrict__ variable_upper_bound,
                                                        int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double rc = (objective[i] - dual_product[i]) * variable_rescaling[i] / objective_vector_rescaling;

        if (!isfinite(variable_lower_bound[i]))
        {
            rc = fmin(rc, 0.0);
        }
        if (!isfinite(variable_upper_bound[i]))
        {
            rc = fmax(rc, 0.0);
        }
        reduced_cost[i] = rc;
    }
}

