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

#include "utils.h"
#include <cub/cub.cuh>
#include <math.h>

#ifndef CUHSDLP_VERSION
#define CUHSDLP_VERSION "unknown"
#endif

__global__ void compute_objective_norm_entries_kernel(
    const double *__restrict__ obj_vec,
    double *__restrict__ entries,
    int n_vars,
    int use_linf);
__global__ void compute_bound_norm_entries_kernel(
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    double *__restrict__ entries,
    int n_cons,
    int use_linf);
__global__ void compute_residual_kernel(
    double *__restrict__ primal_res,
    const double *__restrict__ primal_prod,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    double *__restrict__ dual_res,
    const double *__restrict__ dual_prod,
    const double *__restrict__ dual_slack,
    const double *__restrict__ obj_vec,
    const double *__restrict__ cons_rescaling,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    const double *__restrict__ primal_sol,
    const double *__restrict__ hsd_scalars_d,
    double *__restrict__ homogeneous_dual_res,
    double *__restrict__ constraint_recession_res,
    double *__restrict__ variable_recession_res,
    int n_cons,
    int n_vars);
__global__ void compute_support_function_contrib_kernel(
    double *__restrict__ contrib,
    const double *__restrict__ y,
    const double *__restrict__ lb,
    const double *__restrict__ ub,
    int n);
__global__ void compute_residual_metrics_kernel(
    pdhg_metrics_t *metrics_d,
    const double *pdhg_reduce_d,
    const double *farkas_reduce_d,
    const double *hsd_scalars_d,
    double constraint_bound_rescaling,
    double objective_vector_rescaling,
    double original_bound_norm,
    double original_objective_norm,
    double original_constraint_norm,
    double objective_constant);
__global__ void normalize_power_iteration_vector_kernel(
    double *normalized_vector_d,
    const double *vector_d,
    const double *vector_norm_d,
    int n);
__global__ void compute_power_iteration_residual_kernel(
    double *residual_d,
    const double *vector_d,
    const double *normalized_vector_d,
    const double *sigma_squared_d,
    int n);
__global__ void abs_vector_inplace_kernel(
    double *__restrict__ arr, int n);
static void reduce_device_vector(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output,
    bool use_linf);
const double HOST_ONE = 1.0;
const double HOST_ZERO = 0.0;

void *safe_malloc(size_t size)
{
    if (size == 0)
        size = 1;

    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        perror("Fatal error: malloc failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_calloc(size_t num, size_t size)
{
    if (num == 0 || size == 0)
    {
        num = 1;
        size = 1;
    }

    void *ptr = calloc(num, size);
    if (ptr == NULL)
    {
        perror("Fatal error: calloc failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_realloc(void *ptr, size_t new_size)
{
    if (new_size == 0)
        new_size = 1;

    void *tmp = realloc(ptr, new_size);
    if (!tmp)
    {
        perror("Fatal error: realloc failed");
        exit(EXIT_FAILURE);
    }
    return tmp;
}

void set_default_parameters(pdhg_parameters_t *params)
{
    params->presolve = true;
    params->verbose = true;
    params->optimality_norm = NORM_TYPE_L2;
    params->reflection_coefficient = 1.0;
    params->termination_evaluation_frequency = 200;

    params->scaling_params.l_inf_ruiz_iterations = 10;
    params->scaling_params.has_pock_chambolle_alpha = true;
    params->scaling_params.pock_chambolle_alpha = 1.0;
    params->scaling_params.bound_objective_rescaling = true;
    params->scaling_params.matrix_zero_tol = 1e-9;

    params->power_iter_params.max_iter = 5000;
    params->power_iter_params.tol = 1e-4;

    params->step_size_params.step_size_factor = 0.998;
    params->step_size_params.objective_coupling_budget = 0.05;
    params->step_size_params.k_p = 0.99;
    params->step_size_params.k_i = 0.01;
    params->step_size_params.k_d = 0.0;
    params->step_size_params.i_smooth = 0.3;

    params->cone_params.max_iter = 30;
    params->cone_params.tol = 1e-6;

    params->termination_criteria.eps_optimal_relative = 1e-4;
    params->termination_criteria.eps_feasible_relative = 1e-4;
    params->termination_criteria.time_sec_limit = 3600.0;
    params->termination_criteria.iteration_limit = INT32_MAX;

    params->restart_params.artificial_restart_threshold = 0.36;
    params->restart_params.sufficient_reduction_for_restart = 0.2;
    params->restart_params.necessary_reduction_for_restart = 0.5;
}

const char *termination_reason_to_string(termination_reason_t reason)
{
    switch (reason)
    {
    case TERMINATION_REASON_OPTIMAL:
        return "OPTIMAL";
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
        return "PRIMAL_INFEASIBLE";
    case TERMINATION_REASON_DUAL_INFEASIBLE:
        return "DUAL_INFEASIBLE";
    case TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED:
        return "INFEASIBLE_OR_UNBOUNDED";
    case TERMINATION_REASON_TIME_LIMIT:
        return "TIME_LIMIT";
    case TERMINATION_REASON_ITERATION_LIMIT:
        return "ITERATION_LIMIT";
    case TERMINATION_REASON_NUMERICAL_FAILURE:
        return "NUMERICAL_FAILURE";
    case TERMINATION_REASON_UNSPECIFIED:
        return "UNSPECIFIED";
    default:
        return "UNKNOWN";
    }
}

#define PRINT_DIFF_INT(name, current, default_val) \
    do { \
        if ((current) != (default_val)) { \
            printf("  %-18s : %d\n", name, current); \
        } \
    } while(0)

#define PRINT_DIFF_DBL(name, current, default_val) \
    do { \
        if (fabs((current) - (default_val)) > 1e-9) { \
            printf("  %-18s : %.1e\n", name, (double)(current)); \
        } \
    } while(0)

#define PRINT_DIFF_BOOL(name, current, default_val) \
    do { \
        if ((current) != (default_val)) { \
            printf("  %-18s : %s\n", name, (current) ? "on" : "off"); \
        } \
    } while(0)

static void filter_constraint_matrix_entries(lp_problem_t *problem, const pdhg_parameters_t *params)
{
    const int num_rows = problem->num_constraints;
    const int nnz = problem->constraint_matrix_num_nonzeros;

    if (num_rows == 0 || nnz == 0)
        return;

    const int *row_ptr = problem->constraint_matrix_row_pointers;
    const int *col_ind = problem->constraint_matrix_col_indices;
    const double *vals = problem->constraint_matrix_values;

    int filtered_nnz = 0;
    for (int i = 0; i < num_rows; ++i)
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
            if (fabs(vals[k]) > params->scaling_params.matrix_zero_tol)
                ++filtered_nnz;

    if (filtered_nnz == nnz)
        return;

    const size_t alloc_nnz = (filtered_nnz > 0) ? (size_t)filtered_nnz : 1;
    int *new_row_ptr = (int *)safe_malloc((size_t)(num_rows + 1) * sizeof(int));
    int *new_col_ind = (int *)safe_malloc(alloc_nnz * sizeof(int));
    double *new_vals = (double *)safe_malloc(alloc_nnz * sizeof(double));

    int pos = 0;
    new_row_ptr[0] = 0;
    for (int i = 0; i < num_rows; ++i)
    {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        {
            if (fabs(vals[k]) <= params->scaling_params.matrix_zero_tol)
                continue;
            new_col_ind[pos] = col_ind[k];
            new_vals[pos] = vals[k];
            ++pos;
        }
        new_row_ptr[i + 1] = pos;
    }

    free(problem->constraint_matrix_row_pointers);
    free(problem->constraint_matrix_col_indices);
    free(problem->constraint_matrix_values);

    problem->constraint_matrix_row_pointers = new_row_ptr;
    problem->constraint_matrix_col_indices = new_col_ind;
    problem->constraint_matrix_values = new_vals;
    problem->constraint_matrix_num_nonzeros = filtered_nnz;

    if (((nnz - filtered_nnz) > 0) && params->verbose)
    {
        printf("Dropped %d near-zero %s (|value| <= %.1e) from the constraint matrix\n",
               nnz - filtered_nnz,
               (nnz - filtered_nnz == 1 ? "entry" : "entries"),
               params->scaling_params.matrix_zero_tol);
    }
}

void print_initial_info(const pdhg_parameters_t *params,
                        lp_problem_t *problem)
{
    pdhg_parameters_t default_params;
    set_default_parameters(&default_params);

    filter_constraint_matrix_entries(problem, params);

    if (!params->verbose)
    {
        return;
    }
    printf("-----------------------------------------------------------"
           "-----------------------------------------------------------------\n");
    printf("%*scuHSDLP v%s\n", (124 - 14) / 2, "", CUHSDLP_VERSION);
    printf("%*sA GPU-Accelerated First-Order LP Solver\n", (124 - 39) / 2, "");
    printf("%*s(c) Haihao Lu, Massachusetts Institute of Technology, 2026\n", (124 - 58) / 2, "");
    printf("-----------------------------------------------------------"
           "-----------------------------------------------------------------\n");
    printf("\n");

    printf(
        "Problem: %d rows, %d columns, %d nonzeros\n",
        problem->num_constraints,
        problem->num_variables,
        problem->constraint_matrix_num_nonzeros);

    printf("Settings:\n");
    printf("  iter_limit         : %d\n",
           params->termination_criteria.iteration_limit);
    printf("  time_limit         : %.2f sec\n",
           params->termination_criteria.time_sec_limit);
    printf("  eps_opt            : %.1e\n",
           params->termination_criteria.eps_optimal_relative);
    printf("  eps_feas           : %.1e\n",
           params->termination_criteria.eps_feasible_relative);
    if (params->optimality_norm != default_params.optimality_norm)
    {
        printf("  optimality_norm    : %s\n",
               params->optimality_norm == NORM_TYPE_L_INF ? "L_inf" : "L2");
    }

    PRINT_DIFF_INT("l_inf_ruiz_iter",
                   params->scaling_params.l_inf_ruiz_iterations,
                   default_params.scaling_params.l_inf_ruiz_iterations);
    PRINT_DIFF_DBL("pock_chambolle_alpha",
                   params->scaling_params.pock_chambolle_alpha,
                   default_params.scaling_params.pock_chambolle_alpha);
    PRINT_DIFF_BOOL("has_pock_chambolle_alpha",
                    params->scaling_params.has_pock_chambolle_alpha,
                    default_params.scaling_params.has_pock_chambolle_alpha);
    PRINT_DIFF_BOOL("bound_obj_rescaling",
                    params->scaling_params.bound_objective_rescaling,
                    default_params.scaling_params.bound_objective_rescaling);
    PRINT_DIFF_INT("power_iter_max_iter",
                   params->power_iter_params.max_iter,
                   default_params.power_iter_params.max_iter);
    PRINT_DIFF_DBL("power_iter_tol",
                   params->power_iter_params.tol,
                   default_params.power_iter_params.tol);
    PRINT_DIFF_INT("evaluation_freq",
                   params->termination_evaluation_frequency,
                   default_params.termination_evaluation_frequency);
    PRINT_DIFF_BOOL("presolve",
                    params->presolve,
                    default_params.presolve);
    PRINT_DIFF_DBL("matrix_zero_tol",
                   params->scaling_params.matrix_zero_tol,
                   default_params.scaling_params.matrix_zero_tol);
    PRINT_DIFF_INT("cone_newton_iter",
                   params->cone_params.max_iter,
                   default_params.cone_params.max_iter);
    PRINT_DIFF_DBL("cone_newton_tol",
                   params->cone_params.tol,
                   default_params.cone_params.tol);
}

#undef PRINT_DIFF_INT
#undef PRINT_DIFF_DBL
#undef PRINT_DIFF_BOOL

void compute_objective_and_bound_norms(
    pdhg_solver_state_t *state,
    norm_type_t optimality_norm,
    double *out_objective_norm,
    double *out_bound_norm)
{
    const int use_linf = (optimality_norm == NORM_TYPE_L_INF);

    double *norm_entries_vars = state->primal_vec_buffer_d;
    double *norm_entries_cons = state->dual_vec_buffer_d;

    compute_objective_norm_entries_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->objective_vector, norm_entries_vars, state->num_variables, use_linf);
    CUDA_CHECK(cudaGetLastError());

    compute_bound_norm_entries_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->constraint_lower_bound, state->constraint_upper_bound,
        norm_entries_cons, state->num_constraints, use_linf);
    CUDA_CHECK(cudaGetLastError());

    reduce_device_vector(state, norm_entries_vars, state->num_variables,
                         state->precond_objective_norm_mp, use_linf);
    reduce_device_vector(state, norm_entries_cons, state->num_constraints,
                         state->precond_bound_norm_mp, use_linf);

    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));

    if (!use_linf) {
        *state->precond_objective_norm_mp = sqrt(*state->precond_objective_norm_mp);
        *state->precond_bound_norm_mp = sqrt(*state->precond_bound_norm_mp);
    }

    *out_objective_norm = *state->precond_objective_norm_mp;
    *out_bound_norm = *state->precond_bound_norm_mp;
}

__global__ void compute_objective_norm_entries_kernel(
    const double *__restrict__ obj_vec,
    double *__restrict__ entries,
    int n_vars,
    int use_linf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vars) return;

    double v = obj_vec[i];
    entries[i] = use_linf ? fabs(v) : (v * v);
}

__global__ void compute_bound_norm_entries_kernel(
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    double *__restrict__ entries,
    int n_cons,
    int use_linf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons) return;

    double Li = cons_lb[i];
    double Ui = cons_ub[i];
    double acc = 0.0;
    if (use_linf)
    {
        if (isfinite(Li) && Li != Ui) acc = fmax(acc, fabs(Li));
        if (isfinite(Ui)) acc = fmax(acc, fabs(Ui));
    }
    else
    {
        if (isfinite(Li) && Li != Ui) acc += Li * Li;
        if (isfinite(Ui)) acc += Ui * Ui;
    }
    entries[i] = acc;
}

double estimate_maximum_singular_value(
    pdhg_solver_state_t *state,
    const power_iteration_parameters_t *power_iter_params)
{
    const int residual_check_interval = 5;
    int m = state->num_constraints;
    double *power_iteration_vector_d = state->power_iteration_vector_d;
    double *power_iteration_normalized_vector_d = state->power_iteration_normalized_vector_d;

    size_t curand_count = ((size_t)m + 1) & ~(size_t)1;
    CURAND_CHECK(curandGenerateNormalDouble(state->curand_gen, power_iteration_vector_d, curand_count, 0.0, 1.0));

    *state->power_iter_sigma_squared_mp = 1.0;
    *state->power_iter_residual_norm_mp = INFINITY;

    for (int i = 0; i < power_iter_params->max_iter; ++i)
    {
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A,
            m,
            power_iteration_vector_d,
            1,
            state->power_iteration_vector_norm_scalar_d));

        normalize_power_iteration_vector_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            power_iteration_normalized_vector_d,
            power_iteration_vector_d,
            state->power_iteration_vector_norm_scalar_d,
            m);
        CUDA_CHECK(cudaGetLastError());

        CUSPARSE_CHECK(cusparseSpMV(
            state->sparse_handle_A, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &HOST_ONE, state->matAt, state->spmv_power_iteration_normalized_vector, &HOST_ZERO, state->spmv_power_iteration_product,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

        CUSPARSE_CHECK(cusparseSpMV(
            state->sparse_handle_A, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &HOST_ONE, state->matA, state->spmv_power_iteration_product, &HOST_ZERO, state->spmv_power_iteration_vector,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

        CUBLAS_CHECK(cublasDdot(
            state->blas_handle_A,
            m,
            power_iteration_normalized_vector_d,
            1,
            power_iteration_vector_d,
            1,
            state->power_iter_sigma_squared_mp));

        compute_power_iteration_residual_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            power_iteration_normalized_vector_d,
            power_iteration_vector_d,
            power_iteration_normalized_vector_d,
            state->power_iter_sigma_squared_mp,
            m);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A,
            m,
            power_iteration_normalized_vector_d,
            1,
            state->power_iter_residual_norm_mp));

        if (((i + 1) % residual_check_interval) == 0 || (i + 1) == power_iter_params->max_iter)
        {
            CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
            if (*state->power_iter_residual_norm_mp < power_iter_params->tol)
                break;
        }
    }

    return sqrt(*state->power_iter_sigma_squared_mp);
}

int get_print_frequency(int iter)
{
    int print_interval = 10;
    long long iteration_threshold = 1000;

    while (iter >= iteration_threshold)
    {
        print_interval *= 10;
        iteration_threshold *= 10;
    }
    return print_interval;
}

bool should_do_adaptive_restart(pdhg_solver_state_t *solver_state,
                                const restart_parameters_t *restart_params,
                                int termination_evaluation_frequency)
{
    bool do_restart = false;
    if (solver_state->total_count == termination_evaluation_frequency)
    {
        do_restart = true;
    }
    else if (solver_state->total_count > termination_evaluation_frequency)
    {
        if (*solver_state->fixed_point_error_mp <= restart_params->sufficient_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            do_restart = true;
        }
        if (*solver_state->fixed_point_error_mp <= restart_params->necessary_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            if (*solver_state->fixed_point_error_mp > solver_state->last_trial_fixed_point_error)
            {
                do_restart = true;
            }
        }
        if (solver_state->inner_count >= restart_params->artificial_restart_threshold * solver_state->total_count)
        {
            do_restart = true;
        }
    }
    solver_state->last_trial_fixed_point_error = *solver_state->fixed_point_error_mp;
    return do_restart;
}

bool optimality_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria)
{
    if (state->metrics_h->pdhg_homogeneous_scale * criteria->eps_feasible_relative <= 1.0e-15)
        return false;
    return state->metrics_h->relative_primal_residual < criteria->eps_feasible_relative &&
           state->metrics_h->relative_dual_residual   < criteria->eps_feasible_relative &&
           state->metrics_h->relative_objective_gap   < criteria->eps_optimal_relative;
}

bool primal_infeasibility_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria)
{
    if (state->metrics_h->dual_solution_norm * criteria->eps_feasible_relative <= 1.0e-15)
        return false;
    return state->metrics_h->relative_primal_infeasibility_residual  < criteria->eps_feasible_relative * 1e-2 &&
           state->metrics_h->relative_primal_infeasibility_objective > 1e-4;
}

bool dual_infeasibility_criteria_met(
    const pdhg_solver_state_t *state,
    const termination_criteria_t *criteria)
{
    if (state->metrics_h->primal_solution_norm * criteria->eps_feasible_relative <= 1.0e-15)
        return false;
    return state->metrics_h->relative_dual_infeasibility_residual  < criteria->eps_feasible_relative * 1e-2 &&
           state->metrics_h->relative_dual_infeasibility_objective < -1e-4;
}

void check_termination_criteria(
    pdhg_solver_state_t *solver_state,
    const termination_criteria_t *criteria)
{
    solver_state->cumulative_time_sec = (double)(clock() - solver_state->start_time) / CLOCKS_PER_SEC;

    if (optimality_criteria_met(solver_state, criteria))
    {
        solver_state->termination_reason = TERMINATION_REASON_OPTIMAL;
        return;
    }
    if (primal_infeasibility_criteria_met(solver_state, criteria))
    {
        solver_state->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
        return;
    }
    if (dual_infeasibility_criteria_met(solver_state, criteria))
    {
        solver_state->termination_reason = TERMINATION_REASON_DUAL_INFEASIBLE;
        return;
    }
    if (solver_state->total_count >= criteria->iteration_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        return;
    }
    if (solver_state->cumulative_time_sec >= criteria->time_sec_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_TIME_LIMIT;
        return;
    }
}

void compute_residual(pdhg_solver_state_t *state, norm_type_t optimality_norm)
{
    const bool use_linf = (optimality_norm == NORM_TYPE_L_INF);

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle_A,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE,
        state->matA,
        state->spmv_pdhg_primal_solution,
        &HOST_ZERO,
        state->spmv_primal_product,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2,
        state->primal_spmv_buffer));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle_At,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE,
        state->matAt,
        state->spmv_pdhg_dual_solution,
        &HOST_ZERO,
        state->spmv_dual_product,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2,
        state->dual_spmv_buffer));

    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));

    compute_residual_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->primal_residual_d,
        state->primal_product_d,
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->dual_residual_d,
        state->dual_product_d,
        state->pdhg_dual_slack_d,
        state->objective_vector,
        state->constraint_rescaling_d,
        state->variable_rescaling_d,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->pdhg_primal_solution_d,
        state->hsd_scalars_d,
        state->homogeneous_dual_residual_d,
        state->constraint_recession_residual_d,
        state->variable_recession_residual_d,
        state->num_constraints,
        state->num_variables);
    CUDA_CHECK(cudaGetLastError());

    if (use_linf)
    {
        abs_vector_inplace_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->primal_residual_d, state->num_constraints);
        abs_vector_inplace_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->dual_residual_d, state->num_variables);
        abs_vector_inplace_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->homogeneous_dual_residual_d, state->num_variables);
        abs_vector_inplace_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->constraint_recession_residual_d, state->num_constraints);
        abs_vector_inplace_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->variable_recession_residual_d, state->num_variables);

        reduce_device_vector(state, state->primal_residual_d, state->num_constraints,
                             state->pdhg_reduce_d + PDHG_PRIMAL_NORM, true);
        reduce_device_vector(state, state->dual_residual_d, state->num_variables,
                             state->pdhg_reduce_d + PDHG_DUAL_NORM, true);
        reduce_device_vector(state, state->homogeneous_dual_residual_d, state->num_variables,
                             state->farkas_reduce_d + FARKAS_HOMOGENEOUS_DUAL_NORM, true);
        reduce_device_vector(state, state->constraint_recession_residual_d, state->num_constraints,
                             state->farkas_reduce_d + FARKAS_CONSTRAINT_RECESSION, true);
        reduce_device_vector(state, state->variable_recession_residual_d, state->num_variables,
                             state->farkas_reduce_d + FARKAS_VARIABLE_RECESSION, true);

        CUDA_CHECK(cudaMemcpyAsync(state->primal_vec_buffer_d, state->pdhg_primal_solution_d,
                                   state->var_bytes, cudaMemcpyDeviceToDevice, state->stream_A));
        abs_vector_inplace_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->primal_vec_buffer_d, state->num_variables);
        reduce_device_vector(state, state->primal_vec_buffer_d, state->num_variables,
                             &state->metrics_d->primal_solution_norm, true);

        CUDA_CHECK(cudaMemcpyAsync(state->dual_vec_buffer_d, state->pdhg_dual_solution_d,
                                   state->con_bytes, cudaMemcpyDeviceToDevice, state->stream_A));
        abs_vector_inplace_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->dual_vec_buffer_d, state->num_constraints);
        reduce_device_vector(state, state->dual_vec_buffer_d, state->num_constraints,
                             state->farkas_reduce_d + FARKAS_DUAL_SOLUTION_NORM, true);

        CUDA_CHECK(cudaMemcpyAsync(state->primal_vec_buffer_d, state->pdhg_dual_slack_d,
                                   state->var_bytes, cudaMemcpyDeviceToDevice, state->stream_A));
        abs_vector_inplace_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->primal_vec_buffer_d, state->num_variables);
        reduce_device_vector(state, state->primal_vec_buffer_d, state->num_variables,
                             state->farkas_reduce_d + FARKAS_DUAL_SLACK_NORM, true);
    }
    else
    {
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_constraints,
            state->primal_residual_d, 1,
            state->pdhg_reduce_d + PDHG_PRIMAL_NORM));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_variables,
            state->dual_residual_d, 1,
            state->pdhg_reduce_d + PDHG_DUAL_NORM));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_variables,
            state->homogeneous_dual_residual_d, 1,
            state->farkas_reduce_d + FARKAS_HOMOGENEOUS_DUAL_NORM));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_constraints,
            state->constraint_recession_residual_d, 1,
            state->farkas_reduce_d + FARKAS_CONSTRAINT_RECESSION));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_variables,
            state->variable_recession_residual_d, 1,
            state->farkas_reduce_d + FARKAS_VARIABLE_RECESSION));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_variables,
            state->pdhg_primal_solution_d, 1,
            &state->metrics_d->primal_solution_norm));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_constraints,
            state->pdhg_dual_solution_d, 1,
            state->farkas_reduce_d + FARKAS_DUAL_SOLUTION_NORM));
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A, state->num_variables,
            state->pdhg_dual_slack_d, 1,
            state->farkas_reduce_d + FARKAS_DUAL_SLACK_NORM));
    }

    CUBLAS_CHECK(cublasDdot(
        state->blas_handle_A,
        state->num_variables,
        state->objective_vector, 1,
        state->pdhg_primal_solution_d, 1,
        state->pdhg_reduce_d + PDHG_PRIMAL_CROSS));

    compute_support_function_contrib_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->dual_vec_buffer_d,
        state->pdhg_dual_solution_d,
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->num_constraints);
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace,
        state->cub_reduce_workspace_bytes,
        state->dual_vec_buffer_d,
        state->farkas_reduce_d + FARKAS_SUPPORT_CONSTRAINT,
        state->num_constraints, state->stream_A));

    compute_support_function_contrib_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->primal_vec_buffer_d,
        state->pdhg_dual_slack_d,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->num_variables);
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace,
        state->cub_reduce_workspace_bytes,
        state->primal_vec_buffer_d,
        state->farkas_reduce_d + FARKAS_SUPPORT_VARIABLE,
        state->num_variables, state->stream_A));

    compute_residual_metrics_kernel<<<1, 1, 0, state->stream_A>>>(
        state->metrics_d,
        state->pdhg_reduce_d,
        state->farkas_reduce_d,
        state->hsd_scalars_d,
        state->constraint_bound_rescaling,
        state->objective_vector_rescaling,
        state->original_bound_norm,
        state->original_objective_norm,
        state->original_constraint_norm,
        state->objective_constant);

    CUDA_CHECK(cudaMemcpyAsync(state->metrics_h, state->metrics_d,
                               sizeof(pdhg_metrics_t),
                               cudaMemcpyDeviceToHost,
                               state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
}

static void reduce_device_vector(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output,
    bool use_linf)
{
    if (use_linf)
        CUDA_CHECK(cub::DeviceReduce::Max(state->cub_reduce_workspace,
                                          state->cub_reduce_workspace_bytes,
                                          input, output, length, state->stream_A));
    else
        CUDA_CHECK(cub::DeviceReduce::Sum(state->cub_reduce_workspace,
                                          state->cub_reduce_workspace_bytes,
                                          input, output, length, state->stream_A));
    CUDA_CHECK(cudaGetLastError());
}

__global__ void compute_residual_kernel(
    double *__restrict__ primal_res,
    const double *__restrict__ primal_prod,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    double *__restrict__ dual_res,
    const double *__restrict__ dual_prod,
    const double *__restrict__ dual_slack,
    const double *__restrict__ obj_vec,
    const double *__restrict__ cons_rescaling,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ var_lb,
    const double *__restrict__ var_ub,
    const double *__restrict__ primal_sol,
    const double *__restrict__ hsd_scalars_d,
    double *__restrict__ homogeneous_dual_res,
    double *__restrict__ constraint_recession_res,
    double *__restrict__ variable_recession_res,
    int n_cons,
    int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double tau = hsd_scalars_d[HSD_IDX(HSD_SCALE, HSD_PDHG)];
    double inv_tau = (fabs(tau) > 1e-30) ? 1.0 / tau : 0.0;

    if (i < n_cons)
    {
        double ax = primal_prod[i];
        double ax_hat = ax * inv_tau;
        double clamped = fmax(cons_lb[i], fmin(ax_hat, cons_ub[i]));
        primal_res[i] = (ax_hat - clamped) * cons_rescaling[i];

        double recc_lo = isfinite(cons_lb[i]) ? 0.0 : -INFINITY;
        double recc_hi = isfinite(cons_ub[i]) ? 0.0 : INFINITY;
        double ax_recc_clamped = fmax(recc_lo, fmin(ax, recc_hi));
        constraint_recession_res[i] = (ax - ax_recc_clamped) * cons_rescaling[i];
    }
    else if (i < n_cons + n_vars)
    {
        int idx = i - n_cons;
        double aty = dual_prod[idx];
        double yv  = dual_slack[idx];
        double yc_hat = aty * inv_tau;
        double yv_hat = yv * inv_tau;
        dual_res[idx] = (obj_vec[idx] - yc_hat - yv_hat) * var_rescaling[idx];

        homogeneous_dual_res[idx] = (aty + yv) * var_rescaling[idx];

        double x = primal_sol[idx];
        double recc_lo = isfinite(var_lb[idx]) ? 0.0 : -INFINITY;
        double recc_hi = isfinite(var_ub[idx]) ? 0.0 : INFINITY;
        double x_recc_clamped = fmax(recc_lo, fmin(x, recc_hi));
        variable_recession_res[idx] = (x - x_recc_clamped) * var_rescaling[idx];
    }
}

__global__ void compute_support_function_contrib_kernel(
    double *__restrict__ contrib,
    const double *__restrict__ y,
    const double *__restrict__ lb,
    const double *__restrict__ ub,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double yi = y[i];
    double c = 0.0;
    if (yi > 0.0 && isfinite(lb[i])) 
    {
        c = lb[i] * yi;
    }
    else if (yi < 0.0 && isfinite(ub[i])) 
    {
        c = ub[i] * yi;
    }
    contrib[i] = c;
}

__global__ void compute_residual_metrics_kernel(
    pdhg_metrics_t *metrics_d,
    const double *pdhg_reduce_d,
    const double *farkas_reduce_d,
    const double *hsd_scalars_d,
    double constraint_bound_rescaling,
    double objective_vector_rescaling,
    double original_bound_norm,
    double original_objective_norm,
    double original_constraint_norm,
    double objective_constant)
{
    double primal_scale = hsd_scalars_d[HSD_IDX(HSD_SCALE, HSD_PDHG)];
    double inv_primal_scale = (fabs(primal_scale) > 1e-30) ? 1.0 / primal_scale : 0.0;
    double pdhg_dual_constraint_offset = hsd_scalars_d[HSD_IDX(HSD_DUAL_CONSTRAINT, HSD_PDHG)];
    double pdhg_dual_bound_offset = hsd_scalars_d[HSD_IDX(HSD_DUAL_BOUND, HSD_PDHG)];
    double absolute_primal_residual = pdhg_reduce_d[PDHG_PRIMAL_NORM] / constraint_bound_rescaling;
    double absolute_dual_residual = pdhg_reduce_d[PDHG_DUAL_NORM] / objective_vector_rescaling;
    double objective_dot_primal = pdhg_reduce_d[PDHG_PRIMAL_CROSS];
    double scaled_dual_constraint_offset = pdhg_dual_constraint_offset * inv_primal_scale;
    double scaled_dual_bound_offset = pdhg_dual_bound_offset * inv_primal_scale;

    double primal_objective_value = objective_dot_primal * inv_primal_scale
                                  / (constraint_bound_rescaling * objective_vector_rescaling)
                                  + objective_constant;
    double dual_objective_value = - (scaled_dual_constraint_offset + scaled_dual_bound_offset)
                                / (constraint_bound_rescaling * objective_vector_rescaling)
                                + objective_constant;
    double objective_gap = fabs(primal_objective_value - dual_objective_value);
    double relative_primal_residual = absolute_primal_residual / (1.0 + original_bound_norm);
    double relative_dual_residual = absolute_dual_residual / (1.0 + original_objective_norm);
    double relative_objective_gap = objective_gap / (1.0 + fabs(primal_objective_value) + fabs(dual_objective_value));

    metrics_d->absolute_primal_residual = absolute_primal_residual;
    metrics_d->relative_primal_residual = relative_primal_residual;
    metrics_d->absolute_dual_residual = absolute_dual_residual;
    metrics_d->relative_dual_residual = relative_dual_residual;
    metrics_d->primal_objective = primal_objective_value;
    metrics_d->dual_objective = dual_objective_value;
    metrics_d->objective_gap = objective_gap;
    metrics_d->relative_objective_gap = relative_objective_gap;
    metrics_d->initial_homogeneous_scale = hsd_scalars_d[HSD_IDX(HSD_SCALE, HSD_INITIAL)];
    metrics_d->current_homogeneous_scale = hsd_scalars_d[HSD_IDX(HSD_SCALE, HSD_CURRENT)];
    metrics_d->pdhg_homogeneous_scale    = hsd_scalars_d[HSD_IDX(HSD_SCALE, HSD_PDHG)];
    metrics_d->pdhg_dual_constraint_offset = hsd_scalars_d[HSD_IDX(HSD_DUAL_CONSTRAINT, HSD_PDHG)];
    metrics_d->pdhg_dual_bound_offset      = hsd_scalars_d[HSD_IDX(HSD_DUAL_BOUND, HSD_PDHG)];

    double yc_norm = farkas_reduce_d[FARKAS_DUAL_SOLUTION_NORM];
    double yv_norm = farkas_reduce_d[FARKAS_DUAL_SLACK_NORM];
    double dual_solution_norm = sqrt(yc_norm * yc_norm + yv_norm * yv_norm);
    double primal_solution_norm = metrics_d->primal_solution_norm;

    metrics_d->dual_solution_norm = dual_solution_norm;

    double homogeneous_dual_norm = farkas_reduce_d[FARKAS_HOMOGENEOUS_DUAL_NORM];
    double inv_dual_norm = (dual_solution_norm > 1.0e-30) ? 1.0 / dual_solution_norm : 0.0;
    double abs_p_infeas_residual = homogeneous_dual_norm * inv_dual_norm / objective_vector_rescaling;
    double rel_p_infeas_residual = abs_p_infeas_residual / (1.0 + original_constraint_norm);
    metrics_d->relative_primal_infeasibility_residual = rel_p_infeas_residual;

    double support_total = farkas_reduce_d[FARKAS_SUPPORT_CONSTRAINT]
                         + farkas_reduce_d[FARKAS_SUPPORT_VARIABLE];
    double support_normalized = support_total * inv_dual_norm;
    double rel_p_infeas_objective = support_normalized / (1.0 + original_bound_norm);
    metrics_d->relative_primal_infeasibility_objective = rel_p_infeas_objective;

    double constr_recc = farkas_reduce_d[FARKAS_CONSTRAINT_RECESSION] / constraint_bound_rescaling;
    double var_recc    = farkas_reduce_d[FARKAS_VARIABLE_RECESSION] / objective_vector_rescaling;
    double inv_primal_norm = (primal_solution_norm > 1.0e-30) ? 1.0 / primal_solution_norm : 0.0;
    double rel_constr_recc = (constr_recc * inv_primal_norm) / (1.0 + original_bound_norm);
    double rel_var_recc    = (var_recc    * inv_primal_norm) / (1.0 + original_bound_norm);
    double rel_d_infeas_residual = fmax(rel_constr_recc, rel_var_recc);
    metrics_d->relative_dual_infeasibility_residual = rel_d_infeas_residual;

    double c_dot_x = pdhg_reduce_d[PDHG_PRIMAL_CROSS];
    double rel_d_infeas_objective = (c_dot_x * inv_primal_norm)
                                  / ((1.0 + original_objective_norm) * objective_vector_rescaling);
    metrics_d->relative_dual_infeasibility_objective = rel_d_infeas_objective;
}
__global__ void normalize_power_iteration_vector_kernel(
    double *normalized_vector_d,
    const double *vector_d,
    const double *vector_norm_d,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double vector_norm = *vector_norm_d;
    normalized_vector_d[i] = (vector_norm > 0.0) ? (vector_d[i] / vector_norm) : 0.0;
}

__global__ void compute_power_iteration_residual_kernel(
    double *residual_d,
    const double *vector_d,
    const double *normalized_vector_d,
    const double *sigma_squared_d,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double sigma_squared = *sigma_squared_d;
    residual_d[i] = vector_d[i] - sigma_squared * normalized_vector_d[i];
}

__global__ void abs_vector_inplace_kernel(double *__restrict__ arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        arr[i] = fabs(arr[i]);
}

void display_iteration_stats(const pdhg_solver_state_t *state, bool verbose, bool is_restart)
{
    if (!verbose)
    {
        return;
    }
    if (is_restart || state->total_count % get_print_frequency(state->total_count) == 0)
    {
        printf("%c%6d %.1e | %8.1e  %8.1e | %.1e %.1e %.1e | %.1e %.1e %.1e | %.1e %.1e %.1e | %7.1e\n",
               is_restart ? 'R' : ' ',
               state->total_count,
               state->cumulative_time_sec,
               state->metrics_h->primal_objective,
               state->metrics_h->dual_objective,
               state->metrics_h->absolute_primal_residual,
               state->metrics_h->absolute_dual_residual,
               state->metrics_h->objective_gap,
               state->metrics_h->relative_primal_residual,
               state->metrics_h->relative_dual_residual,
               state->metrics_h->relative_objective_gap,
               state->step_size_primal,
               state->step_size_dual,
               state->step_size_homogeneous,
               state->metrics_h->current_homogeneous_scale);
    }
}

void pdhg_final_log(
    const cuhsdlp_result_t *result,
    const pdhg_parameters_t *params)
{
    if (params->verbose)
    {
        printf("-----------------------------------------------------------"
           "-----------------------------------------------------------------\n");
        printf("Solution Summary\n");
        printf("  Status                 : %s\n", termination_reason_to_string(result->termination_reason));
        if (params->presolve)
        {
            printf("  Presolve time          : %.3g sec\n", result->presolve_time);
        }
        printf("  Precondition time      : %.5g sec\n", result->rescaling_time_sec);
        printf("  Solve time             : %.3g sec\n", result->cumulative_time_sec);
        printf("  Iterations             : %d\n", result->total_count);
        printf("  Primal objective       : %.10g\n", result->primal_objective_value);
        printf("  Dual objective         : %.10g\n", result->dual_objective_value);
        printf("  Objective gap          : %.3e\n", result->relative_objective_gap);
        printf("  Primal infeas          : %.3e\n", result->relative_primal_residual);
        printf("  Dual infeas            : %.3e\n", result->relative_dual_residual);
        // printf("  Restarts               : %d\n", result->restart_count);
        // printf("  Homogeneous scale      : %.3e\n", result->homogeneous_scale);
        // printf("  Step size (primal)     : %.3e\n", result->final_step_size_primal);
        // printf("  Step size (dual)       : %.3e\n", result->final_step_size_dual);
        // printf("  Step size (homogeneous): %.3e\n", result->final_step_size_homogeneous);
    }
}

__global__ void fill_ones_kernel(double *__restrict__ arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        arr[i] = 1.0;
}
