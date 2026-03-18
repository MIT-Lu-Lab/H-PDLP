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
#include <math.h>
#include <cub/cub.cuh>

#ifndef CUHSDLP_VERSION
#define CUHSDLP_VERSION "unknown"
#endif

__global__ void compute_objective_norm_entries_kernel(
    const double *__restrict__ objective_vector,
    double *__restrict__ entries,
    int n_vars,
    int use_linf);
__global__ void compute_bound_norm_entries_kernel(
    const double *__restrict__ constraint_lower_bound,
    const double *__restrict__ constraint_upper_bound,
    double *__restrict__ entries,
    int n_cons,
    int use_linf);
__global__ void compute_residual_kernel(
    double *__restrict__ primal_residual,
    const double *__restrict__ primal_product,
    const double *__restrict__ constraint_lower_bound,
    const double *__restrict__ constraint_upper_bound,
    const double *__restrict__ dual_solution,
    double *__restrict__ dual_residual,
    const double *__restrict__ dual_product,
    const double *__restrict__ dual_slack,
    const double *__restrict__ objective_vector,
    const double *__restrict__ constraint_rescaling,
    const double *__restrict__ variable_rescaling,
    double *__restrict__ dual_obj_contribution,
    const double *__restrict__ const_lb_finite,
    const double *__restrict__ const_ub_finite,
    int num_constraints,
    int num_variables);
__global__ void abs_vector_inplace_kernel(
    double *__restrict__ arr, int n);

const double HOST_ONE = 1.0;
const double HOST_ZERO = 0.0;

void *safe_malloc(size_t size)
{
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
    {
        free(ptr);
        return NULL;
    }
    void *tmp = realloc(ptr, new_size);
    if (!tmp)
    {
        perror("Fatal error: realloc failed");
        exit(EXIT_FAILURE);
    }
    return tmp;
}

int dense_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind,
                 double **vals, int *nnz_out)
{
    int m = desc->m, n = desc->n;
    double tol = (desc->zero_tolerance > 0) ? desc->zero_tolerance : 1e-12;

    int nnz = 0;
    for (int i = 0; i < m * n; ++i)
    {
        if (fabs(desc->data.dense.A[i]) > tol)
            ++nnz;
    }

    *row_ptr = (int *)safe_malloc((size_t)(m + 1) * sizeof(int));
    *col_ind = (int *)safe_malloc((size_t)nnz * sizeof(int));
    *vals = (double *)safe_malloc((size_t)nnz * sizeof(double));

    int nz = 0;
    for (int i = 0; i < m; ++i)
    {
        (*row_ptr)[i] = nz;
        for (int j = 0; j < n; ++j)
        {
            double v = desc->data.dense.A[i * n + j];
            if (fabs(v) > tol)
            {
                (*col_ind)[nz] = j;
                (*vals)[nz] = v;
                ++nz;
            }
        }
    }
    (*row_ptr)[m] = nz;
    *nnz_out = nz;
    return 0;
}

int csc_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind,
               double **vals, int *nnz_out)
{
    const int m = desc->m, n = desc->n;
    const int *col_ptr = desc->data.csc.col_ptr;
    const int *row_ind = desc->data.csc.row_ind;
    const double *v = desc->data.csc.vals;

    const double tol = (desc->zero_tolerance > 0) ? desc->zero_tolerance : 0.0;

    *row_ptr = (int *)safe_malloc((size_t)(m + 1) * sizeof(int));
    for (int i = 0; i <= m; ++i)
        (*row_ptr)[i] = 0;

    int eff_nnz = 0;
    for (int j = 0; j < n; ++j)
    {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
        {
            int ri = row_ind[k];
            if (ri < 0 || ri >= m)
            {
                fprintf(stderr, "[interface] CSC: row index out of range\n");
                free(*row_ptr);
                return -1;
            }
            double val = v[k];
            if (tol > 0 && fabs(val) <= tol)
                continue;
            ++((*row_ptr)[ri + 1]);
            ++eff_nnz;
        }
    }

    for (int i = 0; i < m; ++i)
        (*row_ptr)[i + 1] += (*row_ptr)[i];

    *col_ind = (int *)safe_malloc((size_t)eff_nnz * sizeof(int));
    *vals = (double *)safe_malloc((size_t)eff_nnz * sizeof(double));

    int *next = (int *)safe_malloc((size_t)m * sizeof(int));
    for (int i = 0; i < m; ++i)
        next[i] = (*row_ptr)[i];

    for (int j = 0; j < n; ++j)
    {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
        {
            int ri = row_ind[k];
            double val = v[k];
            if (tol > 0 && fabs(val) <= tol)
                continue;
            int pos = next[ri]++;
            (*col_ind)[pos] = j;
            (*vals)[pos] = val;
        }
    }

    free(next);
    *nnz_out = eff_nnz;
    return 0;
}

int coo_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind,
               double **vals, int *nnz_out)
{
    const int m = desc->m, n = desc->n;
    const int nnz_in = desc->data.coo.nnz;
    const int *r = desc->data.coo.row_ind;
    const int *c = desc->data.coo.col_ind;
    const double *v = desc->data.coo.vals;
    const double tol = (desc->zero_tolerance > 0) ? desc->zero_tolerance : 0.0;

    int nnz = 0;
    if (tol > 0)
    {
        for (int k = 0; k < nnz_in; ++k)
            if (fabs(v[k]) > tol)
                ++nnz;
    }
    else
    {
        nnz = nnz_in;
    }

    *row_ptr = (int *)safe_malloc((size_t)(m + 1) * sizeof(int));
    *col_ind = (int *)safe_malloc((size_t)nnz * sizeof(int));
    *vals = (double *)safe_malloc((size_t)nnz * sizeof(double));

    for (int i = 0; i <= m; ++i)
        (*row_ptr)[i] = 0;
    if (tol > 0)
    {
        for (int k = 0; k < nnz_in; ++k)
            if (fabs(v[k]) > tol)
            {
                int ri = r[k];
                if (ri < 0 || ri >= m)
                {
                    fprintf(stderr, "[interface] COO: row index out of range\n");
                    free(*row_ptr); free(*col_ind); free(*vals);
                    return -1;
                }
                ++((*row_ptr)[ri + 1]);
            }
    }
    else
    {
        for (int k = 0; k < nnz_in; ++k)
        {
            int ri = r[k];
            if (ri < 0 || ri >= m)
            {
                fprintf(stderr, "[interface] COO: row index out of range\n");
                free(*row_ptr); free(*col_ind); free(*vals);
                return -1;
            }
            ++((*row_ptr)[ri + 1]);
        }
    }

    for (int i = 0; i < m; ++i)
        (*row_ptr)[i + 1] += (*row_ptr)[i];

    int *next = (int *)safe_malloc((size_t)m * sizeof(int));
    for (int i = 0; i < m; ++i)
        next[i] = (*row_ptr)[i];

    if (tol > 0)
    {
        for (int k = 0; k < nnz_in; ++k)
        {
            if (fabs(v[k]) <= tol)
                continue;
            int ri = r[k], cj = c[k];
            if (cj < 0 || cj >= n)
            {
                fprintf(stderr, "[interface] COO: col index out of range\n");
                free(next); free(*row_ptr); free(*col_ind); free(*vals);
                return -1;
            }
            int pos = next[ri]++;
            (*col_ind)[pos] = cj;
            (*vals)[pos] = v[k];
        }
    }
    else
    {
        for (int k = 0; k < nnz_in; ++k)
        {
            int ri = r[k], cj = c[k];
            if (cj < 0 || cj >= n)
            {
                fprintf(stderr, "[interface] COO: col index out of range\n");
                free(next);
                free(*row_ptr);
                free(*col_ind);
                free(*vals);
                return -1;
            }
            int pos = next[ri]++;
            (*col_ind)[pos] = cj;
            (*vals)[pos] = v[k];
        }
    }

    free(next);
    *nnz_out = nnz;
    return 0;
}

void set_default_parameters(pdhg_parameters_t *params)
{
    params->l_inf_ruiz_iterations = 10;
    params->has_pock_chambolle_alpha = true;
    params->pock_chambolle_alpha = 1.0;
    params->bound_objective_rescaling = true;
    params->verbose = true;
    params->termination_evaluation_frequency = 200;
    params->reflection_coefficient = 1.0;

    params->sv_max_iter = 5000;
    params->sv_tol = 1e-4;

    params->termination_criteria.eps_optimal_relative = 1e-4;
    params->termination_criteria.eps_feasible_relative = 1e-4;
    params->termination_criteria.time_sec_limit = 3600.0;
    params->termination_criteria.iteration_limit = INT32_MAX;

    params->restart_params.artificial_restart_threshold = 0.36;
    params->restart_params.sufficient_reduction_for_restart = 0.2;
    params->restart_params.necessary_reduction_for_restart = 0.5;
    params->restart_params.k_p = 0.99;
    params->restart_params.k_i = 0.01;
    params->restart_params.k_d = 0.0;
    params->restart_params.i_smooth = 0.3;

    params->optimality_norm = NORM_TYPE_L2;
    params->presolve = true;
    params->matrix_zero_tol = 1e-9;
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
            if (fabs(vals[k]) > params->matrix_zero_tol)
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
            if (fabs(vals[k]) <= params->matrix_zero_tol)
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
               params->matrix_zero_tol);
    }
}

void print_initial_info(const pdhg_parameters_t *params,
                        lp_problem_t *problem)
{
    pdhg_parameters_t default_params;
    set_default_parameters(&default_params);
    if (!params->verbose)
    {
        return;
    }
    printf("-------------------------------------------------------------------------------------\n");
    printf("                              cuHSDLP v%s\n", CUHSDLP_VERSION);
    printf("                   A GPU-Accelerated First-Order LP Solver\n");
    printf("          (c) Haihao Lu, Massachusetts Institute of Technology, 2026\n");
    printf("-------------------------------------------------------------------------------------\n");
    printf("\n");

    filter_constraint_matrix_entries(problem, params);

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
                   params->l_inf_ruiz_iterations,
                   default_params.l_inf_ruiz_iterations);
    PRINT_DIFF_DBL("pock_chambolle_alpha",
                   params->pock_chambolle_alpha,
                   default_params.pock_chambolle_alpha);
    PRINT_DIFF_BOOL("has_pock_chambolle_alpha",
                    params->has_pock_chambolle_alpha,
                    default_params.has_pock_chambolle_alpha);
    PRINT_DIFF_BOOL("bound_obj_rescaling",
                    params->bound_objective_rescaling,
                    default_params.bound_objective_rescaling);
    PRINT_DIFF_INT("sv_max_iter",
                   params->sv_max_iter,
                   default_params.sv_max_iter);
    PRINT_DIFF_DBL("sv_tol",
                   params->sv_tol,
                   default_params.sv_tol);
    PRINT_DIFF_INT("evaluation_freq",
                   params->termination_evaluation_frequency,
                   default_params.termination_evaluation_frequency);
    PRINT_DIFF_BOOL("presolve",
                    params->presolve,
                    default_params.presolve);
    PRINT_DIFF_DBL("matrix_zero_tol",
                   params->matrix_zero_tol,
                   default_params.matrix_zero_tol);
}

#undef PRINT_DIFF_INT
#undef PRINT_DIFF_DBL
#undef PRINT_DIFF_BOOL

void compute_objective_and_bound_norms(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    const int use_linf = (params->optimality_norm == NORM_TYPE_L_INF);

    double *entries_vars = state->dual_residual;
    double *entries_cons = state->primal_residual;
    double *d_out = state->scalar_pool_d;

    compute_objective_norm_entries_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->objective_vector, entries_vars, state->num_variables, use_linf);
    CUDA_CHECK(cudaGetLastError());

    compute_bound_norm_entries_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound, state->constraint_upper_bound,
        entries_cons, state->num_constraints, use_linf);
    CUDA_CHECK(cudaGetLastError());

    size_t tb = state->reduce_workspace_bytes;
    if (use_linf)
        CUDA_CHECK(cub::DeviceReduce::Max(state->reduce_workspace_d, tb, entries_vars, d_out, state->num_variables));
    else
        CUDA_CHECK(cub::DeviceReduce::Sum(state->reduce_workspace_d, tb, entries_vars, d_out, state->num_variables));
    CUDA_CHECK(cudaGetLastError());

    double obj_reduce = 0.0;
    CUDA_CHECK(cudaMemcpy(&obj_reduce, d_out, sizeof(double), cudaMemcpyDeviceToHost));
    state->objective_vector_norm = use_linf ? obj_reduce : sqrt(obj_reduce);

    tb = state->reduce_workspace_bytes;
    if (use_linf)
        CUDA_CHECK(cub::DeviceReduce::Max(state->reduce_workspace_d, tb, entries_cons, d_out, state->num_constraints));
    else
        CUDA_CHECK(cub::DeviceReduce::Sum(state->reduce_workspace_d, tb, entries_cons, d_out, state->num_constraints));
    CUDA_CHECK(cudaGetLastError());

    double con_reduce = 0.0;
    CUDA_CHECK(cudaMemcpy(&con_reduce, d_out, sizeof(double), cudaMemcpyDeviceToHost));
    state->constraint_bound_norm = use_linf ? con_reduce : sqrt(con_reduce);
}

__global__ void compute_objective_norm_entries_kernel(
    const double *__restrict__ objective_vector,
    double *__restrict__ entries,
    int n_vars,
    int use_linf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vars) return;

    double v = objective_vector[i];
    entries[i] = use_linf ? fabs(v) : (v * v);
}

__global__ void compute_bound_norm_entries_kernel(
    const double *__restrict__ constraint_lower_bound,
    const double *__restrict__ constraint_upper_bound,
    double *__restrict__ entries,
    int n_cons,
    int use_linf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons) return;

    double Li = constraint_lower_bound[i];
    double Ui = constraint_upper_bound[i];
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
    cusparseHandle_t sparse_handle,
    cublasHandle_t blas_handle,
    curandGenerator_t curand_gen,
    const cu_sparse_matrix_csr_t *A,
    const cu_sparse_matrix_csr_t *AT,
    int max_iterations, double tolerance)
{
    int m = A->num_rows;
    int n = A->num_cols;
    double *eigenvector_d, *next_eigenvector_d, *dual_product_d;

    CUDA_CHECK(cudaMalloc(&eigenvector_d, ((size_t)m + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&next_eigenvector_d, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dual_product_d, n * sizeof(double)));

    CURAND_CHECK(curandGenerateNormalDouble(curand_gen, eigenvector_d, ((size_t)m + 1) & ~(size_t)1, 0.0, 1.0));

    double sigma_max_sq = 1.0;
    const double one = 1.0;
    const double zero = 0.0;

    CUBLAS_CHECK(cublasSetPointerMode(blas_handle, CUBLAS_POINTER_MODE_HOST));

    cusparseSpMatDescr_t matA, matAT;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA, A->num_rows, A->num_cols, A->num_nonzeros, A->row_ptr, A->col_ind,
        A->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matAT, AT->num_rows, AT->num_cols, AT->num_nonzeros, AT->row_ptr,
        AT->col_ind, AT->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseDnVecDescr_t vecEigen, vecNextEigen, vecDual;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecEigen, m, eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecNextEigen, m, next_eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDual, n, dual_product_d, CUDA_R_64F));

    void *dBufferAT = NULL;
    void *dBufferA = NULL;
    size_t bufferSizeAT = 0, bufferSizeA = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matAT,
        vecNextEigen, &zero, vecDual, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
        &bufferSizeAT));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecDual,
        &zero, vecEigen, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &bufferSizeA));

    CUDA_CHECK(cudaMalloc(&dBufferAT, bufferSizeAT));
    CUDA_CHECK(cudaMalloc(&dBufferA, bufferSizeA));

    for (int i = 0; i < max_iterations; ++i)
    {
        CUDA_CHECK(cudaMemcpy(next_eigenvector_d, eigenvector_d, m * sizeof(double), cudaMemcpyDeviceToDevice));
        double eigenvector_norm;
        CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, m, next_eigenvector_d, 1, &eigenvector_norm));

        double inv_eigenvector_norm = 1.0 / eigenvector_norm;
        CUBLAS_CHECK(cublasDscal(blas_handle, m, &inv_eigenvector_norm, next_eigenvector_d, 1));

        CUSPARSE_CHECK(cusparseSpMV(
            sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, matAT, vecNextEigen, &zero, vecDual,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, dBufferAT));

        CUSPARSE_CHECK(cusparseSpMV(
            sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, matA, vecDual, &zero, vecEigen,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, dBufferA));

        CUBLAS_CHECK(cublasDdot(blas_handle, m, next_eigenvector_d, 1, eigenvector_d, 1, &sigma_max_sq));

        double neg_sigma_sq = - sigma_max_sq;
        CUBLAS_CHECK(cublasDscal(blas_handle, m, &neg_sigma_sq, next_eigenvector_d, 1));
        CUBLAS_CHECK(cublasDaxpy(blas_handle, m, &one, eigenvector_d, 1, next_eigenvector_d, 1));

        double residual_norm;
        CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, m, next_eigenvector_d, 1, &residual_norm));

        if (residual_norm < tolerance)
            break;
    }

    CUDA_CHECK(cudaFree(dBufferAT));
    CUDA_CHECK(cudaFree(dBufferA));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroySpMat(matAT));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecEigen));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecNextEigen));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecDual));
    CUDA_CHECK(cudaFree(eigenvector_d));
    CUDA_CHECK(cudaFree(next_eigenvector_d));
    CUDA_CHECK(cudaFree(dual_product_d));

    CUBLAS_CHECK(cublasSetPointerMode(blas_handle, CUBLAS_POINTER_MODE_DEVICE));
    return sqrt(sigma_max_sq);
}

int get_print_frequency(int iter)
{
    int step = 10;
    long long threshold = 1000;

    while (iter >= threshold)
    {
        step *= 10;
        threshold *= 10;
    }
    return step;
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
        if (solver_state->fixed_point_error <=
            restart_params->sufficient_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            do_restart = true;
        }
        if (solver_state->fixed_point_error <=
            restart_params->necessary_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            if (solver_state->fixed_point_error > solver_state->last_trial_fixed_point_error)
            {
                do_restart = true;
            }
        }
        if (solver_state->inner_count >= restart_params->artificial_restart_threshold * solver_state->total_count)
        {
            do_restart = true;
        }
    }
    solver_state->last_trial_fixed_point_error = solver_state->fixed_point_error;
    return do_restart;
}

bool optimality_criteria_met(
    const pdhg_solver_state_t *state,
    double rel_opt_tol,
    double rel_feas_tol)
{
    return state->relative_dual_residual < rel_feas_tol &&
           state->relative_primal_residual < rel_feas_tol &&
           state->relative_objective_gap < rel_opt_tol;
}

void check_termination_criteria(
    pdhg_solver_state_t *solver_state,
    const termination_criteria_t *criteria)
{
    solver_state->cumulative_time_sec = (double)(clock() - solver_state->start_time) / CLOCKS_PER_SEC;

    if (optimality_criteria_met(solver_state, criteria->eps_optimal_relative, criteria->eps_feasible_relative))
    {
        solver_state->termination_reason = TERMINATION_REASON_OPTIMAL;
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
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, state->pdhg_primal_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->pdhg_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle_A,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE,
        state->matA,
        state->vec_primal_sol,
        &HOST_ZERO,
        state->vec_primal_prod,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2,
        state->primal_spmv_buffer));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle_At,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE,
        state->matAt,
        state->vec_dual_sol,
        &HOST_ZERO,
        state->vec_dual_prod,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG2,
        state->dual_spmv_buffer));

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaEventRecord(state->event_At, state->stream_At));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_A, state->event_At, 0));

    compute_residual_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->primal_residual,
        state->primal_product,
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->pdhg_dual_solution,
        state->dual_residual,
        state->dual_product,
        state->dual_slack,
        state->objective_vector,
        state->constraint_rescaling,
        state->variable_rescaling,
        state->primal_slack,
        state->constraint_lower_bound_finite_val,
        state->constraint_upper_bound_finite_val,
        state->num_constraints,
        state->num_variables);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(state->event_A, state->stream_A));
    CUDA_CHECK(cudaStreamWaitEvent(state->stream_At, state->event_A, 0));

    if (optimality_norm == NORM_TYPE_L_INF)
    {
        abs_vector_inplace_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->primal_residual, state->num_constraints);
        abs_vector_inplace_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
            state->dual_residual, state->num_variables);

        size_t tb = state->reduce_workspace_bytes;
        CUDA_CHECK(cub::DeviceReduce::Max(state->reduce_workspace_d, tb,
            state->primal_residual, state->scalar_pool_d + 0, state->num_constraints, state->stream_A));
        tb = state->reduce_workspace_bytes;
        CUDA_CHECK(cub::DeviceReduce::Max(state->reduce_workspace_d, tb,
            state->dual_residual, state->scalar_pool_d + 1, state->num_variables, state->stream_A));

        CUBLAS_CHECK(cublasDdot(
            state->blas_handle_A,
            state->num_variables,
            state->objective_vector, 1,
            state->pdhg_primal_solution, 1,
            state->scalar_pool_d + 2));

        CUBLAS_CHECK(cublasDdot(
            state->blas_handle_A,
            state->num_variables,
            state->dual_slack, 1,
            state->pdhg_primal_solution, 1,
            state->scalar_pool_d + 3));

        CUBLAS_CHECK(cublasDdot(
            state->blas_handle_At,
            state->num_constraints,
            state->primal_slack, 1,
            state->ones_dual_d, 1,
            state->scalar_pool_d + 4));

        CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
        CUDA_CHECK(cudaStreamSynchronize(state->stream_At));

        double h_buf[5];
        CUDA_CHECK(cudaMemcpy(h_buf, state->scalar_pool_d, 5 * sizeof(double),
                               cudaMemcpyDeviceToHost));

        state->absolute_primal_residual = h_buf[0] / state->constraint_bound_rescaling;
        state->absolute_dual_residual = h_buf[1] / state->objective_vector_rescaling;
        state->primal_objective_value =
            h_buf[2] /
                (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
            state->objective_constant;
        state->dual_objective_value =
            (h_buf[3] + h_buf[4]) /
                (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
            state->objective_constant;
    }
    else
    {
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_A,
            state->num_constraints,
            state->primal_residual, 1,
            state->scalar_pool_d + 0));

        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle_At,
            state->num_variables,
            state->dual_residual, 1,
            state->scalar_pool_d + 1));

        CUBLAS_CHECK(cublasDdot(
            state->blas_handle_A,
            state->num_variables,
            state->objective_vector, 1,
            state->pdhg_primal_solution, 1,
            state->scalar_pool_d + 2));

        CUBLAS_CHECK(cublasDdot(
            state->blas_handle_A,
            state->num_variables,
            state->dual_slack, 1,
            state->pdhg_primal_solution, 1,
            state->scalar_pool_d + 3));

        CUBLAS_CHECK(cublasDdot(
            state->blas_handle_At,
            state->num_constraints,
            state->primal_slack, 1,
            state->ones_dual_d, 1,
            state->scalar_pool_d + 4));

        CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
        CUDA_CHECK(cudaStreamSynchronize(state->stream_At));

        double h_buf[5];
        CUDA_CHECK(cudaMemcpy(h_buf, state->scalar_pool_d, 5 * sizeof(double),
                               cudaMemcpyDeviceToHost));

        state->absolute_primal_residual = h_buf[0] / state->constraint_bound_rescaling;
        state->absolute_dual_residual = h_buf[1] / state->objective_vector_rescaling;
        state->primal_objective_value =
            h_buf[2] /
                (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
            state->objective_constant;
        state->dual_objective_value =
            (h_buf[3] + h_buf[4]) /
                (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
            state->objective_constant;
    }

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);

    state->relative_primal_residual =
        state->absolute_primal_residual / (1.0 + state->constraint_bound_norm);
    state->relative_dual_residual =
        state->absolute_dual_residual / (1.0 + state->objective_vector_norm);
    state->relative_objective_gap =
        state->objective_gap /
        (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
}

__global__ void compute_residual_kernel(
    double *__restrict__ primal_residual,
    const double *__restrict__ primal_product,
    const double *__restrict__ constraint_lower_bound,
    const double *__restrict__ constraint_upper_bound,
    const double *__restrict__ dual_solution,
    double *__restrict__ dual_residual,
    const double *__restrict__ dual_product,
    const double *__restrict__ dual_slack,
    const double *__restrict__ objective_vector,
    const double *__restrict__ constraint_rescaling,
    const double *__restrict__ variable_rescaling,
    double *__restrict__ dual_obj_contribution,
    const double *__restrict__ const_lb_finite,
    const double *__restrict__ const_ub_finite,
    int num_constraints,
    int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {
        double clamped_val = fmax(constraint_lower_bound[i], fmin(primal_product[i], constraint_upper_bound[i]));
        primal_residual[i] = (primal_product[i] - clamped_val) * constraint_rescaling[i];

        dual_obj_contribution[i] =
            fmax(dual_solution[i], 0.0) * const_lb_finite[i] + fmin(dual_solution[i], 0.0) * const_ub_finite[i];
    }
    else if (i < num_constraints + num_variables)
    {
        int idx = i - num_constraints;
        dual_residual[idx] = (objective_vector[idx] - dual_product[idx] - dual_slack[idx]) * variable_rescaling[idx];
    }
}

__global__ void abs_vector_inplace_kernel(double *__restrict__ arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        arr[i] = fabs(arr[i]);
}

void display_iteration_stats(const pdhg_solver_state_t *state, bool verbose)
{
    if (!verbose)
    {
        return;
    }
    if (state->total_count % get_print_frequency(state->total_count) == 0)
    {
        printf("%6d %.1e | %8.1e  %8.1e | %.1e %.1e %.1e | %.1e %.1e %.1e \n",
               state->total_count,
               state->cumulative_time_sec,
               state->primal_objective_value,
               state->dual_objective_value,
               state->absolute_primal_residual,
               state->absolute_dual_residual,
               state->objective_gap,
               state->relative_primal_residual,
               state->relative_dual_residual,
               state->relative_objective_gap);
    }
}

void pdhg_final_log(
    const cuhsdlp_result_t *result,
    const pdhg_parameters_t *params)
{
    if (params->verbose)
    {
        printf("-------------------------------------------------------------------------------------\n");
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
    }
}


__global__ void fill_ones_kernel(double *__restrict__ arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        arr[i] = 1.0;
}
