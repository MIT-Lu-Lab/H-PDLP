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
#include <thrust/iterator/transform_iterator.h>

#ifndef CUHSDLP_VERSION
#define CUHSDLP_VERSION "unknown"
#endif

const double HOST_ONE = 1.0;
const double HOST_ZERO = 0.0;

#define MATRIX_LARGE_VALUE 1e15
#define OBJECTIVE_LARGE_VALUE 1e20

static __global__ void compute_objective_norm_entries_kernel(
    const double *__restrict__ obj_vec,
    double *__restrict__ entries,
    int n_vars,
    int use_linf);
static __global__ void compute_bound_norm_entries_kernel(
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    double *__restrict__ entries,
    int n_cons,
    int use_linf);

void filter_constraint_matrix_entries(
    lp_problem_t *working_problem,
    const lp_problem_t *original_problem,
    const pdhg_parameters_t *params)
{
    const int num_rows = original_problem->num_constraints;
    const int nnz = original_problem->constraint_matrix_num_nonzeros;

    if (num_rows == 0 || nnz == 0)
        return;

    const int *row_ptr = original_problem->constraint_matrix_row_pointers;
    const int *col_ind = original_problem->constraint_matrix_col_indices;
    const double *vals = original_problem->constraint_matrix_values;

    int filtered_nnz = 0;
    int num_large_entries = 0;
    double max_large_entry = 0.0;
    for (int i = 0; i < num_rows; ++i)
    {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        {
            const double abs_value = fabs(vals[k]);
            if (abs_value > params->scaling_params.matrix_zero_tol)
                ++filtered_nnz;
            if (abs_value >= MATRIX_LARGE_VALUE)
            {
                ++num_large_entries;
                if (abs_value > max_large_entry)
                    max_large_entry = abs_value;
            }
        }
    }

    if (num_large_entries > 0)
    {
        fprintf(stderr,
                "WARNING: %d constraint matrix %s |value| >= %.1e (largest %.3e): "
                "the problem is badly scaled.\n",
                num_large_entries,
                (num_large_entries == 1 ? "entry has" : "entries have"),
                MATRIX_LARGE_VALUE,
                max_large_entry);
    }

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

    working_problem->constraint_matrix_row_pointers = new_row_ptr;
    working_problem->constraint_matrix_col_indices = new_col_ind;
    working_problem->constraint_matrix_values = new_vals;
    working_problem->constraint_matrix_num_nonzeros = filtered_nnz;

    if (params->verbose)
    {
        printf("Dropped %d near-zero %s (|value| <= %.1e) from the constraint matrix\n",
               nnz - filtered_nnz,
               (nnz - filtered_nnz == 1 ? "entry" : "entries"),
               params->scaling_params.matrix_zero_tol);
    }
}

void warn_on_large_objective_coefficients(const lp_problem_t *problem)
{
    int num_large_coefficients = 0;
    double max_large_coefficient = 0.0;
    for (int j = 0; j < problem->num_variables; ++j)
    {
        const double abs_value = fabs(problem->objective_vector[j]);
        if (abs_value >= OBJECTIVE_LARGE_VALUE)
        {
            ++num_large_coefficients;
            if (abs_value > max_large_coefficient)
                max_large_coefficient = abs_value;
        }
    }

    if (num_large_coefficients > 0)
    {
        fprintf(stderr,
                "WARNING: %d objective %s |value| >= %.1e (largest %.3e): "
                "the problem is badly scaled.\n",
                num_large_coefficients,
                (num_large_coefficients == 1 ? "coefficient has" : "coefficients have"),
                OBJECTIVE_LARGE_VALUE,
                max_large_coefficient);
    }
}

static bool is_infinite_lower_bound(double bound, double infinite_bound)
{
    return bound <= -infinite_bound;
}

static bool is_infinite_upper_bound(double bound, double infinite_bound)
{
    return bound >= infinite_bound;
}

void replace_large_bounds_with_infinity(
    const double *original_lower,
    const double *original_upper,
    double **working_lower,
    double **working_upper,
    int length,
    double infinite_bound)
{
    bool has_large_bound = false;
    for (int i = 0; i < length && !has_large_bound; ++i)
        has_large_bound = is_infinite_lower_bound(original_lower[i], infinite_bound) ||
                          is_infinite_upper_bound(original_upper[i], infinite_bound);

    if (!has_large_bound)
        return;

    double *new_lower = (double *)safe_malloc((size_t)length * sizeof(double));
    double *new_upper = (double *)safe_malloc((size_t)length * sizeof(double));
    for (int i = 0; i < length; ++i)
    {
        new_lower[i] = is_infinite_lower_bound(original_lower[i], infinite_bound)
                           ? -INFINITY
                           : original_lower[i];
        new_upper[i] = is_infinite_upper_bound(original_upper[i], infinite_bound)
                           ? INFINITY
                           : original_upper[i];
    }

    *working_lower = new_lower;
    *working_upper = new_upper;
}

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

void allocate_zeroed_device_buffer(void **dst, size_t bytes)
{
    const size_t allocation_bytes = bytes > 0 ? bytes : 1;
    CUDA_CHECK(cudaMalloc(dst, allocation_bytes));
    CUDA_CHECK(cudaMemset(*dst, 0, allocation_bytes));
}

void allocate_and_copy_to_device(
    void **dst,
    const void *src,
    size_t bytes)
{
    const size_t allocation_bytes = bytes > 0 ? bytes : 1;
    CUDA_CHECK(cudaMalloc(dst, allocation_bytes));
    if (bytes > 0)
    {
        CUDA_CHECK(cudaMemcpy(
            *dst,
            src,
            bytes,
            cudaMemcpyHostToDevice));
    }
}

void allocate_cub_reduce_workspace(pdhg_solver_state_t *state)
{
    const int max_reduction_length =
        state->num_variables + state->num_constraints;
    size_t sum_bytes = 0;
    size_t max_bytes = 0;
    size_t abs_max_bytes = 0;
    double *null_values = NULL;
    auto absolute_values =
        thrust::make_transform_iterator(
            null_values,
            absolute_value_transform_t{});

    CUDA_CHECK(cub::DeviceReduce::Sum(
        NULL,
        sum_bytes,
        null_values,
        null_values,
        max_reduction_length));
    CUDA_CHECK(cub::DeviceReduce::Max(
        NULL,
        max_bytes,
        null_values,
        null_values,
        max_reduction_length));
    CUDA_CHECK(cub::DeviceReduce::Max(
        NULL,
        abs_max_bytes,
        absolute_values,
        null_values,
        max_reduction_length));

    size_t workspace_bytes =
        sum_bytes > max_bytes ? sum_bytes : max_bytes;
    if (abs_max_bytes > workspace_bytes)
        workspace_bytes = abs_max_bytes;
    state->cub_reduce_workspace_bytes = workspace_bytes;
    CUDA_CHECK(cudaMalloc(
        &state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes));
}

void compute_lp_problem_norms(
    const lp_problem_t *problem,
    norm_type_t optimality_norm,
    double infinite_bound,
    double *out_objective_norm,
    double *out_bound_norm)
{
    const int n_vars = problem->num_variables;
    const int n_cons = problem->num_constraints;
    const bool use_linf = (optimality_norm == NORM_TYPE_L_INF);

    double obj_acc = 0.0;
    for (int i = 0; i < n_vars; ++i)
    {
        double v = problem->objective_vector[i];
        if (use_linf) obj_acc = fmax(obj_acc, fabs(v));
        else          obj_acc += v * v;
    }
    if (!use_linf) obj_acc = sqrt(obj_acc);

    double bnd_acc = 0.0;
    for (int i = 0; i < n_cons; ++i)
    {
        double Li = problem->constraint_lower_bound[i];
        double Ui = problem->constraint_upper_bound[i];
        bool finite_Li = isfinite(Li) && !is_infinite_lower_bound(Li, infinite_bound);
        bool finite_Ui = isfinite(Ui) && !is_infinite_upper_bound(Ui, infinite_bound);
        if (use_linf)
        {
            if (finite_Li && Li != Ui) bnd_acc = fmax(bnd_acc, fabs(Li));
            if (finite_Ui)             bnd_acc = fmax(bnd_acc, fabs(Ui));
        }
        else
        {
            if (finite_Li && Li != Ui) bnd_acc += Li * Li;
            if (finite_Ui)             bnd_acc += Ui * Ui;
        }
    }
    if (!use_linf) bnd_acc = sqrt(bnd_acc);

    *out_objective_norm = obj_acc;
    *out_bound_norm     = bnd_acc;
}

void compute_lp_problem_norms_device(
    pdhg_solver_state_t *state,
    norm_type_t optimality_norm,
    double *out_objective_norm,
    double *out_bound_norm)
{
    const int use_linf = (optimality_norm == NORM_TYPE_L_INF);

    double *norm_entries_vars =
        state->variable_reduction_entries_d;
    double *norm_entries_cons =
        state->constraint_reduction_entries_d;

    compute_objective_norm_entries_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->objective_vector_d, norm_entries_vars, state->num_variables, use_linf);
    CUDA_CHECK(cudaGetLastError());

    compute_bound_norm_entries_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->constraint_lower_bound_d, state->constraint_upper_bound_d,
        norm_entries_cons, state->num_constraints, use_linf);
    CUDA_CHECK(cudaGetLastError());

    if (use_linf)
    {
        reduce_device_max(state, norm_entries_vars, state->num_variables,
                          state->objective_norm_workspace_mp);
        reduce_device_max(state, norm_entries_cons, state->num_constraints,
                          state->bound_norm_workspace_mp);
    }
    else
    {
        reduce_device_sum(state, norm_entries_vars, state->num_variables,
                          state->objective_norm_workspace_mp);
        reduce_device_sum(state, norm_entries_cons, state->num_constraints,
                          state->bound_norm_workspace_mp);
    }

    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));

    if (!use_linf) {
        *state->objective_norm_workspace_mp =
            sqrt(*state->objective_norm_workspace_mp);
        *state->bound_norm_workspace_mp =
            sqrt(*state->bound_norm_workspace_mp);
    }

    *out_objective_norm = *state->objective_norm_workspace_mp;
    *out_bound_norm = *state->bound_norm_workspace_mp;
}

__global__ void compute_delta_kernel(
    const double *__restrict__ initial_primal,
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
        delta_primal[i] = pdhg_primal[i] - initial_primal[i];
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        delta_dual[idx] = pdhg_dual[idx] - initial_dual[idx];
    }
}

static __global__ void compute_objective_norm_entries_kernel(
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

static __global__ void compute_bound_norm_entries_kernel(
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

void reduce_device_sum(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output)
{
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        input,
        output,
        length,
        state->stream_A));
    CUDA_CHECK(cudaGetLastError());
}

void reduce_device_max(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output)
{
    CUDA_CHECK(cub::DeviceReduce::Max(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        input,
        output,
        length,
        state->stream_A));
    CUDA_CHECK(cudaGetLastError());
}

void reduce_device_abs_max(
    pdhg_solver_state_t *state,
    const double *input,
    int length,
    double *output)
{
    auto abs_input =
        thrust::make_transform_iterator(
            input,
            absolute_value_transform_t{});
    CUDA_CHECK(cub::DeviceReduce::Max(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        abs_input,
        output,
        length,
        state->stream_A));
    CUDA_CHECK(cudaGetLastError());
}
