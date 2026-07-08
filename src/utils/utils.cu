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

void filter_constraint_matrix_entries(lp_problem_t *problem, const pdhg_parameters_t *params)
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

void compute_lp_problem_norms(
    const lp_problem_t *problem,
    norm_type_t optimality_norm,
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
        if (use_linf)
        {
            if (isfinite(Li) && Li != Ui) bnd_acc = fmax(bnd_acc, fabs(Li));
            if (isfinite(Ui))             bnd_acc = fmax(bnd_acc, fabs(Ui));
        }
        else
        {
            if (isfinite(Li) && Li != Ui) bnd_acc += Li * Li;
            if (isfinite(Ui))             bnd_acc += Ui * Ui;
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
