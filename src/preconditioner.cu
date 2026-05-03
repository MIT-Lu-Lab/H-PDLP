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

#include "preconditioner.h"
#include "power_iteration.h"
#include "utils.h"
#include <cub/device/device_reduce.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define SCALING_EPSILON 1e-12

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
static void compute_objective_and_bound_norms(
    pdhg_solver_state_t *state,
    norm_type_t optimality_norm,
    double *out_objective_norm,
    double *out_bound_norm);
__global__ void scale_variables_kernel(
    double *__restrict__ obj_vec,
    double *__restrict__ var_lb,
    double *__restrict__ var_ub,
    double *__restrict__ init_primal_sol,
    const double *__restrict__ var_rescaling,
    int n_vars);
__global__ void scale_constraints_kernel(
    double *__restrict__ cons_lb,
    double *__restrict__ cons_ub,
    double *__restrict__ init_dual_sol,
    const double *__restrict__ cons_rescaling,
    int n_cons);
__global__ void scale_csr_nnz_kernel(
    const int *__restrict__ cons_row_ind,
    const int *__restrict__ cons_col_ind,
    double *__restrict__ cons_val,
    double *__restrict__ cons_transpose_val,
    const int *__restrict__ cons_to_transpose_pos,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ cons_rescaling,
    int n_nnzs);
__global__ void compute_csr_row_absmax_kernel(
    const int *__restrict__ row_ptr,
    const double *__restrict__ matrix_vals,
    int n_rows,
    double *__restrict__ row_absmax_values);
__global__ void compute_csr_row_powsum_kernel(
    const int *__restrict__ row_ptr,
    const double *__restrict__ matrix_vals,
    int n_rows,
    double degree,
    double *__restrict__ row_powsum_values);
__global__ void compute_bound_objective_scales_kernel(
    double *cons_scale_d,
    double *obj_scale_d,
    const double *bound_norm,
    const double *objective_norm);
__global__ void clamp_sqrt_and_accum_kernel(
    double *__restrict__ scaling_factors,
    double *__restrict__ cumulative_rescaling,
    int n_vars);
__global__ void compute_bound_contrib_kernel(
    double *__restrict__ contrib,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    int n_cons);
__global__ void scale_bounds_kernel(
    double *__restrict__ cons_lb,
    double *__restrict__ cons_ub,
    double *__restrict__ init_dual_sol,
    const double *__restrict__ cons_scale_d,
    const double *__restrict__ obj_scale_d,
    int n_cons);
__global__ void scale_objective_kernel(
    double *__restrict__ obj_vec,
    double *__restrict__ var_lb,
    double *__restrict__ var_ub,
    double *__restrict__ init_primal_sol,
    const double *__restrict__ cons_scale_d,
    const double *__restrict__ obj_scale_d,
    int n_vars);
static void scale_problem(
    pdhg_solver_state_t *state,
    double *constraint_rescaling,
    double *variable_rescaling);
static void ruiz_rescaling(
    pdhg_solver_state_t *state,
    int num_iters,
    double *constraint_rescaling,
    double *variable_rescaling);
static void pock_chambolle_rescaling(
    pdhg_solver_state_t *state,
    const double alpha,
    double *constraint_rescaling,
    double *variable_rescaling);
static void bound_objective_rescaling(
    pdhg_solver_state_t *state);

static void scale_problem(
    pdhg_solver_state_t *state,
    double *constraint_rescaling,
    double *variable_rescaling)
{
    int num_variables = state->num_variables;
    int num_constraints = state->num_constraints;
    int num_nonzeros = state->constraint_matrix->num_nonzeros;

    scale_variables_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->objective_vector,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->initial_primal_solution_d,
        variable_rescaling,
        num_variables);
    CUDA_CHECK(cudaGetLastError());

    scale_constraints_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->initial_dual_solution_d,
        constraint_rescaling,
        num_constraints);
    CUDA_CHECK(cudaGetLastError());

    scale_csr_nnz_kernel<<<state->num_blocks_nnz, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->row_ind,
        state->constraint_matrix->col_ind,
        state->constraint_matrix->val,
        state->constraint_matrix_t->val,
        state->constraint_matrix->transpose_map,
        variable_rescaling,
        constraint_rescaling,
        num_nonzeros);
    CUDA_CHECK(cudaGetLastError());
}

static void ruiz_rescaling(
    pdhg_solver_state_t *state,
    int num_iterations,
    double *constraint_rescaling,
    double *variable_rescaling)
{
    const int num_constraints = state->num_constraints;
    const int num_variables = state->num_variables;

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        compute_csr_row_absmax_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            state->constraint_matrix->row_ptr,
            state->constraint_matrix->val,
            num_constraints,
            constraint_rescaling);
        clamp_sqrt_and_accum_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            constraint_rescaling,
            state->constraint_rescaling_d,
            num_constraints);

        compute_csr_row_absmax_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->constraint_matrix_t->row_ptr,
            state->constraint_matrix_t->val,
            num_variables,
            variable_rescaling);
        clamp_sqrt_and_accum_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            variable_rescaling,
            state->variable_rescaling_d,
            num_variables);

        scale_problem(state, constraint_rescaling, variable_rescaling);
    }
}

static void pock_chambolle_rescaling(
    pdhg_solver_state_t *state,
    const double alpha,
    double *constraint_rescaling,
    double *variable_rescaling)
{
    const int num_constraints = state->num_constraints;
    const int num_variables = state->num_variables;

    compute_csr_row_powsum_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->row_ptr,
        state->constraint_matrix->val,
        num_constraints,
        alpha,
        constraint_rescaling);
    clamp_sqrt_and_accum_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        constraint_rescaling,
        state->constraint_rescaling_d,
        num_constraints);

    compute_csr_row_powsum_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->val,
        num_variables,
        2.0 - alpha,
        variable_rescaling);
    clamp_sqrt_and_accum_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        variable_rescaling,
        state->variable_rescaling_d,
        num_variables);

    scale_problem(state, constraint_rescaling, variable_rescaling);
}

static void bound_objective_rescaling(
    pdhg_solver_state_t *state)
{
    const int num_constraints = state->num_constraints;
    const int num_variables = state->num_variables;

    double *bound_contrib_d = state->dual_vec_buffer_d;
    compute_bound_contrib_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        bound_contrib_d,
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        num_constraints);

    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace,
        state->cub_reduce_workspace_bytes,
        bound_contrib_d,
        state->precond_bound_norm_mp,
        num_constraints,
        state->stream_A));

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A,
        state->num_variables,
        state->objective_vector,
        1,
        state->precond_objective_norm_mp));

    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    *state->precond_bound_norm_mp = sqrt(*state->precond_bound_norm_mp);

    compute_bound_objective_scales_kernel<<<1, 1, 0, state->stream_A>>>(
        state->constraint_bound_rescaling_d,
        state->objective_vector_rescaling_d,
        state->precond_bound_norm_mp,
        state->precond_objective_norm_mp);

    scale_bounds_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->initial_dual_solution_d,
        state->constraint_bound_rescaling_d,
        state->objective_vector_rescaling_d,
        num_constraints);

    scale_objective_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->objective_vector,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->initial_primal_solution_d,
        state->constraint_bound_rescaling_d,
        state->objective_vector_rescaling_d,
        num_variables);

    CUDA_CHECK(cudaMemcpyAsync(&state->constraint_bound_rescaling,
                               state->constraint_bound_rescaling_d,
                               sizeof(double), cudaMemcpyDeviceToHost, state->stream_A));
    CUDA_CHECK(cudaMemcpyAsync(&state->objective_vector_rescaling,
                               state->objective_vector_rescaling_d,
                               sizeof(double), cudaMemcpyDeviceToHost, state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
}

__global__ void compute_bound_objective_scales_kernel(
    double *cons_scale_d,
    double *obj_scale_d,
    const double *bound_norm,
    const double *objective_norm)
{
    *cons_scale_d = 1.0 / (bound_norm[0] + 1.0);
    *obj_scale_d = 1.0 / (objective_norm[0] + 1.0);
}

void rescale_problem(
    const pdhg_parameters_t *params,
    pdhg_solver_state_t *state)
{
    const scaling_parameters_t *scaling_params = &params->scaling_params;
    const power_iteration_parameters_t *power_iter_params = &params->power_iter_params;
    norm_type_t optimality_norm = params->optimality_norm;
    bool verbose = params->verbose;

    if (verbose)
    {
        printf("\nPreconditioning...\n");
    }

    int num_variables = state->num_variables;
    int num_constraints = state->num_constraints;

    compute_objective_and_bound_norms(state, optimality_norm,
        &state->original_objective_norm, &state->original_bound_norm);
    state->original_constraint_norm = estimate_maximum_singular_value(state, power_iter_params);

    clock_t start_rescaling = clock();

    fill_ones_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_rescaling_d, 
        num_constraints);
    fill_ones_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->variable_rescaling_d, 
        num_variables);

    double *constraint_rescaling = NULL;
    double *variable_rescaling = NULL;
    CUDA_CHECK(cudaMalloc(&constraint_rescaling, num_constraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&variable_rescaling, num_variables * sizeof(double)));

    if (scaling_params->l_inf_ruiz_iterations > 0)
    {
        if (verbose)
        {
            printf("  Ruiz scaling (%d iterations)\n", scaling_params->l_inf_ruiz_iterations);
        }
        ruiz_rescaling(state, scaling_params->l_inf_ruiz_iterations, constraint_rescaling, variable_rescaling);
    }
    if (scaling_params->has_pock_chambolle_alpha)
    {
        if (verbose)
        {
            printf("  Pock-Chambolle scaling (alpha=%.4f)\n", scaling_params->pock_chambolle_alpha);
        }
        pock_chambolle_rescaling(state, scaling_params->pock_chambolle_alpha, constraint_rescaling, variable_rescaling);
    }

    state->constraint_bound_rescaling = 1.0;
    state->objective_vector_rescaling = 1.0;
    CUDA_CHECK(cudaMemcpy(state->constraint_bound_rescaling_d, &HOST_ONE, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->objective_vector_rescaling_d, &HOST_ONE, sizeof(double), cudaMemcpyHostToDevice));
    if (scaling_params->bound_objective_rescaling)
    {
        if (verbose)
        {
            printf("  Bound-objective scaling\n");
        }
        bound_objective_rescaling(state);
    }

    state->rescaling_time_sec = (double)(clock() - start_rescaling) / CLOCKS_PER_SEC;

    compute_objective_and_bound_norms(state, optimality_norm, &state->rescaled_objective_norm, &state->rescaled_bound_norm);
    state->rescaled_constraint_norm = estimate_maximum_singular_value(state, power_iter_params);

    CUDA_CHECK(cudaFree(constraint_rescaling));
    CUDA_CHECK(cudaFree(variable_rescaling));

    if (verbose)
    {
        printf("  %-20s %15s %15s\n", "", "original", "rescaled");
        printf("  %-20s %15.6e %15.6e\n", "sigma_max(A)",
               state->original_constraint_norm, state->rescaled_constraint_norm);
        printf("  %-20s %15.6e %15.6e\n", "||c||",
               state->original_objective_norm, state->rescaled_objective_norm);
        printf("  %-20s %15.6e %15.6e\n", "||b||",
               state->original_bound_norm, state->rescaled_bound_norm);
        printf("  %-20s : %.3g sec\n", "precondition time", state->rescaling_time_sec);
        printf("\n");
    }
}

__global__ void scale_variables_kernel(double *__restrict__ obj_vec,
                                       double *__restrict__ var_lb,
                                       double *__restrict__ var_ub,
                                       double *__restrict__ init_primal_sol,
                                       const double *__restrict__ var_rescaling,
                                       int n_vars)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_vars)
        return;
    double dj = var_rescaling[j];
    double inv_dj = 1.0 / dj;
    obj_vec[j] *= inv_dj;
    var_lb[j] *= dj;
    var_ub[j] *= dj;
    init_primal_sol[j] *= dj;
}

__global__ void scale_constraints_kernel(double *__restrict__ cons_lb,
                                         double *__restrict__ cons_ub,
                                         double *__restrict__ init_dual_sol,
                                         const double *__restrict__ cons_rescaling,
                                         int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons)
        return;
    double ei = cons_rescaling[i];
    double inv_ei = 1.0 / ei;
    cons_lb[i] *= inv_ei;
    cons_ub[i] *= inv_ei;
    init_dual_sol[i] *= ei;
}

__global__ void scale_csr_nnz_kernel(
    const int *__restrict__ cons_row_ind,
    const int *__restrict__ cons_col_ind,
    double *__restrict__ cons_val,
    double *__restrict__ cons_transpose_val,
    const int *__restrict__ cons_to_transpose_pos,
    const double *__restrict__ var_rescaling,
    const double *__restrict__ cons_rescaling,
    int n_nnzs)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x;
         k < n_nnzs;
         k += gridDim.x * blockDim.x)
    {
        int i = cons_row_ind[k];
        int j = cons_col_ind[k];
        double scale = 1.0 / (var_rescaling[j] * cons_rescaling[i]);
        cons_val[k] *= scale;
        cons_transpose_val[cons_to_transpose_pos[k]] *= scale;
    }
}

__global__ void compute_csr_row_absmax_kernel(
    const int *__restrict__ row_ptr,
    const double *__restrict__ matrix_vals,
    int n_rows,
    double *__restrict__ row_absmax_values)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows)
        return;
    int s = row_ptr[i], e = row_ptr[i + 1];
    double m = 0.0;
    for (int k = s; k < e; ++k)
    {
        double v = fabs(matrix_vals[k]);
        if (!isfinite(v))
            v = 0.0;
        if (v > m)
            m = v;
    }
    row_absmax_values[i] = m;
}

__device__ __forceinline__ double pow_fast(
    double v, 
    double p)
{
    if (p == 2.0)
        return v * v;
    if (p == 1.0)
        return v;
    if (p == 0.5)
        return sqrt(v);
    return pow(v, p);
}

__global__ void compute_csr_row_powsum_kernel(
    const int *__restrict__ row_ptr,
    const double *__restrict__ matrix_vals,
    int n_rows,
    double degree,
    double *__restrict__ row_powsum_values)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows)
        return;
    int s = row_ptr[i], e = row_ptr[i + 1];
    double acc = 0.0;
    for (int k = s; k < e; ++k)
    {
        double v = fabs(matrix_vals[k]);
        if (!isfinite(v))
            v = 0.0;
        acc += pow_fast(v, degree);
    }
    row_powsum_values[i] = acc;
}

__global__ void clamp_sqrt_and_accum_kernel(
    double *__restrict__ scaling_factors,
    double *__restrict__ cumulative_rescaling,
    int n_vars)
{
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n_vars; t += blockDim.x * gridDim.x)
    {
        double v = scaling_factors[t];
        double s = (v < SCALING_EPSILON) ? 1.0 : sqrt(v);
        cumulative_rescaling[t] *= s;
        scaling_factors[t] = s;
    }
}

__global__ void compute_bound_contrib_kernel(
    double *__restrict__ contrib,
    const double *__restrict__ cons_lb,
    const double *__restrict__ cons_ub,
    int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons)
        return;

    double Li = cons_lb[i];
    double Ui = cons_ub[i];
    bool fL = isfinite(Li);
    bool fU = isfinite(Ui);

    double acc = 0.0;

    if (fL && (!fU || fabs(Li - Ui) > SCALING_EPSILON))
        acc += Li * Li;
    if (fU)
        acc += Ui * Ui;

    contrib[i] = acc;
}

__global__ void scale_bounds_kernel(
    double *__restrict__ cons_lb,
    double *__restrict__ cons_ub,
    double *__restrict__ init_dual_sol,
    const double *__restrict__ cons_scale_d,
    const double *__restrict__ obj_scale_d,
    int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons)
        return;
    double cons_scale = *cons_scale_d;
    double obj_scale = *obj_scale_d;
    cons_lb[i] *= cons_scale;
    cons_ub[i] *= cons_scale;
    init_dual_sol[i] *= obj_scale;
}

__global__ void scale_objective_kernel(
    double *__restrict__ obj_vec,
    double *__restrict__ var_lb,
    double *__restrict__ var_ub,
    double *__restrict__ init_primal_sol,
    const double *__restrict__ cons_scale_d,
    const double *__restrict__ obj_scale_d,
                                       int n_vars)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_vars)
        return;
    double cons_scale = *cons_scale_d;
    double obj_scale = *obj_scale_d;
    var_lb[j] *= cons_scale;
    var_ub[j] *= cons_scale;
    obj_vec[j] *= obj_scale;
    init_primal_sol[j] *= cons_scale;
}

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

static void compute_objective_and_bound_norms(
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

