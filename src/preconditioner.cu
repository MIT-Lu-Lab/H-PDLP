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
#include "setup.h"
#include "utils.h"
#include <cub/device/device_reduce.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/system/cuda/execution_policy.h>
#include <time.h>

#define SCALING_EPSILON 1e-12

#define GEOMETRIC_MEAN_MULTIPLIER_MAX 1e20

#define MAX_LOG2_VECTOR_WIDTH 5

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
__global__ void compute_bound_contribution_kernel(
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
__global__ void compute_csr_max_row_length_kernel(
    const int *__restrict__ row_ptr,
    int n_rows,
    int *__restrict__ max_row_length);
__global__ void update_geometric_mean_multiplier_kernel(
    const int *__restrict__ row_ptr,
    const int *__restrict__ col_ind,
    const double *__restrict__ matrix_vals,
    int n_rows,
    int log2_width,
    const double *__restrict__ other_multiplier,
    double *__restrict__ multiplier);
__global__ void finalize_geometric_mean_scaling_kernel(
    const double *__restrict__ multiplier,
    double *__restrict__ scaling_factors,
    double *__restrict__ cumulative_rescaling,
    int n);
static void scale_problem(
    pdhg_solver_state_t *state,
    double *constraint_rescaling,
    double *variable_rescaling);
static void geometric_mean_rescaling(
    pdhg_solver_state_t *state,
    int num_iters,
    double *constraint_rescaling,
    double *variable_rescaling,
    double *constraint_multiplier,
    double *variable_multiplier);
static int compute_log2_vector_width(
    long long num_nonzeros,
    int num_rows,
    int longest_row);
static void compute_longest_csr_rows(
    pdhg_solver_state_t *state,
    int *longest);
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

    scale_variables_kernel<<<
        state->num_blocks_primal,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->objective_vector_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
        state->initial_primal_solution_d,
        variable_rescaling,
        num_variables);
    CUDA_CHECK(cudaGetLastError());

    scale_constraints_kernel<<<
        state->num_blocks_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        state->initial_dual_solution_d,
        constraint_rescaling,
        num_constraints);
    CUDA_CHECK(cudaGetLastError());

    scale_csr_nnz_kernel<<<
        state->num_blocks_nnz,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
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

static void geometric_mean_rescaling(
    pdhg_solver_state_t *state,
    int num_iterations,
    double *constraint_rescaling,
    double *variable_rescaling,
    double *constraint_multiplier,
    double *variable_multiplier)
{
    const int num_constraints = state->num_constraints;
    const int num_variables = state->num_variables;

    thrust::fill_n(
        thrust::cuda::par.on(state->stream_A),
        constraint_multiplier,
        num_constraints,
        1.0);
    thrust::fill_n(
        thrust::cuda::par.on(state->stream_A),
        variable_multiplier,
        num_variables,
        1.0);

    int longest[2] = {0, 0};
    compute_longest_csr_rows(state, longest);

    const long long num_nonzeros = state->constraint_matrix->num_nonzeros;
    const int row_log2 = compute_log2_vector_width(num_nonzeros, num_constraints, longest[0]);
    const int col_log2 = compute_log2_vector_width(num_nonzeros, num_variables, longest[1]);
    const int row_blocks =
        (int)((((long long)num_constraints << row_log2) + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);
    const int col_blocks =
        (int)((((long long)num_variables << col_log2) + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        update_geometric_mean_multiplier_kernel<<<
            row_blocks,
            THREADS_PER_BLOCK,
            0,
            state->stream_A>>>(
            state->constraint_matrix->row_ptr,
            state->constraint_matrix->col_ind,
            state->constraint_matrix->val,
            num_constraints,
            row_log2,
            variable_multiplier,
            constraint_multiplier);
        update_geometric_mean_multiplier_kernel<<<
            col_blocks,
            THREADS_PER_BLOCK,
            0,
            state->stream_A>>>(
            state->constraint_matrix_t->row_ptr,
            state->constraint_matrix_t->col_ind,
            state->constraint_matrix_t->val,
            num_variables,
            col_log2,
            constraint_multiplier,
            variable_multiplier);
    }

    finalize_geometric_mean_scaling_kernel<<<
        state->num_blocks_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        constraint_multiplier,
        constraint_rescaling,
        state->constraint_rescaling_d,
        num_constraints);
    finalize_geometric_mean_scaling_kernel<<<
        state->num_blocks_primal,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        variable_multiplier,
        variable_rescaling,
        state->variable_rescaling_d,
        num_variables);

    scale_problem(state, constraint_rescaling, variable_rescaling);
}

static int compute_log2_vector_width(
    long long num_nonzeros,
    int num_rows,
    int longest_row)
{
    if (longest_row >= 512)
        return MAX_LOG2_VECTOR_WIDTH;
    if (num_rows <= 0)
        return 0;

    const long long mean_nnz = (num_nonzeros + num_rows - 1) / num_rows;
    int log2_width = 0;
    while (log2_width < MAX_LOG2_VECTOR_WIDTH && (1LL << log2_width) < mean_nnz)
        ++log2_width;
    return log2_width;
}

static void compute_longest_csr_rows(
    pdhg_solver_state_t *state,
    int *longest)
{
    int *longest_d = NULL;
    CUDA_CHECK(cudaMalloc(&longest_d, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(longest_d, 0, 2 * sizeof(int), state->stream_A));

    compute_csr_max_row_length_kernel<<<
        state->num_blocks_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->constraint_matrix->row_ptr,
        state->num_constraints,
        longest_d);
    compute_csr_max_row_length_kernel<<<
        state->num_blocks_primal,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->constraint_matrix_t->row_ptr,
        state->num_variables,
        longest_d + 1);

    CUDA_CHECK(cudaMemcpyAsync(
        longest,
        longest_d,
        2 * sizeof(int),
        cudaMemcpyDeviceToHost,
        state->stream_A));
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    CUDA_CHECK(cudaFree(longest_d));
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
        compute_csr_row_absmax_kernel<<<
            state->num_blocks_dual,
            THREADS_PER_BLOCK,
            0,
            state->stream_A>>>(
            state->constraint_matrix->row_ptr,
            state->constraint_matrix->val,
            num_constraints,
            constraint_rescaling);
        clamp_sqrt_and_accum_kernel<<<
            state->num_blocks_dual,
            THREADS_PER_BLOCK,
            0,
            state->stream_A>>>(
            constraint_rescaling,
            state->constraint_rescaling_d,
            num_constraints);

        compute_csr_row_absmax_kernel<<<
            state->num_blocks_primal,
            THREADS_PER_BLOCK,
            0,
            state->stream_A>>>(
            state->constraint_matrix_t->row_ptr,
            state->constraint_matrix_t->val,
            num_variables,
            variable_rescaling);
        clamp_sqrt_and_accum_kernel<<<
            state->num_blocks_primal,
            THREADS_PER_BLOCK,
            0,
            state->stream_A>>>(
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

    compute_csr_row_powsum_kernel<<<
        state->num_blocks_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->constraint_matrix->row_ptr,
        state->constraint_matrix->val,
        num_constraints,
        alpha,
        constraint_rescaling);
    clamp_sqrt_and_accum_kernel<<<
        state->num_blocks_dual,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        constraint_rescaling,
        state->constraint_rescaling_d,
        num_constraints);

    compute_csr_row_powsum_kernel<<<
        state->num_blocks_primal,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
        state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->val,
        num_variables,
        2.0 - alpha,
        variable_rescaling);
    clamp_sqrt_and_accum_kernel<<<
        state->num_blocks_primal,
        THREADS_PER_BLOCK,
        0,
        state->stream_A>>>(
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

    double *bound_contrib_d =
        state->constraint_reduction_entries_d;
    compute_bound_contribution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        bound_contrib_d,
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        num_constraints);

    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        bound_contrib_d,
        state->bound_norm_workspace_mp,
        num_constraints,
        state->stream_A));

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        state->blas_handle_A,
        state->num_variables,
        state->objective_vector_d,
        1,
        state->objective_norm_workspace_mp));

    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    *state->bound_norm_workspace_mp =
        sqrt(*state->bound_norm_workspace_mp);

    compute_bound_objective_scales_kernel<<<1, 1, 0, state->stream_A>>>(
        state->constraint_bound_rescaling_d,
        state->objective_vector_rescaling_d,
        state->bound_norm_workspace_mp,
        state->objective_norm_workspace_mp);

    scale_bounds_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->constraint_lower_bound_d,
        state->constraint_upper_bound_d,
        state->initial_dual_solution_d,
        state->constraint_bound_rescaling_d,
        state->objective_vector_rescaling_d,
        num_constraints);

    scale_objective_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream_A>>>(
        state->objective_vector_d,
        state->variable_lower_bound_d,
        state->variable_upper_bound_d,
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

    clock_t start_rescaling = clock();

    thrust::fill_n(
        thrust::cuda::par.on(state->stream_A),
        state->constraint_rescaling_d,
        num_constraints,
        1.0);
    thrust::fill_n(
        thrust::cuda::par.on(state->stream_A),
        state->variable_rescaling_d,
        num_variables,
        1.0);

    double *constraint_rescaling = NULL;
    double *variable_rescaling = NULL;
    const bool needs_equilibration_workspace =
        scaling_params->geometric_mean_iterations > 0 ||
        scaling_params->l_inf_ruiz_iterations > 0 ||
        scaling_params->has_pock_chambolle_alpha;
    if (needs_equilibration_workspace)
    {
        CUDA_CHECK(cudaMalloc(
            &constraint_rescaling,
            (size_t)((num_constraints > 0)
                         ? num_constraints
                         : 1) *
                sizeof(double)));
        CUDA_CHECK(cudaMalloc(
            &variable_rescaling,
            (size_t)((num_variables > 0)
                         ? num_variables
                         : 1) *
                sizeof(double)));
    }

    if (scaling_params->geometric_mean_iterations > 0)
    {
        if (verbose)
        {
            printf("  Geometric-mean scaling (%d iterations)\n", scaling_params->geometric_mean_iterations);
        }

        double *constraint_multiplier = NULL;
        double *variable_multiplier = NULL;
        CUDA_CHECK(cudaMalloc(
            &constraint_multiplier,
            (size_t)((num_constraints > 0) ? num_constraints : 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
            &variable_multiplier,
            (size_t)((num_variables > 0) ? num_variables : 1) * sizeof(double)));

        geometric_mean_rescaling(
            state,
            scaling_params->geometric_mean_iterations,
            constraint_rescaling,
            variable_rescaling,
            constraint_multiplier,
            variable_multiplier);

        CUDA_CHECK(cudaFree(constraint_multiplier));
        CUDA_CHECK(cudaFree(variable_multiplier));
    }
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

    initialize_spmv_plans(state);

    compute_lp_problem_norms_device(state, optimality_norm, &state->rescaled_objective_norm, &state->rescaled_bound_norm);
    if (state->constraint_matrix->num_nonzeros == 0)
    {
        state->rescaled_constraint_norm = 0.0;
    }
    else
    {
        sv_estimator_opts_t opts = {};
        opts.max_iter = power_iter_params->max_iter;
        opts.tol = power_iter_params->tol;
        sv_estimator_ctx_t *estimator = sv_estimator_create(
            state->sparse_handle_A,
            state->blas_handle_A,
            state->stream_A,
            state->curand_gen,
            state->constraint_matrix,
            state->constraint_matrix_t,
            state->num_constraints,
            state->num_variables);
        const sv_estimator_result_t estimate =
            sv_estimator_run(estimator, &opts);
        sv_estimator_free(estimator);
        state->rescaled_constraint_norm = estimate.max_singular_value;
    }
    state->rescaling_time_sec = (double)(clock() - start_rescaling) / CLOCKS_PER_SEC;

    if (constraint_rescaling)
        CUDA_CHECK(cudaFree(constraint_rescaling));
    if (variable_rescaling)
        CUDA_CHECK(cudaFree(variable_rescaling));

    if (verbose)
    {
        if (params->diagnose)
        {
            printf("  %-20s %15s %15s\n", "", "original", "rescaled");
            printf("  %-20s %15s %15.6e\n", "||A||",
                   "-", state->rescaled_constraint_norm);
            printf("  %-20s %15.6e %15.6e\n", "||c||",
                   state->original_objective_norm, state->rescaled_objective_norm);
            printf("  %-20s %15.6e %15.6e\n", "||b||",
                   state->original_bound_norm, state->rescaled_bound_norm);
        }
        printf("  %-20s : %.3g sec\n", "precondition time", state->rescaling_time_sec);
        printf("\n");
    }
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

__global__ void compute_bound_objective_scales_kernel(
    double *cons_scale_d,
    double *obj_scale_d,
    const double *bound_norm,
    const double *objective_norm)
{
    *cons_scale_d = 1.0 / (bound_norm[0] + 1.0);
    *obj_scale_d = 1.0 / (objective_norm[0] + 1.0);
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

__global__ void compute_bound_contribution_kernel(
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

__global__ void compute_csr_max_row_length_kernel(
    const int *__restrict__ row_ptr,
    int n_rows,
    int *__restrict__ max_row_length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows)
        atomicMax(max_row_length, row_ptr[i + 1] - row_ptr[i]);
}

__global__ void update_geometric_mean_multiplier_kernel(
    const int *__restrict__ row_ptr,
    const int *__restrict__ col_ind,
    const double *__restrict__ matrix_vals,
    int n_rows,
    int log2_width,
    const double *__restrict__ other_multiplier,
    double *__restrict__ multiplier)
{
    const int width = 1 << log2_width;
    const int groups_per_block = blockDim.x >> log2_width;
    const int row = blockIdx.x * groups_per_block + (threadIdx.x >> log2_width);
    const int lane = threadIdx.x & (width - 1);

    double lo = INFINITY;
    double hi = 0.0;
    if (row < n_rows)
    {
        const int s = row_ptr[row], e = row_ptr[row + 1];
        for (int k = s + lane; k < e; k += width)
        {
            const double scaled = fabs(matrix_vals[k]) * other_multiplier[col_ind[k]];
            if (!(scaled > 0.0 && isfinite(scaled)))
                continue;
            lo = fmin(lo, scaled);
            hi = fmax(hi, scaled);
        }
    }

    for (int offset = width >> 1; offset > 0; offset >>= 1)
    {
        lo = fmin(lo, __shfl_down_sync(0xffffffffu, lo, offset, width));
        hi = fmax(hi, __shfl_down_sync(0xffffffffu, hi, offset, width));
    }

    if (lane != 0 || row >= n_rows || hi <= 0.0)
        return;

    const double updated = 1.0 / (sqrt(lo) * sqrt(hi));
    multiplier[row] = fmin(
        fmax(updated, 1.0 / GEOMETRIC_MEAN_MULTIPLIER_MAX),
        GEOMETRIC_MEAN_MULTIPLIER_MAX);
}

__global__ void finalize_geometric_mean_scaling_kernel(
    const double *__restrict__ multiplier,
    double *__restrict__ scaling_factors,
    double *__restrict__ cumulative_rescaling,
    int n)
{
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += blockDim.x * gridDim.x)
    {
        const double divisor = 1.0 / multiplier[t];
        cumulative_rescaling[t] *= divisor;
        scaling_factors[t] = divisor;
    }
}
