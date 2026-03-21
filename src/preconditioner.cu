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
#include "utils.h"
#include <cub/device/device_reduce.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define SCALING_EPSILON 1e-12

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
    int nnz);
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
    int n_cons,
    double cons_scale,
    double obj_scale);
__global__ void scale_objective_kernel(
    double *__restrict__ obj_vec,
    double *__restrict__ var_lb,
    double *__restrict__ var_ub,
    double *__restrict__ init_primal_sol,
    int n_vars,
    double cons_scale,
    double obj_scale);
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

    scale_variables_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->objective_vector,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->initial_primal_solution,
        variable_rescaling,
        num_variables);
    CUDA_CHECK(cudaGetLastError());

    scale_constraints_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->initial_dual_solution,
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
        state->constraint_matrix->num_nonzeros);
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
            state->constraint_rescaling,
            num_constraints);

        compute_csr_row_absmax_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->constraint_matrix_t->row_ptr,
            state->constraint_matrix_t->val,
            num_variables,
            variable_rescaling);
        clamp_sqrt_and_accum_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            variable_rescaling,
            state->variable_rescaling,
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
        state->constraint_rescaling,
        num_constraints);

    compute_csr_row_powsum_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->val,
        num_variables,
        2.0 - alpha,
        variable_rescaling);
    clamp_sqrt_and_accum_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        variable_rescaling,
        state->variable_rescaling,
        num_variables);

    scale_problem(state, constraint_rescaling, variable_rescaling);
}

static void bound_objective_rescaling(
    pdhg_solver_state_t *state)
{
    const int num_constraints = state->num_constraints;
    const int num_variables = state->num_variables;

    double *bound_contrib_d = state->dual_vec_buffer;
    compute_bound_contrib_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        bound_contrib_d,
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        num_constraints);

    size_t cub_buffer_bytes = state->cub_reduce_workspace_bytes;
    CUDA_CHECK(cub::DeviceReduce::Sum(state->cub_reduce_workspace, cub_buffer_bytes,
                                       bound_contrib_d, state->scalar_pool_d, num_constraints));

    double bnd_norm_sq_h = 0.0;
    CUDA_CHECK(cudaMemcpy(&bnd_norm_sq_h, state->scalar_pool_d, sizeof(double), cudaMemcpyDeviceToHost));
    double bnd_norm = sqrt(bnd_norm_sq_h);

    double obj_norm = 0.0;
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle_A, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle_A,
                             state->num_variables,
                             state->objective_vector, 1,
                             &obj_norm));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle_A, CUBLAS_POINTER_MODE_DEVICE));

    double constraint_scale = 1.0 / (bnd_norm + 1.0);
    double objective_scale = 1.0 / (obj_norm + 1.0);

    scale_bounds_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->initial_dual_solution,
        num_constraints,
        constraint_scale,
        objective_scale);

    scale_objective_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->objective_vector,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->initial_primal_solution,
        num_variables,
        constraint_scale,
        objective_scale);

    state->constraint_bound_rescaling = constraint_scale;
    state->objective_vector_rescaling = objective_scale;
}

void rescale_problem(
    const pdhg_parameters_t *params,
    pdhg_solver_state_t *state)
{
    if (params->verbose)
    {
        printf("\nPreconditioning...\n");
    }

    int num_variables = state->num_variables;
    int num_constraints = state->num_constraints;

    clock_t start_rescaling = clock();

    fill_ones_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_rescaling, num_constraints);
    fill_ones_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->variable_rescaling, num_variables);

    double *constraint_rescaling = NULL;
    double *variable_rescaling = NULL;
    CUDA_CHECK(cudaMalloc(&constraint_rescaling, num_constraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&variable_rescaling, num_variables * sizeof(double)));

    if (params->l_inf_ruiz_iterations > 0)
    {
        if (params->verbose)
        {
            printf("  Ruiz scaling (%d iterations)\n", params->l_inf_ruiz_iterations);
        }
        ruiz_rescaling(state, params->l_inf_ruiz_iterations, constraint_rescaling, variable_rescaling);
    }
    if (params->has_pock_chambolle_alpha)
    {
        if (params->verbose)
        {
            printf("  Pock-Chambolle scaling (alpha=%.4f)\n", params->pock_chambolle_alpha);
        }
        pock_chambolle_rescaling(state, params->pock_chambolle_alpha, constraint_rescaling, variable_rescaling);
    }

    state->constraint_bound_rescaling = 1.0;
    state->objective_vector_rescaling = 1.0;
    if (params->bound_objective_rescaling)
    {
        if (params->verbose)
        {
            printf("  Bound-objective scaling\n");
        }
        bound_objective_rescaling(state);
    }

    state->rescaling_time_sec = (double)(clock() - start_rescaling) / CLOCKS_PER_SEC;

    CUDA_CHECK(cudaFree(constraint_rescaling));
    CUDA_CHECK(cudaFree(variable_rescaling));

    if (params->verbose)
    {
        printf("  %-15s : %.3g sec\n", "precondition time", state->rescaling_time_sec);
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

__global__ void scale_csr_nnz_kernel(const int *__restrict__ cons_row_ind,
                                     const int *__restrict__ cons_col_ind,
                                     double *__restrict__ cons_val,
                                     double *__restrict__ cons_transpose_val,
                                     const int *__restrict__ cons_to_transpose_pos,
                                     const double *__restrict__ var_rescaling,
                                     const double *__restrict__ cons_rescaling,
                                     int nnz)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x;
         k < nnz;
         k += gridDim.x * blockDim.x)
    {
        int i = cons_row_ind[k];
        int j = cons_col_ind[k];
        double scale = 1.0 / (var_rescaling[j] * cons_rescaling[i]);
        cons_val[k] *= scale;
        cons_transpose_val[cons_to_transpose_pos[k]] *= scale;
    }
}

__global__ void compute_csr_row_absmax_kernel(const int *__restrict__ row_ptr,
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

__device__ __forceinline__ double pow_fast(double v, double p)
{
    if (p == 2.0)
        return v * v;
    if (p == 1.0)
        return v;
    if (p == 0.5)
        return sqrt(v);
    return pow(v, p);
}

__global__ void compute_csr_row_powsum_kernel(const int *__restrict__ row_ptr,
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

__global__ void clamp_sqrt_and_accum_kernel(double *__restrict__ scaling_factors,
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

__global__ void compute_bound_contrib_kernel(double *__restrict__ contrib,
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

__global__ void scale_bounds_kernel(double *__restrict__ cons_lb,
                                    double *__restrict__ cons_ub,
                                    double *__restrict__ init_dual_sol,
                                    int n_cons,
                                    double cons_scale,
                                    double obj_scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons)
        return;
    cons_lb[i] *= cons_scale;
    cons_ub[i] *= cons_scale;
    init_dual_sol[i] *= obj_scale;
}

__global__ void scale_objective_kernel(double *__restrict__ obj_vec,
                                       double *__restrict__ var_lb,
                                       double *__restrict__ var_ub,
                                       double *__restrict__ init_primal_sol,
                                       int n_vars,
                                       double cons_scale,
                                       double obj_scale)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_vars)
        return;
    var_lb[j] *= cons_scale;
    var_ub[j] *= cons_scale;
    obj_vec[j] *= obj_scale;
    init_primal_sol[j] *= cons_scale;
}

