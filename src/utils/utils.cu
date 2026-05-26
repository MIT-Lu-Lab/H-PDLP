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

struct AbsTransform
{
    __host__ __device__ double operator()(const double &x) const
    {
        return fabs(x);
    }
};

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
    auto abs_input = thrust::make_transform_iterator(input, AbsTransform{});
    CUDA_CHECK(cub::DeviceReduce::Max(
        state->cub_reduce_workspace_d,
        state->cub_reduce_workspace_bytes,
        abs_input,
        output,
        length,
        state->stream_A));
    CUDA_CHECK(cudaGetLastError());
}

void compute_lp_problem_norms(
    const lp_problem_t *problem,
    norm_type_t optimality_norm,
    double *out_objective_norm,
    double *out_bound_norm,
    double *out_constraint_frobenius_norm)
{
    const int n_vars = problem->num_variables;
    const int n_cons = problem->num_constraints;
    const int nnz    = problem->constraint_matrix_num_nonzeros;
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

    double frob_acc = 0.0;
    for (int i = 0; i < nnz; ++i)
    {
        double v = problem->constraint_matrix_values[i];
        frob_acc += v * v;
    }

    *out_objective_norm            = obj_acc;
    *out_bound_norm                = bnd_acc;
    *out_constraint_frobenius_norm = sqrt(frob_acc);
}
