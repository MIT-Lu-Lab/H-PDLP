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
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_LARGE_VALUE 1e15
#define OBJECTIVE_LARGE_VALUE 1e20

static bool entry_is_near_zero(double value, double matrix_zero_tol);
static bool is_infinite_lower_bound(double bound, double infinite_bound);
static bool is_infinite_upper_bound(double bound, double infinite_bound);

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

void filter_constraint_matrix_entries(
    const lp_problem_t *original_problem,
    const pdhg_parameters_t *params,
    int **owned_row_pointers,
    int **owned_col_indices,
    double **owned_values)
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
            if (!entry_is_near_zero(vals[k], params->scaling_params.matrix_zero_tol))
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
    *owned_row_pointers = new_row_ptr;
    *owned_col_indices = new_col_ind;
    *owned_values = new_vals;

    int pos = 0;
    new_row_ptr[0] = 0;
    for (int i = 0; i < num_rows; ++i)
    {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        {
            if (entry_is_near_zero(vals[k], params->scaling_params.matrix_zero_tol))
                continue;
            new_col_ind[pos] = col_ind[k];
            new_vals[pos] = vals[k];
            ++pos;
        }
        new_row_ptr[i + 1] = pos;
    }

    if (params->verbose)
    {
        printf("Dropped %d near-zero %s (|value| <= %.1e) from the constraint matrix\n",
               nnz - filtered_nnz,
               (nnz - filtered_nnz == 1 ? "entry" : "entries"),
               params->scaling_params.matrix_zero_tol);
    }
}

static bool entry_is_near_zero(double value, double matrix_zero_tol)
{
    return fabs(value) <= matrix_zero_tol;
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

void replace_large_bounds_with_infinity(
    const double *original_lower,
    const double *original_upper,
    double **working_lower,
    double **working_upper,
    int length,
    double infinite_bound)
{
    bool needs_bound_conversion = false;
    for (int i = 0; i < length && !needs_bound_conversion; ++i)
        needs_bound_conversion =
            (isfinite(original_lower[i]) &&
             is_infinite_lower_bound(original_lower[i], infinite_bound)) ||
            (isfinite(original_upper[i]) &&
             is_infinite_upper_bound(original_upper[i], infinite_bound));

    if (!needs_bound_conversion)
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

static bool is_infinite_lower_bound(double bound, double infinite_bound)
{
    return bound <= -infinite_bound;
}

static bool is_infinite_upper_bound(double bound, double infinite_bound)
{
    return bound >= infinite_bound;
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
