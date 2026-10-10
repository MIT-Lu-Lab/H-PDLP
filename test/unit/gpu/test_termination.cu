#include "cuda_macros.h"
#include "cuda_utils.h"
#include "gpu_support.h"
#include "hpdlp.h"
#include "internal_types.h"
#include "setup.h"
#include "termination_checker.h"
#include "test_checks.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__global__ void compute_fixed_point_error_kernel(
    double *, const fixed_point_reductions_t *,
    const double *, const double *, const double *, const double *,
    double, double, double);

static void check_fixed_point_error(void)
{
    const double inputs[][3] = {
        {1.0, 2.0, 0.0},
        {1.0, 1.0, 1.0 + 1e-15},
        {1.0, 1.0, NAN},
        {1.0, 1.0, INFINITY}
    };
    const double expected[] = {sqrt(5.0), 0.0, INFINITY, INFINITY};
    fixed_point_reductions_t *reductions;
    CUDA_CHECK(cudaMallocManaged(&reductions, sizeof(*reductions)));
    double *values;
    CUDA_CHECK(cudaMallocManaged(&values, 5 * sizeof(double)));
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 5; ++j)
            values[j] = 0.0;
        reductions->delta_primal_norm = inputs[i][0];
        reductions->delta_dual_norm = inputs[i][1];
        reductions->operator_cross_term = inputs[i][2];
        compute_fixed_point_error_kernel<<<1, 1>>>(
            values + 4, reductions,
            values, values + 1, values + 2, values + 3,
            1.0, 1.0, 1.0);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (values[4] != expected[i])
        {
            printf("FAIL: FPI case %d: expected %g, got %g\n", i, expected[i], values[4]);
            ++failures;
        }
    }
    CUDA_CHECK(cudaFree(values));
    CUDA_CHECK(cudaFree(reductions));
}

static void check_solution_norms(void)
{
    const double objective[] = {0.0, 0.0};
    const double lower[] = {-INFINITY, -INFINITY};
    const double upper[] = {INFINITY, INFINITY};
    const double constant = 0.0;
    const int row_ptr[] = {0, 1, 2};
    const int col_ind[] = {0, 1};
    const double values[] = {1.0, 1.0};
    matrix_desc_t matrix = {};
    matrix.m = matrix.n = 2;
    matrix.fmt = matrix_csr;
    matrix.data.csr.nnz = 2;
    matrix.data.csr.row_ptr = row_ptr;
    matrix.data.csr.col_ind = col_ind;
    matrix.data.csr.vals = values;
    lp_problem_t *problem = create_lp_problem(
        objective, &matrix, lower, upper, lower, upper, &constant);
    check("solution norm fixture builds", problem != NULL);
    if (!problem)
        return;

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = false;
    params.presolve = false;
    disable_rescaling(&params);
    pdhg_solver_state_t *state = create_solver_state(problem, &params);

    const double negative[] = {-3., -2.};
    CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution_d, negative, sizeof(negative), cudaMemcpyHostToDevice));
    reduce_device_max(state, state->pdhg_primal_solution_d, 2, state->hsd_primal_scale_pdhg_d);
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    double reduced_value;
    CUDA_CHECK(cudaMemcpy(&reduced_value, state->hsd_primal_scale_pdhg_d, sizeof(double), cudaMemcpyDeviceToHost));
    check("ordinary maximum preserves negative values", reduced_value == -2.);
    reduce_device_abs_max(state, state->pdhg_primal_solution_d, 0, state->hsd_primal_scale_pdhg_d);
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    CUDA_CHECK(cudaMemcpy(&reduced_value, state->hsd_primal_scale_pdhg_d, sizeof(double), cudaMemcpyDeviceToHost));
    check("empty absolute maximum is zero", reduced_value == 0.);
    const double one = 1.;
    CUDA_CHECK(cudaMemcpy(state->hsd_primal_scale_pdhg_d, &one, sizeof(double), cudaMemcpyHostToDevice));

    const double primal[][2] = {{3.0, -4.0}, {0.0, 0.0}};
    const double dual[][2] = {{5.0, -12.0}, {0.0, 0.0}};
    const norm_type_t norms[] = {NORM_TYPE_L_INF, NORM_TYPE_L2};
    for (norm_type_t norm : norms)
    {
        params.optimality_norm = norm;
        for (int sample = 0; sample < 2; ++sample)
        {
            CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution_d, primal[sample],
                                  sizeof(primal[sample]), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution_d, dual[sample],
                                  sizeof(dual[sample]), cudaMemcpyHostToDevice));
            compute_termination_metrics(state, &params);
            const double expected_primal = sample ? 0.0 : (norm == NORM_TYPE_L_INF ? 4.0 : 5.0);
            const double expected_dual = sample ? 0.0 : (norm == NORM_TYPE_L_INF ? 12.0 : 13.0);
            if (!(fabs(state->metrics_h->primal_solution_norm - expected_primal) < 1e-12) ||
                !(fabs(state->metrics_h->dual_solution_norm - expected_dual) < 1e-12))
            {
                fprintf(stderr, "solution norms: norm=%d sample=%d primal=%g dual=%g\n",
                        (int)norm, sample, state->metrics_h->primal_solution_norm,
                        state->metrics_h->dual_solution_norm);
                ++failures;
            }
        }
    }
    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}

static void check_termination(double scale, int iteration_limit, int certificate,
                              termination_reason_t expected)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    pdhg_metrics_t metrics = {};
    metrics.pdhg_primal_scale = scale;
    if (certificate)
    {
        metrics.normalized_homogeneous_ratio = 1;
        metrics.dual_infeasibility_homogeneous_margin = 1;
    }
    pdhg_solver_state_t state = {};
    state.metrics_h = &metrics;
    state.control.start_time = clock();
    if (iteration_limit)
        state.control.total_count = params.termination_criteria.iteration_limit;
    check_termination_criteria(&state, &params.termination_criteria);
    if (state.control.termination_reason != expected)
    {
        fprintf(stderr, "FAIL tau=%.17g limit=%d certificate=%d reason=%d expected=%d\n",
                scale, iteration_limit, certificate, state.control.termination_reason, expected);
        ++failures;
    }
}

static void check_scale_termination(void)
{
    check_termination(-1, 0, 0, TERMINATION_REASON_UNSPECIFIED);
    check_termination(0, 0, 0, TERMINATION_REASON_UNSPECIFIED);
    check_termination(1e-51, 0, 0, TERMINATION_REASON_UNSPECIFIED);
    check_termination(1e-50, 0, 0, TERMINATION_REASON_UNSPECIFIED);
    check_termination(nextafter(1e-50, INFINITY), 0, 0, TERMINATION_REASON_OPTIMAL);
    check_termination(1, 0, 0, TERMINATION_REASON_OPTIMAL);
    check_termination(NAN, 0, 0, TERMINATION_REASON_NUMERICAL_FAILURE);
    check_termination(INFINITY, 0, 0, TERMINATION_REASON_NUMERICAL_FAILURE);
    check_termination(0, 1, 0, TERMINATION_REASON_ITERATION_LIMIT);
    check_termination(0, 0, 1, TERMINATION_REASON_DUAL_INFEASIBLE);
}

int main(void)
{
    const int status = test_gpu_status();
    if (status) return status;
    check_fixed_point_error();
    check_solution_norms();
    check_scale_termination();
    return failures != 0;
}
