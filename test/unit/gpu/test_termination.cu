#include "cuda_macros.h"
#include "cuda_utils.h"
#include "gpu_support.h"
#include "hpdlp.h"
#include "internal_types.h"
#include "result.h"
#include "setup.h"
#include "solver.h"
#include "termination_checker.h"
#include "test_checks.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void check_fixed_point_error(void)
{
    const double inputs[][3] = {
        {1.0, 2.0, 0.0},
        {1.0, 1.0, 1.0 + 1e-15},
        {1.0, 1.0, NAN},
        {1.0, 1.0, INFINITY}
    };
    const double expected[] = {sqrt(5.0), 0.0, INFINITY, INFINITY};
    fixed_point_reductions_t reductions = {};
    pdhg_solver_state_t state = {};
    state.fixed_point_reductions_mp = &reductions;
    state.control.step_size_base = 1.0;
    state.control.primal_weight = 1.0;
    step_size_parameters_t step_size_params = {};
    step_size_params.step_size_factor = 1.0;
    for (int i = 0; i < 4; ++i)
    {
        reductions.primal_delta_norm = inputs[i][0];
        reductions.dual_delta_norm = inputs[i][1];
        reductions.operator_cross_term = inputs[i][2];
        compute_fixed_point_error(&state, &step_size_params);
        if (state.control.fixed_point_error != expected[i])
        {
            fprintf(stderr,
                    "FAIL: FPI case %d: expected %g, got %g\n",
                    i,
                    expected[i],
                    state.control.fixed_point_error);
            ++failures;
        }
    }
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
    CUDA_CHECK(cudaMemcpy(state->primal_pdhg_solution_d, negative, sizeof(negative), cudaMemcpyHostToDevice));
    reduce_device_abs_max(state, state->primal_pdhg_solution_d, 2, state->hsd_primal_pdhg_scale_d);
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    double reduced_value;
    CUDA_CHECK(cudaMemcpy(&reduced_value, state->hsd_primal_pdhg_scale_d, sizeof(double), cudaMemcpyDeviceToHost));
    check("absolute maximum of negative values is their largest magnitude", reduced_value == 3.);
    reduce_device_abs_max(state, state->primal_pdhg_solution_d, 0, state->hsd_primal_pdhg_scale_d);
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
    CUDA_CHECK(cudaMemcpy(&reduced_value, state->hsd_primal_pdhg_scale_d, sizeof(double), cudaMemcpyDeviceToHost));
    check("empty absolute maximum is zero", reduced_value == 0.);
    const double one = 1.;
    CUDA_CHECK(cudaMemcpy(state->hsd_primal_pdhg_scale_d, &one, sizeof(double), cudaMemcpyHostToDevice));

    const double primal[][2] = {{3.0, -4.0}, {0.0, 0.0}};
    const double dual[][2] = {{5.0, -12.0}, {0.0, 0.0}};
    const norm_type_t norms[] = {NORM_TYPE_L_INF, NORM_TYPE_L2};
    for (norm_type_t norm : norms)
    {
        params.optimality_norm = norm;
        for (int sample = 0; sample < 2; ++sample)
        {
            CUDA_CHECK(cudaMemcpy(state->primal_pdhg_solution_d, primal[sample],
                                  sizeof(primal[sample]), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->dual_pdhg_solution_d, dual[sample],
                                  sizeof(dual[sample]), cudaMemcpyHostToDevice));
            compute_termination_metrics(state, &params);
            const double expected_primal = sample ? 0.0 : (norm == NORM_TYPE_L_INF ? 4.0 : 5.0);
            const double expected_dual = sample ? 0.0 : (norm == NORM_TYPE_L_INF ? 12.0 : 13.0);
            if (!(fabs(state->metrics_h->primal_solution_norm - expected_primal) < 1e-12) ||
                !(fabs(state->metrics_h->dual_solution_norm - expected_dual) < 1e-12))
            {
                fprintf(stderr, "FAIL: solution norms: norm=%d sample=%d primal=%g dual=%g\n",
                        (int)norm, sample, state->metrics_h->primal_solution_norm,
                        state->metrics_h->dual_solution_norm);
                ++failures;
            }
        }
    }
    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}

static void check_termination(double scale, bool at_iteration_limit,
                              termination_reason_t certificate,
                              termination_reason_t expected)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    pdhg_metrics_t metrics = {};
    metrics.primal_pdhg_scale = scale;
    if (certificate != TERMINATION_REASON_UNSPECIFIED)
        metrics.normalized_homogeneous_ratio = 1;
    if (certificate == TERMINATION_REASON_PRIMAL_INFEASIBLE)
    {
        metrics.primal_infeasibility_iterate.margin = 1;
        metrics.primal_infeasibility_iterate.relative_margin = 1;
    }
    if (certificate == TERMINATION_REASON_DUAL_INFEASIBLE)
    {
        metrics.dual_infeasibility_iterate.margin = 1;
        metrics.dual_infeasibility_iterate.relative_margin = 1;
    }
    pdhg_solver_state_t state = {};
    state.metrics_h = &metrics;
    state.control.start_time = clock();
    if (at_iteration_limit)
        state.control.total_count = params.termination_criteria.iteration_limit;
    check_termination_criteria(&state, &params.termination_criteria);
    if (state.control.termination_reason != expected)
    {
        fprintf(stderr, "FAIL: tau=%.17g limit=%d certificate=%d reason=%d expected=%d\n",
                scale, at_iteration_limit, certificate, state.control.termination_reason, expected);
        ++failures;
    }
}

static void check_scale_termination(void)
{
    check_termination(-1, false, TERMINATION_REASON_UNSPECIFIED, TERMINATION_REASON_UNSPECIFIED);
    check_termination(0, false, TERMINATION_REASON_UNSPECIFIED, TERMINATION_REASON_UNSPECIFIED);
    check_termination(1e-51, false, TERMINATION_REASON_UNSPECIFIED, TERMINATION_REASON_UNSPECIFIED);
    check_termination(1e-50, false, TERMINATION_REASON_UNSPECIFIED, TERMINATION_REASON_UNSPECIFIED);
    check_termination(nextafter(1e-50, INFINITY), false, TERMINATION_REASON_UNSPECIFIED, TERMINATION_REASON_OPTIMAL);
    check_termination(1, false, TERMINATION_REASON_UNSPECIFIED, TERMINATION_REASON_OPTIMAL);
    check_termination(NAN, false, TERMINATION_REASON_UNSPECIFIED, TERMINATION_REASON_NUMERICAL_FAILURE);
    check_termination(INFINITY, false, TERMINATION_REASON_UNSPECIFIED, TERMINATION_REASON_NUMERICAL_FAILURE);
    check_termination(0, true, TERMINATION_REASON_UNSPECIFIED, TERMINATION_REASON_ITERATION_LIMIT);
    check_termination(0, false, TERMINATION_REASON_PRIMAL_INFEASIBLE, TERMINATION_REASON_PRIMAL_INFEASIBLE);
    check_termination(0, false, TERMINATION_REASON_DUAL_INFEASIBLE, TERMINATION_REASON_DUAL_INFEASIBLE);
}

static void check_ray_candidate(
    int m, int n, const double *a, const double *c,
    const double *row_lower, const double *row_upper,
    const double *lower, const double *upper,
    const double *primal, const double *dual,
    certificate_source_t source, bool scaled, termination_reason_t expected)
{
    matrix_desc_t matrix = {};
    matrix.m = m;
    matrix.n = n;
    matrix.fmt = matrix_dense;
    matrix.data.dense.A = a;
    lp_problem_t *problem = create_lp_problem(
        c, &matrix, row_lower, row_upper, lower, upper, NULL);
    check("ray fixture builds", problem != NULL);
    if (!problem) return;

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = params.presolve = false;
    if (!scaled) disable_rescaling(&params);
    pdhg_solver_state_t *state = create_solver_state(problem, &params);
    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *y = (double *)malloc((size_t)m * sizeof(double));
    CUDA_CHECK(cudaMemcpy(x, state->variable_rescaling_d,
                          (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y, state->constraint_rescaling_d,
                          (size_t)m * sizeof(double), cudaMemcpyDeviceToHost));
    for (int j = 0; j < n; ++j)
        x[j] *= primal[j] * state->constraint_bound_rescaling;
    for (int i = 0; i < m; ++i)
        y[i] *= dual[i] * state->objective_vector_rescaling;
    const bool iterate = source == CERTIFICATE_SOURCE_ITERATE;
    CUDA_CHECK(cudaMemcpy(state->primal_reflected_solution_d, x,
                          (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state->dual_reflected_solution_d, y,
                          (size_t)m * sizeof(double), cudaMemcpyHostToDevice));
    if (source != CERTIFICATE_SOURCE_STEP)
    {
        CUDA_CHECK(cudaMemcpy(state->primal_pdhg_solution_d, x,
                              (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->dual_pdhg_solution_d, y,
                              (size_t)m * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (iterate)
    {
        CUDA_CHECK(cudaMemcpy(state->primal_initial_solution_d, x,
                              (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->dual_initial_solution_d, y,
                              (size_t)m * sizeof(double), cudaMemcpyHostToDevice));
    }
    const double tau = iterate ? 0.0 : 1.0;
    CUDA_CHECK(cudaMemcpy(state->hsd_primal_pdhg_scale_d, &tau,
                          sizeof(double), cudaMemcpyHostToDevice));
    compute_termination_metrics(state, &params);
    // Exercise the requested source independently of HSD routing.
    state->metrics_h->normalized_homogeneous_ratio = iterate ? 1.0 : 0.5;
    state->control.start_time = clock();
    check_termination_criteria(state, &params.termination_criteria);
    if (state->control.termination_reason != expected)
    {
        fprintf(stderr, "FAIL: ray candidate (source=%d scaled=%d): got %d expected %d\n",
                source, scaled, state->control.termination_reason, expected);
        ++failures;
    }
    else if (expected == TERMINATION_REASON_DUAL_INFEASIBLE)
    {
        hpdlp_result_t *result = create_result_from_state(state, problem);
        double objective = 0;
        for (int j = 0; j < n; ++j)
            objective += c[j] * result->primal_solution[j];
        check("exported primal ray has unit objective margin", fabs(objective + 1) < 1e-10);
        check("primal ray retains its selected source",
              state->control.certificate_source == source);
        hpdlp_result_free(result);
    }
    else if (expected == TERMINATION_REASON_PRIMAL_INFEASIBLE)
    {
        hpdlp_result_t *result = create_result_from_state(state, problem);
        for (int j = 0; j < n; ++j)
        {
            double residual = result->reduced_cost[j];
            for (int i = 0; i < m; ++i)
                residual += a[i * n + j] * result->dual_solution[i];
            check("exported dual ray satisfies stationarity", fabs(residual) < 1e-10);
        }
        check("dual ray retains its selected source",
              state->control.certificate_source == source);
        hpdlp_result_free(result);
    }
    free(x);
    free(y);
    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}

static void check_ray_certificates(void)
{
    const double zero[] = {0, 0, 0};
    const double free_lower[] = {-INFINITY, -INFINITY, -INFINITY};
    const double free_upper[] = {INFINITY, INFINITY, INFINITY};
    for (int s = 0; s < CERTIFICATE_SOURCE_COUNT; ++s)
    {
        const certificate_source_t source = (certificate_source_t)s;
        for (int scaled = 0; scaled < 2; ++scaled)
        {
            const double a[] = {1, 1, 0, 0, 0, 1e10};
            const double c[] = {-1e12, -1.001e12, 0};
            const double row_lower[] = {-INFINITY, 0}, row_upper[] = {2, 0};
            const double bad_direction[] = {0, 1, 0};
            check_ray_candidate(2, 3, a, c, row_lower, row_upper, zero, free_upper,
                                bad_direction, zero, source, scaled,
                                TERMINATION_REASON_UNSPECIFIED);

            const double null_a[] = {0, 1};
            const double decreasing[] = {-1, 0}, null_direction[] = {1, 0};
            check_ray_candidate(1, 2, null_a, decreasing, zero, zero, zero, free_upper,
                                null_direction, zero, source, scaled,
                                TERMINATION_REASON_DUAL_INFEASIBLE);

            const double cancelling_a[] = {0, 0};
            const double cancelling_c[] = {1, -1 - 1e-12}, direction[] = {1, 1};
            check_ray_candidate(1, 2, cancelling_a, cancelling_c, zero, zero, zero, free_upper,
                                direction, zero, source, scaled,
                                TERMINATION_REASON_UNSPECIFIED);

            const double dual_a[] = {1, 1}, dual_c[] = {1};
            const double lower[] = {2, -INFINITY}, upper[] = {INFINITY, 1};
            const double dual_ray[] = {1, -1};
            check_ray_candidate(2, 1, dual_a, dual_c, lower, upper, free_lower, free_upper,
                                zero, dual_ray, source, scaled,
                                TERMINATION_REASON_PRIMAL_INFEASIBLE);
            const double bounded_upper[] = {1}, bound_lower[] = {2}, positive[] = {1};
            check_ray_candidate(1, 1, positive, dual_c, bound_lower, free_upper, free_lower, bounded_upper,
                                zero, positive, source, scaled,
                                TERMINATION_REASON_PRIMAL_INFEASIBLE);
            const double zero_a[] = {0}, unit_lower[] = {1};
            check_ray_candidate(1, 1, zero_a, dual_c, unit_lower, free_upper, free_lower, free_upper,
                                zero, positive, source, scaled,
                                TERMINATION_REASON_PRIMAL_INFEASIBLE);
            const double cancelling_lower[] = {1 + 1e-12, -INFINITY};
            check_ray_candidate(2, 1, dual_a, dual_c, cancelling_lower, upper, free_lower, free_upper,
                                zero, dual_ray, source, scaled,
                                TERMINATION_REASON_UNSPECIFIED);

            const double huge_lower[] = {1e12};
            check_ray_candidate(1, 1, positive, dual_c, huge_lower, free_upper, free_lower, free_upper,
                                zero, positive, source, scaled,
                                TERMINATION_REASON_UNSPECIFIED);
        }
    }
    const double a[] = {0}, c[] = {1e12}, negative[] = {-1};
    check_ray_candidate(1, 1, a, c, zero, zero, zero, free_upper,
                        negative, zero, CERTIFICATE_SOURCE_ITERATE, false,
                        TERMINATION_REASON_UNSPECIFIED);

    // Finite candidate data can still overflow in A*d and |A|*|d|.
    const double huge_a[] = {1e200}, huge_direction[] = {1e200}, decreasing[] = {-1};
    check_ray_candidate(1, 1, huge_a, decreasing, zero, free_upper, zero, free_upper,
                        huge_direction, zero, CERTIFICATE_SOURCE_ITERATE, false,
                        TERMINATION_REASON_UNSPECIFIED);

    // The row residual 5e-9 is small relative to the activity 64 but not
    // relative to the objective margin 1, so the certificate is accepted.
    double long_a[65], long_c[65] = {}, long_direction[65], long_lower[65], long_upper[65];
    long_c[0] = -1;
    for (int j = 0; j < 65; ++j)
    {
        long_a[j] = j < 32 ? 1 : (j < 64 ? -1 : 5e-9);
        long_direction[j] = 1;
        long_lower[j] = 0;
        long_upper[j] = INFINITY;
    }
    for (int s = 0; s < CERTIFICATE_SOURCE_COUNT; ++s)
        check_ray_candidate(1, 65, long_a, long_c, zero, zero, long_lower, long_upper,
                            long_direction, zero, (certificate_source_t)s, false,
                            TERMINATION_REASON_DUAL_INFEASIBLE);
}

int main(void)
{
    const int status = test_gpu_status();
    if (status) return status;
    check_fixed_point_error();
    check_solution_norms();
    check_scale_termination();
    check_ray_certificates();
    return failures != 0;
}
