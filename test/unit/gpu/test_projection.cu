#include "active_set_boost.h"
#include "cone_projector.h"
#include "cuda_macros.h"
#include "gpu_support.h"
#include "hpdlp.h"
#include "setup.h"
#include "termination_checker.h"
#include "test_checks.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { WINDOW_ITERATIONS = 10, EVALUATION_ITERATION = 20 };

static const double INITIAL_PRIMAL = 10.0;
static const double PRIMAL_STEP = 0.1;
static const double HOMOGENEOUS_STEP = 0.1;
static const double EXPECTED_SCALE = 200.0 / 101.0;
static const double EXPECTED_SHADOW = 990.0 / 101.0;
static const double UNWRITTEN_SCALE = -7.0;
static const double UNWRITTEN_OBSERVATION = -123.0;
static const double CLAMPED_GRADIENT = -3.0;
static const double EXPECTED_CLAMPED_MULTIPLIER = 3.0;




static void check_close(const char *what, double actual, double expected)
{
    if (!(fabs(actual - expected) < 1e-10))
    {
        printf("  FAIL: %s: expected %.17g, got %.17g\n",
               what, expected, actual);
        ++failures;
    }
}

static double read_device_double(const double *device_value)
{
    double value = NAN;
    CUDA_CHECK(cudaMemcpy(
        &value,
        device_value,
        sizeof(value),
        cudaMemcpyDeviceToHost));
    return value;
}

static lp_problem_t *create_projection_fixture(int num_variables)
{
    double *objective = (double *)calloc((size_t)num_variables, sizeof(double));
    double *variable_lower =
        (double *)calloc((size_t)num_variables, sizeof(double));
    double *variable_upper =
        (double *)calloc((size_t)num_variables, sizeof(double));
    if (!objective || !variable_lower || !variable_upper)
    {
        free(objective);
        free(variable_lower);
        free(variable_upper);
        return NULL;
    }

    objective[0] = 1.0;
    variable_upper[0] = INFINITY;
    for (int j = 1; j < num_variables; ++j)
        variable_upper[j] = 0.0;

    const int row_ptr[] = {0, 1};
    const int col_ind[] = {0};
    const double values[] = {1.0};
    const double constraint_bound[] = {0.0};
    const double objective_constant = 0.0;

    matrix_desc_t matrix = {};
    matrix.m = 1;
    matrix.n = num_variables;
    matrix.fmt = matrix_csr;
    matrix.data.csr.nnz = 1;
    matrix.data.csr.row_ptr = row_ptr;
    matrix.data.csr.col_ind = col_ind;
    matrix.data.csr.vals = values;

    lp_problem_t *problem = create_lp_problem(
        objective,
        &matrix,
        constraint_bound,
        constraint_bound,
        variable_lower,
        variable_upper,
        &objective_constant);
    free(objective);
    free(variable_lower);
    free(variable_upper);
    return problem;
}

static void set_test_parameters(pdhg_parameters_t *params)
{
    set_default_parameters(params);
    params->verbose = false;
    params->presolve = false;
    params->scaling_params.geometric_mean_iterations = 0;
    params->scaling_params.l_inf_ruiz_iterations = 0;
    params->scaling_params.has_pock_chambolle_alpha = false;
    params->scaling_params.bound_objective_rescaling = false;
    params->cone_params.tol = 1e-12;
    params->active_set_params.window_iterations = WINDOW_ITERATIONS;
}

static void initialize_projection_state(
    pdhg_solver_state_t *state,
    int num_variables)
{
    double *primal = (double *)calloc((size_t)num_variables, sizeof(double));
    primal[0] = INITIAL_PRIMAL;
    CUDA_CHECK(cudaMemcpy(
        state->initial_primal_solution_d,
        primal,
        state->var_bytes,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->current_primal_solution_d,
        primal,
        state->var_bytes,
        cudaMemcpyHostToDevice));
    free(primal);
    CUDA_CHECK(cudaMemset(state->dual_product_d, 0, state->var_bytes));
    if (num_variables > 1)
        CUDA_CHECK(cudaMemcpy(
            state->dual_product_d + 1,
            &CLAMPED_GRADIENT,
            sizeof(CLAMPED_GRADIENT),
            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(
        state->pdhg_dual_solution_d, 0, state->con_bytes));

    double hsd_scalars[HSD_SCALARS_COUNT] = {};
    hsd_scalars[HSD_PRIMAL_SCALE_INITIAL] = 1.0;
    hsd_scalars[HSD_PRIMAL_SCALE_CURRENT] = 1.0;
    hsd_scalars[HSD_PRIMAL_SCALE_PDHG] = UNWRITTEN_SCALE;
    hsd_scalars[HSD_PRIMAL_SCALE_REFLECTED] = UNWRITTEN_SCALE;
    hsd_scalars[HSD_DUAL_SCALE_INITIAL] = 1.0;
    hsd_scalars[HSD_DUAL_SCALE_CURRENT] = 1.0;
    hsd_scalars[HSD_DUAL_SCALE_PDHG] = 1.0;
    hsd_scalars[HSD_DUAL_SCALE_REFLECTED] = 0.0;
    CUDA_CHECK(cudaMemcpy(
        state->hsd_scalars_d,
        hsd_scalars,
        sizeof(hsd_scalars),
        cudaMemcpyHostToDevice));

    state->control.step_size_primal = PRIMAL_STEP;
    CUDA_CHECK(cudaMemcpy(
        state->control.step_size_primal_d,
        &PRIMAL_STEP,
        sizeof(PRIMAL_STEP),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->control.step_size_homogeneous_d,
        &HOMOGENEOUS_STEP,
        sizeof(HOMOGENEOUS_STEP),
        cudaMemcpyHostToDevice));
}

static void run_primal_resolvent(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    bool is_major)
{
    update_primal_box_cone_resolvent(
        &params->cone_params,
        &state->primal_box_cone_state,
        state->current_primal_solution_d,
        state->reflected_primal_solution_d,
        state->pdhg_primal_solution_d,
        state->pdhg_primal_projection_multiplier_d,
        state->initial_primal_solution_d,
        state->dual_product_d,
        state->objective_vector_d,
        state->control.step_size_primal_d,
        state->control.step_size_homogeneous_d,
        state->hsd_dual_scale_reflected_d,
        state->control.inner_count_d,
        1,
        1.0,
        state->hsd_primal_scale_initial_d,
        state->hsd_primal_scale_current_d,
        state->hsd_primal_scale_pdhg_d,
        state->hsd_primal_scale_reflected_d,
        is_major,
        state->stream_A);
    CUDA_CHECK(cudaStreamSynchronize(state->stream_A));
}

static void check_major_observation(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    int num_variables)
{
    run_primal_resolvent(state, params, true);

    check_close("primal scale follows the closed-form resolvent",
                read_device_double(state->hsd_primal_scale_pdhg_d),
                EXPECTED_SCALE);
    check_close("primal shadow follows the closed-form resolvent",
                read_device_double(state->pdhg_primal_solution_d),
                EXPECTED_SHADOW);
    check_close("an interior projection records a zero multiplier",
                read_device_double(state->pdhg_primal_projection_multiplier_d),
                0.0);
    if (num_variables > 1)
        check_close("a clamped projection records its bound multiplier",
                    read_device_double(state->pdhg_primal_projection_multiplier_d + 1),
                    EXPECTED_CLAMPED_MULTIPLIER);
}

static void check_interior_column_retained(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    int num_variables)
{
    compute_termination_metrics(state, params);
    check_close("interior projection contributes no bound multiplier",
                read_device_double(state->pdhg_reduced_cost_d),
                0.0);
    check_close("the primary candidate retains the interior stationarity residual",
                state->metrics_h->absolute_dual_residual,
                1.0);
    if (num_variables > 1)
        check_close("termination reconstructs the fixed column reduced cost",
                    read_device_double(state->pdhg_reduced_cost_d + 1),
                    0.0);
    check_close("termination preserves the interior projection multiplier",
                read_device_double(state->pdhg_primal_projection_multiplier_d),
                0.0);
    if (num_variables > 1)
        check_close("termination preserves the clamped projection multiplier",
                    read_device_double(state->pdhg_primal_projection_multiplier_d + 1),
                    EXPECTED_CLAMPED_MULTIPLIER);

    state->control.total_count = EVALUATION_ITERATION;
    update_active_set_masks(state, params);
    bool retained = false;
    int last_free_iter = -1;
    CUDA_CHECK(cudaMemcpy(
        &retained,
        state->active_set.free_variable_mask_d,
        sizeof(retained),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &last_free_iter,
        state->active_set.variable_last_free_iter_d,
        sizeof(last_free_iter),
        cudaMemcpyDeviceToHost));
    check("an interior primal projection retains the first column", retained);
    check("an interior primal projection refreshes last-free iteration",
          last_free_iter == EVALUATION_ITERATION);
}

static void check_non_major_preserves_observation(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    int num_variables)
{
    double *unwritten_observation =
        (double *)malloc((size_t)num_variables * sizeof(double));
    double *observation_after_non_major =
        (double *)malloc((size_t)num_variables * sizeof(double));
    for (int j = 0; j < num_variables; ++j)
        unwritten_observation[j] = UNWRITTEN_OBSERVATION - j;
    CUDA_CHECK(cudaMemcpy(
        state->pdhg_primal_projection_multiplier_d,
        unwritten_observation,
        state->var_bytes,
        cudaMemcpyHostToDevice));

    run_primal_resolvent(state, params, false);

    CUDA_CHECK(cudaMemcpy(
        observation_after_non_major,
        state->pdhg_primal_projection_multiplier_d,
        state->var_bytes,
        cudaMemcpyDeviceToHost));
    bool observation_preserved = true;
    for (int j = 0; j < num_variables; ++j)
        observation_preserved &=
            observation_after_non_major[j] == unwritten_observation[j];
    check("non-major projection preserves the major projection multiplier",
          observation_preserved);

    free(observation_after_non_major);
    free(unwritten_observation);
}

static void check_projection_path(int num_variables, bool expect_cooperative)
{
    pdhg_parameters_t params;
    set_test_parameters(&params);
    lp_problem_t *problem = create_projection_fixture(num_variables);
    if (!problem)
    {
        check("projection fixture allocation", false);
        return;
    }

    pdhg_solver_state_t *state =
        (pdhg_solver_state_t *)calloc(1, sizeof(pdhg_solver_state_t));
    initialize_solver_state(state, problem, problem, &params);
    check(expect_cooperative ? "large fixture uses cooperative projector"
                             : "scalar fixture uses single-block projector",
          expect_cooperative
              ? state->primal_box_cone_state.num_blocks_newton > 1
              : state->primal_box_cone_state.num_blocks_newton == 1);

    initialize_projection_state(state, num_variables);
    check_major_observation(state, &params, num_variables);
    check_interior_column_retained(state, &params, num_variables);
    check_non_major_preserves_observation(state, &params, num_variables);

    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}

static void check_projection_observation(void)
{
    check_projection_path(1, false);
    check_projection_path(4 * THREADS_PER_BLOCK + 1, true);
}

static void check_projection(int num_blocks, int max_iter, double current_scale,
                             double homogeneous_gradient, double expected_scale)
{
    const int n = num_blocks == 1 ? 1 : 4 * THREADS_PER_BLOCK + 1;
    double *vectors, *scalars;
    int *count;
    newton_terms_t *partials;
    CUDA_CHECK(cudaMallocManaged((void **)&vectors, 9 * (size_t)n * sizeof(double), cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged((void **)&scalars, 10 * sizeof(double), cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged((void **)&count, sizeof(int), cudaMemAttachGlobal));
    CUDA_CHECK(cudaMalloc((void **)&partials, 2 * num_blocks * sizeof(newton_terms_t)));
    for (int i = 0; i < 9 * n; ++i) vectors[i] = 0;
    vectors[6 * n] = 1;
    vectors[8 * n] = 1;
    scalars[0] = 1;
    scalars[1] = 1;
    scalars[2] = homogeneous_gradient;
    scalars[3] = 1;
    scalars[4] = current_scale;
    scalars[5] = NAN;
    scalars[6] = NAN;
    scalars[7] = scalars[8] = scalars[9] = 1;
    *count = 10;
    box_newton_state_t state = {};
    state.scale_history_d = scalars + 7;
    state.lower_bound_d = vectors + 7 * n;
    state.upper_bound_d = vectors + 8 * n;
    state.num_elements = n;
    state.num_blocks_newton = num_blocks;
    state.newton_partials_d = partials;
    cone_parameters_t params = {max_iter, 1e-6};
    update_primal_box_cone_resolvent(&params, &state,
        vectors, vectors + n, vectors + 2 * n, vectors + 3 * n,
        vectors + 4 * n, vectors + 5 * n, vectors + 6 * n,
        scalars, scalars + 1, scalars + 2, count, 1, 1,
        scalars + 3, scalars + 4, scalars + 5, scalars + 6, true, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double scale_tolerance = params.tol * fmax(1.0, fabs(expected_scale));
    int valid = isfinite(scalars[5]) && scalars[5] >= 0 &&
        fabs(scalars[5] - expected_scale) <= scale_tolerance &&
        fabs(scalars[6] - (2 * expected_scale - current_scale)) <= 2 * scale_tolerance;
    for (int i = 0; i < n; ++i)
        valid = valid && isfinite(vectors[2 * n + i]) && vectors[2 * n + i] == 0;
    if (!valid)
    {
        fprintf(stderr, "FAIL blocks=%d max_iter=%d current_tau=%g gradient=%g tau=%.17g x=%.17g expected_tau=%g\n",
            num_blocks, max_iter, current_scale, homogeneous_gradient, scalars[5], vectors[2 * n], expected_scale);
        ++failures;
    }
    CUDA_CHECK(cudaFree(partials));
    CUDA_CHECK(cudaFree(count));
    CUDA_CHECK(cudaFree(scalars));
    CUDA_CHECK(cudaFree(vectors));
}

static void check_homogeneous_scale(void)
{
    for (int blocks = 1; blocks <= 2; ++blocks)
    {
        check_projection(blocks, 1, -1, -1, 0);
        check_projection(blocks, 100, -1, -1, 0);
        check_projection(blocks, 100, -1, 3, 2);
        check_projection(blocks, 100, 1, -3, 0);
        check_projection(blocks, 100, 1, 1, 2);
    }
}

int main(void)
{
    const int status = test_gpu_status();
    if (status) return status;
    check_projection_observation();
    check_homogeneous_scale();
    return failures != 0;
}
