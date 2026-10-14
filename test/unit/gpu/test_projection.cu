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
static const double EXPECTED_SCALE = 1.0;
static const double EXPECTED_SHADOW = 9.9;
static const double UNWRITTEN_SCALE = -7.0;
static const double UNWRITTEN_OBSERVATION = -123.0;
static const double CLAMPED_GRADIENT = -3.0;
static const double EXPECTED_CLAMPED_MULTIPLIER = 3.0;

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
    disable_rescaling(params);
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
        state->primal_initial_solution_d,
        primal,
        state->var_bytes,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->primal_current_solution_d,
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
        state->dual_pdhg_solution_d, 0, state->con_bytes));

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
        state->primal_current_solution_d,
        state->primal_reflected_solution_d,
        state->primal_pdhg_solution_d,
        state->primal_pdhg_projection_multiplier_d,
        state->primal_initial_solution_d,
        state->dual_product_d,
        state->objective_vector_d,
        state->control.step_size_primal_d,
        state->control.step_size_homogeneous_d,
        state->hsd_dual_reflected_scale_d,
        state->control.inner_count_d,
        1,
        1.0,
        state->hsd_primal_initial_scale_d,
        state->hsd_primal_current_scale_d,
        state->hsd_primal_pdhg_scale_d,
        state->hsd_primal_reflected_scale_d,
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

    check_close("primal scale is capped at one",
                read_device_double(state->hsd_primal_pdhg_scale_d),
                EXPECTED_SCALE, 1e-10);
    check_close("primal shadow follows the closed-form resolvent",
                read_device_double(state->primal_pdhg_solution_d),
                EXPECTED_SHADOW, 1e-10);
    check_close("an interior projection records a zero multiplier",
                read_device_double(state->primal_pdhg_projection_multiplier_d),
                0.0, 1e-10);
    if (num_variables > 1)
        check_close("a clamped projection records its bound multiplier",
                    read_device_double(state->primal_pdhg_projection_multiplier_d + 1),
                    EXPECTED_CLAMPED_MULTIPLIER, 1e-10);
}

static void check_interior_column_retained(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    int num_variables)
{
    compute_termination_metrics(state, params);
    check_close("interior projection contributes no bound multiplier",
                read_device_double(state->pdhg_reduced_cost_d),
                0.0, 1e-10);
    check_close("the primary candidate retains the interior stationarity residual",
                state->metrics_h->absolute_dual_residual,
                1.0, 1e-10);
    if (num_variables > 1)
        check_close("termination reconstructs the fixed column reduced cost",
                    read_device_double(state->pdhg_reduced_cost_d + 1),
                    0.0, 1e-10);
    check_close("termination preserves the interior projection multiplier",
                read_device_double(state->primal_pdhg_projection_multiplier_d),
                0.0, 1e-10);
    if (num_variables > 1)
        check_close("termination preserves the clamped projection multiplier",
                    read_device_double(state->primal_pdhg_projection_multiplier_d + 1),
                    EXPECTED_CLAMPED_MULTIPLIER, 1e-10);

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
    const double previous_scale = read_device_double(state->hsd_primal_pdhg_scale_d);
    const double previous_shadow = read_device_double(state->primal_pdhg_solution_d);
    double *unwritten_observation =
        (double *)malloc((size_t)num_variables * sizeof(double));
    double *observation_after_non_major =
        (double *)malloc((size_t)num_variables * sizeof(double));
    for (int j = 0; j < num_variables; ++j)
        unwritten_observation[j] = UNWRITTEN_OBSERVATION - j;
    CUDA_CHECK(cudaMemcpy(
        state->primal_pdhg_projection_multiplier_d,
        unwritten_observation,
        state->var_bytes,
        cudaMemcpyHostToDevice));

    run_primal_resolvent(state, params, false);

    CUDA_CHECK(cudaMemcpy(
        observation_after_non_major,
        state->primal_pdhg_projection_multiplier_d,
        state->var_bytes,
        cudaMemcpyDeviceToHost));
    bool observation_preserved = true;
    for (int j = 0; j < num_variables; ++j)
        observation_preserved &=
            observation_after_non_major[j] == unwritten_observation[j];
    check("non-major projection preserves the major projection multiplier",
          observation_preserved);
    check_close("non-major projection preserves the major scale",
                read_device_double(state->hsd_primal_pdhg_scale_d), previous_scale, 0.0);
    check_close("non-major projection preserves the major primal point",
                read_device_double(state->primal_pdhg_solution_d), previous_shadow, 0.0);

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

    pdhg_solver_state_t *state = create_solver_state(problem, &params);
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

typedef struct
{
    const char *name;
    int num_variables;
    int max_iter;
    bool zero_terms;

    double primal_step;
    double homogeneous_step;
    double current_scale;
    double homogeneous_gradient;

    double current[2];
    double gradient[2];
    double objective[2];
    double lower[2];
    double upper[2];

    double expected_scale;
    double expected_primal[2];
} projection_case_t;

static void check_projection(int num_blocks, const projection_case_t *test)
{
    const int n = num_blocks == 1 ? test->num_variables : 4 * THREADS_PER_BLOCK + 1;
    double *vectors, *scalars;
    int *count;
    newton_terms_t *partials;
    CUDA_CHECK(cudaMallocManaged((void **)&vectors, 9 * (size_t)n * sizeof(double), cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged((void **)&scalars, 10 * sizeof(double), cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged((void **)&count, sizeof(int), cudaMemAttachGlobal));
    CUDA_CHECK(cudaMalloc((void **)&partials, 2 * num_blocks * sizeof(newton_terms_t)));

    double *current = vectors;
    double *reflected = vectors + n;
    double *evaluation = vectors + 2 * n;
    double *multiplier = vectors + 3 * n;
    double *initial = vectors + 4 * n;
    double *gradient = vectors + 5 * n;
    double *objective = vectors + 6 * n;
    double *lower = vectors + 7 * n;
    double *upper = vectors + 8 * n;
    for (int i = 0; i < 9 * n; ++i) vectors[i] = 0.0;
    for (int i = 0; i < test->num_variables; ++i)
    {
        current[i] = initial[i] = test->current[i];
        gradient[i] = test->gradient[i];
        objective[i] = test->objective[i];
        lower[i] = test->lower[i];
        upper[i] = test->upper[i];
    }
    double *primal_step = scalars;
    double *homogeneous_step = scalars + 1;
    double *homogeneous_gradient = scalars + 2;
    double *initial_scale = scalars + 3;
    double *current_scale = scalars + 4;
    double *evaluation_scale = scalars + 5;
    double *reflected_scale = scalars + 6;
    double *scale_history = scalars + 7;
    *primal_step = test->primal_step;
    *homogeneous_step = test->homogeneous_step;
    *homogeneous_gradient = test->homogeneous_gradient;
    *initial_scale = 1.0;
    *current_scale = test->current_scale;
    *evaluation_scale = *reflected_scale = NAN;
    scale_history[0] = scale_history[1] = scale_history[2] = 1.0;
    *count = 10;

    box_newton_state_t state = {};
    state.scale_history_d = scale_history;
    state.lower_bound_d = lower;
    state.upper_bound_d = upper;
    state.num_elements = n;
    state.num_blocks_newton = num_blocks;
    state.newton_terms_are_zero = test->zero_terms;
    state.newton_partials_d = partials;
    const cone_parameters_t params = {test->max_iter, 1e-12};
    update_primal_box_cone_resolvent(&params, &state,
        current, reflected, evaluation, multiplier, initial, gradient, objective,
        primal_step, homogeneous_step, homogeneous_gradient, count, 1, 1.0,
        initial_scale, current_scale, evaluation_scale, reflected_scale, true, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    const double expected_reflected = 2.0 * test->expected_scale - test->current_scale;
    const double weight = 11.0 / 12.0;
    const double expected_halpern = weight * expected_reflected + (1.0 - weight);
    const double tol = 1e-10;
    bool valid = isfinite(*evaluation_scale) &&
        *evaluation_scale >= 0.0 && *evaluation_scale <= 1.0 &&
        fabs(*evaluation_scale - test->expected_scale) <= tol &&
        fabs(*reflected_scale - expected_reflected) <= tol &&
        fabs(*current_scale - expected_halpern) <= tol;
    for (int i = 0; i < n; ++i)
    {
        const double expected = i < test->num_variables ? test->expected_primal[i] : 0.0;
        valid = valid && isfinite(evaluation[i]) && fabs(evaluation[i] - expected) <= tol;
    }
    if (!valid)
    {
        fprintf(stderr,
            "FAIL: %s blocks=%d tau=%.17g x=%.17g reflected=%.17g halpern=%.17g "
            "expected_tau=%.17g expected_x=%.17g\n",
            test->name, num_blocks, *evaluation_scale, evaluation[0],
            *reflected_scale, *current_scale, test->expected_scale, test->expected_primal[0]);
        ++failures;
    }
    CUDA_CHECK(cudaFree(partials));
    CUDA_CHECK(cudaFree(count));
    CUDA_CHECK(cudaFree(scalars));
    CUDA_CHECK(cudaFree(vectors));
}

static void check_homogeneous_scale(void)
{
    const projection_case_t cases[] = {
        {"lower endpoint with one Newton step", 1, 1, false,
         1.0, 1.0, -1.0, -1.0,
         {0.0}, {0.0}, {1.0}, {0.0}, {1.0},
         0.0, {0.0}},
        {"lower endpoint", 1, 100, false,
         1.0, 1.0, 1.0, -3.0,
         {0.0}, {0.0}, {1.0}, {0.0}, {1.0},
         0.0, {0.0}},
        {"upper endpoint with reflection above one", 1, 100, false,
         1.0, 1.0, -1.0, 3.0,
         {0.0}, {0.0}, {1.0}, {0.0}, {1.0},
         1.0, {0.0}},
        {"upper endpoint", 1, 100, false,
         1.0, 1.0, 1.0, 1.0,
         {0.0}, {0.0}, {1.0}, {0.0}, {1.0},
         1.0, {0.0}},
        {"interior scale", 1, 100, false,
         1.0, 1.0, -0.5, 0.0,
         {0.0}, {0.0}, {1.0}, {0.0}, {1.0},
         0.5, {0.0}},
        {"zero-term interior scale", 1, 100, true,
         0.5, 0.5, 0.2, 0.0,
         {0.0}, {0.0}, {0.0}, {-INFINITY}, {INFINITY},
         0.7, {0.0}},
        {"zero-term upper endpoint", 1, 100, true,
         0.5, 0.5, 0.8, 0.0,
         {0.0}, {0.0}, {0.0}, {-INFINITY}, {INFINITY},
         1.0, {0.0}},
        {"zero-term lower endpoint", 1, 100, true,
         0.5, 0.5, -1.0, 0.0,
         {0.0}, {0.0}, {0.0}, {-INFINITY}, {INFINITY},
         0.0, {0.0}},
        {"coupled primal recomputed at cap", 1, 100, false,
         0.5, 0.5, 0.8, 0.0,
         {3.0}, {0.0}, {1.0}, {-INFINITY}, {INFINITY},
         1.0, {2.5}},
        {"coupled interior scale", 1, 100, false,
         0.5, 0.5, 0.0, 0.0,
         {0.0}, {0.0}, {1.0}, {-INFINITY}, {INFINITY},
         0.4, {-0.2}},
        {"escape non-informative fixed point", 2, 100, false,
         0.1, 0.5, 0.0, 0.0,
         {0.0, 0.0}, {-2.0, -1.0}, {-2.0, -1.0}, {1.0, -2.0}, {2.0, -1.0},
         1.0 / 52.0, {1.0 / 52.0, -1.0 / 26.0}},
    };
    for (int blocks = 1; blocks <= 2; ++blocks)
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
            check_projection(blocks, &cases[i]);
}

int main(void)
{
    const int status = test_gpu_status();
    if (status) return status;
    check_projection_observation();
    check_homogeneous_scale();
    return failures != 0;
}
