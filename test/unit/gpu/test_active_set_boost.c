#include "active_set_boost.h"
#include "gpu_support.h"
#include "hopdlp.h"
#include "power_iteration.h"
#include "setup.h"
#include "test_checks.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>

enum { HADAMARD_ORDER = 16,
       NUM_VARIABLES = 2 * HADAMARD_ORDER,
       NUM_CONSTRAINTS = HADAMARD_ORDER };




static double hadamard_entry(int row, int column)
{
    int bits = 0;
    int overlap = row & column;
    while (overlap)
    {
        bits ^= overlap & 1;
        overlap >>= 1;
    }
    return bits ? -1.0 : 1.0;
}

static lp_problem_t *create_hadamard_problem(void)
{
    static double objective[NUM_VARIABLES];
    static int row_ptr[NUM_CONSTRAINTS + 1];
    static int col_ind[NUM_CONSTRAINTS * NUM_VARIABLES];
    static double vals[NUM_CONSTRAINTS * NUM_VARIABLES];
    static double constraint_lower[NUM_CONSTRAINTS];
    static double constraint_upper[NUM_CONSTRAINTS];
    static double variable_lower[NUM_VARIABLES];
    static double variable_upper[NUM_VARIABLES];
    const double objective_constant = 0.0;

    for (int j = 0; j < NUM_VARIABLES; ++j)
    {
        objective[j] = (j < HADAMARD_ORDER) ? 0.0 : 1.0;
        variable_lower[j] = 0.0;
        variable_upper[j] = 1.0;
    }

    int nnz = 0;
    for (int i = 0; i < NUM_CONSTRAINTS; ++i)
    {
        row_ptr[i] = nnz;
        double rhs = 0.0;
        for (int j = 0; j < HADAMARD_ORDER; ++j)
        {
            col_ind[nnz] = j;
            vals[nnz] = hadamard_entry(i, j);
            rhs += 0.5 * vals[nnz];
            ++nnz;
        }
        for (int j = 0; j < HADAMARD_ORDER; ++j)
        {
            col_ind[nnz] = HADAMARD_ORDER + j;
            vals[nnz] = 1.0;
            ++nnz;
        }
        constraint_lower[i] = rhs;
        constraint_upper[i] = rhs;
    }
    row_ptr[NUM_CONSTRAINTS] = nnz;

    matrix_desc_t A_desc;
    A_desc.m = NUM_CONSTRAINTS;
    A_desc.n = NUM_VARIABLES;
    A_desc.fmt = matrix_csr;
    A_desc.data.csr.nnz = nnz;
    A_desc.data.csr.row_ptr = row_ptr;
    A_desc.data.csr.col_ind = col_ind;
    A_desc.data.csr.vals = vals;

    return create_lp_problem(
        objective, &A_desc, constraint_lower, constraint_upper,
        variable_lower, variable_upper, &objective_constant);
}

static void set_test_parameters(
    pdhg_parameters_t *params,
    bool step_size_boost)
{
    set_default_parameters(params);
    params->verbose = false;
    params->presolve = false;
    params->termination_evaluation_frequency = 20;
    params->termination_criteria.eps_optimal_relative = 1e-8;
    params->termination_criteria.eps_feasible_relative = 1e-8;
    params->termination_criteria.iteration_limit = 200000;
    params->active_set_params.step_size_boost = step_size_boost;
    params->active_set_params.window_iterations = 20;
}

static int solve_and_check(const char *label, const pdhg_parameters_t *params)
{
    lp_problem_t *problem = create_hadamard_problem();
    if (problem == NULL)
    {
        printf("  FAIL: %s could not build the problem\n", label);
        ++failures;
        return 0;
    }
    hopdlp_result_t *result = solve_lp_problem(problem, params);
    if (result == NULL || result->primal_solution == NULL)
    {
        printf("  FAIL: %s produced no solution\n", label);
        ++failures;
        lp_problem_free(problem);
        return 0;
    }

    double max_free_error = 0.0;
    double max_clamped_value = 0.0;
    for (int j = 0; j < HADAMARD_ORDER; ++j)
    {
        double free_error = fabs(result->primal_solution[j] - 0.5);
        if (free_error > max_free_error)
            max_free_error = free_error;
        double clamped_value = fabs(result->primal_solution[HADAMARD_ORDER + j]);
        if (clamped_value > max_clamped_value)
            max_clamped_value = clamped_value;
    }
    printf("%-14s iterations %6d  objective %11.3e  max|x-0.5| %.2e  max|y| %.2e"
           "  raises %d  reverts %d\n",
           label, result->total_count, result->primal_objective_value,
           max_free_error, max_clamped_value,
           result->active_set_raise_count, result->active_set_revert_count);

    char message[128];
    snprintf(message, sizeof(message), "%s terminates OPTIMAL", label);
    check(message, result->termination_reason == TERMINATION_REASON_OPTIMAL);
    snprintf(message, sizeof(message), "%s reaches the closed-form objective 0", label);
    check(message, fabs(result->primal_objective_value) < 1e-6);
    snprintf(message, sizeof(message), "%s leaves the free block at 0.5", label);
    check(message, max_free_error < 1e-5);
    snprintf(message, sizeof(message), "%s clamps the dense block at zero", label);
    check(message, max_clamped_value < 1e-6);

    const int raise_count = result->active_set_raise_count;
    hopdlp_result_free(result);
    lp_problem_free(problem);
    return raise_count;
}

static void check_active_set_resources(bool step_size_boost)
{
    pdhg_parameters_t params;
    set_test_parameters(&params, step_size_boost);
    lp_problem_t *problem = create_hadamard_problem();
    pdhg_solver_state_t *state = initialize_solver_state(problem, problem, &params);
    active_set_state_t *active_set = &state->active_set;

    check("active-set constraint observations follow enablement",
          (active_set->constraint_dual_projection_input_d != NULL) == step_size_boost);
    check("primal projection multipliers are available without the boost",
          state->pdhg_primal_projection_multiplier_d != NULL);
    check("active-set variable history follows enablement",
          (active_set->variable_last_free_iter_d != NULL) == step_size_boost);
    check("active-set constraint history follows enablement",
          (active_set->constraint_last_binding_iter_d != NULL) == step_size_boost);
    check("active-set variable mask follows enablement",
          (active_set->free_variable_mask_d != NULL) == step_size_boost);
    check("active-set constraint mask follows enablement",
          (active_set->binding_constraint_mask_d != NULL) == step_size_boost);
    check("active-set mask change counts follow enablement",
          (active_set->mask_changes_d != NULL) == step_size_boost);
    check("active-set raise-point anchor follows enablement",
              (active_set->anchor.primal_solution_d != NULL) == step_size_boost &&
              (active_set->anchor.dual_solution_d != NULL) == step_size_boost &&
              (active_set->anchor.hsd_scalars_d != NULL) == step_size_boost &&
              (active_set->anchor.primal_projection_multiplier_d != NULL) == step_size_boost);

    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}

static void check_zero_revert_budget_disables_resources(void)
{
    pdhg_parameters_t params;
    set_test_parameters(&params, true);
    params.active_set_params.max_reverts = 0;
    lp_problem_t *problem = create_hadamard_problem();
    pdhg_solver_state_t *state = initialize_solver_state(problem, problem, &params);
    active_set_state_t *active_set = &state->active_set;

    check("a zero revert budget allocates no active-set buffers",
          active_set->constraint_dual_projection_input_d == NULL &&
              active_set->variable_last_free_iter_d == NULL &&
              active_set->constraint_last_binding_iter_d == NULL &&
              active_set->free_variable_mask_d == NULL &&
              active_set->binding_constraint_mask_d == NULL &&
              active_set->mask_changes_d == NULL &&
              active_set->anchor.primal_solution_d == NULL &&
              active_set->anchor.dual_solution_d == NULL &&
              active_set->anchor.hsd_scalars_d == NULL &&
              active_set->anchor.primal_projection_multiplier_d == NULL);

    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}

static void check_estimator_owns_work_vectors(void)
{
    pdhg_parameters_t params;
    set_test_parameters(&params, true);
    lp_problem_t *problem = create_hadamard_problem();
    pdhg_solver_state_t *state = initialize_solver_state(problem, problem, &params);

    double primal_product[NUM_CONSTRAINTS];
    double dual_product[NUM_VARIABLES];
    for (int i = 0; i < NUM_CONSTRAINTS; ++i)
        primal_product[i] = 17.0 + i;
    for (int j = 0; j < NUM_VARIABLES; ++j)
        dual_product[j] = 31.0 + j;

    CUDA_CHECK(cudaMemcpy(
        state->primal_product_d, primal_product, sizeof(primal_product),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->dual_product_d, dual_product, sizeof(dual_product),
        cudaMemcpyHostToDevice));

    sv_estimator_opts_t opts = {};
    opts.max_iter = 1;
    opts.tol = params.power_iter_params.tol;
    sv_estimator_ctx_t *estimator = sv_estimator_create(
        state->sparse_handle_A,
        state->blas_handle_A,
        state->stream_A,
        state->curand_gen,
        state->constraint_matrix,
        state->constraint_matrix_t,
        state->num_constraints,
        state->num_variables);
    sv_estimator_run(estimator, &opts);
    sv_estimator_free(estimator);

    double primal_product_after[NUM_CONSTRAINTS];
    double dual_product_after[NUM_VARIABLES];
    CUDA_CHECK(cudaMemcpy(
        primal_product_after, state->primal_product_d,
        sizeof(primal_product_after), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        dual_product_after, state->dual_product_d,
        sizeof(dual_product_after), cudaMemcpyDeviceToHost));

    bool work_vectors_unchanged = true;
    for (int i = 0; i < NUM_CONSTRAINTS; ++i)
        work_vectors_unchanged &= primal_product_after[i] == primal_product[i];
    for (int j = 0; j < NUM_VARIABLES; ++j)
        work_vectors_unchanged &= dual_product_after[j] == dual_product[j];
    check("singular-value estimator owns its work vectors", work_vectors_unchanged);

    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}

static void check_empty_mask_skips_estimation(bool empty_variables)
{
    pdhg_parameters_t params;
    set_test_parameters(&params, true);
    lp_problem_t *problem = create_hadamard_problem();
    pdhg_solver_state_t *state = initialize_solver_state(problem, problem, &params);
    active_set_state_t *active_set = &state->active_set;
    state->control.step_size_base = 1.0;
    state->control.step_size = 1.0;
    bool *mask = empty_variables ? active_set->free_variable_mask_d
                                 : active_set->binding_constraint_mask_d;
    int *count = empty_variables ? &active_set->free_variable_count
                                : &active_set->binding_constraint_count;
    const int full_count = *count;
    CUDA_CHECK(cudaMemset(mask, 0, (size_t)full_count * sizeof(bool)));
    *count = 0;
    active_set->mask_changes_since_estimate = full_count;

    try_raise_active_set_step_size(state, &params);
    try_raise_active_set_step_size(state, &params);
    check(empty_variables ? "empty variable mask skips repeated norm estimates"
                          : "empty constraint mask skips repeated norm estimates",
          active_set->power_iteration_count == 0);
    check("an empty mask leaves the step unchanged", state->control.step_size == 1.0);

    CUDA_CHECK(cudaMemset(mask, 1, (size_t)full_count * sizeof(bool)));
    *count = full_count;
    active_set->mask_changes_since_estimate += full_count;
    try_raise_active_set_step_size(state, &params);
    check("norm estimation resumes when the mask becomes nonempty",
          active_set->power_iteration_count > 0);

    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}

static void check_estimator_retry_policy(bool degenerate)
{
    const double objective[] = {1.0, 1.0};
    const int row_ptr[] = {0, 1, 2};
    const int col_ind[] = {0, 1};
    const double values[] = {1.0, 2.0};
    const double lower[] = {0.0, 0.0};
    const double upper[] = {1.0, 1.0};
    const double constant = 0.0;
    matrix_desc_t matrix = {0};
    matrix.m = matrix.n = 2;
    matrix.fmt = matrix_csr;
    matrix.data.csr.nnz = 2;
    matrix.data.csr.row_ptr = row_ptr;
    matrix.data.csr.col_ind = col_ind;
    matrix.data.csr.vals = values;
    pdhg_parameters_t params;
    set_test_parameters(&params, true);
    params.scaling_params.geometric_mean_iterations = 0;
    params.scaling_params.l_inf_ruiz_iterations = 0;
    params.scaling_params.has_pock_chambolle_alpha = false;
    params.scaling_params.bound_objective_rescaling = false;
    lp_problem_t *problem = create_lp_problem(
        objective, &matrix, lower, upper, lower, upper, &constant);
    pdhg_solver_state_t *state = initialize_solver_state(problem, problem, &params);
    active_set_state_t *active_set = &state->active_set;
    state->control.step_size_base = 0.01;
    state->control.step_size = 0.01;
    if (degenerate)
    {
        const bool rows[] = {true, false};
        const bool columns[] = {false, true};
        CUDA_CHECK(cudaMemcpy(active_set->binding_constraint_mask_d,
                              rows, sizeof(rows), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(active_set->free_variable_mask_d,
                              columns, sizeof(columns), cudaMemcpyHostToDevice));
        active_set->binding_constraint_count = 1;
        active_set->free_variable_count = 1;
    }
    else
    {
        params.power_iter_params.max_iter = 1;
        params.power_iter_params.tol = 1e-30;
    }
    active_set->mask_changes_since_estimate = 2;
    try_raise_active_set_step_size(state, &params);
    const int first_iterations = active_set->power_iteration_count;
    check("the retry fixture actually runs the estimator", first_iterations > 0);
    try_raise_active_set_step_size(state, &params);
    check(degenerate ? "an unchanged zero operator is not estimated again"
                     : "an unfinished estimate continues with unchanged masks",
          degenerate ? active_set->power_iteration_count == first_iterations
                     : active_set->power_iteration_count > first_iterations);
    check("neither a zero nor an unfinished estimate raises the step",
          state->control.step_size == 0.01);
    if (degenerate)
    {
        const bool columns[] = {true, true};
        const active_set_mask_changes_t changes = {1, 0, 0, 0};
        CUDA_CHECK(cudaMemcpy(active_set->free_variable_mask_d,
                              columns, sizeof(columns), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(active_set->mask_changes_d,
                              &changes, sizeof(changes), cudaMemcpyHostToDevice));
        try_raise_active_set_step_size(state, &params);
        check("mask changes allow estimation after a degenerate result",
              active_set->power_iteration_count > first_iterations);
        check("the new nonzero operator can raise the step",
              state->control.step_size > 0.01);
    }
    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}

static void check_active_set_revert_state(void)
{
    pdhg_parameters_t params;
    set_test_parameters(&params, true);
    lp_problem_t *problem = create_hadamard_problem();
    pdhg_solver_state_t *state = initialize_solver_state(problem, problem, &params);
    state->control.step_size_base = 1.0;
    state->control.step_size = 2.0;
    state->control.total_count = 123;
    state->control.inner_count = 17;
    state->control.restart_count = 4;
    state->active_set.raise_checked = true;

    double anchor_primal[NUM_VARIABLES];
    double anchor_multiplier[NUM_VARIABLES];
    double anchor_dual[NUM_CONSTRAINTS];
    double existing_reduced_cost[NUM_VARIABLES];
    double anchor_hsd_scalars[HSD_SCALARS_COUNT];
    for (int j = 0; j < NUM_VARIABLES; ++j)
        anchor_primal[j] = 10.0 + j;
    for (int j = 0; j < NUM_VARIABLES; ++j)
        anchor_multiplier[j] = 70.0 + j;
    for (int j = 0; j < NUM_VARIABLES; ++j)
        existing_reduced_cost[j] = 50.0 + j;
    for (int i = 0; i < NUM_CONSTRAINTS; ++i)
        anchor_dual[i] = 20.0 + i;
    for (int i = 0; i < HSD_SCALARS_COUNT; ++i)
        anchor_hsd_scalars[i] = 30.0 + i;

    CUDA_CHECK(cudaMemcpy(
        state->active_set.anchor.primal_solution_d,
        anchor_primal,
        sizeof(anchor_primal),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->active_set.anchor.dual_solution_d,
        anchor_dual,
        sizeof(anchor_dual),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->pdhg_reduced_cost_d,
        existing_reduced_cost,
        sizeof(existing_reduced_cost),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->active_set.anchor.hsd_scalars_d,
        anchor_hsd_scalars,
        sizeof(anchor_hsd_scalars),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->active_set.anchor.primal_projection_multiplier_d,
        anchor_multiplier,
        sizeof(anchor_multiplier),
        cudaMemcpyHostToDevice));
    state->active_set.anchor.primal_weight = 3.0;
    state->active_set.anchor.pid.error_sum = 5.0;
    state->active_set.anchor.pid.last_error = 6.0;
    state->active_set.anchor.pid.has_last_error = true;

    const int total_count = state->control.total_count;

    revert_active_set_step_size(state, &params);

    double restored_initial_primal[NUM_VARIABLES];
    double restored_current_primal[NUM_VARIABLES];
    double restored_pdhg_primal[NUM_VARIABLES];
    double restored_initial_dual[NUM_CONSTRAINTS];
    double restored_current_dual[NUM_CONSTRAINTS];
    double restored_pdhg_dual[NUM_CONSTRAINTS];
    double restored_reduced_cost[NUM_VARIABLES];
    double restored_multiplier[NUM_VARIABLES];
    double restored_hsd_scalars[HSD_SCALARS_COUNT];
    double restored_cone_scale_history[BOX_CONE_SCALE_COUNT];
    CUDA_CHECK(cudaMemcpy(
        restored_initial_primal, state->initial_primal_solution_d,
        sizeof(restored_initial_primal), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        restored_current_primal, state->current_primal_solution_d,
        sizeof(restored_current_primal), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        restored_pdhg_primal, state->pdhg_primal_solution_d,
        sizeof(restored_pdhg_primal), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        restored_initial_dual, state->initial_dual_solution_d,
        sizeof(restored_initial_dual), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        restored_current_dual, state->current_dual_solution_d,
        sizeof(restored_current_dual), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        restored_pdhg_dual, state->pdhg_dual_solution_d,
        sizeof(restored_pdhg_dual), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        restored_reduced_cost, state->pdhg_reduced_cost_d,
        sizeof(restored_reduced_cost), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        restored_hsd_scalars, state->hsd_scalars_d,
        sizeof(restored_hsd_scalars), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        restored_cone_scale_history, state->dual_box_cone_state.scale_history_d,
        sizeof(restored_cone_scale_history), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        restored_multiplier, state->pdhg_primal_projection_multiplier_d,
        sizeof(restored_multiplier), cudaMemcpyDeviceToHost));

    bool iterates_restored = true;
    bool reduced_cost_untouched = true;
    bool multipliers_restored = true;
    for (int j = 0; j < NUM_VARIABLES; ++j)
    {
        iterates_restored &= restored_initial_primal[j] == anchor_primal[j];
        iterates_restored &= restored_current_primal[j] == anchor_primal[j];
        iterates_restored &= restored_pdhg_primal[j] == anchor_primal[j];
        reduced_cost_untouched &=
            restored_reduced_cost[j] == existing_reduced_cost[j];
        multipliers_restored &= restored_multiplier[j] == anchor_multiplier[j];
    }
    for (int i = 0; i < NUM_CONSTRAINTS; ++i)
    {
        iterates_restored &= restored_initial_dual[i] == anchor_dual[i];
        iterates_restored &= restored_current_dual[i] == anchor_dual[i];
        iterates_restored &= restored_pdhg_dual[i] == anchor_dual[i];
    }
    bool hsd_state_restored = true;
    for (int i = 0; i < HSD_SCALARS_COUNT; ++i)
        hsd_state_restored &= restored_hsd_scalars[i] == anchor_hsd_scalars[i];
    for (int i = 0; i < BOX_CONE_SCALE_COUNT; ++i)
        hsd_state_restored &= restored_cone_scale_history[i] ==
            anchor_hsd_scalars[HSD_DUAL_CONE_SCALE_HISTORY + i];

    check("a revert restores the base step",
          state->control.step_size == state->control.step_size_base);
    check("a revert restores every iterate from the raise point",
          iterates_restored);
    check("a direct revert leaves derived reduced cost untouched",
          reduced_cost_untouched);
    check("a revert restores the raise-point HSD state",
          hsd_state_restored);
    check("a revert restores the raise-point primal projection multipliers",
          multipliers_restored);
    check("a revert caps later raises below the diverged step",
          state->active_set.step_size_ceiling ==
              params.active_set_params.divergence_ceiling_ratio * 2.0);
    check("a revert consumes exactly one budget slot",
          state->active_set.revert_count == 1);
    check("a revert invalidates the previous raise decision",
          !state->active_set.raise_checked);
    check("a revert restores the raise-point primal weight",
          state->control.primal_weight == 3.0);
    check("a revert restores the raise-point PID state",
          state->control.pid.error_sum == 5.0 &&
              state->control.pid.last_error == 6.0 &&
              state->control.pid.has_last_error);
    check("a revert leaves iteration bookkeeping to the solver",
          state->control.total_count == total_count &&
              state->control.inner_count == 17 &&
              state->control.restart_count == 4);

    pdhg_solver_state_free(state);
    lp_problem_free(problem);
}

static int check_active_set_boost(void)
{
    pdhg_parameters_t params;

    set_test_parameters(&params, false);
    const int raises_without_boost = solve_and_check("boost off", &params);

    set_test_parameters(&params, true);
    const int raises_with_boost = solve_and_check("boost on", &params);

    set_test_parameters(&params, true);
    params.active_set_params.max_reverts = 0;
    const int raises_without_revert_budget = solve_and_check("no reverts", &params);

    check("the disabled controller reports no raise", raises_without_boost == 0);
    check("the operative submatrix earns at least one raise", raises_with_boost >= 1);
    check("a zero revert budget disables the boost", raises_without_revert_budget == 0);
    check_active_set_resources(false);
    check_active_set_resources(true);
    check_zero_revert_budget_disables_resources();
    check_estimator_owns_work_vectors();
    check_empty_mask_skips_estimation(true);
    check_empty_mask_skips_estimation(false);
    check_estimator_retry_policy(true);
    check_estimator_retry_policy(false);
    check_active_set_revert_state();

    printf("test_active_set_boost: %d failure(s)\n", failures);
    return failures != 0;
}

int main(void)
{
    const int status = test_gpu_status();
    if (status) return status;
    int failed = 0;
    failed |= check_active_set_boost();
    return failed || failures != 0;
}
