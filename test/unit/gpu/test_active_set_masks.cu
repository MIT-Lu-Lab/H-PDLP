#include "cuda_macros.h"
#include "gpu_support.h"
#include "internal_types.h"
#include "test_checks.h"
#include <math.h>
#include <stdio.h>

__global__ void update_binding_constraint_mask_kernel(
    const double *, const double *, const double *, const double *,
    int *, bool *, active_set_mask_changes_t *, double, int, int, int);
__global__ void update_free_variable_mask_kernel(
    const double *, const double *, const double *, const double *,
    const double *, int *, bool *, active_set_mask_changes_t *, double,
    int, int, int);

enum { NUM_ROWS = 10, NUM_COLUMNS = 12, NUM_ZERO_SCALE_COLUMNS = 4, NUM_GAUGES = 2 };
enum { CURRENT_ITER = 100, WINDOW_START_ITER = 0 };
enum { ROW_WITH_TINY_UPPER_SLACK = 3, COLUMN_AT_LOWER_MARGIN = 2 };

static const double RELATIVE_TOL = 1e-8;
static const double CONE_SCALE = 4.0;
static const double PRIMAL_SCALE = 0.25;
static const double GAUGES[NUM_GAUGES] = {1e-6, 1e6};

struct row_case
{
    double lower;
    double upper;
    double dual_projection_input;
    int retained;
};

struct column_case
{
    double lower;
    double upper;
    double objective;
    double tolerance_factor;
    int retained;
};

struct zero_scale_column_case
{
    const char *description;
    double lower;
    double upper;
    double multiplier;
    int retained;
};

static const struct row_case ROWS[NUM_ROWS] = {
    {0.0, 0.0, 0.0, 1},
    {-1.0, 1.0, 0.0, 0},
    {-1.0, 1.0, -1.0, 1},
    {-1.0, 1.0, -1.0 + 1e-12, 1},
    {-1.0, 1.0, -1.0 + 1e-6, 0},
    {-1.0, 1.0, 1.0 - 1e-12, 1},
    {-INFINITY, INFINITY, -5.0, 0},
    {0.0, INFINITY, -1e-12, 1},
    {0.0, INFINITY, -1.0, 0},
    {-INFINITY, 0.0, 1.0, 0}};

static const struct column_case COLUMNS[NUM_COLUMNS] = {
    {0.0, 0.0, 0.0, 0.0, 0},
    {0.0, 1.0, 1.0, -2.0, 0},
    {0.0, 1.0, 2.0, -1.0, 1},
    {0.0, 1.0, 1.0, -0.5, 1},
    {-1.0, 0.0, 1.0, 2.0, 0},
    {-1.0, 0.0, 2.0, 1.0, 1},
    {-1.0, 0.0, 1.0, 0.5, 1},
    {2.0, 5.0, 1.0, 1e12, 1},
    {3.0, 3.0, 1.0, 0.0, 1},
    {0.0, INFINITY, 1.0, 2.0, 1},
    {0.0, 1.0, 1.0, INFINITY, 1},
    {-1.0, 0.0, 1.0, -INFINITY, 1}};

static const struct zero_scale_column_case ZERO_SCALE_COLUMNS[NUM_ZERO_SCALE_COLUMNS] = {
    {"zero primal scale clamps a lower-bound column with a positive multiplier",
     0.0, 1.0, 1.0, 0},
    {"zero primal scale retains a lower-bound column at zero multiplier",
     0.0, 1.0, 0.0, 1},
    {"zero primal scale clamps an upper-bound column with a negative multiplier",
     -1.0, 0.0, -1.0, 0},
    {"zero primal scale retains an upper-bound column at zero multiplier",
     -1.0, 0.0, 0.0, 1}};

static double project_dual_response(double q, double lower, double upper, double scale)
{
    const double box_lower = isfinite(lower) ? lower * scale : -INFINITY;
    const double box_upper = isfinite(upper) ? upper * scale : INFINITY;
    return q + fmax(box_lower, fmin(-q, box_upper));
}

struct mask_buffers
{
    double *row_dual_projection_input;
    double *cone_scale;
    double *row_lower;
    double *row_upper;
    double *column_primal_projection_multiplier;
    double *column_objective;
    double *primal_scale;
    double *column_lower;
    double *column_upper;
    int *constraint_last_binding_iter;
    int *variable_last_free_iter;
    bool *binding_constraint_mask;
    bool *free_variable_mask;
    active_set_mask_changes_t *mask_changes;
};

static void allocate_buffers(struct mask_buffers *b)
{
    CUDA_CHECK(cudaMallocManaged(&b->row_dual_projection_input, NUM_ROWS * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&b->cone_scale, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&b->row_lower, NUM_ROWS * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&b->row_upper, NUM_ROWS * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(
        &b->column_primal_projection_multiplier,
        NUM_COLUMNS * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&b->column_objective, NUM_COLUMNS * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&b->primal_scale, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&b->column_lower, NUM_COLUMNS * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&b->column_upper, NUM_COLUMNS * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(
        &b->constraint_last_binding_iter,
        NUM_ROWS * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(
        &b->variable_last_free_iter,
        NUM_COLUMNS * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(
        &b->binding_constraint_mask,
        NUM_ROWS * sizeof(bool)));
    CUDA_CHECK(cudaMallocManaged(
        &b->free_variable_mask,
        NUM_COLUMNS * sizeof(bool)));
    CUDA_CHECK(cudaMallocManaged(
        &b->mask_changes,
        sizeof(active_set_mask_changes_t)));
}

static void release_buffers(struct mask_buffers *b)
{
    CUDA_CHECK(cudaFree(b->row_dual_projection_input));
    CUDA_CHECK(cudaFree(b->cone_scale));
    CUDA_CHECK(cudaFree(b->row_lower));
    CUDA_CHECK(cudaFree(b->row_upper));
    CUDA_CHECK(cudaFree(b->column_primal_projection_multiplier));
    CUDA_CHECK(cudaFree(b->column_objective));
    CUDA_CHECK(cudaFree(b->primal_scale));
    CUDA_CHECK(cudaFree(b->column_lower));
    CUDA_CHECK(cudaFree(b->column_upper));
    CUDA_CHECK(cudaFree(b->constraint_last_binding_iter));
    CUDA_CHECK(cudaFree(b->variable_last_free_iter));
    CUDA_CHECK(cudaFree(b->binding_constraint_mask));
    CUDA_CHECK(cudaFree(b->free_variable_mask));
    CUDA_CHECK(cudaFree(b->mask_changes));
}

static void fill_rows(struct mask_buffers *b, double gauge)
{
    *b->cone_scale = CONE_SCALE * gauge;
    for (int i = 0; i < NUM_ROWS; ++i)
    {
        b->row_lower[i] = ROWS[i].lower;
        b->row_upper[i] = ROWS[i].upper;
        b->row_dual_projection_input[i] =
            ROWS[i].dual_projection_input * CONE_SCALE * gauge;
        b->constraint_last_binding_iter[i] = -1;
        b->binding_constraint_mask[i] = false;
    }
}

static void fill_columns(
    struct mask_buffers *b,
    double gauge,
    double relative_tol)
{
    *b->primal_scale = PRIMAL_SCALE * gauge;
    for (int j = 0; j < NUM_COLUMNS; ++j)
    {
        b->column_lower[j] = COLUMNS[j].lower;
        b->column_upper[j] = COLUMNS[j].upper;
        b->column_objective[j] = COLUMNS[j].objective;
        const double multiplier_tolerance =
            relative_tol *
            fmax(1.0, fabs(b->column_objective[j])) *
            (*b->primal_scale);
        b->column_primal_projection_multiplier[j] =
            -COLUMNS[j].tolerance_factor * multiplier_tolerance;
        b->variable_last_free_iter[j] = -1;
        b->free_variable_mask[j] = false;
    }
}

static void run_row_mask_kernel(struct mask_buffers *b, double relative_tol)
{
    update_binding_constraint_mask_kernel<<<1, 256>>>(
        b->row_dual_projection_input,
        b->cone_scale,
        b->row_lower,
        b->row_upper,
        b->constraint_last_binding_iter,
        b->binding_constraint_mask,
        b->mask_changes,
        relative_tol,
        CURRENT_ITER,
        WINDOW_START_ITER,
        NUM_ROWS);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void run_column_mask_kernel(
    struct mask_buffers *b,
    double relative_tol,
    int num_columns)
{
    update_free_variable_mask_kernel<<<1, 256>>>(
        b->column_primal_projection_multiplier,
        b->column_objective,
        b->primal_scale,
        b->column_lower,
        b->column_upper,
        b->variable_last_free_iter,
        b->free_variable_mask,
        b->mask_changes,
        relative_tol,
        CURRENT_ITER,
        WINDOW_START_ITER,
        num_columns);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void run_masks(
    struct mask_buffers *b,
    double gauge,
    double relative_tol)
{
    fill_rows(b, gauge);
    fill_columns(b, gauge, relative_tol);
    *b->mask_changes = {};
    run_row_mask_kernel(b, relative_tol);
    run_column_mask_kernel(b, relative_tol, NUM_COLUMNS);
}

static void check_nonbinding_rows_have_vanishing_response(void)
{
    for (int i = 0; i < NUM_ROWS; ++i)
    {
        if (ROWS[i].lower == ROWS[i].upper || ROWS[i].retained)
            continue;
        const double q = ROWS[i].dual_projection_input * CONE_SCALE;
        const double response =
            project_dual_response(q, ROWS[i].lower, ROWS[i].upper, CONE_SCALE);
        char message[160];
        snprintf(message, sizeof(message),
                 "row %d with a vanishing response is not called binding at "
                 "zero margin", i);
        check(message, response == 0.0);
    }
}

static void check_retention_table(struct mask_buffers *b)
{
    run_masks(b, 1.0, RELATIVE_TOL);

    int expected_retained = 0;
    for (int i = 0; i < NUM_ROWS; ++i)
    {
        char message[160];
        snprintf(message, sizeof(message), "row %d retention", i);
        check(message,
              (int)b->binding_constraint_mask[i] == ROWS[i].retained);
        expected_retained += ROWS[i].retained;
    }
    for (int j = 0; j < NUM_COLUMNS; ++j)
    {
        char message[160];
        snprintf(message, sizeof(message), "column %d retention", j);
        check(message, (int)b->free_variable_mask[j] == COLUMNS[j].retained);
        expected_retained += COLUMNS[j].retained;
    }

    check("every retained index is counted as exactly one mask entry",
          b->mask_changes->variables_added +
                  b->mask_changes->constraints_added ==
              expected_retained &&
              b->mask_changes->variables_removed == 0 &&
              b->mask_changes->constraints_removed == 0);
}

static void check_gauge_invariance(struct mask_buffers *b)
{
    run_masks(b, 1.0, RELATIVE_TOL);
    int row_reference[NUM_ROWS];
    int column_reference[NUM_COLUMNS];
    for (int i = 0; i < NUM_ROWS; ++i)
        row_reference[i] = (int)b->binding_constraint_mask[i];
    for (int j = 0; j < NUM_COLUMNS; ++j)
        column_reference[j] = (int)b->free_variable_mask[j];

    for (int g = 0; g < NUM_GAUGES; ++g)
    {
        run_masks(b, GAUGES[g], RELATIVE_TOL);
        for (int i = 0; i < NUM_ROWS; ++i)
        {
            char message[160];
            snprintf(message, sizeof(message),
                     "row %d retention is gauge invariant at factor %g",
                     i, GAUGES[g]);
            check(message,
                  (int)b->binding_constraint_mask[i] == row_reference[i]);
        }
        for (int j = 0; j < NUM_COLUMNS; ++j)
        {
            char message[160];
            snprintf(message, sizeof(message),
                     "column %d retention is gauge invariant at factor %g",
                     j, GAUGES[g]);
            check(message,
                  (int)b->free_variable_mask[j] == column_reference[j]);
        }
    }
}

static void check_zero_tolerance(struct mask_buffers *b)
{
    run_masks(b, 1.0, 0.0);
    check("zero tolerance leaves a row with a tiny upper slack nonbinding",
          !b->binding_constraint_mask[ROW_WITH_TINY_UPPER_SLACK]);
    check("zero tolerance retains a lower-bound column at zero multiplier",
          b->free_variable_mask[COLUMN_AT_LOWER_MARGIN]);
}

static void check_zero_primal_scale(struct mask_buffers *b)
{
    *b->primal_scale = 0.0;
    for (int j = 0; j < NUM_ZERO_SCALE_COLUMNS; ++j)
    {
        b->column_lower[j] = ZERO_SCALE_COLUMNS[j].lower;
        b->column_upper[j] = ZERO_SCALE_COLUMNS[j].upper;
        b->column_objective[j] = 1.0;
        b->column_primal_projection_multiplier[j] =
            ZERO_SCALE_COLUMNS[j].multiplier;
        b->variable_last_free_iter[j] = -1;
        b->free_variable_mask[j] = false;
    }
    *b->mask_changes = {};
    run_column_mask_kernel(b, RELATIVE_TOL, NUM_ZERO_SCALE_COLUMNS);

    for (int j = 0; j < NUM_ZERO_SCALE_COLUMNS; ++j)
        check(ZERO_SCALE_COLUMNS[j].description,
              (int)b->free_variable_mask[j] == ZERO_SCALE_COLUMNS[j].retained);
}

int main(void)
{
    const int status = test_gpu_status();
    if (status) return status;
    struct mask_buffers b;
    allocate_buffers(&b);
    check_nonbinding_rows_have_vanishing_response();
    check_retention_table(&b);
    check_gauge_invariance(&b);
    check_zero_tolerance(&b);
    check_zero_primal_scale(&b);
    release_buffers(&b);
    return failures != 0;
}
