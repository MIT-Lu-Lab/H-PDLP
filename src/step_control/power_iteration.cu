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

#include "power_iteration.h"
#include "cuda_macros.h"
#include "cuda_utils.h"
#include "spmv_backend.h"
#include "utils.h"
#include <math.h>

struct sv_estimator_ctx
{
    cusparseHandle_t sparse_handle;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
    curandGenerator_t curand_gen;
    cusparseSpMatDescr_t matrix;
    cusparseSpMatDescr_t transpose;
    spmv_plan_t matrix_plan;
    spmv_plan_t transpose_plan;
    int num_rows;
    int num_columns;
    int num_blocks_row;
    int num_blocks_column;
    double *iterate_d;
    double *normalized_iterate_d;
    double *transpose_product_d;
    double *iterate_norm_d;
    cusparseDnVecDescr_t spmv_iterate;
    cusparseDnVecDescr_t spmv_normalized_iterate;
    cusparseDnVecDescr_t spmv_transpose_product;
    double *sigma_squared_mp;
    double *residual_norm_mp;
    bool has_warm_start;
};

__global__ void apply_mask_kernel(
    double *__restrict__ vector_d,
    const bool *__restrict__ mask_d,
    int n);
__global__ void seed_zero_warm_start_entries_kernel(
    double *__restrict__ vector_d,
    const bool *__restrict__ mask_d,
    double perturbation_amplitude,
    int n);
__global__ void normalize_power_iteration_vector_kernel(
    double *normalized_vector_d,
    const double *vector_d,
    const double *vector_norm_d,
    int n);
__global__ void replace_normalized_iterate_with_residual_kernel(
    double *normalized_iterate_d,
    const double *iterate_d,
    const double *sigma_squared_d,
    int n);
static void run_power_iteration_step(
    sv_estimator_ctx_t *ctx,
    const sv_estimator_opts_t *opts,
    bool evaluate_residual);
static sv_estimator_status_t classify_power_iteration_outcome(
    const sv_estimator_ctx_t *ctx,
    const sv_estimator_opts_t *opts);

sv_estimator_ctx_t *sv_estimator_create(
    cusparseHandle_t sparse_handle,
    cublasHandle_t blas_handle,
    cudaStream_t stream,
    curandGenerator_t curand_gen,
    const cu_sparse_matrix_csr_t *matrix,
    const cu_sparse_matrix_csr_t *transpose,
    int num_rows,
    int num_columns)
{
    sv_estimator_ctx_t *ctx =
        (sv_estimator_ctx_t *)safe_calloc(1, sizeof(sv_estimator_ctx_t));
    ctx->sparse_handle = sparse_handle;
    ctx->blas_handle = blas_handle;
    ctx->stream = stream;
    ctx->curand_gen = curand_gen;
    ctx->num_rows = num_rows;
    ctx->num_columns = num_columns;
    ctx->num_blocks_row =
        (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    ctx->num_blocks_column =
        (num_columns + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &ctx->matrix,
        num_rows,
        num_columns,
        matrix->num_nonzeros,
        matrix->row_ptr,
        matrix->col_ind,
        matrix->val,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &ctx->transpose,
        num_columns,
        num_rows,
        transpose->num_nonzeros,
        transpose->row_ptr,
        transpose->col_ind,
        transpose->val,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    allocate_zeroed_device_buffer(
        (void **)&ctx->iterate_d,
        (((size_t)num_rows + 1) & ~(size_t)1) * sizeof(double));
    allocate_zeroed_device_buffer(
        (void **)&ctx->normalized_iterate_d,
        (size_t)num_rows * sizeof(double));
    allocate_zeroed_device_buffer(
        (void **)&ctx->transpose_product_d,
        (size_t)num_columns * sizeof(double));
    allocate_zeroed_device_buffer((void **)&ctx->iterate_norm_d, sizeof(double));
    CUDA_CHECK(cudaHostAlloc(&ctx->sigma_squared_mp, sizeof(double), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&ctx->residual_norm_mp, sizeof(double), cudaHostAllocMapped));
    CUSPARSE_CHECK(cusparseCreateDnVec(
        &ctx->spmv_iterate, num_rows, ctx->iterate_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(
        &ctx->spmv_normalized_iterate, num_rows, ctx->normalized_iterate_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(
        &ctx->spmv_transpose_product, num_columns, ctx->transpose_product_d, CUDA_R_64F));
    create_spmv_plan(
        sparse_handle, ctx->transpose, ctx->spmv_normalized_iterate,
        ctx->spmv_transpose_product, &ctx->transpose_plan);
    create_spmv_plan(
        sparse_handle, ctx->matrix, ctx->spmv_transpose_product,
        ctx->spmv_iterate, &ctx->matrix_plan);
    return ctx;
}

sv_estimator_result_t sv_estimator_run(
    sv_estimator_ctx_t *ctx,
    const sv_estimator_opts_t *opts)
{
    sv_estimator_result_t result = {};
    result.status = SV_ESTIMATOR_DEGENERATE;

    if (ctx->num_rows == 0 || ctx->num_columns == 0)
    {
        return result;
    }

    bool random_start = !ctx->has_warm_start;
    if (!random_start && opts->row_mask_d)
    {
        apply_mask_kernel<<<
            ctx->num_blocks_row,
            THREADS_PER_BLOCK,
            0,
            ctx->stream>>>(ctx->iterate_d, opts->row_mask_d, ctx->num_rows);

        CUBLAS_CHECK(cublasDnrm2_v2_64(
            ctx->blas_handle,
            ctx->num_rows,
            ctx->iterate_d,
            1,
            ctx->iterate_norm_d));
        double warm_start_norm = 0.0;
        CUDA_CHECK(cudaMemcpyAsync(
            &warm_start_norm,
            ctx->iterate_norm_d,
            sizeof(double),
            cudaMemcpyDeviceToHost,
            ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

        if (isfinite(warm_start_norm) && warm_start_norm > 0.0)
        {
            const double perturbation_amplitude =
                1e-3 * warm_start_norm / sqrt((double)ctx->num_rows);
            seed_zero_warm_start_entries_kernel<<<
                ctx->num_blocks_row,
                THREADS_PER_BLOCK,
                0,
                ctx->stream>>>(
                ctx->iterate_d,
                opts->row_mask_d,
                perturbation_amplitude,
                ctx->num_rows);
        }
        else
        {
            random_start = true;
        }
    }
    if (random_start)
    {
        const size_t curand_sample_count =
            ((size_t)ctx->num_rows + 1) & ~(size_t)1;
        CURAND_CHECK(curandGenerateNormalDouble(
            ctx->curand_gen,
            ctx->iterate_d,
            curand_sample_count,
            0.0,
            1.0));
        if (opts->row_mask_d)
        {
            apply_mask_kernel<<<
                ctx->num_blocks_row,
                THREADS_PER_BLOCK,
                0,
                ctx->stream>>>(
                ctx->iterate_d,
                opts->row_mask_d,
                ctx->num_rows);
        }
    }

    result.status = SV_ESTIMATOR_MAX_ITER;

    const int evaluation_interval = 5;
    const int num_full_batches = opts->max_iter / evaluation_interval;
    const int remaining_iterations = opts->max_iter % evaluation_interval;

    if (num_full_batches > 0)
    {
        cudaGraph_t graph = NULL;
        cudaGraphExec_t graph_exec = NULL;
        CUDA_CHECK(cudaStreamBeginCapture(
            ctx->stream,
            cudaStreamCaptureModeGlobal));
        for (int k = 0; k < evaluation_interval - 1; ++k)
        {
            run_power_iteration_step(ctx, opts, false);
        }
        run_power_iteration_step(ctx, opts, true);
        CUDA_CHECK(cudaStreamEndCapture(ctx->stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(
            &graph_exec,
            graph,
            NULL,
            NULL,
            0));
        CUDA_CHECK(cudaGraphDestroy(graph));

        for (int batch = 0; batch < num_full_batches; ++batch)
        {
            CUDA_CHECK(cudaGraphLaunch(
                graph_exec,
                ctx->stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            result.iterations += evaluation_interval;
            result.status = classify_power_iteration_outcome(ctx, opts);
            if (result.status != SV_ESTIMATOR_MAX_ITER)
            {
                break;
            }
        }

        CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    }

    if (result.status == SV_ESTIMATOR_MAX_ITER && remaining_iterations > 0)
    {
        for (int k = 0; k < remaining_iterations - 1; ++k)
        {
            run_power_iteration_step(ctx, opts, false);
        }
        run_power_iteration_step(ctx, opts, true);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        result.iterations += remaining_iterations;
        result.status = classify_power_iteration_outcome(ctx, opts);
    }

    if (result.status != SV_ESTIMATOR_DEGENERATE)
    {
        result.max_singular_value = sqrt(*ctx->sigma_squared_mp);
    }
    ctx->has_warm_start = result.status != SV_ESTIMATOR_DEGENERATE;
    return result;
}

static void run_power_iteration_step(
    sv_estimator_ctx_t *ctx,
    const sv_estimator_opts_t *opts,
    bool evaluate_residual)
{
    const int num_rows = ctx->num_rows;
    const int num_columns = ctx->num_columns;

    CUBLAS_CHECK(cublasDnrm2_v2_64(
        ctx->blas_handle,
        num_rows,
        ctx->iterate_d,
        1,
        ctx->iterate_norm_d));
    normalize_power_iteration_vector_kernel<<<
        ctx->num_blocks_row,
        THREADS_PER_BLOCK,
        0,
        ctx->stream>>>(
        ctx->normalized_iterate_d,
        ctx->iterate_d,
        ctx->iterate_norm_d,
        num_rows);
    execute_spmv(
        ctx->sparse_handle,
        ctx->spmv_normalized_iterate,
        ctx->spmv_transpose_product,
        &ctx->transpose_plan);
    if (opts->column_mask_d)
    {
        apply_mask_kernel<<<
            ctx->num_blocks_column,
            THREADS_PER_BLOCK,
            0,
            ctx->stream>>>(
            ctx->transpose_product_d,
            opts->column_mask_d,
            num_columns);
    }
    execute_spmv(
        ctx->sparse_handle,
        ctx->spmv_transpose_product,
        ctx->spmv_iterate,
        &ctx->matrix_plan);
    if (opts->row_mask_d)
    {
        apply_mask_kernel<<<
            ctx->num_blocks_row,
            THREADS_PER_BLOCK,
            0,
            ctx->stream>>>(ctx->iterate_d, opts->row_mask_d, num_rows);
    }
    if (evaluate_residual)
    {
        CUBLAS_CHECK(cublasDdot(
            ctx->blas_handle,
            num_rows,
            ctx->normalized_iterate_d,
            1,
            ctx->iterate_d,
            1,
            ctx->sigma_squared_mp));
        replace_normalized_iterate_with_residual_kernel<<<
            ctx->num_blocks_row,
            THREADS_PER_BLOCK,
            0,
            ctx->stream>>>(
            ctx->normalized_iterate_d,
            ctx->iterate_d,
            ctx->sigma_squared_mp,
            num_rows);
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            ctx->blas_handle,
            num_rows,
            ctx->normalized_iterate_d,
            1,
            ctx->residual_norm_mp));
    }
}

static sv_estimator_status_t classify_power_iteration_outcome(
    const sv_estimator_ctx_t *ctx,
    const sv_estimator_opts_t *opts)
{
    const double sigma_squared = *ctx->sigma_squared_mp;
    if (!(sigma_squared > 0.0) || !isfinite(sigma_squared))
    {
        return SV_ESTIMATOR_DEGENERATE;
    }
    const double residual_norm = *ctx->residual_norm_mp;
    if (!isfinite(residual_norm))
    {
        return SV_ESTIMATOR_DEGENERATE;
    }
    if (opts->early_exit_singular_value_threshold > 0.0 &&
        sqrt(sigma_squared) >= opts->early_exit_singular_value_threshold)
    {
        return SV_ESTIMATOR_EARLY_EXIT;
    }
    if (residual_norm < opts->tol * fmin(1.0, sigma_squared))
    {
        return SV_ESTIMATOR_CONVERGED;
    }
    return SV_ESTIMATOR_MAX_ITER;
}

void sv_estimator_free(sv_estimator_ctx_t *ctx)
{
    if (!ctx)
    {
        return;
    }
    destroy_spmv_plan(&ctx->transpose_plan);
    destroy_spmv_plan(&ctx->matrix_plan);
    CUSPARSE_CHECK(cusparseDestroySpMat(ctx->transpose));
    CUSPARSE_CHECK(cusparseDestroySpMat(ctx->matrix));
    CUSPARSE_CHECK(cusparseDestroyDnVec(ctx->spmv_iterate));
    CUSPARSE_CHECK(cusparseDestroyDnVec(ctx->spmv_normalized_iterate));
    CUSPARSE_CHECK(cusparseDestroyDnVec(ctx->spmv_transpose_product));
    CUDA_CHECK(cudaFree(ctx->iterate_d));
    CUDA_CHECK(cudaFree(ctx->normalized_iterate_d));
    CUDA_CHECK(cudaFree(ctx->transpose_product_d));
    CUDA_CHECK(cudaFree(ctx->iterate_norm_d));
    CUDA_CHECK(cudaFreeHost(ctx->sigma_squared_mp));
    CUDA_CHECK(cudaFreeHost(ctx->residual_norm_mp));
    free(ctx);
}

__global__ void apply_mask_kernel(
    double *__restrict__ vector_d,
    const bool *__restrict__ mask_d,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (!mask_d[i])
        vector_d[i] = 0.0;
}

__global__ void seed_zero_warm_start_entries_kernel(
    double *__restrict__ vector_d,
    const bool *__restrict__ mask_d,
    double perturbation_amplitude,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (!mask_d[i] || vector_d[i] != 0.0)
        return;

    unsigned long long hash =
        (unsigned long long)i + 0x9E3779B97F4A7C15ULL;
    hash = (hash ^ (hash >> 30)) * 0xBF58476D1CE4E5B9ULL;
    hash = (hash ^ (hash >> 27)) * 0x94D049BB133111EBULL;
    hash = hash ^ (hash >> 31);
    const double uniform_sample =
        (double)(hash >> 11) * (1.0 / 9007199254740992.0);
    vector_d[i] = perturbation_amplitude * (uniform_sample - 0.5);
}

__global__ void normalize_power_iteration_vector_kernel(
    double *normalized_vector_d,
    const double *vector_d,
    const double *vector_norm_d,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double vector_norm = *vector_norm_d;
    normalized_vector_d[i] = (vector_norm > 0.0) ? (vector_d[i] / vector_norm) : 0.0;
}

__global__ void replace_normalized_iterate_with_residual_kernel(
    double *normalized_iterate_d,
    const double *iterate_d,
    const double *sigma_squared_d,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double sigma_squared = *sigma_squared_d;
    normalized_iterate_d[i] = iterate_d[i] - sigma_squared * normalized_iterate_d[i];
}
