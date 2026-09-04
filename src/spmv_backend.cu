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

#include "spmv_backend.h"
#include "cuda_macros.h"
#include "utils.h"

const char *spmv_backend_name(void)
{
    return CUHSDLP_HAS_SPMVOP ? "cusparseSpMVOp" : "cusparseSpMV";
}

void create_spmv_plan(
    cusparseHandle_t handle,
    cusparseSpMatDescr_t matrix,
    cusparseDnVecDescr_t input,
    cusparseDnVecDescr_t output,
    spmv_plan_t *plan)
{
    size_t buffer_size = 0;
    plan->matrix = matrix;
#if CUHSDLP_HAS_SPMVOP
    cusparseSpMVOpDescr_t spmvop_descriptor = NULL;
    cusparseSpMVOpPlan_t spmvop_plan = NULL;
    CUSPARSE_CHECK(cusparseSpMVOp_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix, input, output, output,
        CUDA_R_64F, CUSPARSE_SPMVOP_ALG2, &buffer_size));
    CUDA_CHECK(cudaMalloc(&plan->buffer, buffer_size));
    CUSPARSE_CHECK(cusparseSpMVOp_createDescr(
        handle, &spmvop_descriptor, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix,
        input, output, output, CUDA_R_64F, CUSPARSE_SPMVOP_ALG2, plan->buffer));
    CUSPARSE_CHECK(cusparseSpMVOp_createPlan(handle, spmvop_descriptor, &spmvop_plan, NULL, 0));
    plan->descriptor = (void *)spmvop_descriptor;
    plan->plan = (void *)spmvop_plan;
#else
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, matrix, input,
        &HOST_ZERO, output, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &buffer_size));
    CUDA_CHECK(cudaMalloc(&plan->buffer, buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, matrix, input,
        &HOST_ZERO, output, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, plan->buffer));
    plan->descriptor = NULL;
    plan->plan = NULL;
#endif
}

void execute_spmv(
    cusparseHandle_t handle,
    cusparseDnVecDescr_t input,
    cusparseDnVecDescr_t output,
    const spmv_plan_t *plan)
{
#if CUHSDLP_HAS_SPMVOP
    CUSPARSE_CHECK(cusparseSpMVOp(
        handle, (cusparseSpMVOpPlan_t)plan->plan, &HOST_ONE, &HOST_ZERO, input, output, output));
#else
    CUSPARSE_CHECK(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, plan->matrix, input,
        &HOST_ZERO, output, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, plan->buffer));
#endif
}

void destroy_spmv_plan(spmv_plan_t *plan)
{
#if CUHSDLP_HAS_SPMVOP
    if (plan->plan)
        CUSPARSE_CHECK(cusparseSpMVOp_destroyPlan((cusparseSpMVOpPlan_t)plan->plan));
    if (plan->descriptor)
        CUSPARSE_CHECK(cusparseSpMVOp_destroyDescr((cusparseSpMVOpDescr_t)plan->descriptor));
#endif
    plan->plan = NULL;
    plan->descriptor = NULL;
    if (plan->buffer)
        CUDA_CHECK(cudaFree(plan->buffer));
    plan->buffer = NULL;
    plan->matrix = NULL;
}
