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

#pragma once

#include "cusparse_compat.h"
#include "internal_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    const char *spmv_backend_name(void);

    void create_spmv_plan(
        cusparseHandle_t handle,
        cusparseSpMatDescr_t matrix,
        cusparseDnVecDescr_t input,
        cusparseDnVecDescr_t output,
        spmv_plan_t *plan);

    void execute_spmv(
        cusparseHandle_t handle,
        cusparseDnVecDescr_t input,
        cusparseDnVecDescr_t output,
        const spmv_plan_t *plan);

    void destroy_spmv_plan(spmv_plan_t *plan);

#ifdef __cplusplus
}
#endif
