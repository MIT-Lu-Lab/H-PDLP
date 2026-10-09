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

#include "error_handling.h"
#include <cublas_v2.h>

#define CUBLAS_CHECK(call)                                      \
    do                                                          \
    {                                                           \
        cublasStatus_t status = call;                           \
        if (status != CUBLAS_STATUS_SUCCESS)                    \
            raise_error("cuBLAS error at %s:%d: %s", __FILE__,  \
                        __LINE__, cublasGetStatusName(status)); \
    } while (0)
