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

#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#define CUSPARSE_CHECK(call)                                           \
    do                                                                 \
    {                                                                  \
        cusparseStatus_t status = call;                                \
        if (status != CUSPARSE_STATUS_SUCCESS)                         \
        {                                                              \
            fprintf(stderr, "cuSPARSE Error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cusparseGetErrorName(status));           \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)
