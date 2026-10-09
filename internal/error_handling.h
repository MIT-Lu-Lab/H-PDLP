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

#include <setjmp.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        jmp_buf env;
        bool is_releasing;
    } error_scope_t;

    void enter_error_scope(error_scope_t *scope);
    void leave_error_scope(void);
    void report_error(const char *format, ...);
    void report_warning(const char *format, ...);
    void raise_error(const char *format, ...);

#ifdef __cplusplus
}
#endif
