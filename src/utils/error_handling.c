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

#include "error_handling.h"
#include "hpdlp.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#define ERROR_MESSAGE_SIZE 512

static _Thread_local error_scope_t *current_scope;
static _Thread_local char last_error[ERROR_MESSAGE_SIZE];

static void vreport_error(const char *format, va_list args);

void enter_error_scope(error_scope_t *scope)
{
    last_error[0] = '\0';
    scope->is_releasing = false;
    current_scope = scope;
}

void leave_error_scope(void)
{
    current_scope = NULL;
}

void report_error(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vreport_error(format, args);
    va_end(args);
}

static void vreport_error(const char *format, va_list args)
{
    vsnprintf(last_error, sizeof(last_error), format, args);
    fprintf(stderr, "ERROR: %s\n", last_error);
}

void report_warning(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "WARNING: ");
    vfprintf(stderr, format, args);
    fputc('\n', stderr);
    va_end(args);
}

void raise_error(const char *format, ...)
{
    if (current_scope && current_scope->is_releasing)
        return;

    va_list args;
    va_start(args, format);
    vreport_error(format, args);
    va_end(args);

    if (!current_scope)
        exit(EXIT_FAILURE);
    longjmp(current_scope->env, 1);
}

const char *hpdlp_last_error(void)
{
    return last_error;
}
