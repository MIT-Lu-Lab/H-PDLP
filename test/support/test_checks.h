#pragma once

#include <math.h>
#include <stdio.h>

static int failures;

static inline void check(const char *label, int condition)
{
    if (!condition)
    {
        fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}

#define CHECK(condition) check(#condition, (condition))
