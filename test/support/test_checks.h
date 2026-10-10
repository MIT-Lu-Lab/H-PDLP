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

static inline void check_close(const char *label, double actual, double expected,
                               double tolerance)
{
    if (!isfinite(actual) || fabs(actual - expected) > tolerance * (1.0 + fabs(expected)))
    {
        fprintf(stderr, "FAIL: %s: got %.17g, expected %.17g\n", label, actual, expected);
        ++failures;
    }
}

#define CHECK(condition) check(#condition, (condition))
