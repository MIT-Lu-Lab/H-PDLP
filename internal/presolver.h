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

#include "PSLP_API.h"
#include "PSLP_stats.h"
#include "cuhsdlp.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        Presolver *presolver;
        Settings *settings;
        lp_problem_t *reduced_problem;
        bool problem_solved_during_presolve;
        double presolve_time;
        char presolve_status;
    } cuhsdlp_presolve_info_t;

    cuhsdlp_presolve_info_t *pslp_presolve(const lp_problem_t *original_prob, bool verbose);

    cuhsdlp_result_t *create_result_from_presolve(const cuhsdlp_presolve_info_t *info, const lp_problem_t *original_prob);
    const char *get_presolve_status_str(int status);

    void pslp_postsolve(const cuhsdlp_presolve_info_t *info,
                        cuhsdlp_result_t *reduced_result,
                        const lp_problem_t *original_prob);

    void cuhsdlp_presolve_info_free(cuhsdlp_presolve_info_t *info);

#ifdef __cplusplus
}
#endif
