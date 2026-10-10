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

#include "hpdlp.h"
#include "display.h"
#include "mps_parser.h"
#include "presolver.h"
#include "utils.h"
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void print_usage(const char *prog_name);
static bool parse_int_arg(const char *option_name, const char *value, int *out);
static bool parse_double_arg(const char *option_name, const char *value, double *out);
static bool is_writable_directory(const char *path);
static char *extract_instance_name(const char *filename);
static void strip_suffix(char *name, const char *suffix);
static bool save_solver_summary(const hpdlp_result_t *result, const char *output_dir,
                                const char *instance_name);
static FILE *open_output_file(const char *output_dir, const char *instance_name,
                              const char *suffix);
static char *get_output_path(const char *output_dir, const char *instance_name,
                             const char *suffix);
static bool save_solution(const double *data, int size, const char *output_dir,
                          const char *instance_name, const char *suffix);

int main(int argc, char *argv[])
{
    pdhg_parameters_t params;
    set_default_parameters(&params);

    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"no_presolve", no_argument, 0, 1001},
        {"l_inf_ruiz_iter", required_argument, 0, 1002},
        {"no_pock_chambolle", no_argument, 0, 1003},
        {"pock_chambolle_alpha", required_argument, 0, 1004},
        {"no_bound_obj_rescaling", no_argument, 0, 1005},
        {"step_size_factor", required_argument, 0, 1006},
        {"no_active_set_step_size_boost", no_argument, 0, 1007},
        {"no_primal_weight_safeguard", no_argument, 0, 1008},
        {"singular_value_max_iter", required_argument, 0, 1009},
        {"singular_value_tol", required_argument, 0, 1010},
        {"cone_newton_max_iter", required_argument, 0, 1011},
        {"cone_newton_tol", required_argument, 0, 1012},
        {"time_limit", required_argument, 0, 1013},
        {"iter_limit", required_argument, 0, 1014},
        {"eps_opt", required_argument, 0, 1015},
        {"eps_feas", required_argument, 0, 1016},
        {"eps_farkas_homogeneous", required_argument, 0, 1017},
        {"eps_farkas_displacement", required_argument, 0, 1018},
        {"eval_freq", required_argument, 0, 1019},
        {"opt_norm", required_argument, 0, 1020},
        {"verbose", no_argument, 0, 'v'},
        {"quiet", no_argument, 0, 'q'},
        {"diagnose", no_argument, 0, 'd'},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "hvqd", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'h':
            print_usage(argv[0]);
            return 0;
        case 'v':
            params.verbose = true;
            break;
        case 'q':
            params.verbose = false;
            break;
        case 'd':
            params.diagnose = true;
            break;
        case 1001:
            params.presolve = false;
            break;
        case 1002:
            if (!parse_int_arg("--l_inf_ruiz_iter", optarg,
                               &params.scaling_params.l_inf_ruiz_iterations))
                return 1;
            break;
        case 1003:
            params.scaling_params.has_pock_chambolle_alpha = false;
            break;
        case 1004:
            if (!parse_double_arg("--pock_chambolle_alpha", optarg,
                                  &params.scaling_params.pock_chambolle_alpha))
                return 1;
            break;
        case 1005:
            params.scaling_params.bound_objective_rescaling = false;
            break;
        case 1006:
            if (!parse_double_arg("--step_size_factor", optarg,
                                  &params.step_size_params.step_size_factor))
                return 1;
            break;
        case 1007:
            params.active_set_params.step_size_boost = false;
            break;
        case 1008:
            params.step_size_params.primal_weight_safeguard = false;
            break;
        case 1009:
            if (!parse_int_arg("--singular_value_max_iter", optarg,
                               &params.power_iter_params.max_iter))
                return 1;
            break;
        case 1010:
            if (!parse_double_arg("--singular_value_tol", optarg,
                                  &params.power_iter_params.tol))
                return 1;
            break;
        case 1011:
            if (!parse_int_arg("--cone_newton_max_iter", optarg,
                               &params.cone_params.max_iter))
                return 1;
            break;
        case 1012:
            if (!parse_double_arg("--cone_newton_tol", optarg,
                                  &params.cone_params.tol))
                return 1;
            break;
        case 1013:
            if (!parse_double_arg("--time_limit", optarg,
                                  &params.termination_criteria.time_sec_limit))
                return 1;
            break;
        case 1014:
            if (!parse_int_arg("--iter_limit", optarg,
                               &params.termination_criteria.iteration_limit))
                return 1;
            break;
        case 1015:
            if (!parse_double_arg("--eps_opt", optarg,
                                  &params.termination_criteria.eps_optimal_relative))
                return 1;
            break;
        case 1016:
            if (!parse_double_arg("--eps_feas", optarg,
                                  &params.termination_criteria.eps_feasible_relative))
                return 1;
            break;
        case 1017:
            if (!parse_double_arg("--eps_farkas_homogeneous", optarg,
                                  &params.termination_criteria.eps_farkas_homogeneous))
                return 1;
            break;
        case 1018:
            if (!parse_double_arg("--eps_farkas_displacement", optarg,
                                  &params.termination_criteria.eps_farkas_displacement))
                return 1;
            break;
        case 1019:
            if (!parse_int_arg("--eval_freq", optarg,
                               &params.termination_evaluation_frequency))
                return 1;
            break;
        case 1020:
        {
            const char *norm_str = optarg;
            if (strcmp(norm_str, "l2") == 0)
                params.optimality_norm = NORM_TYPE_L2;
            else if (strcmp(norm_str, "linf") == 0)
                params.optimality_norm = NORM_TYPE_L_INF;
            else
            {
                fprintf(stderr, "ERROR: '--opt_norm' must be 'l2' or 'linf'\n");
                return 1;
            }
        }
        break;
        case '?':
            return 1;
        }
    }

    if (argc - optind != 2)
    {
        fprintf(stderr, "ERROR: missing input file or output directory\n\n");
        print_usage(argv[0]);
        return 1;
    }

    const char *filename = argv[optind];
    const char *output_dir = argv[optind + 1];
    if (!is_writable_directory(output_dir))
    {
        fprintf(stderr, "ERROR: output directory '%s' does not exist or is not writable\n",
                output_dir);
        return 1;
    }

    char *instance_name = extract_instance_name(filename);
    if (instance_name == NULL)
        return 1;

    lp_problem_t *problem = read_mps_file(filename);

    if (problem == NULL)
    {
        free(instance_name);
        return 1;
    }

    hpdlp_result_t *result = solve_lp_problem(problem, &params);

    int exit_code = 0;
    if (result == NULL)
        exit_code = 1;
    else
    {
        if (!save_solver_summary(result, output_dir, instance_name))
            exit_code = 1;
        if (!save_solution(result->primal_solution, problem->num_variables, output_dir,
                           instance_name, "_primal_solution.txt"))
            exit_code = 1;
        if (!save_solution(result->dual_solution, problem->num_constraints, output_dir,
                           instance_name, "_dual_solution.txt"))
            exit_code = 1;
        if (result->termination_reason == TERMINATION_REASON_NUMERICAL_FAILURE)
            exit_code = 1;
        hpdlp_result_free(result);
    }

    lp_problem_free(problem);
    free(instance_name);

    return exit_code;
}

static void print_usage(const char *prog_name)
{
    pdhg_parameters_t defaults;
    set_default_parameters(&defaults);

    fprintf(stderr, "Usage: %s [OPTIONS] <mps_file> <output_dir>\n\n", prog_name);

    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  <mps_file>               Path to the input problem in MPS format (.mps or .mps.gz).\n");
    fprintf(stderr, "  <output_dir>             Directory where output files will be saved.\n\n");

    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help                            Display this help message.\n");
    fprintf(stderr, "\n  Preprocessing/scaling:\n");
    fprintf(stderr, "      --no_presolve                     Disable presolve.\n");
    fprintf(stderr, "      --l_inf_ruiz_iter <int>           Iterations for L-inf Ruiz rescaling (default: %d).\n",
            defaults.scaling_params.l_inf_ruiz_iterations);
    fprintf(stderr, "      --no_pock_chambolle               Disable Pock-Chambolle rescaling.\n");
    fprintf(stderr, "      --pock_chambolle_alpha <float>    Value for Pock-Chambolle alpha (default: %g).\n",
            defaults.scaling_params.pock_chambolle_alpha);
    fprintf(stderr, "      --no_bound_obj_rescaling          Disable bound objective rescaling.\n");
    fprintf(stderr, "\n  Solver algorithm:\n");
    fprintf(stderr, "      --singular_value_max_iter <int>   Max iterations for singular value estimation (default: %d).\n",
            defaults.power_iter_params.max_iter);
    fprintf(stderr, "      --singular_value_tol <float>      Tolerance for singular value estimation (default: %g).\n",
            defaults.power_iter_params.tol);
    fprintf(stderr, "      --step_size_factor <float>        Step size scaling factor in (0, 1) (default: %g).\n",
            defaults.step_size_params.step_size_factor);
    fprintf(stderr, "      --no_primal_weight_safeguard      Disable primal-weight safeguarding (default: enabled).\n");
    fprintf(stderr, "      --cone_newton_max_iter <int>      Max Newton iterations for cone projection (default: %d).\n",
            defaults.cone_params.max_iter);
    fprintf(stderr, "      --cone_newton_tol <float>         Newton tolerance for cone projection (default: %g).\n",
            defaults.cone_params.tol);
    fprintf(stderr, "      --no_active_set_step_size_boost   Disable active-set step-size boosting (default: enabled).\n");
    fprintf(stderr, "\n  Termination:\n");
    fprintf(stderr, "      --opt_norm <norm_type>            Norm for optimality criteria: l2 or linf (default: l2).\n");
    fprintf(stderr, "      --eval_freq <int>                 Termination evaluation frequency (default: %d).\n",
            defaults.termination_evaluation_frequency);
    fprintf(stderr, "      --eps_opt <tolerance>             Relative optimality tolerance (default: %g).\n",
            defaults.termination_criteria.eps_optimal_relative);
    fprintf(stderr, "      --eps_feas <tolerance>            Relative feasibility tolerance (default: %g).\n",
            defaults.termination_criteria.eps_feasible_relative);
    fprintf(stderr, "      --eps_farkas_homogeneous <tol>    Homogeneous-certificate infeasibility ratio (default: %g).\n",
            defaults.termination_criteria.eps_farkas_homogeneous);
    fprintf(stderr, "      --eps_farkas_displacement <tol>   Displacement-certificate infeasibility ratio (default: %g).\n",
            defaults.termination_criteria.eps_farkas_displacement);
    fprintf(stderr, "      --time_limit <seconds>            Time limit in seconds (default: %g).\n",
            defaults.termination_criteria.time_sec_limit);
    fprintf(stderr, "      --iter_limit <iterations>         Iteration limit (default: %d).\n",
            defaults.termination_criteria.iteration_limit);
    fprintf(stderr, "\n  Output:\n");
    fprintf(stderr, "  -v, --verbose                         Enable verbose logging (default: true).\n");
    fprintf(stderr, "  -q, --quiet                           Disable verbose logging.\n");
    fprintf(stderr, "  -d, --diagnose                        Add HSD scale, step-size, weight and norm columns.\n");
}

static bool parse_int_arg(const char *option_name, const char *value, int *out)
{
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        parsed < INT_MIN || parsed > INT_MAX)
    {
        fprintf(stderr, "ERROR: invalid integer for '%s': '%s'\n", option_name, value);
        return false;
    }

    *out = (int)parsed;
    return true;
}

static bool parse_double_arg(const char *option_name, const char *value, double *out)
{
    char *end = NULL;
    errno = 0;
    double parsed = strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0')
    {
        fprintf(stderr, "ERROR: invalid floating-point value for '%s': '%s'\n",
                option_name, value);
        return false;
    }

    *out = parsed;
    return true;
}

static bool is_writable_directory(const char *path)
{
    struct stat info;
    return stat(path, &info) == 0 && S_ISDIR(info.st_mode) && access(path, W_OK) == 0;
}

static char *extract_instance_name(const char *filename)
{
    char *filename_copy = strdup(filename);
    if (filename_copy == NULL)
    {
        fprintf(stderr, "ERROR: out of memory\n");
        return NULL;
    }

    char *base = basename(filename_copy);
    strip_suffix(base, ".gz");
    strip_suffix(base, ".mps");

    char *instance_name = strdup(base);
    free(filename_copy);
    if (instance_name == NULL)
        fprintf(stderr, "ERROR: out of memory\n");
    return instance_name;
}

static void strip_suffix(char *name, const char *suffix)
{
    size_t name_len = strlen(name);
    size_t suffix_len = strlen(suffix);
    if (name_len > suffix_len &&
        strcmp(name + name_len - suffix_len, suffix) == 0)
        name[name_len - suffix_len] = '\0';
}

static bool save_solver_summary(const hpdlp_result_t *result, const char *output_dir,
                                const char *instance_name)
{
    FILE *outfile = open_output_file(output_dir, instance_name, "_summary.txt");
    if (outfile == NULL)
        return false;

    fprintf(outfile, "Termination Reason: %s\n",
            termination_reason_to_string(result->termination_reason));
    if (result->presolve_time > 0.0)
    {
        fprintf(outfile, "Presolve Status: %s\n", get_presolve_status_str(result->presolve_status));
        fprintf(outfile, "Presolve Time (sec): %e\n", result->presolve_time);
        fprintf(outfile, "Reduced Rows: %d\n", result->num_reduced_constraints);
        fprintf(outfile, "Reduced Columns: %d\n", result->num_reduced_variables);
        fprintf(outfile, "Reduced Nonzeros: %d\n", result->num_reduced_nonzeros);
    }
    fprintf(outfile, "Precondition time (sec): %e\n", result->rescaling_time_sec);
    fprintf(outfile, "Runtime (sec): %e\n", result->cumulative_time_sec);
    fprintf(outfile, "Iterations Count: %d\n", result->total_count);
    fprintf(outfile, "Primal Objective Value: %e\n", result->primal_objective_value);
    fprintf(outfile, "Dual Objective Value: %e\n", result->dual_objective_value);
    fprintf(outfile, "Relative Primal Residual: %e\n", result->relative_primal_residual);
    fprintf(outfile, "Relative Dual Residual: %e\n", result->relative_dual_residual);
    fprintf(outfile, "Absolute Objective Gap: %e\n", result->objective_gap);
    fprintf(outfile, "Relative Objective Gap: %e\n", result->relative_objective_gap);
    fprintf(outfile, "Restarts: %d\n", result->restart_count);
    fprintf(outfile, "Rows: %d\n", result->num_constraints);
    fprintf(outfile, "Columns: %d\n", result->num_variables);
    fprintf(outfile, "Nonzeros: %d\n", result->num_nonzeros);
    return fclose(outfile) == 0;
}

static FILE *open_output_file(const char *output_dir, const char *instance_name,
                              const char *suffix)
{
    char *file_path = get_output_path(output_dir, instance_name, suffix);

    FILE *outfile = fopen(file_path, "w");
    if (outfile == NULL)
        fprintf(stderr, "ERROR: could not open '%s' for writing: %s\n",
                file_path, strerror(errno));

    free(file_path);
    return outfile;
}

static char *get_output_path(const char *output_dir, const char *instance_name,
                             const char *suffix)
{
    size_t path_len =
        strlen(output_dir) + strlen(instance_name) + strlen(suffix) + 2;
    char *full_path = safe_malloc(path_len * sizeof(char));
    snprintf(full_path, path_len, "%s/%s%s", output_dir, instance_name, suffix);
    return full_path;
}

static bool save_solution(const double *data, int size, const char *output_dir,
                          const char *instance_name, const char *suffix)
{
    if (data == NULL)
        return true;

    FILE *outfile = open_output_file(output_dir, instance_name, suffix);
    if (outfile == NULL)
        return false;

    for (int i = 0; i < size; ++i)
        fprintf(outfile, "%.10g\n", data[i]);

    return fclose(outfile) == 0;
}
