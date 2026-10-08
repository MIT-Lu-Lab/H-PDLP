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

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum
    {
        TERMINATION_REASON_UNSPECIFIED,
        TERMINATION_REASON_OPTIMAL,
        TERMINATION_REASON_PRIMAL_INFEASIBLE,
        TERMINATION_REASON_DUAL_INFEASIBLE,
        TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED,
        TERMINATION_REASON_TIME_LIMIT,
        TERMINATION_REASON_ITERATION_LIMIT,
        TERMINATION_REASON_NUMERICAL_FAILURE
    } termination_reason_t;

    typedef enum
    {
        NORM_TYPE_L2 = 0,
        NORM_TYPE_L_INF = 1
    } norm_type_t;

    typedef enum
    {
        OBJECTIVE_SENSE_MINIMIZE = 0,
        OBJECTIVE_SENSE_MAXIMIZE = 1
    } objective_sense_t;

    typedef struct
    {
        int num_variables;
        int num_constraints;
        double *variable_lower_bound;
        double *variable_upper_bound;
        double *objective_vector;
        double objective_constant;
        objective_sense_t objective_sense;

        int *constraint_matrix_row_pointers;
        int *constraint_matrix_col_indices;
        double *constraint_matrix_values;
        int constraint_matrix_num_nonzeros;

        double *constraint_lower_bound;
        double *constraint_upper_bound;

        double *primal_start;
        double *dual_start;
    } lp_problem_t;

    typedef struct
    {
        int geometric_mean_iterations;
        int l_inf_ruiz_iterations;
        bool has_pock_chambolle_alpha;
        double pock_chambolle_alpha;
        bool bound_objective_rescaling;
        double matrix_zero_tol;
        double infinite_bound;
    } scaling_parameters_t;

    typedef struct
    {
        int max_iter;
        double tol;
    } power_iteration_parameters_t;

    typedef struct
    {
        double step_size_factor;
        double k_p;
        double k_i;
        double k_d;
        double i_smooth;
        bool primal_weight_safeguard;
    } step_size_parameters_t;

    typedef struct
    {
        int max_iter;
        double tol;
    } cone_parameters_t;

    typedef struct
    {
        double artificial_restart_threshold;
        double sufficient_reduction_for_restart;
        double necessary_reduction_for_restart;
    } restart_parameters_t;

    typedef struct
    {
        bool step_size_boost;
        int window_iterations;
        double relative_tol;
        double safety_factor;
        double min_raise_ratio;
        double reestimate_change_ratio;
        double divergence_ceiling_ratio;
        double divergence_margin;
        int max_reverts;
    } active_set_parameters_t;

    typedef struct
    {
        double eps_optimal_relative;
        double eps_feasible_relative;
        double eps_farkas_homogeneous;
        double eps_farkas_displacement;
        double time_sec_limit;
        int iteration_limit;
    } termination_criteria_t;

    typedef struct
    {
        bool presolve;
        bool verbose;
        bool diagnose;
        norm_type_t optimality_norm;
        double reflection_coefficient;
        int termination_evaluation_frequency;

        scaling_parameters_t scaling_params;
        power_iteration_parameters_t power_iter_params;
        step_size_parameters_t step_size_params;
        cone_parameters_t cone_params;
        termination_criteria_t termination_criteria;
        restart_parameters_t restart_params;
        active_set_parameters_t active_set_params;
    } pdhg_parameters_t;

    typedef struct
    {
        int num_variables;
        int num_constraints;
        int num_nonzeros;

        int num_reduced_variables;
        int num_reduced_constraints;
        int num_reduced_nonzeros;

        double *primal_solution;
        double *dual_solution;
        double *reduced_cost;

        int total_count;
        double rescaling_time_sec;
        double cumulative_time_sec;
        double presolve_time;
        int presolve_status;

        double absolute_primal_residual;
        double relative_primal_residual;
        double absolute_dual_residual;
        double relative_dual_residual;
        double primal_objective_value;
        double dual_objective_value;
        double objective_gap;
        double relative_objective_gap;
        double farkas_margin;
        double farkas_residual;
        termination_reason_t termination_reason;

        int restart_count;
        double homogeneous_scale;
        double final_step_size_primal;
        double final_step_size_dual;
        double final_step_size_homogeneous;
        int active_set_raise_count;
        int active_set_revert_count;
        int active_set_power_iteration_count;
    } hpdlp_result_t;

    typedef enum
    {
        matrix_dense = 0,
        matrix_csr = 1,
        matrix_csc = 2,
        matrix_coo = 3
    } matrix_format_t;

    typedef struct
    {
        int m;
        int n;
        matrix_format_t fmt;

        union MatrixData
        {
            struct MatrixDense
            {
                const double *A;
            } dense;

            struct MatrixCSR
            {
                int nnz;
                const int *row_ptr;
                const int *col_ind;
                const double *vals;
            } csr;

            struct MatrixCSC
            {
                int nnz;
                const int *col_ptr;
                const int *row_ind;
                const double *vals;
            } csc;

            struct MatrixCOO
            {
                int nnz;
                const int *row_ind;
                const int *col_ind;
                const double *vals;
            } coo;
        } data;
    } matrix_desc_t;

#ifdef __cplusplus
}
#endif
