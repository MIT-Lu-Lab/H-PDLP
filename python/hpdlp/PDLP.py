# Copyright 2025 Haihao Lu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Objective-sense and termination-status constants, plus parameter-name aliases."""

# Objective
MINIMIZE = 1
MAXIMIZE = -1

# Status codes (must match status_to_code in python_bindings/_core_bindings.cpp)
OPTIMAL                 = 0
PRIMAL_INFEASIBLE       = 1
DUAL_INFEASIBLE         = 2
TIME_LIMIT              = 3
ITERATION_LIMIT         = 4
INFEASIBLE_OR_UNBOUNDED = 5
NUMERICAL_FAILURE       = 7
UNSPECIFIED             = -1


# parameter name alias
_PARAM_ALIAS = {
    # general
    "Presolve": "presolve",
    "OutputFlag": "verbose",
    "LogToConsole": "verbose",
    "Diagnose": "diagnose",
    "OptimalityNorm": "optimality_norm",
    "ReflectionCoeff": "reflection_coefficient",
    "NormalizationStrength": "normalization_strength",
    "TermCheckFreq": "termination_evaluation_frequency",
    # scaling
    "GeoMeanIters": "geometric_mean_iterations",
    "RuizIters": "l_inf_ruiz_iterations",
    "UsePCAlpha": "has_pock_chambolle_alpha",
    "PCAlpha": "pock_chambolle_alpha",
    "BoundObjRescaling": "bound_objective_rescaling",
    "MatrixZeroTol": "matrix_zero_tol",
    "InfiniteBound": "infinite_bound",
    # singular value estimation (power method)
    "SVMaxIter": "sv_max_iter",
    "SVTol": "sv_tol",
    # step size and primal weight
    "StepSizeFactor": "step_size_factor",
    "RestartKp": "k_p",
    "Ki": "k_i",
    "Kd": "k_d",
    "ISmooth": "i_smooth",
    "PrimalWeightSafeguard": "primal_weight_safeguard",
    # cone projection
    "ConeNewtonMaxIter": "cone_newton_max_iter",
    "ConeNewtonTol": "cone_newton_tol",
    # termination criteria
    "OptimalityTol": "eps_optimal_relative",
    "FeasibilityTol": "eps_feasible_relative",
    "FarkasHomogeneousTol": "eps_farkas_homogeneous",
    "FarkasDisplacementTol": "eps_farkas_displacement",
    "TimeLimit": "time_sec_limit",
    "IterationLimit": "iteration_limit",
    # restarts
    "RestartArtificialThresh": "artificial_restart_threshold",
    "RestartSufficientReduction": "sufficient_reduction_for_restart",
    "RestartNecessaryReduction": "necessary_reduction_for_restart",
    # active-set step boost
    "ActiveSetBoost": "active_set_boost",
    "ASBWindowIter": "asb_window_iter",
    "ASBRelativeTol": "asb_relative_tol",
    "ASBSafetyFactor": "asb_safety_factor",
    "ASBMinRaiseRatio": "asb_min_raise_ratio",
    "ASBReestimateChangeRatio": "asb_reestimate_change_ratio",
    "ASBDivergenceCeilingRatio": "asb_divergence_ceiling_ratio",
    "ASBDivergenceMargin": "asb_divergence_margin",
    "ASBMaxReverts": "asb_max_reverts",
}
