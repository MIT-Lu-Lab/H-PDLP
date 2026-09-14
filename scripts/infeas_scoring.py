"""Classify a solver's answer on an infeasible or unbounded benchmark instance."""
from infeas_datasets import is_unbounded

# Certificates for each half of the pair; _INACCURATE is a certificate SCS/ABIP hold at a limit without full tolerance.
PRIMAL_CERT = {"PRIMAL_INFEASIBLE", "kInfeasible", "PRIMAL_INFEASIBLE_INACCURATE"}
DUAL_CERT = {"DUAL_INFEASIBLE", "kUnbounded", "kUnboundedOrInfeasible", "DUAL_INFEASIBLE_INACCURATE"}   # HiGHS sets kUnboundedOrInfeasible on proven dual infeasibility
# Infeasible-or-unbounded without saying which, as Gurobi barrier reports it.
EITHER_CERT = {"INFEASIBLE_OR_UNBOUNDED"}
OPTIMAL_CLAIM = {"OPTIMAL", "kOptimal"}
# Infeasible on both the primal and the dual side, so either certificate is correct.
BOTH_INFEASIBLE = {"cplex1", "cplex1_if", "cplex1_ub"}


def infeasible_outcome(status, instance):
    """Classify an answer as detected, false_optimal, opposite or no_answer; <name>_ub is unbounded, others infeasible."""
    if status in OPTIMAL_CLAIM:
        return "false_optimal"
    if instance in BOTH_INFEASIBLE:
        correct = status in PRIMAL_CERT | DUAL_CERT | EITHER_CERT
    elif is_unbounded(instance):
        correct = status in DUAL_CERT | EITHER_CERT
    else:
        correct = status in PRIMAL_CERT | EITHER_CERT
    if correct:
        return "detected"
    if status in PRIMAL_CERT | DUAL_CERT:
        return "opposite"
    return "no_answer"
