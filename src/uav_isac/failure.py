from __future__ import annotations

from enum import Enum
from typing import Iterable

import cvxpy as cp
import numpy as np

from .covariance import CovarianceSlotResult


class FailureCode(str, Enum):
    INFEASIBLE_FIXED_GEOMETRY_SENSING = "INFEASIBLE_FIXED_GEOMETRY_SENSING"
    INFEASIBLE_FIXED_GEOMETRY_COMMUNICATION = "INFEASIBLE_FIXED_GEOMETRY_COMMUNICATION"
    INFEASIBLE_ACTIVE_CONSTRAINTS = "INFEASIBLE_ACTIVE_CONSTRAINTS"
    SOLVER_INACCURATE = "SOLVER_INACCURATE"
    SOLVER_EXCEPTION = "SOLVER_EXCEPTION"
    PSD_VIOLATION = "PSD_VIOLATION"
    PCRB_RESIDUAL_VIOLATION = "PCRB_RESIDUAL_VIOLATION"
    RANK_RECOVERY_FAILED = "RANK_RECOVERY_FAILED"
    KINEMATIC_INFEASIBILITY = "KINEMATIC_INFEASIBILITY"
    # Proposition 5: joint illumination bound above P_max in a slot whose UAV state
    # is fixed by the boundary conditions -> infeasible for every trajectory.
    INFEASIBLE_CERTIFIED_BOUNDARY = "INFEASIBLE_CERTIFIED_BOUNDARY"
    MISSING_OUTPUT = "MISSING_OUTPUT"
    UNKNOWN_NUMERICAL_FAILURE = "UNKNOWN_NUMERICAL_FAILURE"


NUMERICAL_FAILURE_CODES = {
    FailureCode.SOLVER_INACCURATE,
    FailureCode.SOLVER_EXCEPTION,
    FailureCode.PSD_VIOLATION,
    FailureCode.PCRB_RESIDUAL_VIOLATION,
    FailureCode.RANK_RECOVERY_FAILED,
    FailureCode.MISSING_OUTPUT,
    FailureCode.UNKNOWN_NUMERICAL_FAILURE,
}


def classify_covariance_failure(
    results: Iterable[CovarianceSlotResult],
    *,
    worst_sensing_ratio: float | None = None,
    worst_communication_ratio: float | None = None,
    psd_tolerance: float = 1e-6,
    pcrb_residual_tolerance: float = 1e-3,
    ratio_tolerance: float = 1e-3,
) -> FailureCode:
    results = list(results)
    if worst_sensing_ratio is not None and np.isfinite(worst_sensing_ratio):
        if worst_sensing_ratio > 1.0 + ratio_tolerance:
            return FailureCode.INFEASIBLE_FIXED_GEOMETRY_SENSING
    if worst_communication_ratio is not None and np.isfinite(worst_communication_ratio):
        if worst_communication_ratio > 1.0 + ratio_tolerance:
            return FailureCode.INFEASIBLE_FIXED_GEOMETRY_COMMUNICATION
    if any(str(item.status).lower().startswith("error") for item in results):
        return FailureCode.SOLVER_EXCEPTION
    if any(
        np.isfinite(min(item.min_w_eigenvalue, item.min_s_eigenvalue))
        and min(item.min_w_eigenvalue, item.min_s_eigenvalue) < -psd_tolerance
        for item in results
    ):
        return FailureCode.PSD_VIOLATION
    if any(
        np.isfinite(item.pcrb_residual_m2)
        and item.pcrb_residual_m2 > pcrb_residual_tolerance
        for item in results
    ):
        return FailureCode.PCRB_RESIDUAL_VIOLATION
    statuses = {str(item.status).lower() for item in results}
    explicit_infeasible = {
        str(cp.INFEASIBLE).lower(), "infeasible", "unbounded",
    }
    if statuses & explicit_infeasible:
        return FailureCode.INFEASIBLE_ACTIVE_CONSTRAINTS
    if "optimal_inaccurate" in statuses or "infeasible_inaccurate" in statuses:
        return FailureCode.SOLVER_INACCURATE
    return FailureCode.UNKNOWN_NUMERICAL_FAILURE
