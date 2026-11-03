from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import math
import numbers


PHYSICAL_INFEASIBILITY_CODES = {
    "INFEASIBLE_FIXED_GEOMETRY_SENSING",
    "INFEASIBLE_FIXED_GEOMETRY_COMMUNICATION",
    "INFEASIBLE_ACTIVE_CONSTRAINTS",
    "KINEMATIC_INFEASIBILITY",
}

NUMERICAL_FAILURE_CODES = {
    "SOLVER_INACCURATE",
    "SOLVER_EXCEPTION",
    "PSD_VIOLATION",
    "PCRB_RESIDUAL_VIOLATION",
    "RANK_RECOVERY_FAILED",
    "MISSING_OUTPUT",
    "UNKNOWN_NUMERICAL_FAILURE",
}

PAPER_GRADE_CLASSES = {
    "FEASIBLE",
    "INFEASIBLE_ACTIVE_CONSTRAINTS",
    "OUT_OF_SAMPLE_CONSTRAINT_VIOLATION",
}


def as_bool(value: Any, default: bool = False) -> bool:
    """Parse CSV booleans without treating NaN as True.

    ``bool(float('nan'))`` evaluates to True in Python, which was the root cause of
    the CP6-v2 validator reporting 16/16 accepted for S4.  This helper is the only
    boolean parser used by CP6-v3 validation/reuse scripts.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, bool):
            return scalar
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", "", "nan", "none", "null"}:
            return False
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
    except TypeError:
        pass
    if isinstance(value, numbers.Number):
        return bool(value)
    return default


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def classify_run(row: Mapping[str, Any]) -> str:
    """Classify one CP6 run without turning physical infeasibility into a validator error.

    Sensitivity sweeps are allowed to contain physically infeasible operating points.
    Numerical failures, in contrast, are unresolved and must be re-run or diagnosed
    before the corresponding scenario is declared paper-grade.
    """
    if as_bool(row.get("accepted", False)):
        return "FEASIBLE"

    code = _text(row, "failure_code").upper()
    if code in PHYSICAL_INFEASIBILITY_CODES:
        return "INFEASIBLE_ACTIVE_CONSTRAINTS"
    if code in NUMERICAL_FAILURE_CODES:
        return "NUMERICAL_UNRESOLVED"

    reasons = {x.strip() for x in _text(row, "rejection_reasons").split(";") if x.strip()}
    numerical_gates = {"rank_ok", "psd_ok", "enforcement_ok", "power_ok", "kinematic_equalities_ok", "kinematic_inequalities_ok"}
    performance_gates = {"communication_ok", "sensing_ok"}

    if reasons:
        numeric_bad = reasons & numerical_gates
        perf_bad = reasons & performance_gates
        if "rank_ok" in numeric_bad and "sensing_ok" in perf_bad:
            return "UNRESOLVED_RANK_AND_CHANCE"
        if numeric_bad:
            return "NUMERICAL_UNRESOLVED"
        if perf_bad:
            return "OUT_OF_SAMPLE_CONSTRAINT_VIOLATION"

    status = _text(row, "status").lower()
    success = as_bool(row.get("success", False))
    if status == "failed" or not success:
        return "UNKNOWN_UNRESOLVED"
    return "UNKNOWN_UNRESOLVED"


def is_paper_grade_class(label: str) -> bool:
    return label in PAPER_GRADE_CLASSES


def requires_rerun(label: str) -> bool:
    return label in {
        "NUMERICAL_UNRESOLVED",
        "UNRESOLVED_RANK_AND_CHANCE",
        "UNKNOWN_UNRESOLVED",
    }
