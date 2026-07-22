from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Tuple

import numpy as np

EPS = 1e-12


def _sample_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        round(float(row.get("frequency_ghz", 0.0)), 9),
        round(float(row.get("theta_inc_deg", 0.0)), 9),
        round(float(row.get("theta_scat_deg", 0.0)), 9),
    )


def _ensure_properties_len(props: List[Any], n: int = 6) -> List[str]:
    out = [str(p) for p in list(props or [])]
    if len(out) < n:
        out.extend([""] * (n - len(out)))
    return out



def scale_snapshot_panel_density(snapshot: Dict[str, Any], fine_factor: float) -> Dict[str, Any]:
    """
    Return a deep-copied geometry snapshot with increased panel density.

    This scales the per-segment N/discretization property in `properties[1]` while
    preserving all geometric/material flags.

    When ``properties[1]`` is empty or zero (auto-density mode), the solver uses
    DEFAULT_PANELS_PER_WAVELENGTH.  The fine snapshot switches to an explicit
    panels-per-wavelength value (negative N) scaled by ``fine_factor`` so the
    convergence comparison is meaningful.
    """

    try:
        from rcs_solver import DEFAULT_PANELS_PER_WAVELENGTH
    except ImportError:
        DEFAULT_PANELS_PER_WAVELENGTH = 20

    factor = float(fine_factor)
    if factor <= 1.0:
        raise ValueError("fine_factor must be > 1.0.")

    out = copy.deepcopy(snapshot)
    segments = list(out.get("segments", []) or [])
    for seg in segments:
        props = _ensure_properties_len(seg.get("properties", []), 6)
        raw = str(props[1]).strip()
        try:
            base_n = int(round(float(raw or 0)))
        except Exception:
            base_n = 0

        if base_n == 0:
            # Auto-density mode: solver uses DEFAULT_PANELS_PER_WAVELENGTH ppw.
            # Switch to explicit negative-N (panels-per-wavelength) scaled by factor.
            fine_ppw = max(2, int(math.ceil(DEFAULT_PANELS_PER_WAVELENGTH * factor)))
            props[1] = str(-fine_ppw)
        elif base_n > 0:
            # Explicit panel count: scale directly.
            fine_n = max(base_n + 1, int(math.ceil(base_n * factor)))
            props[1] = str(fine_n)
        else:
            # Already in panels-per-wavelength mode (negative N): scale the ppw value.
            base_ppw = abs(base_n)
            fine_ppw = max(base_ppw + 1, int(math.ceil(base_ppw * factor)))
            props[1] = str(-fine_ppw)

        seg["properties"] = props
        seg["seg_type"] = props[0] if props[0] else seg.get("seg_type")
    out["segments"] = segments
    return out



def evaluate_mesh_convergence(
    base_result: Dict[str, Any],
    fine_result: Dict[str, Any],
    rms_limit_db: float,
    max_abs_limit_db: float,
) -> Dict[str, Any]:
    """
    Compare two solve results point-by-point in dB.

    Matching is done on (frequency_ghz, theta_inc_deg, theta_scat_deg).
    """

    base_samples = list(base_result.get("samples", []) or [])
    fine_samples = list(fine_result.get("samples", []) or [])
    if not base_samples or not fine_samples:
        raise ValueError("Both base_result and fine_result must contain samples.")

    fine_by_key = {_sample_key(row): row for row in fine_samples}
    deltas_db: List[float] = []
    missing: List[Tuple[float, float, float]] = []

    for row in base_samples:
        key = _sample_key(row)
        other = fine_by_key.get(key)
        if other is None:
            missing.append(key)
            continue
        base_db = float(row.get("rcs_db", 10.0 * math.log10(max(float(row.get("rcs_linear", EPS)), EPS))))
        fine_db = float(other.get("rcs_db", 10.0 * math.log10(max(float(other.get("rcs_linear", EPS)), EPS))))
        if math.isfinite(base_db) and math.isfinite(fine_db):
            deltas_db.append(base_db - fine_db)

    if missing:
        raise ValueError(
            f"Mesh convergence comparison failed because {len(missing)} sample point(s) were missing in fine_result."
        )
    if not deltas_db:
        raise ValueError("Mesh convergence comparison produced no finite overlapping dB samples.")

    deltas = np.asarray(deltas_db, dtype=float)
    abs_deltas = np.abs(deltas)
    rms_db = float(np.sqrt(np.mean(deltas * deltas)))
    max_abs_db = float(np.max(abs_deltas))
    passed = (rms_db <= float(rms_limit_db)) and (max_abs_db <= float(max_abs_limit_db))

    violations: List[str] = []
    if rms_db > float(rms_limit_db):
        violations.append(f"RMS dB delta {rms_db:.6g} exceeds limit {float(rms_limit_db):.6g}")
    if max_abs_db > float(max_abs_limit_db):
        violations.append(
            f"Max |dB| delta {max_abs_db:.6g} exceeds limit {float(max_abs_limit_db):.6g}"
        )

    return {
        "passed": bool(passed),
        "sample_count": int(len(deltas)),
        "rms_db": rms_db,
        "max_abs_db": max_abs_db,
        "mean_db": float(np.mean(deltas)),
        "median_abs_db": float(np.median(abs_deltas)),
        "limits": {
            "rms_db": float(rms_limit_db),
            "max_abs_db": float(max_abs_limit_db),
        },
        "violations": violations,
        "reason": "; ".join(violations) if violations else "mesh convergence passed",
    }
