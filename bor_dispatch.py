"""
BoR dispatch (phase 4): solve .geo geometry snapshots with the BoR-MoM solver.

The drawing's (x, y) plane is reinterpreted as the (rho, z) half-plane:
x = rho (distance from the rotation axis, must be >= 0) and y = z (the
rotation axis, drawn vertically).  A closed body of revolution is an OPEN
generatrix polyline whose two endpoints lie ON the axis, traversed from the
+z end (nose) to the -z end (tail) so the left-of-travel normal faces the
exterior (BOR_CONVENTIONS.md).  Wrong traversal or non-axis endpoints are
hard preflight errors, never silently corrected — same philosophy as the 2D
solver's orientation checks.

Supported material configurations (segment TYPE semantics shared with the
2D solver / MaterialLibrary):

  * all TYPE 2 (PEC, optionally with IBC flags incl. tapers and mat.N
    tables)                      -> CFIE (pure PEC) / IBC-EFIE (any Z_s != 0)
  * all TYPE 3, one pos_mat     -> homogeneous penetrable body (PMCHWT)
  * TYPE 3 outer + TYPE 4 core,
    matching pos_mat            -> coated PEC (multi-region PMCHWT)

Anything else (TYPE 1 sheets, TYPE 5 interior interfaces, mixed PEC +
dielectric on one generatrix) raises with a named error.

Entry points mirror rcs_solver.solve_monostatic_rcs_2d so the GUI, the
runners, and grim_io work unchanged; sigma is 3-D RCS in m^2 (dBsm).
"""

import cmath
import math
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from bor_kernels import C0
from bor_solver import (solve_bor, solve_bor_dielectric, solve_bor_coated_pec,
                        solve_bor_partial_coating, solve_bor_coated2_pec,
                        solve_bor_coated_n_pec, solve_bor_coating_patch,
                        _causal_medium)
from rcs_solver import MaterialLibrary

DEFAULT_ELEMENTS_PER_WAVELENGTH = 20
MAX_ELEMENTS_DEFAULT = 50_000


def _unit_scale_to_meters(units: str) -> float:
    value = str(units or "").strip().lower()
    if value in {"inch", "inches", "in"}:
        return 0.0254
    if value in {"meter", "meters", "m"}:
        return 1.0
    raise ValueError(f"Unsupported geometry units '{units}'. Use inches or meters.")


def _canonical_bor_polarization(label: Optional[str]) -> str:
    """BoR polarizations are true 3-D VV (theta-pol) / HH (phi-pol).  The
    2D solver's elevation-cut aliases map TM->HH and TE->VV, and the same
    mapping is kept here so .geo/GUI/grim plumbing is shared."""

    text = str(label or "").strip().upper()
    if text in {"VV", "V", "TE", "VERTICAL"} or not text:
        return "VV"
    if text in {"HH", "H", "TM", "HORIZONTAL"}:
        return "HH"
    raise ValueError(f"Unsupported polarization '{label}'. Use VV or HH.")


def _parse_flag(tok: Any, default: int = 0) -> int:
    try:
        text = str(tok).strip().lower()
        if text.startswith("mat."):
            text = text[4:]
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _parse_int(tok: Any, default: int = 0) -> int:
    try:
        text = str(tok).strip()
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


class _SegChain:
    """One segment's polyline in scaled (rho, z) coordinates."""

    def __init__(self, name: str, seg_type: int, n_prop: int, ibc_flag: int,
                 pos_mat: int, neg_mat: int, pts: np.ndarray):
        self.name = name
        self.seg_type = seg_type
        self.n_prop = n_prop
        self.ibc_flag = ibc_flag
        self.pos_mat = pos_mat
        self.neg_mat = neg_mat
        self.pts = pts                       # (N, 2) columns rho, z
        d = np.diff(pts, axis=0)
        self.prim_lengths = np.hypot(d[:, 0], d[:, 1])
        self.length = float(np.sum(self.prim_lengths))


def _chains_from_snapshot(snapshot: Dict[str, Any], scale: float) -> List[_SegChain]:
    chains: List[_SegChain] = []
    for seg_idx, seg in enumerate(snapshot.get("segments", []) or []):
        props = list(seg.get("properties", []) or [])
        seg_type = _parse_flag(
            props[0] if len(props) > 0 and str(props[0]).strip() else seg.get("seg_type", 2), 2)
        n_prop = _parse_int(props[1] if len(props) > 1 else 0, 0)
        ibc_flag = _parse_flag(props[2] if len(props) > 2 else 0)
        pos_mat = _parse_flag(props[3] if len(props) > 3 else 0)
        neg_mat = _parse_flag(props[4] if len(props) > 4 else 0)
        pts: List[Tuple[float, float]] = []
        for i, pair in enumerate(list(seg.get("point_pairs", []) or [])):
            x1 = float(pair.get("x1", 0.0)) * scale
            y1 = float(pair.get("y1", 0.0)) * scale
            x2 = float(pair.get("x2", 0.0)) * scale
            y2 = float(pair.get("y2", 0.0)) * scale
            if i == 0:
                pts.append((x1, y1))
            elif math.hypot(x1 - pts[-1][0], y1 - pts[-1][1]) > 0:
                # primitives inside one segment must chain head-to-tail
                raise ValueError(
                    f"Segment '{seg.get('name', seg_idx)}': primitives do not chain "
                    f"head-to-tail at ({x1 / scale:.6g}, {y1 / scale:.6g}).")
            pts.append((x2, y2))
        if len(pts) < 2:
            continue
        chains.append(_SegChain(str(seg.get("name", f"segment_{seg_idx + 1}")),
                                seg_type, n_prop, ibc_flag, pos_mat, neg_mat,
                                np.asarray(pts, dtype=float)))
    if not chains:
        raise ValueError("Geometry contains no usable segments.")
    return chains


def _stitch_generatrix(chains: List[_SegChain], what: str,
                       tol: float) -> List[_SegChain]:
    """Order chains head-to-tail into a single generatrix run.  Junctions
    must match END of one chain to START of the next AS DRAWN (reversing a
    segment silently would flip its normal and its taper direction)."""

    if len(chains) == 1:
        return chains

    def key(p) -> Tuple[int, int]:
        return (int(round(p[0] / tol)), int(round(p[1] / tol)))

    start_of = {}
    end_of = {}
    for c in chains:
        ks, ke = key(c.pts[0]), key(c.pts[-1])
        if ks in start_of or ke in end_of:
            raise ValueError(f"The {what} segments do not form a single chain "
                             "(two segments start or end at the same point).")
        start_of[ks] = c
        end_of[ke] = c
    heads = [c for c in chains if key(c.pts[0]) not in end_of]
    if len(heads) != 1:
        raise ValueError(
            f"The {what} segments do not chain head-to-tail into one generatrix. "
            "Check that consecutive segments share endpoints and that each "
            "segment is drawn in the same traversal direction (a start-to-start "
            "or end-to-end meeting means one segment's endpoint order must be "
            "reversed).")
    ordered = [heads[0]]
    while True:
        nxt = start_of.get(key(ordered[-1].pts[-1]))
        if nxt is None:
            break
        if nxt is ordered[0]:
            raise ValueError(f"The {what} segments form a closed loop in the "
                             "(rho, z) plane; a BoR generatrix must be an open "
                             "polyline with both endpoints on the axis.")
        ordered.append(nxt)
    if len(ordered) != len(chains):
        raise ValueError(f"The {what} segments split into multiple disconnected "
                         "chains; expected one generatrix.")
    return ordered


def _preflight_generatrix(ordered: List[_SegChain], what: str, tol: float) -> None:
    pts = np.vstack([ordered[0].pts] + [c.pts[1:] for c in ordered[1:]])
    rho, z = pts[:, 0], pts[:, 1]
    if np.any(rho < -tol):
        bad = pts[np.argmin(rho)]
        raise ValueError(
            f"The {what} generatrix crosses the rotation axis (rho = x = "
            f"{bad[0]:.6g} < 0). Draw the half-profile entirely at x >= 0.")
    if rho[0] > tol or rho[-1] > tol:
        raise ValueError(
            f"The {what} generatrix endpoints must lie ON the rotation axis "
            f"(x = 0) to close the body of revolution; got start rho = "
            f"{rho[0]:.6g}, end rho = {rho[-1]:.6g}. Open BoR shells are not "
            "supported in phase 4.")
    if z[0] <= z[-1]:
        raise ValueError(
            f"The {what} generatrix must be traversed from the +z (top) axis "
            "end to the -z (bottom) axis end so the left-of-travel normal "
            f"faces the exterior; it is drawn bottom-to-top (z {z[0]:.6g} -> "
            f"{z[-1]:.6g}). Reverse the segment endpoint order.")


def _element_count(n_prop: int, prim_len: float, lam_target: float) -> int:
    if prim_len <= 0.0:
        return 1
    if n_prop > 0:
        return max(1, n_prop)
    n_wave = abs(n_prop) if n_prop < 0 else DEFAULT_ELEMENTS_PER_WAVELENGTH
    target = max(lam_target / max(1, n_wave), prim_len / 2000.0)
    return max(1, int(math.ceil(prim_len / target)))


def _mesh_generatrix(ordered: List[_SegChain], lam_target: float,
                     max_elements: int, axis_tol: float):
    """Subdivide the ordered chains into elements.  Returns (points [Nn,2],
    elem_seg [Ne] chain index, elem_arc_s [Ne] normalized arc position of the
    element midpoint along its own segment — the taper coordinate)."""

    points: List[Tuple[float, float]] = []
    elem_seg: List[int] = []
    elem_arc: List[float] = []
    for ci, c in enumerate(ordered):
        arc0 = 0.0
        for pi in range(len(c.pts) - 1):
            p0, p1 = c.pts[pi], c.pts[pi + 1]
            plen = c.prim_lengths[pi]
            cnt = _element_count(c.n_prop, plen, lam_target)
            for i in range(cnt):
                q0 = p0 + (p1 - p0) * (i / cnt)
                if not points:
                    points.append(tuple(q0))
                elem_seg.append(ci)
                elem_arc.append((arc0 + plen * (i + 0.5) / cnt) / max(c.length, 1e-300))
                q1 = p0 + (p1 - p0) * ((i + 1) / cnt)
                points.append(tuple(q1))
            arc0 += plen
    pts = np.asarray(points, dtype=float)
    # snap near-axis coordinates exactly onto the axis (Generatrix requires rho >= 0)
    pts[np.abs(pts[:, 0]) <= axis_tol, 0] = 0.0
    pts[:, 0] = np.maximum(pts[:, 0], 0.0)
    if len(pts) - 1 > max_elements:
        raise ValueError(f"BoR mesh would need {len(pts) - 1} elements "
                         f"(> max {max_elements}). Reduce frequency or density.")
    return pts, np.asarray(elem_seg, dtype=int), np.asarray(elem_arc, dtype=float)


def _classify(chains: List[_SegChain]) -> str:
    types = {c.seg_type for c in chains}
    if types == {2}:
        return "conductor"
    if types == {3}:
        if len({c.pos_mat for c in chains}) != 1:
            raise ValueError("All TYPE 3 segments of a homogeneous BoR body "
                             "must reference the same pos_mat material.")
        return "dielectric"
    if types in ({3, 4}, {2, 3, 4}):
        pm3 = {c.pos_mat for c in chains if c.seg_type == 3}
        pm4 = {c.pos_mat for c in chains if c.seg_type == 4}
        if len(pm3) != 1 or len(pm4) != 1 or pm3 != pm4:
            raise ValueError("Coated BoR: the TYPE 3 interface and the TYPE 4 "
                             "covered core must reference the same pos_mat "
                             "coating material.")
        if 2 in types:
            return "partial"
        return "coated"
    if types == {3, 4, 5}:
        pm5 = {(c.pos_mat, c.neg_mat) for c in chains if c.seg_type == 5}
        if len(pm5) > 1:
            return "layered_n"
        outer_flag, inner_flag = next(iter(pm5))
        if outer_flag <= 0 or inner_flag <= 0 or outer_flag == inner_flag:
            raise ValueError("TYPE 5 needs distinct positive pos_mat (outer "
                             "layer) and neg_mat (inner layer) flags.")
        pm4 = {c.pos_mat for c in chains if c.seg_type == 4}
        if pm4 != {inner_flag}:
            raise ValueError("Layered BoR: the TYPE 4 core's pos_mat must be "
                             "the TYPE 5 interface's neg_mat (inner layer).")
        for c in chains:
            if c.seg_type == 3 and c.pos_mat not in (outer_flag, inner_flag):
                raise ValueError("Layered BoR: every TYPE 3 pos_mat must be "
                                 "the outer-layer or inner-layer flag.")
        return "layered"
    unsupported = types - {2, 3, 4, 5}
    if unsupported:
        raise ValueError(f"Segment TYPE(s) {sorted(unsupported)} are not "
                         "supported by the BoR solver (supported: TYPE 2 "
                         "PEC/IBC, TYPE 3 dielectric, TYPE 3+4 coated, "
                         "TYPE 2+3+4 partially coated, TYPE 3+5+4 layered).")
    raise ValueError("Unsupported BoR material combination: TYPE 2 "
                     "conductors can only mix with dielectric interfaces via "
                     "the TYPE 2+3+4 partial-coating layout.")


def _stitch_pieces(chains: List[_SegChain], what: str,
                   tol: float) -> List[List[_SegChain]]:
    """Order chains head-to-tail into MULTIPLE maximal open runs (used for
    the bare-conductor pieces of a partial coating)."""

    def key(p) -> Tuple[int, int]:
        return (int(round(p[0] / tol)), int(round(p[1] / tol)))

    start_of = {}
    end_of = {}
    for c in chains:
        ks, ke = key(c.pts[0]), key(c.pts[-1])
        if ks in start_of or ke in end_of:
            raise ValueError(f"Two {what} segments start or end at the same point.")
        start_of[ks] = c
        end_of[ke] = c
    heads = [c for c in chains if key(c.pts[0]) not in end_of]
    runs: List[List[_SegChain]] = []
    used = 0
    for head in heads:
        run = [head]
        while True:
            nxt = start_of.get(key(run[-1].pts[-1]))
            if nxt is None or nxt is head:
                break
            run.append(nxt)
        runs.append(run)
        used += len(run)
    if used != len(chains):
        raise ValueError(f"The {what} segments contain a closed loop or a "
                         "branching junction; expected open head-to-tail runs.")
    return runs


def solve_monostatic_rcs_bor(
    geometry_snapshot: Dict[str, Any],
    frequencies_ghz: List[float],
    elevations_deg: List[float],
    polarization: str,
    geometry_units: str = "inches",
    material_base_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    mesh_reference_ghz: Optional[float] = None,
    cfie_alpha: float = 0.5,
    n_modes: Optional[int] = None,
    mode_tol: float = 1e-6,
    max_elements: int = MAX_ELEMENTS_DEFAULT,
    workers: Optional[int] = None,
    abort_event: Optional[threading.Event] = None,
    table_precision: str = "auto",
    assembly: str = "auto",
    expand_to_360: bool = False,
) -> Dict[str, Any]:
    """
    Monostatic 3-D RCS (m^2 / dBsm) of an axisymmetric body described by a
    .geo geometry snapshot.  `elevations_deg` are ASPECT angles measured from
    the +z rotation axis (0 = nose-on, 90 = broadside, 180 = tail-on); the
    same argument name as the 2D entry point is kept so callers are drop-in.
    Both polarizations are solved (they share each mode's factorization);
    `polarization` selects which one the samples report.

    expand_to_360=True mirrors the samples about the axis to fill the full
    polar cut: sigma(360 - theta) = sigma(theta) — EXACT for a body of
    revolution (rotating the problem 180 deg about z maps the body, the
    directions, and the polarization basis onto themselves), including the
    complex amplitudes.  The seam directions 0/360 and 180 are not
    duplicated.  Note this is a property of the axisymmetric MODEL: it does
    not conjure the effect of any non-axisymmetric feature the BoR cannot
    represent, and it is NOT the nose<->tail flip (theta -> 180 - theta),
    which is only valid for fore-aft symmetric bodies.
    """

    if not frequencies_ghz:
        raise ValueError("At least one frequency is required.")
    if not elevations_deg:
        raise ValueError("At least one aspect angle is required.")
    frequencies = [float(f) for f in frequencies_ghz]
    if any(f <= 0.0 for f in frequencies):
        raise ValueError("Frequencies must be positive GHz values.")
    aspects = [float(a) for a in elevations_deg]
    if any(a < 0.0 or a > 180.0 for a in aspects):
        raise ValueError("Aspect angles must lie in [0, 180] degrees from +z.")
    pol = _canonical_bor_polarization(polarization)
    scale = _unit_scale_to_meters(geometry_units)
    base_dir = material_base_dir or os.getcwd()
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    materials = MaterialLibrary.from_entries(
        geometry_snapshot.get("ibcs", []) or [],
        geometry_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )

    chains = _chains_from_snapshot(geometry_snapshot, scale)
    kind = _classify(chains)
    diag = max(float(np.ptp(np.vstack([c.pts for c in chains]), axis=0).max()), 1e-9)
    tol = max(1e-12, 1e-9 * diag)
    axis_tol = 1e-6 * diag

    if kind == "coated":
        outer_chains = _stitch_generatrix([c for c in chains if c.seg_type == 3],
                                          "outer-interface (TYPE 3)", tol)
        core_chains = _stitch_generatrix([c for c in chains if c.seg_type == 4],
                                         "core (TYPE 4)", tol)
        _preflight_generatrix(outer_chains, "outer-interface", axis_tol)
        _preflight_generatrix(core_chains, "core", axis_tol)
        groups = [outer_chains, core_chains]
    elif kind == "partial":
        iface_chains = _stitch_generatrix([c for c in chains if c.seg_type == 3],
                                          "coating-interface (TYPE 3)", tol)
        cov_chains = _stitch_generatrix([c for c in chains if c.seg_type == 4],
                                        "covered-core (TYPE 4)", tol)
        bare_runs = _stitch_pieces([c for c in chains if c.seg_type == 2],
                                   "bare-conductor (TYPE 2)", tol)
        # The PEC core as a whole (covered + bare pieces, joined at the
        # coating-termination junctions) must still be a valid closed body.
        merged = _stitch_generatrix(
            cov_chains + [c for run in bare_runs for c in run],
            "PEC core (TYPE 2 + TYPE 4)", tol)
        _preflight_generatrix(merged, "PEC core", axis_tol)
        groups = [iface_chains, cov_chains, bare_runs]
    elif kind == "layered":
        outer_flag, inner_flag = next(iter(
            {(c.pos_mat, c.neg_mat) for c in chains if c.seg_type == 5}))
        mid5_chains = _stitch_generatrix([c for c in chains if c.seg_type == 5],
                                         "layer-interface (TYPE 5)", tol)
        core_chains = _stitch_generatrix([c for c in chains if c.seg_type == 4],
                                         "core (TYPE 4)", tol)
        patch_chains = _stitch_generatrix(
            [c for c in chains if c.seg_type == 3 and c.pos_mat == outer_flag],
            "outer-interface (TYPE 3, outer layer)", tol)
        bare_mid_runs = _stitch_pieces(
            [c for c in chains if c.seg_type == 3 and c.pos_mat == inner_flag],
            "exposed-inner-interface (TYPE 3, inner layer)", tol)
        _preflight_generatrix(core_chains, "core", axis_tol)
        merged_mid = _stitch_generatrix(
            mid5_chains + [c for run in bare_mid_runs for c in run],
            "inner-layer interface (TYPE 5 + TYPE 3)", tol)
        _preflight_generatrix(merged_mid, "inner-layer interface", axis_tol)
        if not bare_mid_runs:
            _preflight_generatrix(patch_chains, "outer interface", axis_tol)
        groups = [patch_chains, mid5_chains, bare_mid_runs, core_chains,
                  (outer_flag, inner_flag)]
    elif kind == "layered_n":
        # N-layer full stack: walk the TYPE 5 pos->neg flag chain from the
        # TYPE 3 outer interface's pos_mat down to the TYPE 4 core's pos_mat.
        t3 = [c for c in chains if c.seg_type == 3]
        top_flags = {c.pos_mat for c in t3}
        if len(top_flags) != 1:
            raise ValueError("N-layer stacks (multiple TYPE 5 flag pairs) "
                             "support full coverage only: all TYPE 3 chains "
                             "must reference the outermost layer flag "
                             "(patch layouts are limited to two layers).")
        f_top = next(iter(top_flags))
        pm4 = {c.pos_mat for c in chains if c.seg_type == 4}
        if len(pm4) != 1:
            raise ValueError("The TYPE 4 core segments must share one pos_mat.")
        f_bottom = next(iter(pm4))
        pair_map = {}
        for (po, ne) in {(c.pos_mat, c.neg_mat) for c in chains if c.seg_type == 5}:
            if po in pair_map:
                raise ValueError(f"Two TYPE 5 interfaces claim outer flag {po}.")
            pair_map[po] = ne
        flag_order = [f_top]
        while flag_order[-1] != f_bottom:
            nxt = pair_map.pop(flag_order[-1], None)
            if nxt is None:
                raise ValueError(
                    f"Layer-flag chain broken: no TYPE 5 interface has "
                    f"pos_mat {flag_order[-1]} (walking outer flag "
                    f"{f_top} toward core flag {f_bottom}).")
            flag_order.append(nxt)
        if pair_map:
            raise ValueError(f"TYPE 5 interfaces with flags {sorted(pair_map)} "
                             "are not part of the outer-to-core layer chain.")
        iface_groups = [_stitch_generatrix(t3, "outer-interface (TYPE 3)", tol)]
        for po, ne in zip(flag_order[:-1], flag_order[1:]):
            seg5 = [c for c in chains if c.seg_type == 5
                    and (c.pos_mat, c.neg_mat) == (po, ne)]
            iface_groups.append(_stitch_generatrix(
                seg5, f"layer-interface (TYPE 5, {po}|{ne})", tol))
        core_chains = _stitch_generatrix([c for c in chains if c.seg_type == 4],
                                         "core (TYPE 4)", tol)
        for gi, g in enumerate(iface_groups):
            _preflight_generatrix(g, f"interface {gi}", axis_tol)
        _preflight_generatrix(core_chains, "core", axis_tol)
        groups = [iface_groups, core_chains, flag_order]
    else:
        ordered = _stitch_generatrix(chains, "TYPE 2" if kind == "conductor"
                                     else "TYPE 3", tol)
        _preflight_generatrix(ordered, "body", axis_tol)
        groups = [ordered]

    def check_abort():
        if abort_event is not None and abort_event.is_set():
            raise InterruptedError("Solve cancelled by user.")

    samples: List[Dict[str, Any]] = []
    per_freq_meta: List[Dict[str, Any]] = []
    formulation_label = ""
    total_steps = len(frequencies)

    for fi, freq_ghz in enumerate(frequencies):
        check_abort()
        freq_hz = freq_ghz * 1e9
        mesh_ghz = float(mesh_reference_ghz) if mesh_reference_ghz else freq_ghz
        lam0 = C0 / (mesh_ghz * 1e9)

        def report(modes_done, m_cap):
            if progress_callback is not None:
                try:
                    progress_callback(fi, total_steps,
                                     f"{freq_ghz:g} GHz: mode {modes_done}/{m_cap}")
                except Exception:
                    pass

        if kind == "conductor":
            ordered = groups[0]
            pts, elem_seg, elem_arc = _mesh_generatrix(ordered, lam0,
                                                       max_elements, axis_tol)
            zs_elem = np.zeros(len(pts) - 1, dtype=complex)
            for ei in range(len(zs_elem)):
                c = ordered[elem_seg[ei]]
                if c.ibc_flag > 0:
                    zs_elem[ei] = materials.get_impedance(
                        c.ibc_flag, freq_ghz, arc_s=float(elem_arc[ei]))
            has_ibc = bool(np.any(np.abs(zs_elem) > 0.0))
            form = "efie" if (has_ibc or cfie_alpha <= 0.0) else "cfie"
            out = solve_bor(pts, freq_hz, aspects, formulation=form,
                            cfie_alpha=cfie_alpha,
                            zs=zs_elem if has_ibc else None,
                            n_modes=n_modes, mode_tol=mode_tol,
                            workers=workers, progress=report,
                            check_abort=check_abort,
                            table_precision=table_precision,
                            assembly=assembly)
            formulation_label = ("BoR-MoM IBC-EFIE (Leontovich)" if has_ibc
                                 else f"BoR-MoM PEC {form.upper()}")
        elif kind == "dielectric":
            eps, mu = materials.get_medium(groups[0][0].pos_mat, freq_ghz)
            m_idx, _ = _causal_medium(eps, mu)
            pts, _, _ = _mesh_generatrix(groups[0], lam0 / max(1.0, abs(m_idx)),
                                         max_elements, axis_tol)
            out = solve_bor_dielectric(pts, freq_hz, aspects, eps, mu,
                                       n_modes=n_modes, mode_tol=mode_tol,
                                       workers=workers, progress=report,
                                       check_abort=check_abort)
            formulation_label = "BoR-MoM PMCHWT (homogeneous dielectric)"
        elif kind == "coated":
            eps, mu = materials.get_medium(groups[0][0].pos_mat, freq_ghz)
            m_idx, _ = _causal_medium(eps, mu)
            lam_in = lam0 / max(1.0, abs(m_idx))
            pts_o, _, _ = _mesh_generatrix(groups[0], lam_in, max_elements, axis_tol)
            pts_c, _, _ = _mesh_generatrix(groups[1], lam_in, max_elements, axis_tol)
            out = solve_bor_coated_pec(pts_o, pts_c, freq_hz, aspects, eps, mu,
                                       n_modes=n_modes, mode_tol=mode_tol,
                                       workers=workers, progress=report,
                                       check_abort=check_abort)
            formulation_label = "BoR-MoM PMCHWT coated PEC (multi-region)"
        elif kind == "partial":
            eps, mu = materials.get_medium(groups[0][0].pos_mat, freq_ghz)
            m_idx, _ = _causal_medium(eps, mu)
            lam_in = lam0 / max(1.0, abs(m_idx))
            pts_i, _, _ = _mesh_generatrix(groups[0], lam_in, max_elements, axis_tol)
            pts_c, _, _ = _mesh_generatrix(groups[1], lam_in, max_elements, axis_tol)
            bare_pts = []
            bare_zs = []
            any_ibc = False
            for run in groups[2]:
                pts_b, elem_seg, elem_arc = _mesh_generatrix(
                    run, lam0, max_elements, axis_tol)
                zs_elem = np.zeros(len(pts_b) - 1, dtype=complex)
                for ei in range(len(zs_elem)):
                    c = run[elem_seg[ei]]
                    if c.ibc_flag > 0:
                        zs_elem[ei] = materials.get_impedance(
                            c.ibc_flag, freq_ghz, arc_s=float(elem_arc[ei]))
                bare_pts.append(pts_b)
                has = bool(np.any(np.abs(zs_elem) > 0.0))
                any_ibc |= has
                bare_zs.append(zs_elem if has else None)
            out = solve_bor_partial_coating(pts_i, pts_c, bare_pts, freq_hz,
                                            aspects, eps, mu, bare_zs=bare_zs,
                                            n_modes=n_modes,
                                            mode_tol=mode_tol, workers=workers,
                                            progress=report,
                                            check_abort=check_abort)
            for w in out.get("warnings", []):
                materials.warn_once(w)
            formulation_label = ("BoR-MoM PMCHWT partial coating "
                                 f"({out['n_junctions']} junction(s)"
                                 f"{', IBC bare' if any_ibc else ''})")
        elif kind == "layered":
            outer_flag, inner_flag = groups[4]
            eps_o, mu_o = materials.get_medium(outer_flag, freq_ghz)
            eps_i, mu_i = materials.get_medium(inner_flag, freq_ghz)
            m_o, _ = _causal_medium(eps_o, mu_o)
            m_i, _ = _causal_medium(eps_i, mu_i)
            lam_in = lam0 / max(1.0, abs(m_o), abs(m_i))
            pts_p, _, _ = _mesh_generatrix(groups[0], lam_in, max_elements, axis_tol)
            pts_m, _, _ = _mesh_generatrix(groups[1], lam_in, max_elements, axis_tol)
            pts_c, _, _ = _mesh_generatrix(groups[3], lam_in, max_elements, axis_tol)
            bare_pts = [_mesh_generatrix(run, lam_in, max_elements, axis_tol)[0]
                        for run in groups[2]]
            if bare_pts:
                out = solve_bor_coating_patch(pts_p, pts_m, bare_pts, pts_c,
                                              freq_hz, aspects, eps_i, mu_i,
                                              eps_o, mu_o, n_modes=n_modes,
                                              mode_tol=mode_tol, workers=workers,
                                              progress=report,
                                              check_abort=check_abort)
                formulation_label = ("BoR-MoM PMCHWT coating patch "
                                     f"({out['n_junctions']} junction(s))")
            else:
                out = solve_bor_coated2_pec(pts_p, pts_m, pts_c, freq_hz,
                                            aspects, eps_i, mu_i, eps_o, mu_o,
                                            n_modes=n_modes, mode_tol=mode_tol,
                                            workers=workers, progress=report,
                                            check_abort=check_abort)
                formulation_label = "BoR-MoM PMCHWT two-layer coated PEC"
        elif kind == "layered_n":
            iface_groups, core_chains, flag_order = groups
            media = [materials.get_medium(fl, freq_ghz) for fl in flag_order]
            idxs = [abs(_causal_medium(e, u)[0]) for (e, u) in media]
            lam_in = lam0 / max([1.0] + idxs)
            iface_pts = [_mesh_generatrix(g, lam_in, max_elements, axis_tol)[0]
                         for g in iface_groups]
            pts_c, _, _ = _mesh_generatrix(core_chains, lam_in, max_elements,
                                           axis_tol)
            # solver wants eps/mu INNERMOST first; flag_order is outer->inner
            eps_list = [media[i][0] for i in range(len(media) - 1, -1, -1)]
            mu_list = [media[i][1] for i in range(len(media) - 1, -1, -1)]
            out = solve_bor_coated_n_pec(iface_pts, pts_c, freq_hz, aspects,
                                         eps_list, mu_list, n_modes=n_modes,
                                         mode_tol=mode_tol, workers=workers,
                                         progress=report,
                                         check_abort=check_abort)
            formulation_label = (f"BoR-MoM PMCHWT {len(iface_pts)}-layer "
                                 "coated PEC")

        for w in out.get("warnings", []) or []:
            materials.warn_once(str(w))
        sig = out["sigma_vv"] if pol == "VV" else out["sigma_hh"]
        amp = out["amp_vv"] if pol == "VV" else out["amp_hh"]
        residual = float(out.get("linear_residual", 0.0))
        for ai, aspect in enumerate(aspects):
            lin = max(float(sig[ai]), 1e-30)
            a_val = complex(amp[ai])
            samples.append({
                "frequency_ghz": float(freq_ghz),
                "theta_inc_deg": float(aspect),
                "theta_scat_deg": float(aspect),
                "rcs_linear": lin,
                "rcs_db": 10.0 * math.log10(lin),
                "rcs_amp_real": float(a_val.real),
                "rcs_amp_imag": float(a_val.imag),
                "rcs_amp_phase_deg": float(math.degrees(cmath.phase(a_val))),
                "linear_residual": residual,
            })
        per_freq_meta.append({
            "frequency_ghz": float(freq_ghz),
            "modes_used": int(out["modes_used"]),
            "n_unknowns": int(out["n_unknowns"]),
            "linear_residual": residual,
            "max_cond": float(out["max_cond"]) if "max_cond" in out else None,
        })
        if progress_callback is not None:
            try:
                progress_callback(fi + 1, total_steps, f"Solved {freq_ghz:g} GHz")
            except Exception:
                pass

    if expand_to_360:
        mirrored = []
        for s in samples:
            th = float(s["theta_inc_deg"])
            if 0.0 < th < 180.0:            # 0/360 and 180 are seam directions
                m = dict(s)
                m["theta_inc_deg"] = m["theta_scat_deg"] = 360.0 - th
                mirrored.append(m)
        samples = sorted(samples + mirrored,
                         key=lambda s: (s["frequency_ghz"], s["theta_inc_deg"]))

    return {
        "solver": "bor_mom_rcs",
        "scattering_mode": "monostatic",
        "polarization": pol,
        "polarization_export": pol,
        "rcs_log_unit": "dBsm",
        "rcs_linear_quantity": "sigma_3d",
        "samples": samples,
        "metadata": {
            "formulation": formulation_label,
            "geometry_kind": kind,
            "expanded_to_360": bool(expand_to_360),
            "per_frequency": per_freq_meta,
            "cfie_alpha": float(cfie_alpha),
            "workers": int(workers),
            "warnings": list(materials.warnings),
        },
    }


def bor_az_el_grid(res_vv: Dict[str, Any], res_hh: Dict[str, Any],
                   azimuths_deg, elevations_deg,
                   axis_az_deg: float = 0.0,
                   axis_el_deg: float = 0.0) -> Dict[str, Any]:
    """
    Radar-frame monostatic polarimetric grid over (azimuth, elevation) from a
    BoR aspect sweep.

    By axisymmetry the monostatic response depends only on the aspect angle
    theta between the look direction and the rotation axis, and every look is
    a principal-plane look in its own meridian plane, where cross-pol is
    identically zero.  This routine therefore needs only the two co-pol
    COMPLEX amplitudes: for each (az, el) it maps to theta, interpolates
    F_VV(theta), F_HH(theta) linearly in the complex amplitude, and rotates
    the diagonal scattering matrix from the meridian basis into the radar's
    earth-vertical V/H basis — producing radar-frame VV, HH, and the
    cross-pol VH that appears off the waterline:

        S_radar = M^T diag(F_VV, F_HH) M,   M[i,j] = (target basis_i . radar basis_j)

    res_vv / res_hh: solve_monostatic_rcs_bor results for the two
    polarizations of the SAME geometry, frequencies, and aspect grid (both
    solves share each mode's factorization internally, so the second call is
    cheap relative to a new geometry).  The aspect grid must be dense enough
    that linear interpolation of the complex amplitudes resolves the pattern
    lobes.  axis_az/el_deg orient the rotation axis in the earth frame
    (default: horizontal, pointing along azimuth 0).  MONOSTATIC only —
    bistatic out-of-plane cuts are outside the BoR principal-plane result.

    Returns a dict with axes, per-channel complex amplitude and sigma arrays
    of shape [n_az, n_el, n_freq], plus theta/tilt maps.  Export with
    grim_io.save_bor_az_el_grim.

    LABEL CAUTION: the output VV/HH are EARTH-frame (radar) polarizations.
    For a HORIZONTAL axis the waterline meridian plane is horizontal, so on
    the el=0 cut radar-VV equals the solver's HH sweep and vice versa (the
    solver's V is defined by the meridian plane, not by earth vertical).
    The labels coincide for all looks only when the axis is vertical
    (axis_el_deg = +-90), where the tilt is zero everywhere.
    """

    if _canonical_bor_polarization(res_vv.get("polarization")) != "VV" or \
            _canonical_bor_polarization(res_hh.get("polarization")) != "HH":
        raise ValueError("Pass the VV result first and the HH result second.")
    az = np.atleast_1d(np.asarray(azimuths_deg, dtype=float))
    el = np.atleast_1d(np.asarray(elevations_deg, dtype=float))

    def by_freq(res):
        out: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        for s in res["samples"]:
            out.setdefault(float(s["frequency_ghz"]), []).append(
                (float(s["theta_inc_deg"]),
                 complex(s["rcs_amp_real"], s["rcs_amp_imag"])))
        return {f: (np.asarray([t for t, _ in sorted(v)]),
                    np.asarray([a for _, a in sorted(v)]))
                for f, v in out.items()}

    fv, fh = by_freq(res_vv), by_freq(res_hh)
    freqs = sorted(fv)
    if sorted(fh) != freqs:
        raise ValueError("VV and HH results cover different frequencies.")

    def unit(v):
        return v / np.linalg.norm(v)

    def direction(a_deg, e_deg):
        a, e = math.radians(a_deg), math.radians(e_deg)
        return np.array([math.cos(e) * math.cos(a),
                         math.cos(e) * math.sin(a), math.sin(e)])

    ax = direction(axis_az_deg, axis_el_deg)
    zhat = np.array([0.0, 0.0, 1.0])
    tol = 1e-9

    theta_map = np.zeros((len(az), len(el)))
    m_maps = np.zeros((len(az), len(el), 2, 2))
    for i, a_deg in enumerate(az):
        for j, e_deg in enumerate(el):
            d = direction(a_deg, e_deg)
            ct = float(np.clip(np.dot(d, ax), -1.0, 1.0))
            theta_map[i, j] = math.degrees(math.acos(ct))
            c = np.cross(ax, d)
            r = np.cross(zhat, d)
            if np.linalg.norm(c) < tol or np.linalg.norm(r) < tol:
                # axis-aligned look (F_VV = F_HH there by BoR symmetry) or
                # vertical look (radar H undefined): basis choice is moot
                m_maps[i, j] = np.eye(2)
                continue
            h_t = unit(c)
            v_t = unit(np.cross(h_t, d))
            h_r = unit(r)
            v_r = unit(np.cross(h_r, d))
            m_maps[i, j] = [[np.dot(v_t, v_r), np.dot(v_t, h_r)],
                            [np.dot(h_t, v_r), np.dot(h_t, h_r)]]

    shape = (len(az), len(el), len(freqs))
    amp = {ch: np.zeros(shape, dtype=complex) for ch in ("VV", "HH", "VH")}
    for kf, f in enumerate(freqs):
        th_v, av = fv[f]
        th_h, ah = fh[f]
        thq = theta_map.ravel()
        Fv = (np.interp(thq, th_v, av.real) + 1j * np.interp(thq, th_v, av.imag))
        Fh = (np.interp(thq, th_h, ah.real) + 1j * np.interp(thq, th_h, ah.imag))
        M = m_maps.reshape(-1, 2, 2)
        amp["VV"][..., kf] = (M[:, 0, 0] ** 2 * Fv + M[:, 1, 0] ** 2 * Fh
                              ).reshape(len(az), len(el))
        amp["HH"][..., kf] = (M[:, 0, 1] ** 2 * Fv + M[:, 1, 1] ** 2 * Fh
                              ).reshape(len(az), len(el))
        amp["VH"][..., kf] = (M[:, 0, 0] * M[:, 0, 1] * Fv +
                              M[:, 1, 0] * M[:, 1, 1] * Fh
                              ).reshape(len(az), len(el))

    return {
        "azimuths_deg": az.tolist(),
        "elevations_deg": el.tolist(),
        "frequencies_ghz": freqs,
        "axis_az_deg": float(axis_az_deg),
        "axis_el_deg": float(axis_el_deg),
        "theta_map_deg": theta_map,
        "amp": amp,
        "sigma": {ch: 4.0 * math.pi * np.abs(a) ** 2 for ch, a in amp.items()},
        "rcs_log_unit": "dBsm",
        "rcs_linear_quantity": "sigma_3d",
    }


def solve_adaptive_frequency_sweep_bor(
    geometry_snapshot: Dict[str, Any],
    freq_start_ghz: float,
    freq_stop_ghz: float,
    elevations_deg: List[float],
    polarization: str,
    initial_points: int = 11,
    max_refinements: int = 3,
    db_threshold: float = 1.0,
    max_total_points: int = 201,
    **kwargs,
) -> Dict[str, Any]:
    """Adaptive broadband sweep for the BoR solver: uniform seed frequencies,
    then midpoint insertion wherever adjacent samples differ by more than
    db_threshold dB (mirrors rcs_solver.solve_adaptive_frequency_sweep;
    midpoints are rounded exactly like the sample keys so a frequency is
    never re-solved)."""

    if freq_start_ghz <= 0 or freq_stop_ghz <= 0:
        raise ValueError("Frequencies must be positive.")
    if freq_start_ghz >= freq_stop_ghz:
        raise ValueError("freq_start_ghz must be less than freq_stop_ghz.")
    initial_points = max(3, int(initial_points))

    freqs = sorted({round(float(f), 12)
                    for f in np.linspace(freq_start_ghz, freq_stop_ghz, initial_points)})
    all_samples: List[Dict[str, Any]] = []
    freq_to_samples: Dict[float, List[Dict[str, Any]]] = {}
    abort_event = kwargs.get("abort_event")

    def run_freqs(freq_list: List[float]) -> None:
        if not freq_list:
            return
        result = solve_monostatic_rcs_bor(
            geometry_snapshot=geometry_snapshot,
            frequencies_ghz=freq_list,
            elevations_deg=elevations_deg,
            polarization=polarization,
            **kwargs,
        )
        for s in result.get("samples", []):
            f = round(float(s["frequency_ghz"]), 12)
            freq_to_samples.setdefault(f, []).append(s)
            all_samples.append(s)

    run_freqs(freqs)
    refinement_count = 0
    for _ in range(max_refinements):
        if abort_event is not None and abort_event.is_set():
            break
        if len(freqs) >= max_total_points:
            break
        new_freqs: set = set()
        sorted_freqs = sorted(freqs)
        for aspect in elevations_deg:
            db_at = {}
            for f in sorted_freqs:
                for s in freq_to_samples.get(round(f, 12), []):
                    if abs(s["theta_inc_deg"] - aspect) < 0.01:
                        db_at[f] = s["rcs_db"]
                        break
            for i in range(len(sorted_freqs) - 1):
                f0, f1 = sorted_freqs[i], sorted_freqs[i + 1]
                if f0 in db_at and f1 in db_at and abs(db_at[f1] - db_at[f0]) > db_threshold:
                    mid = round(0.5 * (f0 + f1), 12)
                    if mid not in freq_to_samples and mid != f0 and mid != f1:
                        new_freqs.add(mid)
        if not new_freqs:
            break
        remaining = max_total_points - len(freqs)
        if remaining <= 0:
            break
        new_list = sorted(new_freqs)[:remaining]
        run_freqs(new_list)
        freqs = sorted(set(freqs) | set(new_list))
        refinement_count += 1

    return {
        "solver": "bor_mom_rcs",
        "scattering_mode": "monostatic_adaptive",
        "polarization": _canonical_bor_polarization(polarization),
        "polarization_export": _canonical_bor_polarization(polarization),
        "rcs_log_unit": "dBsm",
        "rcs_linear_quantity": "sigma_3d",
        "samples": sorted(all_samples, key=lambda s: (s["frequency_ghz"], s["theta_inc_deg"])),
        "metadata": {
            "formulation": "BoR adaptive frequency sweep",
            "initial_points": initial_points,
            "final_point_count": len(freqs),
            "refinement_count": refinement_count,
            "db_threshold": db_threshold,
            "freq_start_ghz": freq_start_ghz,
            "freq_stop_ghz": freq_stop_ghz,
        },
    }
