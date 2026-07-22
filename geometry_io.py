import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# IBCS row shape constants — see TAPERED_IBC.md for the full spec.
IBC_KINDS = ("constant", "linear", "cosine", "exp")
# Flags > this are interpreted as tabulated via mat.<flag>
TABULATED_FLAG_THRESHOLD = 50


def is_ibc_inline_row(row: List[str]) -> bool:
    """True if the row is in the 6-token inline form (flag kind R_s X_s R_e X_e)."""
    return (
        len(row) == 6
        and str(row[1]).strip().lower() in IBC_KINDS
    )


def is_tabulated_row(row: List[str]) -> bool:
    """True if the row references a tabulated impedance (flag > threshold)."""
    if not row:
        return False
    try:
        return int(row[0]) > TABULATED_FLAG_THRESHOLD
    except (ValueError, TypeError):
        return False


def _validate_ibc_row(tokens: List[str], lineno_for_err: str) -> None:
    """Raise ValueError if the row is not a supported IBC shape.

    Supported shapes (the `flag R X` and `flag taper kind ...` forms are not accepted):
      * Tabulated: a single token `flag` with flag > TABULATED_FLAG_THRESHOLD.
      * Inline: `flag kind R_start X_start R_end X_end` with kind in IBC_KINDS.
        For ``kind == "constant"`` only R_start/X_start matter; the end values
        are placeholders (write 0) and are ignored on read.
    """
    if not tokens:
        return
    try:
        flag = int(tokens[0])
    except (ValueError, TypeError):
        raise ValueError(f"IBC row must start with an integer flag: {lineno_for_err}")
    if flag > TABULATED_FLAG_THRESHOLD:
        if len(tokens) != 1:
            raise ValueError(
                f"Tabulated IBC row (flag>{TABULATED_FLAG_THRESHOLD}) must be a single flag token; "
                f"got {len(tokens)}: {lineno_for_err}"
            )
        return
    if len(tokens) != 6:
        raise ValueError(
            f"Inline IBC row must have 6 tokens (flag kind R_start X_start R_end X_end); "
            f"got {len(tokens)}: {lineno_for_err}"
        )
    kind = tokens[1].strip().lower()
    if kind not in IBC_KINDS:
        raise ValueError(
            f"IBC kind must be one of {IBC_KINDS}; got {tokens[1]!r}: {lineno_for_err}"
        )


@dataclass
class Segment:
    name: str
    seg_type: Optional[str]
    properties: List[str]
    x: List[float]
    y: List[float]


def parse_geometry(text: str) -> Tuple[str, List[Segment], List[List[str]], List[List[str]]]:
    lines = [ln.strip() for ln in text.splitlines()]
    title = "Geometry"
    segments: List[Segment] = []
    ibcs_entries: List[List[str]] = []
    dielectric_entries: List[List[str]] = []

    state = "segments"
    current_name: Optional[str] = None
    current_type: Optional[str] = None
    current_props: List[str] = []
    cur_x: List[float] = []
    cur_y: List[float] = []

    def flush_segment() -> None:
        if current_name is not None:
            segments.append(
                Segment(
                    name=current_name,
                    seg_type=current_type,
                    properties=current_props[:],
                    x=cur_x[:],
                    y=cur_y[:],
                )
            )

    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        low = ln.lower()
        if low.startswith("title"):
            if ":" not in ln:
                raise ValueError(f"Title line must contain ':': {ln}")
            title = ln.split(":", 1)[1].strip() or title
            continue
        if low.startswith("ibcs_resistances:") or low.startswith("ibcs:"):
            if state == "segments":
                flush_segment()
            state = "ibcs"
            continue
        if low.startswith("dielectrics:"):
            if state == "segments":
                flush_segment()
            state = "dielectrics"
            continue

        if state == "segments":
            if low.startswith("segment:"):
                flush_segment()
                parts = ln.split(":", 1)[1].strip().split()
                if not parts:
                    current_name, current_type = "Unnamed", None
                elif len(parts) == 1:
                    current_name, current_type = parts[0], None
                else:
                    current_name, current_type = parts[0], parts[1]
                current_props = []
                cur_x.clear()
                cur_y.clear()
                continue
            if low.startswith("properties:"):
                current_props = ln.split(":", 1)[1].strip().split()
                if len(current_props) != 5:
                    raise ValueError(
                        f"properties line must have exactly 5 fields "
                        f"(type n ibc pos_mat neg_mat); got {len(current_props)}: {ln}"
                    )
                continue

            tokens = ln.split()
            if len(tokens) != 4:
                raise ValueError(f"Geometry line must have 4 numbers, got {len(tokens)} {ln}")
            if current_name is None:
                # Without this, coordinates before the first "Segment:" header
                # were accumulated and then silently dropped by flush_segment.
                raise ValueError(
                    f"Geometry data line appears before the first 'Segment:' header: {ln}"
                )
            try:
                x1, y1, x2, y2 = map(float, tokens)
            except ValueError:
                raise ValueError(f"Geometry line must contain valid numbers: {ln}")
            cur_x.extend([x1, x2])
            cur_y.extend([y1, y2])
        elif state == "ibcs":
            tokens = ln.split()
            if tokens:
                _validate_ibc_row(tokens, ln)
                ibcs_entries.append(list(tokens))
        elif state == "dielectrics":
            tokens = ln.split()
            if tokens:
                dielectric_entries.append(tokens)

    if state == "segments":
        flush_segment()

    return title, segments, ibcs_entries, dielectric_entries


def build_geometry_text(
    title: str,
    segments: List[Segment],
    ibcs_entries: List[List[str]],
    dielectric_entries: List[List[str]],
) -> str:
    lines: List[str] = [f"Title: {title}"]
    for seg in segments:
        props = list(seg.properties)
        # Prefer the segment's declared type; fall back to properties[0] only if
        # seg_type is missing. This keeps load → save idempotent when the Segment:
        # header and properties[0] disagree.
        effective_type = seg.seg_type
        if not effective_type and props and str(props[0]).strip():
            effective_type = props[0]
        if effective_type:
            lines.append(f"Segment: {seg.name} {effective_type}")
        else:
            lines.append(f"Segment: {seg.name}")

        if len(props) < 5:
            props.extend([""] * (5 - len(props)))
        elif len(props) > 5:
            props = props[:5]
        # Blank fields must be written as concrete tokens: space-joined blanks
        # vanish on split(), producing a file parse_geometry rejects ("must
        # have exactly 5 fields").  Blank TYPE falls back to the effective
        # segment type; blank N means auto density (0); blank flags are 0.
        type_token = str(props[0]).strip() if props[0] is not None and str(props[0]).strip() else str(effective_type or "2")
        out_props = [type_token]
        for p in props[1:]:
            token = str(p).strip() if p is not None else ""
            out_props.append(token if token else "0")
        lines.append("properties: " + " ".join(out_props))

        if len(seg.x) != len(seg.y) or len(seg.x) % 2 != 0:
            raise ValueError(
                f"Segment {seg.name} has mismatched or odd number of coordinates."
            )
        for i in range(0, len(seg.x), 2):
            x1, y1, x2, y2 = seg.x[i], seg.y[i], seg.x[i + 1], seg.y[i + 1]
            # repr() of a Python float is the shortest string that round-trips
            # exactly: clean values stay clean ("0.05"), everything else keeps
            # full precision.  The old "%.4f" silently truncated coordinates
            # to 4 decimals (0.1 mm resolution in meters mode), corrupting
            # fine features and breaking closed contours on reload.
            lines.append(f"{float(x1)!r} {float(y1)!r} {float(x2)!r} {float(y2)!r}")

    lines.append("IBCS_Resistances:")
    for row in ibcs_entries:
        lines.append(" ".join(row))
    lines.append("Dielectrics:")
    for row in dielectric_entries:
        lines.append(" ".join(row))
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Orientation / material-side consistency checks (shared by the GUI validator
# and the solver preflight).
#
# Drawing convention (see rcs_solver._apply_user_convention_flip):
#   TYPE 1 sheet:        both sides air; orientation irrelevant.
#   TYPE 2 (air|PEC):    normal (left of travel) points INTO AIR.
#   TYPE 3 (air|diel):   normal points INTO AIR; pos_mat behind.
#   TYPE 4 (diel|PEC):   normal points INTO the pos_mat dielectric; PEC behind.
#   TYPE 5 (diel|diel):  normal points INTO pos_mat; neg_mat behind.
#
# The TM formulations are winding-insensitive, but the TE rows carry a ±1/2
# mass jump tied to the normal direction, so a wrong winding or an
# inconsistent air side silently corrupts TE results.  These checks make
# that a loud, named error instead.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChainSpec:
    """Neutral view of one segment's primitive chain for orientation checks."""

    name: str
    seg_type: int
    pos_mat: int
    points: List[Tuple[float, float]] = field(default_factory=list)


def _chain_is_closed(points: List[Tuple[float, float]], tol: float) -> bool:
    if len(points) < 4:
        return False
    return math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= tol


def _chain_area2(points: List[Tuple[float, float]]) -> float:
    area2 = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        area2 += x0 * y1 - x1 * y0
    return area2


def _point_in_polygon(px: float, py: float, poly: List[Tuple[float, float]]) -> bool:
    """Even-odd ray casting; poly is a closed vertex chain (first ~= last)."""

    inside = False
    for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
        if (y1 > py) != (y2 > py):
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_cross:
                inside = not inside
    return inside


def _rep_point(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """A point strictly on the chain: midpoint of the first primitive."""

    return (0.5 * (points[0][0] + points[1][0]), 0.5 * (points[0][1] + points[1][1]))


def _geometry_tolerance(chains: List["ChainSpec"]) -> float:
    xs = [p[0] for c in chains for p in c.points]
    ys = [p[1] for c in chains for p in c.points]
    if not xs:
        return 1e-9
    diag = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)
    return max(1e-12, 1e-9 * diag)


def check_orientation_consistency(
    chains: List[ChainSpec],
    tol: Optional[float] = None,
) -> List[Tuple[str, int, str]]:
    """
    Validate winding and air-side consistency of a set of segment chains.

    Returns findings as (severity, chain_index, message) with severity in
    {"ERROR", "INFO"}.  ERROR findings mean the geometry, solved as drawn,
    would produce silently wrong TE results:

    1. A closed TYPE 2/3 chain whose winding puts the air side on the wrong
       side for its nesting depth (a top-level body must be CW = normals out
       into unbounded air; a void nested inside a body must be CCW = normals
       into the enclosed air; parity alternates with depth).
    2. A closed TYPE 4 chain enclosed by a matching-pos_mat TYPE 3/5 contour
       (standard coated-body layout) wound so its normal points away from
       the coating.
    3. Two open TYPE 2/3 chains meeting end-to-end or start-to-start at a
       degree-2 junction: their air sides disagree there (e.g. an
       air|dielectric wall chained to an air|PEC wall the wrong way round).
    4. A closed loop stitched head-to-tail from open TYPE 2/3 chains whose
       overall winding is inside-out for its nesting depth (consistent with
       each other, but all reversed together).

    TYPE 5 chains and TYPE 1 sheets are never flagged: for TYPE 5 the
    winding IS the user's pos/neg labeling choice, and TYPE 1 is symmetric.
    """

    findings: List[Tuple[str, int, str]] = []
    if not chains:
        return findings
    if tol is None:
        tol = _geometry_tolerance(chains)

    closed_polys: List[Tuple[int, List[Tuple[float, float]]]] = []  # (chain idx, poly)
    open_air_chains: List[int] = []
    for idx, ch in enumerate(chains):
        if len(ch.points) < 2:
            continue
        if _chain_is_closed(ch.points, tol):
            if abs(_chain_area2(ch.points)) > 0.0:
                closed_polys.append((idx, ch.points))
        elif ch.seg_type in (2, 3):
            open_air_chains.append(idx)

    def _depth(rep: Tuple[float, float], exclude_idx: int) -> int:
        return sum(
            1 for cidx, poly in closed_polys
            if cidx != exclude_idx and _point_in_polygon(rep[0], rep[1], poly)
        )

    # ── 1 & 2: closed-chain winding ──────────────────────────────────────
    for idx, poly in closed_polys:
        ch = chains[idx]
        drawn_ccw = _chain_area2(poly) > 0.0
        if ch.seg_type in (2, 3):
            depth = _depth(_rep_point(poly), idx)
            expected_ccw = (depth % 2) == 1
            if drawn_ccw != expected_ccw:
                where = (
                    "the air side is the unbounded exterior" if depth % 2 == 0
                    else "the air side is the enclosed interior (nested void)"
                )
                findings.append((
                    "ERROR", idx,
                    f"Segment '{ch.name}' (TYPE {ch.seg_type}) is a closed contour drawn "
                    f"{'CCW' if drawn_ccw else 'CW'}, but {where}, so the drawing convention "
                    f"(normal into air) requires {'CCW' if expected_ccw else 'CW'} winding. "
                    "Reverse the segment's endpoint order.",
                ))
        elif ch.seg_type == 4:
            rep = _rep_point(poly)
            has_matching_parent = any(
                cidx != idx
                and chains[cidx].seg_type in (3, 5)
                and chains[cidx].pos_mat == ch.pos_mat
                and _point_in_polygon(rep[0], rep[1], p)
                for cidx, p in closed_polys
            )
            if has_matching_parent and drawn_ccw:
                findings.append((
                    "ERROR", idx,
                    f"Segment '{ch.name}' (TYPE 4) is a closed contour drawn CCW, but the "
                    f"pos_mat={ch.pos_mat} coating lies outside it, so the drawing convention "
                    "(normal into the dielectric) requires CW winding. "
                    "Reverse the segment's endpoint order.",
                ))

    # ── 3: air-side continuity at degree-2 junctions of open T2/T3 chains ─
    def _key(p: Tuple[float, float]) -> Tuple[int, int]:
        return (int(round(p[0] / tol)), int(round(p[1] / tol)))

    # endpoint registry: key -> list of (chain idx, 'start'|'end')
    ends: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
    for idx in open_air_chains:
        pts = chains[idx].points
        ends.setdefault(_key(pts[0]), []).append((idx, "start"))
        ends.setdefault(_key(pts[-1]), []).append((idx, "end"))

    adjacency: Dict[int, List[Tuple[str, int, str]]] = {i: [] for i in open_air_chains}
    for key, members in ends.items():
        if len(members) != 2:
            continue  # free end, or junction of degree > 2 — no pairing implied
        (ia, ea), (ib, eb) = members
        if ia == ib:
            continue
        adjacency[ia].append((ea, ib, eb))
        adjacency[ib].append((eb, ia, ea))
        if ea == eb:  # end-to-end or start-to-start: air sides disagree
            ca, cb = chains[ia], chains[ib]
            findings.append((
                "ERROR", ia,
                f"Segments '{ca.name}' (TYPE {ca.seg_type}) and '{cb.name}' "
                f"(TYPE {cb.seg_type}) meet {ea}-to-{eb} at "
                f"({chains[ia].points[0 if ea == 'start' else -1][0]:.6g}, "
                f"{chains[ia].points[0 if ea == 'start' else -1][1]:.6g}): "
                "their air sides point to opposite sides of the boundary there. "
                "Reverse one segment's endpoint order so the chains run head-to-tail.",
            ))

    # ── 4: winding of loops stitched from consistently-oriented open chains ─
    visited: set = set()
    for start_idx in open_air_chains:
        if start_idx in visited or len(adjacency[start_idx]) != 2:
            continue
        # Walk forward from this chain's end until we return or hit a break.
        loop = [start_idx]
        cur, arrived_via = start_idx, "end"
        ok = True
        while True:
            nxt = next(
                ((jb, eb) for (ea, jb, eb) in adjacency[cur] if ea == arrived_via),
                None,
            )
            if nxt is None:
                ok = False
                break
            jdx, joint_end = nxt
            if jdx == start_idx:
                break
            if jdx in loop or joint_end != "start":
                # revisited chain mid-walk, or an inconsistent junction
                # (already reported by check 3) — skip loop-level check.
                ok = False
                break
            loop.append(jdx)
            cur, arrived_via = jdx, "end"
        visited.update(loop)
        if not ok or len(loop) < 2:
            continue
        stitched: List[Tuple[float, float]] = []
        for idx in loop:
            pts = chains[idx].points
            stitched.extend(pts if not stitched else pts[1:])
        if not _chain_is_closed(stitched, tol) or abs(_chain_area2(stitched)) <= 0.0:
            continue
        drawn_ccw = _chain_area2(stitched) > 0.0
        depth = sum(
            1 for cidx, poly in closed_polys
            if cidx not in loop and _point_in_polygon(*_rep_point(stitched), poly)
        )
        expected_ccw = (depth % 2) == 1
        if drawn_ccw != expected_ccw:
            names = ", ".join(f"'{chains[i].name}'" for i in loop)
            findings.append((
                "ERROR", loop[0],
                f"Segments {names} form a closed boundary drawn "
                f"{'CCW' if drawn_ccw else 'CW'}, but the drawing convention (normal "
                f"into air) requires {'CCW' if expected_ccw else 'CW'} winding for this "
                "loop. Reverse every segment in the loop.",
            ))

    return findings


def chains_from_snapshot_segments(segments: List[Dict[str, Any]]) -> List[ChainSpec]:
    """Build ChainSpecs from solver-snapshot segment dicts (point_pairs form)."""

    chains: List[ChainSpec] = []
    for seg_idx, seg in enumerate(segments):
        props = list(seg.get("properties", []) or [])

        def _flag(tok: Any, default: int = 0) -> int:
            try:
                text = str(tok).strip().lower()
                if text.startswith("mat."):
                    text = text[4:]
                return int(float(text))
            except (ValueError, TypeError):
                return default

        seg_type = _flag(props[0], 2) if len(props) > 0 and str(props[0]).strip() else _flag(seg.get("seg_type", 2), 2)
        pos_mat = _flag(props[3]) if len(props) > 3 else 0
        pts: List[Tuple[float, float]] = []
        for i, pair in enumerate(list(seg.get("point_pairs", []) or [])):
            try:
                x1 = float(pair.get("x1", 0.0)); y1 = float(pair.get("y1", 0.0))
                x2 = float(pair.get("x2", 0.0)); y2 = float(pair.get("y2", 0.0))
            except (TypeError, ValueError):
                continue
            if i == 0:
                pts.append((x1, y1))
            pts.append((x2, y2))
        chains.append(ChainSpec(
            name=str(seg.get("name", f"segment_{seg_idx + 1}")),
            seg_type=seg_type,
            pos_mat=pos_mat,
            points=pts,
        ))
    return chains


def build_geometry_snapshot(
    title: str,
    segments: List[Segment],
    ibcs_entries: List[List[str]],
    dielectric_entries: List[List[str]],
) -> Dict[str, Any]:
    segments_payload = []
    for seg in segments:
        point_pairs = []
        for i in range(0, min(len(seg.x), len(seg.y)), 2):
            if i + 1 >= len(seg.x) or i + 1 >= len(seg.y):
                break
            point_pairs.append(
                {
                    "x1": seg.x[i],
                    "y1": seg.y[i],
                    "x2": seg.x[i + 1],
                    "y2": seg.y[i + 1],
                }
            )
        props = list(seg.properties)
        effective_type = seg.seg_type
        if not effective_type and props and str(props[0]).strip():
            effective_type = props[0]
        segments_payload.append(
            {
                "name": seg.name,
                "seg_type": effective_type,
                "properties": props,
                "point_pairs": point_pairs,
            }
        )

    return {
        "title": title,
        "segment_count": len(segments),
        "segments": segments_payload,
        "ibcs": [list(row) for row in ibcs_entries],
        "dielectrics": [list(row) for row in dielectric_entries],
    }
