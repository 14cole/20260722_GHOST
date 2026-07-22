#!/usr/bin/env python3
"""
Local BoR monostatic RCS aspect sweep — same naming/streaming pattern as
run_local_monostatic.py, but each unit is a true 3-D (dBsm) BoR solve over
the ASPECT angles (degrees from the +z rotation axis: 0 = nose-on,
90 = broadside, 180 = tail-on).

Geometries are .geo half-profiles: x = rho (>= 0), y = z (rotation axis),
drawn from the +z axis end to the -z axis end (see bor_dispatch.py).

Edit the CONFIG block and run:

    python run_local_bor.py

Outputs go to <OUTPUT_DIR>/run_YYYYMMDD_HHMMSS/results/ as
"<POL>_<FREQ:.3f>GHz_<geometry_stem>.grim".  Re-running skips units whose
output already exists.  Each unit uses the solver's threaded mode engine, so
the pool default is few processes x several threads.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

GEOMETRY_DIRS = ["geometries/BOR"]      # every *.geo under these, recursively

FREQUENCIES_GHZ = [1.0, 2.0, 4.0]
ASPECTS_DEG     = [float(a) for a in range(0, 181, 5)]
POLARIZATIONS   = ["VV", "HH"]

OUTPUT_DIR = "rcs_runs_bor"

# Process pool size (units in flight). None -> max(1, cpu_count // 4): each
# unit already runs SOLVER_WORKERS threads for its mode sweep.
WORKERS = None
SOLVER_WORKERS = 4

GEOMETRY_UNITS = "inches"               # "inches" or "meters"
CFIE_ALPHA     = 0.5
BLAS_THREADS_PER_WORKER = 1

# ═══════════════════════════════════════════════════════════════════════════════


def _discover_geometries() -> List[Path]:
    found: List[Path] = []
    seen: set = set()
    for d in GEOMETRY_DIRS:
        root = Path(d)
        if not root.is_dir():
            print(f"  [warn] dir not found: {root}", file=sys.stderr)
            continue
        for p in sorted(root.rglob("*.geo")):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(p)
    return found


def _pin_blas_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


def _unit_output_path(results_dir: Path, unit: Dict[str, Any]) -> Path:
    return results_dir / (f"{unit['polarization']}_"
                          f"{float(unit['frequency_ghz']):.3f}GHz_"
                          f"{unit['geometry_stem']}.grim")


def _solve_and_export(unit, snapshot, material_base, results_dir_str):
    """Pool-worker entry point: solve one unit, export .grim. Idempotent."""
    out_path = _unit_output_path(Path(results_dir_str), unit)
    if out_path.exists():
        return ("skipped", str(out_path))

    from bor_dispatch import solve_monostatic_rcs_bor
    result = solve_monostatic_rcs_bor(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(unit["frequency_ghz"])],
        elevations_deg=[float(a) for a in unit["aspects_deg"]],
        polarization=unit["polarization"],
        geometry_units=GEOMETRY_UNITS,
        material_base_dir=material_base,
        cfie_alpha=CFIE_ALPHA,
        workers=SOLVER_WORKERS,
    )

    from grim_io import export_result_to_grim
    written = export_result_to_grim(
        result, str(out_path),
        source_path=str(snapshot.get("source_path", "") or ""),
        history=(f"run_local_bor.py pol={unit['polarization']} "
                 f"freq={unit['frequency_ghz']}GHz"),
    )
    return ("written", str(written[0]) if written else str(out_path))


def main() -> None:
    from geometry_io import parse_geometry, build_geometry_snapshot

    geometries = _discover_geometries()
    if not geometries:
        sys.exit("ERROR: no geometry files (*.geo) found under GEOMETRY_DIRS.")

    pols = [p.strip().upper() for p in POLARIZATIONS if p and p.strip()]
    if not pols:            sys.exit("ERROR: POLARIZATIONS is empty.")
    if not FREQUENCIES_GHZ: sys.exit("ERROR: FREQUENCIES_GHZ is empty.")
    if not ASPECTS_DEG:     sys.exit("ERROR: ASPECTS_DEG is empty.")

    units: List[Dict[str, Any]] = []
    for geom in geometries:
        for pol in pols:
            for f in FREQUENCIES_GHZ:
                units.append({
                    "geometry":      str(geom.resolve()),
                    "geometry_stem": geom.stem,
                    "polarization":  pol,
                    "frequency_ghz": float(f),
                    "aspects_deg":   [float(a) for a in ASPECTS_DEG],
                })

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(OUTPUT_DIR).resolve() / run_id
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cpu = os.cpu_count() or 1
    n_workers = WORKERS if WORKERS else max(1, cpu // 4)
    n_workers = max(1, min(int(n_workers), len(units)))

    print("=" * 70)
    print("Local BoR monostatic RCS aspect sweep")
    print("=" * 70)
    print(f"  Run dir       : {run_dir}")
    print(f"  Geometries    : {len(geometries)}")
    print(f"  Polarizations : {', '.join(pols)}")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Aspects       : {len(ASPECTS_DEG)}  "
          f"({min(ASPECTS_DEG):g}-{max(ASPECTS_DEG):g} deg from +z axis)")
    print(f"  Units total   : {len(units)}  (geom × freq × pol)")
    print(f"  Workers       : {n_workers} procs × {SOLVER_WORKERS} mode threads "
          f"of {cpu} cpus")
    print("=" * 70, flush=True)

    snapshots: Dict[str, Tuple[Dict[str, Any], str]] = {}
    for u in units:
        gpath = u["geometry"]
        if gpath in snapshots:
            continue
        p = Path(gpath)
        title, segments, ibcs, dielectrics = parse_geometry(p.read_text())
        snap = build_geometry_snapshot(title, segments, ibcs, dielectrics)
        snap["source_path"] = str(p)
        snapshots[gpath] = (snap, str(p.parent))

    _pin_blas_threads(BLAS_THREADS_PER_WORKER)
    t0 = time.time()
    n_done = n_skipped = n_failed = 0
    total = len(units)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_pin_blas_threads,
        initargs=(BLAS_THREADS_PER_WORKER,),
    ) as pool:
        fut_to_unit = {}
        for u in units:
            snap, mat_base = snapshots[u["geometry"]]
            fut = pool.submit(_solve_and_export, u, snap, mat_base, str(results_dir))
            fut_to_unit[fut] = u

        for fut in as_completed(fut_to_unit):
            u = fut_to_unit[fut]
            tag = (f"{u['polarization']} {u['frequency_ghz']:7.3f}GHz "
                   f"{u['geometry_stem']}")
            try:
                status, path = fut.result()
                n_skipped += status == "skipped"
                n_done += status != "skipped"
                idx = n_done + n_skipped + n_failed
                print(f"  [{idx:3d}/{total}] {status:7s}  {tag}  -> "
                      f"{Path(path).name}", flush=True)
            except Exception as exc:
                n_failed += 1
                idx = n_done + n_skipped + n_failed
                print(f"  [{idx:3d}/{total}] FAILED   {tag}: {exc}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done. wrote={n_done}, skipped={n_skipped}, failed={n_failed}.  "
          f"{elapsed:.1f} s elapsed.")
    print(f"  Outputs: {results_dir}/")


if __name__ == "__main__":
    main()
