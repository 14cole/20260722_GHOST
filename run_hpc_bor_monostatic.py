#!/usr/bin/env python3
"""
HPC monostatic BoR RCS sweep driver (SLURM) — the body-of-revolution
counterpart of run_hpc_monostatic.py.

Edit the CONFIG block below and run:

    python run_hpc_bor_monostatic.py

Workflow (mirrors the 2D driver):
- Discover geometry files under FRD_DIR + OPN_DIR (BoR .geo files: x = rho,
  y = z, generatrices traversed +z -> -z; see bor_dispatch).
- Expand into a (geometry × frequency × polarization) unit list. All ASPECT
  angles for a unit are solved in one call (each azimuthal mode is factored
  once; extra aspects are RHS columns).
- Distribute units round-robin across N_NODES × N_JOBS slots, write sbatch
  job-array scripts, submit. Restartable: units whose .grim exists are
  skipped.
- Each unit exports "<POL>_<FREQ:.3f>GHz_<geometry_stem>.grim" (sigma_3d,
  dBsm) with the complex far-field amplitudes preserved.

BoR-specific notes:
- A BoR unit parallelizes INTERNALLY (threads across azimuthal modes and
  streaming-assembly tiles), so the node is divided into a few workers with
  several threads each (WORKERS_PER_UNIT) instead of one process per core.
- The dispatch auto-selects table vs streaming assembly and single/double
  precision by memory estimate; override with ASSEMBLY / TABLE_PRECISION /
  STREAM_BUDGET_GB if needed.
- EXPAND_TO_360 mirrors each aspect sweep about the axis (exact for a BoR).

Radar-frame (azimuth, elevation) polarimetric grids — VV/HH/VH — are built
AUTOMATICALLY during the sweep: as each (geometry, frequency) pair's second
polarization finishes, that worker writes the pair's az/el grids to
<run_dir>/azel/ (AZEL_ENABLE / AZEL_* in the config).  A manual backfill
CLI exists for re-runs:  python run_hpc_bor_monostatic.py --azel <run_dir>

Internal worker invocation (called by SLURM, not by the user):
    python run_hpc_bor_monostatic.py --worker <run_dir> <job_index> <node_index>
"""

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — the only section most users need to edit
# ═══════════════════════════════════════════════════════════════════════════════

FRD_DIR = "geometries/FRD"
OPN_DIR = "geometries/OPN"

# Requested sweep.  ASPECTS are angles from the +z rotation axis (0 =
# nose-on, 90 = broadside, 180 = tail-on).  [0, 180] fully characterizes the
# monostatic response; EXPAND_TO_360 fills the mirrored half in the export.
FREQUENCIES_GHZ = [1.0, 2.0, 3.0]
ASPECTS_DEG     = [float(a) for a in range(0, 181, 3)]
POLARIZATIONS   = ["VV", "HH"]          # keep both if you want --azel grids
EXPAND_TO_360   = False

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "rcs_runs_bor"

# --- Multi-node / multi-submission parallelism -----------------------------
N_NODES = 1
N_JOBS  = 1

# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED — fine tuning (SLURM resources, solver knobs, az/el product)
# ═══════════════════════════════════════════════════════════════════════════════

SLURM_PARTITION = "compute"
SLURM_ACCOUNT   = None
SLURM_QOS       = None
SLURM_TIME      = None
CORES_PER_NODE  = None            # None = whole node via --exclusive
MEM_PER_NODE    = "0"             # "0" = all node memory (see 2D driver notes)
MAX_WORKERS_PER_NODE = None       # cap concurrent UNITS per node (memory-heavy
                                  # units: peak ~ streaming blocks + mode LU)
SLURM_MAIL_TYPE = None
SLURM_MAIL_USER = None
SLURM_EXTRA_SBATCH = []  # type: List[str]
JOB_PROLOGUE = []  # type: List[str]

# --- Solver knobs (mirror bor_dispatch.solve_monostatic_rcs_bor) -----------
GEOMETRY_UNITS          = "inches"       # "inches" or "meters"
CFIE_ALPHA              = 0.5            # closed PEC bodies -> CFIE
N_MODES                 = None           # None = auto (adaptive truncation)
MODE_TOL                = 1e-6
MAX_ELEMENTS            = 50_000
ASSEMBLY                = "auto"         # "auto" | "tables" | "streaming"
TABLE_PRECISION         = "auto"         # "auto" | "single" | "double"
STREAM_BUDGET_GB        = 8.0            # held streaming-block budget per unit
WORKERS_PER_UNIT        = 4              # threads inside one BoR solve (modes
                                         # + streaming tiles); pool size =
                                         # cores // WORKERS_PER_UNIT
BLAS_THREADS_PER_WORKER = 1

# --- Radar-frame (az, el) product ------------------------------------------
# Built AUTOMATICALLY inside the workers: whenever a unit finishes and its
# partner polarization's .grim already exists, that worker writes the
# VV/HH/VH az/el grids for the (geometry, frequency) pair into
# <run_dir>/azel/ (atomic renames, so two nodes racing on the same pair are
# safe; whichever finishes second does the work).  Requires both "VV" and
# "HH" in POLARIZATIONS.  The `--azel <run_dir>` CLI remains only as a
# manual backfill / re-run (e.g. after editing the grid below).
AZEL_ENABLE         = True
AZEL_AZIMUTHS_DEG   = [float(a) for a in range(0, 360, 5)]
AZEL_ELEVATIONS_DEG = [float(e) for e in range(-60, 61, 5)]
AZEL_AXIS_AZ_DEG    = 0.0     # target axis orientation in the earth frame
AZEL_AXIS_EL_DEG    = 0.0     # horizontal, nose toward azimuth 0.  NOTE the
                              # polarization label mapping for horizontal
                              # axes (bor_az_el_grid docstring).

GEOMETRY_EXTS = (".geo",)
PYTHON_EXE    = sys.executable
SUBMIT        = True

# ═══════════════════════════════════════════════════════════════════════════════

_SBATCH = shutil.which("sbatch") or "sbatch"


# ─── shared helpers ────────────────────────────────────────────────────────

def _discover_geometries():
    # type: () -> List[Path]
    found = []   # type: List[Path]
    seen = set()  # type: set
    for d in (FRD_DIR, OPN_DIR):
        root = Path(d)
        if not root.is_dir():
            print(f"  [warn] dir not found: {root}", file=sys.stderr)
            continue
        for ext in GEOMETRY_EXTS:
            for p in sorted(root.rglob(f"*{ext}")):
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                found.append(p)
    return found


def _pin_blas_threads(n):
    # type: (int) -> None
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


def _detect_cores():
    # type: () -> int
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(var, "").strip()
        if v.isdigit() and int(v) > 0:
            return int(v)
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def _unit_output_path(run_dir, unit):
    # type: (Path, Dict[str, Any]) -> Path
    pol  = unit["polarization"]
    freq = float(unit["frequency_ghz"])
    stem = unit["geometry_stem"]
    return run_dir / "results" / f"{pol}_{freq:.3f}GHz_{stem}.grim"


def _azel_out_paths(run_dir, stem, freq):
    # type: (Path, str, float) -> List[Path]
    return [run_dir / "azel" / f"azel_{freq:.3f}GHz_{stem}_{ch}.grim"
            for ch in ("VV", "HH", "VH")]


def _build_azel_pair(run_dir, stem, freq, azel_cfg):
    # type: (Path, str, float, Dict[str, Any]) -> bool
    """Build the radar-frame az/el grids for one (geometry, frequency) pair
    from its VV + HH .grim exports.  Idempotent; atomic via temp + rename so
    concurrent attempts from different nodes cannot interleave writes."""
    outs = _azel_out_paths(run_dir, stem, freq)
    if all(p.exists() for p in outs):
        return False
    vv = run_dir / "results" / f"VV_{freq:.3f}GHz_{stem}.grim"
    hh = run_dir / "results" / f"HH_{freq:.3f}GHz_{stem}.grim"
    if not (vv.exists() and hh.exists()):
        return False

    from bor_dispatch import bor_az_el_grid
    from grim_io import save_bor_az_el_grim
    rv = _result_from_grim(vv, "VV")
    rh = _result_from_grim(hh, "HH")
    grid = bor_az_el_grid(
        rv, rh, azel_cfg["azimuths_deg"], azel_cfg["elevations_deg"],
        axis_az_deg=float(azel_cfg["axis_az_deg"]),
        axis_el_deg=float(azel_cfg["axis_el_deg"]))
    out_dir = run_dir / "azel"
    out_dir.mkdir(exist_ok=True)
    tmp_stem = out_dir / f".tmp_{os.getpid()}_{freq:.3f}GHz_{stem}.grim"
    written = save_bor_az_el_grim(
        grid, str(tmp_stem), source_path=stem,
        history=f"run_hpc_bor_monostatic.py azel {freq}GHz")
    for tmp, final in zip(sorted(written), sorted(str(p) for p in outs)):
        os.replace(tmp, final)
    return True


def _solve_and_export(unit, snapshot, material_base, run_dir_str):
    # type: (Dict[str, Any], Dict[str, Any], str, str) -> Tuple[str, str]
    """Pool-worker entry point: solve one BoR unit, export .grim, and (when
    its partner polarization is already done) write the pair's az/el
    product. Idempotent."""
    run_dir = Path(run_dir_str)
    out_path = _unit_output_path(run_dir, unit)
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
        n_modes=N_MODES,
        mode_tol=MODE_TOL,
        max_elements=MAX_ELEMENTS,
        workers=WORKERS_PER_UNIT,
        table_precision=TABLE_PRECISION,
        assembly=ASSEMBLY,
        expand_to_360=EXPAND_TO_360,
    )
    for w in result["metadata"].get("warnings", []) or []:
        print(f"      [warn] {unit['geometry_stem']}: {w}", flush=True)

    from grim_io import export_result_to_grim
    written = export_result_to_grim(
        result, str(out_path),
        source_path=str(snapshot.get("source_path", "") or ""),
        history=(f"run_hpc_bor_monostatic.py pol={unit['polarization']} "
                 f"freq={unit['frequency_ghz']}GHz"),
    )

    # az/el product: whichever polarization of the pair finishes second
    # builds it (config from the manifest, so all nodes agree).
    if AZEL_ENABLE:
        try:
            manifest = json.loads((run_dir / "manifest.json").read_text())
            cfg = manifest.get("azel_config") or {}
            if cfg and _build_azel_pair(run_dir, unit["geometry_stem"],
                                        float(unit["frequency_ghz"]), cfg):
                print(f"      azel grids written for {unit['geometry_stem']} "
                      f"{unit['frequency_ghz']:.3f}GHz", flush=True)
        except Exception:
            print("      [warn] azel product failed:", flush=True)
            for line in traceback.format_exc().rstrip().splitlines():
                print(f"        {line}", flush=True)

    return ("written", str(written[0]) if written else str(out_path))


def _solve_and_export_star(args):
    # type: (tuple) -> tuple
    u, snap, mat_base, run_dir_str = args
    try:
        status, path = _solve_and_export(u, snap, mat_base, run_dir_str)
        return ("ok", status, path, u)
    except Exception:
        return ("err", traceback.format_exc(), "", u)


# ─── submit mode (user-invoked) ────────────────────────────────────────────

def _build_slurm(script_path, run_dir, job_index):
    # type: (Path, Path, int) -> str
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=bor_{run_dir.name}_j{job_index}",
        f"#SBATCH --array=0-{N_NODES - 1}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --partition={SLURM_PARTITION}",
        f"#SBATCH --output={run_dir}/logs/job{job_index}_task_%A_%a.out",
        f"#SBATCH --error={run_dir}/logs/job{job_index}_task_%A_%a.err",
    ]
    if CORES_PER_NODE is not None:
        lines.append(f"#SBATCH --cpus-per-task={CORES_PER_NODE}")
    else:
        lines.append("#SBATCH --exclusive")
    if MEM_PER_NODE:
        lines.append(f"#SBATCH --mem={MEM_PER_NODE}")
    if SLURM_TIME:
        lines.append(f"#SBATCH --time={SLURM_TIME}")
    if SLURM_ACCOUNT:   lines.append(f"#SBATCH --account={SLURM_ACCOUNT}")
    if SLURM_QOS:       lines.append(f"#SBATCH --qos={SLURM_QOS}")
    if SLURM_MAIL_TYPE: lines.append(f"#SBATCH --mail-type={SLURM_MAIL_TYPE}")
    if SLURM_MAIL_USER: lines.append(f"#SBATCH --mail-user={SLURM_MAIL_USER}")
    for extra in SLURM_EXTRA_SBATCH:
        e = extra.strip()
        if e:
            lines.append(e if e.startswith("#SBATCH") else f"#SBATCH {e}")

    lines += [
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(script_path.parent))}",
        *JOB_PROLOGUE,
        (f"exec {shlex.quote(PYTHON_EXE)} {shlex.quote(str(script_path))} "
         f"--worker {shlex.quote(str(run_dir))} {job_index} "
         f"${{SLURM_ARRAY_TASK_ID}}"),
        "",
    ]
    return "\n".join(lines)


def submit():
    # type: () -> None
    geometries = _discover_geometries()
    if not geometries:
        sys.exit("ERROR: no geometry files (*.geo) found under FRD_DIR or OPN_DIR.")

    pols = [p.strip().upper() for p in POLARIZATIONS if p and p.strip()]
    if not pols:            sys.exit("ERROR: POLARIZATIONS is empty.")
    if not FREQUENCIES_GHZ: sys.exit("ERROR: FREQUENCIES_GHZ is empty.")
    if not ASPECTS_DEG:     sys.exit("ERROR: ASPECTS_DEG is empty.")
    if any(a < 0.0 or a > 180.0 for a in ASPECTS_DEG):
        sys.exit("ERROR: BoR aspects must lie in [0, 180] deg from the +z "
                 "axis (use EXPAND_TO_360 for the mirrored half).")
    if int(N_NODES) < 1 or int(N_JOBS) < 1:
        sys.exit("ERROR: N_NODES and N_JOBS must be >= 1.")

    units = []  # type: List[Dict[str, Any]]
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

    run_id  = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(OUTPUT_DIR).resolve() / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id":          run_id,
        "created":         datetime.now().isoformat(),
        "solver":          "bor_mom_rcs",
        "frd_dir":         str(Path(FRD_DIR).resolve()),
        "opn_dir":         str(Path(OPN_DIR).resolve()),
        "output_dir":      str(run_dir),
        "frequencies_ghz": list(FREQUENCIES_GHZ),
        "aspects_deg":     list(ASPECTS_DEG),
        "polarizations":   pols,
        "expand_to_360":   bool(EXPAND_TO_360),
        "n_nodes":         int(N_NODES),
        "n_jobs":          int(N_JOBS),
        "n_slots":         int(N_NODES) * int(N_JOBS),
        "n_units":         len(units),
        "solver_config": {
            "geometry_units":          GEOMETRY_UNITS,
            "cfie_alpha":              CFIE_ALPHA,
            "n_modes":                 N_MODES,
            "mode_tol":                MODE_TOL,
            "max_elements":            MAX_ELEMENTS,
            "assembly":                ASSEMBLY,
            "table_precision":         TABLE_PRECISION,
            "stream_budget_gb":        STREAM_BUDGET_GB,
            "workers_per_unit":        WORKERS_PER_UNIT,
            "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
            "cores_per_node":          CORES_PER_NODE,
        },
        "azel_config": {
            "enabled":        bool(AZEL_ENABLE),
            "azimuths_deg":   list(AZEL_AZIMUTHS_DEG),
            "elevations_deg": list(AZEL_ELEVATIONS_DEG),
            "axis_az_deg":    AZEL_AXIS_AZ_DEG,
            "axis_el_deg":    AZEL_AXIS_EL_DEG,
        },
        "units": units,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    script_path = Path(__file__).resolve()
    slurm_paths = []  # type: List[Path]
    for j in range(int(N_JOBS)):
        sp = run_dir / f"submit_job{j}.slurm"
        sp.write_text(_build_slurm(script_path, run_dir, j))
        sp.chmod(0o755)
        slurm_paths.append(sp)

    print("=" * 70)
    print("HPC monostatic BoR RCS sweep")
    print("=" * 70)
    print(f"  Run dir       : {run_dir}")
    print(f"  Geometries    : {len(geometries)}")
    print(f"  Polarizations : {', '.join(pols)}")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Aspects       : {len(ASPECTS_DEG)}  (0-180 from the axis"
          f"{', mirrored to 360 on export' if EXPAND_TO_360 else ''})")
    print(f"  Units total   : {len(units)}  (geom × freq × pol)")
    print(f"  Slots         : {N_JOBS} job(s) × {N_NODES} node(s)")
    print(f"  Per unit      : {WORKERS_PER_UNIT} threads (modes + streaming "
          f"tiles), assembly={ASSEMBLY}, precision={TABLE_PRECISION}")
    print(f"  Slurm scripts : {len(slurm_paths)} files in {run_dir}")

    if not SUBMIT or shutil.which("sbatch") is None:
        why = "SUBMIT=False" if not SUBMIT else "[warn] sbatch not on PATH"
        print(f"\n  {why} — submit manually with:")
        for sp in slurm_paths:
            print(f"    sbatch {sp}")
        return

    for sp in slurm_paths:
        print(f"\n  Submitting: sbatch {sp.name}")
        res = subprocess.run(
            [_SBATCH, str(sp)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            sys.exit(f"sbatch failed (exit {res.returncode}):\n"
                     f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        print(f"  {res.stdout.strip()}")

    print(f"\nMonitor with:  squeue -u $USER")
    print(f"Outputs in:    {run_dir}/results/"
          + (f"  (+ az/el grids in {run_dir}/azel/)" if AZEL_ENABLE else ""))


# ─── worker mode (invoked by SLURM) ────────────────────────────────────────

def _slot_units(units, job_index, node_index, n_nodes, n_jobs):
    # type: (List[Dict[str, Any]], int, int, int, int) -> List[Dict[str, Any]]
    n_slots = n_nodes * n_jobs
    slot = job_index * n_nodes + node_index
    if slot < 0 or slot >= n_slots:
        raise ValueError(f"slot {slot} out of range (0..{n_slots - 1})")
    return [u for i, u in enumerate(units) if i % n_slots == slot]


def worker(run_dir_str, job_index, node_index):
    # type: (str, int, int) -> None
    from geometry_io import parse_geometry, build_geometry_snapshot

    run_dir  = Path(run_dir_str).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    units    = manifest["units"]
    n_nodes  = int(manifest.get("n_nodes", 1))
    n_jobs   = int(manifest.get("n_jobs", 1))

    my_units = _slot_units(units, job_index, node_index, n_nodes, n_jobs)
    cores    = _detect_cores()
    # BoR units are internally threaded: divide the node into pool workers of
    # WORKERS_PER_UNIT threads each.
    by_threads = max(1, cores // max(1, WORKERS_PER_UNIT))
    worker_cap = by_threads if MAX_WORKERS_PER_NODE is None else \
        max(1, min(by_threads, int(MAX_WORKERS_PER_NODE)))
    pool_size = max(1, min(worker_cap, len(my_units))) if my_units else 1

    slot_id = job_index * n_nodes + node_index
    print("=" * 70)
    print(f"  Slot {slot_id}/{n_nodes * n_jobs - 1}  "
          f"(job={job_index}, node={node_index})")
    print(f"  Units assigned: {len(my_units)} of {len(units)} total")
    print(f"  Cores: {cores}   pool: {pool_size} × {WORKERS_PER_UNIT} threads "
          f"(BLAS/worker: {BLAS_THREADS_PER_WORKER})")
    print("=" * 70, flush=True)

    if not my_units:
        print("  No work for this slot.")
        return

    snapshots = {}  # type: Dict[str, Tuple[Dict[str, Any], str]]
    for u in my_units:
        gpath = u["geometry"]
        if gpath in snapshots:
            continue
        p = Path(gpath)
        if not p.is_file():
            sys.exit(f"Geometry missing on compute node: {p}")
        title, segments, ibcs, dielectrics = parse_geometry(p.read_text())
        snap = build_geometry_snapshot(title, segments, ibcs, dielectrics)
        snap["source_path"] = str(p)
        snapshots[gpath] = (snap, str(p.parent))

    _pin_blas_threads(BLAS_THREADS_PER_WORKER)
    t0 = time.time()
    n_done = n_skipped = n_failed = 0
    total = len(my_units)
    args_list = [
        (u, snapshots[u["geometry"]][0], snapshots[u["geometry"]][1], str(run_dir))
        for u in my_units
    ]
    with Pool(processes=pool_size,
              initializer=_pin_blas_threads,
              initargs=(BLAS_THREADS_PER_WORKER,),
              maxtasksperchild=1) as pool:
        for idx, result in enumerate(
            pool.imap_unordered(_solve_and_export_star, args_list, chunksize=1),
            start=1,
        ):
            kind, a, b, u = result
            tag = (f"{u['polarization']} {u['frequency_ghz']:7.3f}GHz "
                   f"{u['geometry_stem']}")
            if kind == "ok":
                status, path = a, b
                if status == "skipped":
                    n_skipped += 1
                else:
                    n_done += 1
                print(f"  [{idx:3d}/{total}] {status:7s}  {tag}  -> "
                      f"{Path(path).name}", flush=True)
            else:
                n_failed += 1
                print(f"  [{idx:3d}/{total}] FAILED   {tag}", flush=True)
                for line in str(a).rstrip().splitlines():
                    print(f"      {line}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Slot complete. wrote={n_done}, skipped={n_skipped}, "
          f"failed={n_failed}.  {elapsed:.1f} s elapsed.")


# ─── az/el post-processing mode (login node, after the sweep) ──────────────

def _result_from_grim(path, pol):
    # type: (Path, str) -> Dict[str, Any]
    """Reconstruct the minimal result dict bor_az_el_grid needs from a .grim
    (the exports preserve the complex far-field amplitudes)."""
    import numpy as np
    d = np.load(str(path))
    # NpzFile.get() only exists on numpy >= 1.25; use .files membership so
    # this runs on older HPC numpy builds too.
    if ("raw_complex_amplitude_preserved" not in getattr(d, "files", [])
            or not bool(d["raw_complex_amplitude_preserved"])):
        raise ValueError(f"{path.name}: complex amplitudes were not preserved.")
    samples = []
    az = d["azimuths"]           # aspect angles for BoR exports
    freqs = d["frequencies"]
    ar, ai = d["rcs_amp_real"], d["rcs_amp_imag"]
    for i, th in enumerate(az):
        if float(th) > 180.0:
            continue             # skip the EXPAND_TO_360 mirror half
        for kf, f in enumerate(freqs):
            samples.append({
                "frequency_ghz": float(f),
                "theta_inc_deg": float(th),
                "theta_scat_deg": float(th),
                "rcs_amp_real": float(ar[i, 0, kf, 0]),
                "rcs_amp_imag": float(ai[i, 0, kf, 0]),
                "rcs_linear": float(d["rcs_power"][i, 0, kf, 0]),
                "rcs_db": 10.0 * math.log10(max(float(d["rcs_power"][i, 0, kf, 0]), 1e-30)),
                "rcs_amp_phase_deg": 0.0,
            })
    return {"polarization": pol, "samples": samples}


def azel(run_dir_str):
    # type: (str) -> None
    """Manual backfill: normally unnecessary — the workers build each pair's
    az/el product automatically as the second polarization finishes.  Use
    this only to (re)build after editing the AZEL_* grid in the manifest, or
    if AZEL_ENABLE was off during the sweep."""

    run_dir  = Path(run_dir_str).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = dict(manifest.get("azel_config") or {})
    cfg.setdefault("azimuths_deg", AZEL_AZIMUTHS_DEG)
    cfg.setdefault("elevations_deg", AZEL_ELEVATIONS_DEG)
    cfg.setdefault("axis_az_deg", AZEL_AXIS_AZ_DEG)
    cfg.setdefault("axis_el_deg", AZEL_AXIS_EL_DEG)

    keys = sorted({(u["geometry_stem"], float(u["frequency_ghz"]))
                   for u in manifest["units"]})
    n_done = n_skip = 0
    for stem, freq in keys:
        if _build_azel_pair(run_dir, stem, freq, cfg):
            n_done += 1
            print(f"  {stem} {freq:.3f}GHz -> azel_{freq:.3f}GHz_{stem}_[VV|HH|VH].grim")
        else:
            n_skip += 1
    print(f"\n  az/el grids: {n_done} written, {n_skip} skipped (already "
          f"built or missing a polarization).  Outputs: {run_dir / 'azel'}/")


# ─── entry point ───────────────────────────────────────────────────────────

def main():
    # type: () -> None
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--worker", nargs=3, metavar=("RUN_DIR", "JOB_INDEX", "NODE_INDEX"),
        help="Internal: execute one array-task slice. Invoked by SLURM.",
    )
    ap.add_argument(
        "--azel", metavar="RUN_DIR",
        help="Post-process a completed run into radar-frame (az, el) "
             "polarimetric grids (needs both VV and HH units).",
    )
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
    elif args.azel:
        azel(args.azel)
    else:
        submit()


if __name__ == "__main__":
    main()
