#!/usr/bin/env python3
"""
HPC monostatic RCS sweep driver (SLURM).

Edit the CONFIG block below and run:

    python run_hpc_monostatic.py

Workflow:
- Discover geometry files under FRD_DIR + OPN_DIR.
- Expand into a (geometry × frequency × polarization) unit list. All azimuths
  for a unit are solved in a single solver call (matrix factored once).
- Distribute units round-robin across N_NODES × N_JOBS parallel slots.
- Write N_JOBS sbatch scripts (each one a job array of size N_NODES) and
  submit them. Each array task runs on one node and parallelizes its assigned
  units across the cores SLURM allocated.
- As each unit finishes, its result is exported immediately to
  "<POL>_<FREQ:.3f>GHz_<geometry_stem>.grim" in <run_dir>/results/.

Restartable: a unit whose .grim file already exists is skipped, so you can
cancel and resubmit one job's slice (e.g., move it to a different partition)
without re-doing finished work. The manifest's slot partitioning is fixed
once written, so the other in-flight submissions stay correct.

Internal worker invocation (called by SLURM, not by the user):
    python run_hpc_monostatic.py --worker <run_dir> <job_index> <node_index>
"""

import argparse
import json
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

# Input geometry folders. Every *.geo file found under these paths
# (recursively) is added to the sweep. Source folder is NOT injected into
# output filenames — the geometry filename is preserved verbatim.
FRD_DIR = "geometries/FRD"
OPN_DIR = "geometries/OPN"

# Requested sweep.
FREQUENCIES_GHZ = [2.0, 4.0, 6.0, 8.0, 10.0]
AZIMUTHS_DEG    = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
POLARIZATIONS   = ["VV", "HH"]          # any subset of: VV, HH, TM, TE

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "rcs_runs"

# --- Multi-node / multi-submission parallelism -----------------------------
# Total parallel compute = N_NODES × N_JOBS nodes. Units are split round-robin
# across that many slots. N_JOBS separate sbatch submissions are produced
# (each one a job array of size N_NODES). This lets you put e.g. 2 nodes on
# partition A and 2 nodes on partition B without overlapping work — the
# submissions don't need to talk to each other; the partitioning is
# deterministic from the manifest.
N_NODES = 1
N_JOBS  = 1

# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED — fine tuning (SLURM resources, solver knobs, env setup)
# ═══════════════════════════════════════════════════════════════════════════════

# --- SLURM resources (per array task = one node) ---------------------------
SLURM_PARTITION = "compute"
SLURM_ACCOUNT   = None            # e.g. "my_project"; None to omit
SLURM_QOS       = None
SLURM_TIME      = None            # None = no walltime limit; or "HH:MM:SS"
CORES_PER_NODE  = None            # None = request whole node via --exclusive
                                  # (pool size auto-detected from SLURM env).
                                  # Or set an integer, e.g. 32.
MEM_PER_NODE    = "0"             # "0" = ALL memory of the node (SLURM idiom;
                                  # recommended with --exclusive). None = omit
                                  # the directive -> cluster default applies
                                  # (often DefMemPerCPU ~3.5G x CPUs, which can
                                  # be far less than node RAM and OOM-kill
                                  # workers). Or an explicit value, e.g. "64G".
MAX_WORKERS_PER_NODE = None       # None = one worker per allocated core. Set a
                                  # smaller integer when units are memory-heavy:
                                  # peak node RAM ~ pool_size x per-unit peak
                                  # (dense solve ~ 5*16*N^2 bytes, N = boundary
                                  # nodes ~ 20 x perimeter/lambda). An OOM KILL
                                  # (cgroup) can hang the Pool, unlike a Python
                                  # MemoryError which is caught and logged.
SLURM_MAIL_TYPE = None            # e.g. "END,FAIL"
SLURM_MAIL_USER = None
SLURM_EXTRA_SBATCH = []  # type: List[str]  # raw extra lines, e.g. "--constraint=intel"

JOB_PROLOGUE = []  # type: List[str]

# --- Solver knobs (mirror run_monostatic.py) -------------------------------
GEOMETRY_UNITS          = "inches"       # "inches" or "meters"
SOLVER_METHOD           = "auto"         # "auto" | "direct" | "gmres" | "fmm"
CFIE_ALPHA              = 0.0
MAX_PANELS              = 50_000
BLAS_THREADS_PER_WORKER = 1              # keeps N workers × BLAS threads sane

# --- Geometry discovery & submission ---------------------------------------
GEOMETRY_EXTS = (".geo",)
PYTHON_EXE    = sys.executable           # interpreter used inside the job
SUBMIT        = True                     # False → write .slurm files but don't sbatch

# ═══════════════════════════════════════════════════════════════════════════════

_SBATCH = shutil.which("sbatch") or "sbatch"


# ─── shared helpers ────────────────────────────────────────────────────────

def _discover_geometries():
    # type: () -> List[Path]
    """Return every geometry file under FRD_DIR/OPN_DIR (deduplicated)."""
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
    """Pin BLAS threads via env vars. Called in parent and in each pool worker."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


def _detect_cores():
    # type: () -> int
    """Cores actually allocated to this process. Prefers SLURM, falls back to OS."""
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


def _solve_and_export(unit, snapshot, material_base, run_dir_str):
    # type: (Dict[str, Any], Dict[str, Any], str, str) -> Tuple[str, str]
    """Pool-worker entry point: solve one unit, export .grim. Idempotent."""
    run_dir = Path(run_dir_str)
    out_path = _unit_output_path(run_dir, unit)
    if out_path.exists():
        return ("skipped", str(out_path))

    from rcs_solver import solve_monostatic_rcs_2d
    result = solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(unit["frequency_ghz"])],
        elevations_deg=[float(a) for a in unit["azimuths_deg"]],
        polarization=unit["polarization"],
        geometry_units=GEOMETRY_UNITS,
        material_base_dir=material_base,
        max_panels=MAX_PANELS,
        cfie_alpha=CFIE_ALPHA,
        solver_method=SOLVER_METHOD,
    )

    from grim_io import export_result_to_grim
    written = export_result_to_grim(
        result, str(out_path),
        source_path=str(snapshot.get("source_path", "") or ""),
        history=(f"run_hpc_monostatic.py pol={unit['polarization']} "
                 f"freq={unit['frequency_ghz']}GHz"),
    )
    return ("written", str(written[0]) if written else str(out_path))


def _solve_and_export_star(args):
    # type: (tuple) -> tuple
    """Pool.imap_unordered entry point: unpack args and catch exceptions in-band.

    The full traceback string is returned (not just str(exc)) so the SLURM log
    shows where the failure happened, not just the message.
    """
    u, snap, mat_base, run_dir_str = args
    try:
        status, path = _solve_and_export(u, snap, mat_base, run_dir_str)
        return ("ok", status, path, u)
    except Exception:
        return ("err", traceback.format_exc(), "", u)


# ─── submit mode (user-invoked) ────────────────────────────────────────────

def _build_slurm(script_path, run_dir, job_index):
    # type: (Path, Path, int) -> str
    n_array = N_NODES
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=rcs_{run_dir.name}_j{job_index}",
        f"#SBATCH --array=0-{n_array - 1}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --partition={SLURM_PARTITION}",
        f"#SBATCH --output={run_dir}/logs/job{job_index}_task_%A_%a.out",
        f"#SBATCH --error={run_dir}/logs/job{job_index}_task_%A_%a.err",
    ]
    # Cores: explicit count or --exclusive (whole node; pool auto-detects via SLURM env).
    if CORES_PER_NODE is not None:
        lines.append(f"#SBATCH --cpus-per-task={CORES_PER_NODE}")
    else:
        lines.append("#SBATCH --exclusive")
    # Memory and time are optional; omitting them means no limit.
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
        if not e:
            continue
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
    if not AZIMUTHS_DEG:    sys.exit("ERROR: AZIMUTHS_DEG is empty.")
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
                    "azimuths_deg":  [float(a) for a in AZIMUTHS_DEG],
                })

    run_id  = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(OUTPUT_DIR).resolve() / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id":          run_id,
        "created":         datetime.now().isoformat(),
        "frd_dir":         str(Path(FRD_DIR).resolve()),
        "opn_dir":         str(Path(OPN_DIR).resolve()),
        "output_dir":      str(run_dir),
        "frequencies_ghz": list(FREQUENCIES_GHZ),
        "azimuths_deg":    list(AZIMUTHS_DEG),
        "polarizations":   pols,
        "n_nodes":         int(N_NODES),
        "n_jobs":          int(N_JOBS),
        "n_slots":         int(N_NODES) * int(N_JOBS),
        "n_units":         len(units),
        "solver_config": {
            "geometry_units":          GEOMETRY_UNITS,
            "solver_method":           SOLVER_METHOD,
            "cfie_alpha":              CFIE_ALPHA,
            "max_panels":              MAX_PANELS,
            "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
            "cores_per_node":          CORES_PER_NODE,
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
    print("HPC monostatic RCS sweep")
    print("=" * 70)
    print(f"  Run dir       : {run_dir}")
    print(f"  Geometries    : {len(geometries)}")
    print(f"  Polarizations : {', '.join(pols)}")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Azimuths      : {len(AZIMUTHS_DEG)}")
    print(f"  Units total   : {len(units)}  (geom × freq × pol)")
    print(f"  Slots         : {N_JOBS} job(s) × {N_NODES} node(s) "
          f"= {int(N_JOBS) * int(N_NODES)} parallel nodes")
    cores_str = str(CORES_PER_NODE) if CORES_PER_NODE is not None else "auto (--exclusive)"
    mem_str   = str(MEM_PER_NODE) if MEM_PER_NODE else "unlimited"
    time_str  = str(SLURM_TIME) if SLURM_TIME else "unlimited"
    print(f"  Per node      : {cores_str} cores, {mem_str} RAM, "
          f"{time_str} walltime")
    print(f"  Slurm scripts : {len(slurm_paths)} files in {run_dir}")

    if not SUBMIT:
        print("\n  SUBMIT=False — submit manually with:")
        for sp in slurm_paths:
            print(f"    sbatch {sp}")
        return

    if shutil.which("sbatch") is None:
        print("\n  [warn] sbatch not on PATH. Submit manually:")
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
    print(f"Outputs in:    {run_dir}/results/")


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
    worker_cap = cores if MAX_WORKERS_PER_NODE is None else max(1, int(MAX_WORKERS_PER_NODE))
    pool_size = max(1, min(cores, worker_cap, len(my_units))) if my_units else 1

    slot_id = job_index * n_nodes + node_index
    print("=" * 70)
    print(f"  Slot {slot_id}/{n_nodes * n_jobs - 1}  "
          f"(job={job_index}, node={node_index})")
    print(f"  Units assigned: {len(my_units)} of {len(units)} total")
    print(f"  Cores detected: {cores}   pool size: {pool_size}   "
          f"(BLAS threads/worker: {BLAS_THREADS_PER_WORKER})")
    print("=" * 70, flush=True)

    if not my_units:
        print("  No work for this slot.")
        return

    # Parse each geometry once; share snapshots across all units that use it.
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

    # Pin BLAS in parent; Pool initializer pins it in each worker too (works on
    # all start methods, unlike ProcessPoolExecutor which only got initializer
    # support in Python 3.7).
    _pin_blas_threads(BLAS_THREADS_PER_WORKER)
    t0 = time.time()
    n_done = n_skipped = n_failed = 0
    total = len(my_units)
    args_list = [
        (u, snapshots[u["geometry"]][0], snapshots[u["geometry"]][1], str(run_dir))
        for u in my_units
    ]
    # maxtasksperchild=1: each worker process is replaced after every unit, so
    # memory fragmentation / allocator growth from a big solve can never
    # accumulate across the hundreds of units of a long sweep.
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


# ─── entry point ───────────────────────────────────────────────────────────

def main():
    # type: () -> None
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--worker", nargs=3, metavar=("RUN_DIR", "JOB_INDEX", "NODE_INDEX"),
        help="Internal: execute one array-task slice. Invoked by SLURM.",
    )
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
    else:
        submit()


if __name__ == "__main__":
    main()
