#!/usr/bin/env python3
"""
Plot boundary surface currents for a 2D geometry.

Edit the CONFIG block below and run:

    python plot_surface_currents.py

Left panel: geometry colored by |density| with the incidence direction
marked.  Right panel: |density| and phase versus element index (boundary
order).  Uses rcs_solver.compute_surface_currents (same formulation the
RCS dispatch uses; the label is printed on the figure).
"""

import math
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ═══════════════════════════════════════════════════════════════════════════════

GEOMETRY      = "demo_scene copy.geo"   # *.geo file to solve
FREQ_GHZ      = 18.0                # frequency in GHz
ANGLE_DEG     = 0.0                # incidence angle, coming-from (0 = from +x)
POLARIZATION  = "TM"               # TM/TE (or VV/HH aliases)
UNITS         = "inches"           # "inches" or "meters"
OUT_PNG       = None               # None -> "<geometry>_currents.png"
SHOW_WINDOW   = False              # True -> open an interactive window too

# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib
if not SHOW_WINDOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from geometry_io import parse_geometry, build_geometry_snapshot
from rcs_solver import compute_surface_currents


def main() -> None:
    geo_path = Path(GEOMETRY)
    title, segments, ibcs, dielectrics = parse_geometry(geo_path.read_text())
    snap = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    snap["source_path"] = str(geo_path)

    res = compute_surface_currents(
        snap, frequency_ghz=FREQ_GHZ, elevation_deg=ANGLE_DEG,
        polarization=POLARIZATION, geometry_units=UNITS,
        material_base_dir=str(geo_path.parent),
    )

    cx = np.asarray(res["centers_x"]); cy = np.asarray(res["centers_y"])
    nx = np.asarray(res["normals_x"]); ny = np.asarray(res["normals_y"])
    L = np.asarray(res["lengths"])
    mag = np.asarray(res["density_abs"])
    ph = np.asarray(res["density_phase_deg"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # Geometry colored by |density|: draw each element as a segment.
    tx, ty = ny, -nx        # tangent from normal (rotation)
    p0 = np.column_stack([cx - 0.5 * L * tx, cy - 0.5 * L * ty])
    p1 = np.column_stack([cx + 0.5 * L * tx, cy + 0.5 * L * ty])
    segs = np.stack([p0, p1], axis=1)
    lc = LineCollection(segs, cmap="inferno", linewidths=3.0)
    lc.set_array(mag)
    ax1.add_collection(lc)
    fig.colorbar(lc, ax=ax1, label="|density|")
    span = max(np.ptp(cx), np.ptp(cy), 1e-9)
    d = np.radians(ANGLE_DEG)
    x0 = cx.mean() + 0.75 * span * math.cos(d)
    y0 = cy.mean() + 0.75 * span * math.sin(d)
    ax1.annotate("", xy=(cx.mean() + 0.5 * span * math.cos(d),
                         cy.mean() + 0.5 * span * math.sin(d)),
                 xytext=(x0, y0),
                 arrowprops={"arrowstyle": "-|>", "color": "tab:blue", "lw": 2})
    ax1.text(x0, y0, f"  inc {ANGLE_DEG:g}\N{DEGREE SIGN}", color="tab:blue", ha="left")
    ax1.set_xlim(cx.mean() - 0.9 * span, cx.mean() + 0.9 * span)
    ax1.set_ylim(cy.mean() - 0.9 * span, cy.mean() + 0.9 * span)
    ax1.set_aspect("equal")
    ax1.set_title(f"|density| on boundary — {POLARIZATION} {FREQ_GHZ:g} GHz")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")

    idx = np.arange(len(mag))
    ax2.plot(idx, mag, color="tab:red", lw=1.4)
    ax2.set_xlabel("element index (boundary order)")
    ax2.set_ylabel("|density|", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2b = ax2.twinx()
    ax2b.plot(idx, ph, color="tab:gray", lw=0.8, alpha=0.7)
    ax2b.set_ylabel("phase (deg)", color="tab:gray")
    ax2.set_title(res["formulation"], fontsize=9)

    fig.suptitle(f"{geo_path.name}   [{res['element_count']} elements]")
    fig.tight_layout()

    out = OUT_PNG or str(geo_path.with_suffix("")) + "_currents.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}   ({res['formulation']})")
    if SHOW_WINDOW:
        plt.show()


if __name__ == "__main__":
    main()
