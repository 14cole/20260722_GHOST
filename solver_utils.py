"""
Shared utility functions for the 2D RCS solver project.

Provides shared polarization handling, unit conversion, and constants
used by rcs_solver.py and grim_io.py.
"""

C0 = 299_792_458.0
ETA0 = 376.730313668
EPS = 1e-12


def canonical_polarization(label: str | None) -> str:
    """
    Normalize polarization label to 'TM' or 'TE'.

    Convention: 2D geometries are ELEVATION cuts — the out-of-plane (z) axis
    is HORIZONTAL, so E_z (the TM branch) corresponds to horizontal polarization.

    Accepted aliases:
    - TM, HH, H, HORIZONTAL → 'TM'   (E along horizontal z-axis)
    - TE, VV, V, VERTICAL   → 'TE'   (H along z, E has vertical component)
    - None or empty → 'TM' (default)

    Raises ValueError for unrecognized labels.
    """

    text = str(label or '').strip().upper()
    if text in {'TM', 'HH', 'H', 'HORIZONTAL'}:
        return 'TM'
    if text in {'TE', 'VV', 'V', 'VERTICAL'}:
        return 'TE'
    if not text:
        return 'TM'
    raise ValueError(
        f"Unsupported polarization '{label}'. "
        "Use TM/HH/H/HORIZONTAL or TE/VV/V/VERTICAL."
    )


def primary_polarization_alias(label: str) -> str:
    """Return 'HH' for TM, 'VV' for TE (elevation-cut convention)."""
    return 'HH' if canonical_polarization(label) == 'TM' else 'VV'


def polarization_alias_list(label: str) -> list[str]:
    """Return all accepted aliases for the given polarization."""
    canonical = canonical_polarization(label)
    if canonical == 'TM':
        return ['TM', 'HH', 'H', 'HORIZONTAL']
    return ['TE', 'VV', 'V', 'VERTICAL']


def unit_scale_to_meters(units: str) -> float:
    """Convert geometry unit string to meters scale factor."""
    value = (units or '').strip().lower()
    if value in {'inch', 'inches', 'in'}:
        return 0.0254
    if value in {'meter', 'meters', 'm'}:
        return 1.0
    raise ValueError(f"Unsupported geometry units '{units}'. Use inches or meters.")
