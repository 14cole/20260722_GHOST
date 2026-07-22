"""
Analytic sphere scattering references for the BoR solver (Phase 0).

Exact Mie-series RCS of PEC, impedance (Leontovich), homogeneous
dielectric/magnetic, and coated-PEC spheres, in the project's e^{+jwt}
convention (outgoing waves ~ h_n^(2), lossy media Im(eps), Im(mu) <= 0).
See BOR_CONVENTIONS.md.

Design note: modal coefficients are NOT transcribed from textbook tables
(where a single sign slip can hide for years).  Each mode's coefficients
are obtained by solving the small linear system expressing the boundary
conditions directly, using the Debye-potential matching rules:

    TM (electric) modes:  U continuous,  (k_i/eps_i) U' continuous
    TE (magnetic) modes:  U continuous,  (k_i/mu_i)  U' continuous
    PEC surface:          TM -> U' = 0,  TE -> U = 0
    Leontovich surface:   TM -> eta0 * U'/U =  j Z_s   (outward normal)
                          TE -> eta0 * U /U' =  j Z_s

with U the Riccati radial function of each region.  Correctness is pinned
by independent anchors (Rayleigh dipole limits, geometric optics, optical
theorem, degenerate-limit equalities) rather than by trusting any single
formula source.

All public functions return sigma_3D in m^2.
"""

import math
from typing import Tuple

import numpy as np
from scipy import special as sp

C0 = 299_792_458.0
ETA0 = 376.730313668


# ─────────────────────────────────────────────────────────────────────────────
# Riccati–Bessel machinery (complex-capable)
# ─────────────────────────────────────────────────────────────────────────────

def _riccati_psi(nmax: int, z: complex) -> Tuple[np.ndarray, np.ndarray]:
    """psi_n(z) = z j_n(z) and psi_n'(z), n = 0..nmax."""
    n = np.arange(0, nmax + 1)
    # spherical j_n from cylindrical J_{n+1/2} (complex-capable)
    jn = np.sqrt(np.pi / (2.0 * z)) * sp.jv(n + 0.5, z)
    jnm1 = np.empty_like(jn)
    jnm1[1:] = jn[:-1]
    jnm1[0] = np.cos(z) / z            # j_{-1}(z) = cos(z)/z
    psi = z * jn
    dpsi = z * jnm1 - n * jn           # R'(z) = z f_{n-1} - n f_n
    return psi, dpsi


def _riccati_zeta(nmax: int, z: complex) -> Tuple[np.ndarray, np.ndarray]:
    """zeta_n(z) = z y_n(z) and zeta_n'(z), n = 0..nmax."""
    n = np.arange(0, nmax + 1)
    yn = np.sqrt(np.pi / (2.0 * z)) * sp.yv(n + 0.5, z)
    ynm1 = np.empty_like(yn)
    ynm1[1:] = yn[:-1]
    ynm1[0] = np.sin(z) / z            # y_{-1}(z) = sin(z)/z
    zeta = z * yn
    dzeta = z * ynm1 - n * yn
    return zeta, dzeta


def _riccati_xi(nmax: int, z: complex) -> Tuple[np.ndarray, np.ndarray]:
    """xi_n(z) = psi_n - j zeta_n = z h_n^(2)(z) (outgoing, e^{+jwt})."""
    psi, dpsi = _riccati_psi(nmax, z)
    zeta, dzeta = _riccati_zeta(nmax, z)
    return psi - 1j * zeta, dpsi - 1j * dzeta


def _nmax_for(x: float, pad: int = 12) -> int:
    """Wiscombe-style truncation for size parameter x."""
    x = abs(x)
    return max(6, int(math.ceil(x + 4.05 * x ** (1.0 / 3.0))) + pad)


def _causal_index(eps_r: complex, mu_r: complex) -> complex:
    """m = sqrt(eps*mu) with Im(m) <= 0 (matches the 2D solver's branch)."""
    m = np.sqrt(complex(eps_r) * complex(mu_r))
    if m.imag > 0.0:
        m = -m
    if m.real < 0.0:
        m = -m
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Modal coefficients: exterior expansion U = psi_n(x) + A_n xi_n(x) (TM)
#                                        U = psi_n(x) + B_n xi_n(x) (TE)
# per unit modal incident amplitude.  A_n/B_n solved per mode from BCs.
# ─────────────────────────────────────────────────────────────────────────────

def _coeffs_pec(x: float, nmax: int) -> Tuple[np.ndarray, np.ndarray]:
    psi, dpsi = _riccati_psi(nmax, x)
    xi, dxi = _riccati_xi(nmax, x)
    a = -dpsi / dxi        # TM: U'(a) = 0
    b = -psi / xi          # TE: U(a) = 0
    return a, b


def _coeffs_impedance(x: float, zs: complex, nmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leontovich sphere.  Modal equations (outward normal):
        TM: [dpsi + A dxi] = +(j zs/eta0) [psi + A xi]
        TE: [psi + B xi]  = -(j zs/eta0) [dpsi + B dxi]
    The RELATIVE sign between TM and TE is pinned by Weston's theorem:
    a Z_s = eta0 impedance sphere has identically zero backscatter
    (A_n == B_n for every mode).  The same-sign variant violates it by
    +28 dB above PEC (physically impossible for a passive matched
    absorber) — the exact analog of the TE sign bug found in the 2D
    solver.  zs -> 0 reduces both to the PEC conditions.
    """
    psi, dpsi = _riccati_psi(nmax, x)
    xi, dxi = _riccati_xi(nmax, x)
    g = 1j * complex(zs) / ETA0
    a = -(dpsi - g * psi) / (dxi - g * xi)
    b = -(psi + g * dpsi) / (xi + g * dxi)
    return a, b


def _coeffs_dielectric(x: float, eps_r: complex, mu_r: complex, nmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Homogeneous sphere: interior regular U_i = C psi_n(m x_r).
    Continuity (TM): psi + A xi = C psi_m
                     (dpsi + A dxi) = C (m/eps_r) dpsi_m
    TE: same with (m/mu_r).
    """
    m = _causal_index(eps_r, mu_r)
    psi, dpsi = _riccati_psi(nmax, x)
    xi, dxi = _riccati_xi(nmax, x)
    psim, dpsim = _riccati_psi(nmax, m * x)

    wa = m / complex(eps_r)
    wb = m / complex(mu_r)
    # Eliminate C:  (dpsi + A dxi) * psi_m = w * dpsi_m * (psi + A xi)
    a = -(dpsi * psim - wa * dpsim * psi) / (dxi * psim - wa * dpsim * xi)
    b = -(dpsi * psim - wb * dpsim * psi) / (dxi * psim - wb * dpsim * xi)
    return a, b


def _coeffs_coated_pec(x_out: float, x_core_layer: complex, x_out_layer: complex,
                       eps_r: complex, mu_r: complex, nmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    PEC core (radius b) + one layer (outer radius a).
    Layer radial function: U_L = D psi_n(m k0 r) + E zeta_n(m k0 r).
    PEC core:  TM -> U_L'(m k0 b) = 0,  TE -> U_L(m k0 b) = 0.
    Outer interface: continuity as in the dielectric case.
    """
    m = _causal_index(eps_r, mu_r)
    psi, dpsi = _riccati_psi(nmax, x_out)
    xi, dxi = _riccati_xi(nmax, x_out)
    psiL_a, dpsiL_a = _riccati_psi(nmax, x_out_layer)
    zetaL_a, dzetaL_a = _riccati_zeta(nmax, x_out_layer)
    psiL_b, dpsiL_b = _riccati_psi(nmax, x_core_layer)
    zetaL_b, dzetaL_b = _riccati_zeta(nmax, x_core_layer)

    # TM: core gives E = -D dpsiL_b / dzetaL_b  -> layer combo functions
    fa = psiL_a - (dpsiL_b / dzetaL_b) * zetaL_a
    dfa = dpsiL_a - (dpsiL_b / dzetaL_b) * dzetaL_a
    wa = m / complex(eps_r)
    a = -(dpsi * fa - wa * dfa * psi) / (dxi * fa - wa * dfa * xi)

    # TE: core gives E = -D psiL_b / zetaL_b
    fb = psiL_a - (psiL_b / zetaL_b) * zetaL_a
    dfb = dpsiL_a - (psiL_b / zetaL_b) * dzetaL_a
    wb = m / complex(mu_r)
    b = -(dpsi * fb - wb * dfb * psi) / (dxi * fb - wb * dfb * xi)
    return a, b


# ─────────────────────────────────────────────────────────────────────────────
# Cross sections from modal coefficients
# ─────────────────────────────────────────────────────────────────────────────

def _backscatter_sigma(k: float, a_n: np.ndarray, b_n: np.ndarray) -> float:
    """sigma_back = (lambda^2/4pi) |sum (-1)^n (2n+1)(A_n - B_n)|^2, n>=1."""
    n = np.arange(1, len(a_n))
    s = np.sum(((-1.0) ** n) * (2 * n + 1) * (a_n[1:] - b_n[1:]))
    lam = 2.0 * math.pi / k
    return float((lam ** 2 / (4.0 * math.pi)) * abs(s) ** 2)


def _sigma_scat_ext(k: float, a_n: np.ndarray, b_n: np.ndarray) -> Tuple[float, float]:
    """Total scattering and extinction cross sections (m^2)."""
    n = np.arange(1, len(a_n))
    w = 2 * n + 1
    sca = (2.0 * math.pi / k ** 2) * np.sum(w * (np.abs(a_n[1:]) ** 2 + np.abs(b_n[1:]) ** 2))
    ext = -(2.0 * math.pi / k ** 2) * np.sum(w * np.real(a_n[1:] + b_n[1:]))
    return float(sca), float(ext)


def _pi_tau(nmax: int, cos_t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Angular functions pi_n, tau_n (n = 0..nmax) by recurrence."""
    pi = np.zeros(nmax + 1)
    tau = np.zeros(nmax + 1)
    pi[1] = 1.0
    tau[1] = cos_t
    for n in range(2, nmax + 1):
        pi[n] = ((2 * n - 1) * cos_t * pi[n - 1] - n * pi[n - 2]) / (n - 1)
        tau[n] = n * cos_t * pi[n] - (n + 1) * pi[n - 1]
    return pi, tau


def _bistatic_sigma(k: float, a_n: np.ndarray, b_n: np.ndarray, theta_bis_deg: float) -> Tuple[float, float]:
    """
    Bistatic (sigma_VV, sigma_HH) at bistatic angle theta (180 deg = back).
    S2 pairs A with tau (E-plane), S1 pairs A with pi (H-plane).
    """
    nmax = len(a_n) - 1
    ct = math.cos(math.radians(theta_bis_deg))
    pi_n, tau_n = _pi_tau(nmax, ct)
    n = np.arange(1, nmax + 1)
    w = (2 * n + 1) / (n * (n + 1))
    s1 = np.sum(w * (a_n[1:] * pi_n[1:] + b_n[1:] * tau_n[1:]))
    s2 = np.sum(w * (a_n[1:] * tau_n[1:] + b_n[1:] * pi_n[1:]))
    # sigma = 4 pi |S/k|^2 = (lambda^2/pi) |S|^2.  (At theta=180 the pi/tau
    # values are +-n(n+1)/2, so S = (1/2) * the monostatic sum and this
    # prefactor reproduces the (lambda^2/4pi)|sum|^2 backscatter formula.)
    lam = 2.0 * math.pi / k
    pref = lam ** 2 / math.pi
    return float(pref * abs(s2) ** 2), float(pref * abs(s1) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Public API (sigma in m^2)
# ─────────────────────────────────────────────────────────────────────────────

def sigma_pec_sphere(radius_m: float, freq_hz: float) -> float:
    k = 2.0 * math.pi * freq_hz / C0
    x = k * radius_m
    a, b = _coeffs_pec(x, _nmax_for(x))
    return _backscatter_sigma(k, a, b)


def sigma_impedance_sphere(radius_m: float, freq_hz: float, zs_ohm: complex) -> float:
    k = 2.0 * math.pi * freq_hz / C0
    x = k * radius_m
    a, b = _coeffs_impedance(x, zs_ohm, _nmax_for(x))
    return _backscatter_sigma(k, a, b)


def sigma_dielectric_sphere(radius_m: float, eps_r: complex, mu_r: complex, freq_hz: float) -> float:
    k = 2.0 * math.pi * freq_hz / C0
    x = k * radius_m
    m = _causal_index(eps_r, mu_r)
    nmax = _nmax_for(max(x, abs(m) * x))
    a, b = _coeffs_dielectric(x, eps_r, mu_r, nmax)
    return _backscatter_sigma(k, a, b)


def sigma_coated_pec_sphere(core_radius_m: float, outer_radius_m: float,
                            eps_r: complex, mu_r: complex, freq_hz: float) -> float:
    if not (0.0 < core_radius_m < outer_radius_m):
        raise ValueError("Require 0 < core radius < outer radius.")
    k = 2.0 * math.pi * freq_hz / C0
    x = k * outer_radius_m
    m = _causal_index(eps_r, mu_r)
    nmax = _nmax_for(max(x, abs(m) * x))
    a, b = _coeffs_coated_pec(x, m * k * core_radius_m, m * k * outer_radius_m,
                              eps_r, mu_r, nmax)
    return _backscatter_sigma(k, a, b)


def _coeffs_multilayer_pec(radii_m, eps_list, mu_list, k0: float,
                           nmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    PEC core + L concentric layers.  radii_m = [b, r1, ..., rL] (core radius
    then each layer's OUTER radius, ascending); eps_list/mu_list per layer,
    inner to outer.  Per mode the radial function is
        layer j:   U_j = D_j psi_n(k_j r) + E_j zeta_n(k_j r)
        exterior:  U_0 = psi_n(k0 r) + A xi_n(k0 r)
    matched with the Debye rules (TM: U and (k/eps) U' continuous; TE with
    mu; PEC core: TM U' = 0, TE U = 0) and solved per mode as a small
    linear system — same solve-from-BCs discipline as every other
    coefficient set in this module.
    """

    L = len(eps_list)
    if len(radii_m) != L + 1 or len(mu_list) != L:
        raise ValueError("radii_m must be [core, r1..rL]; eps/mu one per layer.")
    if any(radii_m[i] >= radii_m[i + 1] for i in range(L)):
        raise ValueError("Layer radii must be strictly increasing.")
    ks = [k0 * _causal_index(e, u) for e, u in zip(eps_list, mu_list)]

    a_n = np.zeros(nmax + 1, dtype=complex)
    b_n = np.zeros(nmax + 1, dtype=complex)
    # Riccati tables per (layer, radius) evaluated once for all n.
    psi_t, dpsi_t, zeta_t, dzeta_t = {}, {}, {}, {}
    for j in range(L):
        for r in (radii_m[j], radii_m[j + 1]):
            z = ks[j] * r
            psi_t[(j, r)], dpsi_t[(j, r)] = _riccati_psi(nmax, z)
            zeta_t[(j, r)], dzeta_t[(j, r)] = _riccati_zeta(nmax, z)
    a_out = radii_m[-1]
    psi0, dpsi0 = _riccati_psi(nmax, k0 * a_out)
    xi0, dxi0 = _riccati_xi(nmax, k0 * a_out)

    for pol in ("TM", "TE"):
        wgt = [ks[j] / (k0 * complex(eps_list[j] if pol == "TM" else mu_list[j]))
               for j in range(L)]
        for n in range(1, nmax + 1):
            # unknowns: [D_1, E_1, ..., D_L, E_L, A]
            N = 2 * L + 1
            Amat = np.zeros((N, N), dtype=complex)
            rhs = np.zeros(N, dtype=complex)
            row = 0
            # PEC core in layer 1
            b_r = radii_m[0]
            if pol == "TM":
                Amat[row, 0] = dpsi_t[(0, b_r)][n]
                Amat[row, 1] = dzeta_t[(0, b_r)][n]
            else:
                Amat[row, 0] = psi_t[(0, b_r)][n]
                Amat[row, 1] = zeta_t[(0, b_r)][n]
            row += 1
            # interior layer interfaces r_j between layer j and j+1
            for j in range(L - 1):
                r = radii_m[j + 1]
                Amat[row, 2 * j] = psi_t[(j, r)][n]
                Amat[row, 2 * j + 1] = zeta_t[(j, r)][n]
                Amat[row, 2 * j + 2] = -psi_t[(j + 1, r)][n]
                Amat[row, 2 * j + 3] = -zeta_t[(j + 1, r)][n]
                row += 1
                Amat[row, 2 * j] = wgt[j] * dpsi_t[(j, r)][n]
                Amat[row, 2 * j + 1] = wgt[j] * dzeta_t[(j, r)][n]
                Amat[row, 2 * j + 2] = -wgt[j + 1] * dpsi_t[(j + 1, r)][n]
                Amat[row, 2 * j + 3] = -wgt[j + 1] * dzeta_t[(j + 1, r)][n]
                row += 1
            # outer interface r_L: layer L <-> exterior (unit incident)
            r = a_out
            jL = L - 1
            Amat[row, 2 * jL] = psi_t[(jL, r)][n]
            Amat[row, 2 * jL + 1] = zeta_t[(jL, r)][n]
            Amat[row, N - 1] = -xi0[n]
            rhs[row] = psi0[n]
            row += 1
            Amat[row, 2 * jL] = wgt[jL] * dpsi_t[(jL, r)][n]
            Amat[row, 2 * jL + 1] = wgt[jL] * dzeta_t[(jL, r)][n]
            Amat[row, N - 1] = -dxi0[n]
            rhs[row] = dpsi0[n]
            sol = np.linalg.solve(Amat, rhs)
            if pol == "TM":
                a_n[n] = sol[-1]
            else:
                b_n[n] = sol[-1]
    return a_n, b_n


def sigma_multilayer_pec_sphere(radii_m, eps_list, mu_list,
                                freq_hz: float) -> float:
    """Backscatter RCS (m^2) of a PEC sphere under N concentric coating
    layers.  radii_m = [core, r1, ..., rN] ascending; eps/mu inner->outer."""

    k = 2.0 * math.pi * freq_hz / C0
    x = k * radii_m[-1]
    scale = max([abs(_causal_index(e, u)) for e, u in zip(eps_list, mu_list)] + [1.0])
    a, b = _coeffs_multilayer_pec(list(map(float, radii_m)),
                                  [complex(e) for e in eps_list],
                                  [complex(u) for u in mu_list],
                                  k, _nmax_for(scale * x))
    return _backscatter_sigma(k, a, b)


def sigma_bistatic_pec_sphere(radius_m: float, freq_hz: float, theta_bis_deg: float) -> Tuple[float, float]:
    """(sigma_VV, sigma_HH) at bistatic angle theta (180 = backscatter)."""
    k = 2.0 * math.pi * freq_hz / C0
    x = k * radius_m
    a, b = _coeffs_pec(x, _nmax_for(x))
    return _bistatic_sigma(k, a, b, theta_bis_deg)


def cross_sections_dielectric_sphere(radius_m: float, eps_r: complex, mu_r: complex,
                                     freq_hz: float) -> Tuple[float, float]:
    """(sigma_scattering, sigma_extinction) in m^2 — optical-theorem gates."""
    k = 2.0 * math.pi * freq_hz / C0
    x = k * radius_m
    m = _causal_index(eps_r, mu_r)
    nmax = _nmax_for(max(x, abs(m) * x))
    a, b = _coeffs_dielectric(x, eps_r, mu_r, nmax)
    return _sigma_scat_ext(k, a, b)
