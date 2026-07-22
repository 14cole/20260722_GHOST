"""
BoR modal kernel machinery (Phase 1).

Provides the generatrix discretization and the modal Green's functions

    G_m(p, q) = Int_{-pi}^{pi}  e^{-jkR(xi)} / (4 pi R(xi)) * e^{-jm xi} dxi
    R(xi)^2   = (rho_p - rho_q)^2 + (z_p - z_q)^2 + 4 rho_p rho_q sin^2(xi/2)

in the project's e^{+jwt} convention (see BOR_CONVENTIONS.md).  The
cos(xi)- and sin(xi)-weighted kernels needed by the EFIE dyad follow from
neighbors:  Gc_m = (G_{m-1}+G_{m+1})/2,  Gs_m = (G_{m-1}-G_{m+1})/(2j).

Evaluation strategy:
- Far point pairs: uniform xi sampling + FFT — all modes at once,
  spectrally accurate for the periodic smooth integrand.
- Near point pairs (R can be small): sinh-graded quadrature concentrated
  at xi = 0, evaluated for all modes by direct projection.

Both paths are validated against adaptive reference integration in the
phase-1 gate battery before any solver uses them.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

C0 = 299_792_458.0
ETA0 = 376.730313668
AXIS_TOL = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Generatrix
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Generatrix:
    """Polyline generatrix in the (rho, z) half-plane, rho >= 0.

    Convention (BOR_CONVENTIONS.md): traversed so the left-of-travel normal
    (-z', rho') points into the exterior (air).  For a closed body that
    means from the +z axis end to the -z axis end (sphere: north pole ->
    south pole).
    """

    nodes: np.ndarray            # (Nn, 2) columns rho, z
    elem_n0: np.ndarray = field(init=False)
    elem_n1: np.ndarray = field(init=False)
    lengths: np.ndarray = field(init=False)
    trho: np.ndarray = field(init=False)   # d rho / dt per element
    tz: np.ndarray = field(init=False)     # d z / dt per element

    def __post_init__(self):
        pts = np.asarray(self.nodes, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            raise ValueError("Generatrix needs an (Nn, 2) array of (rho, z) nodes.")
        if np.any(pts[:, 0] < -1e-12):
            raise ValueError("Generatrix rho coordinates must be >= 0.")
        pts[:, 0] = np.maximum(pts[:, 0], 0.0)
        self.nodes = pts
        d = pts[1:] - pts[:-1]
        self.lengths = np.hypot(d[:, 0], d[:, 1])
        if np.any(self.lengths <= 0):
            raise ValueError("Generatrix has a zero-length element.")
        self.trho = d[:, 0] / self.lengths
        self.tz = d[:, 1] / self.lengths
        self.elem_n0 = np.arange(len(self.lengths))
        self.elem_n1 = self.elem_n0 + 1

    @property
    def n_elems(self) -> int:
        return len(self.lengths)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def node_on_axis(self, i: int) -> bool:
        return self.nodes[i, 0] <= AXIS_TOL * max(1.0, float(np.max(self.nodes[:, 0])))


@dataclass
class GaussData:
    """Per-Gauss-point geometry over the whole generatrix."""

    elem: np.ndarray        # element index per point
    s: np.ndarray           # local coordinate in [0,1]
    w: np.ndarray           # weight * element length (integrates dt)
    rho: np.ndarray
    z: np.ndarray
    trho: np.ndarray        # element tangent components at the point
    tz: np.ndarray
    # nodal shape data: T0/T1 = shape values of the element's two nodes,
    # dT0/dT1 = d/dt of (rho * T) for the two nodes (for surface divergence)
    T0: np.ndarray
    T1: np.ndarray
    dRT0: np.ndarray
    dRT1: np.ndarray


def gauss_on_generatrix(gen: Generatrix, order: int = 4) -> GaussData:
    xg, wg = np.polynomial.legendre.leggauss(order)
    s = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    ne = gen.n_elems
    E = np.repeat(np.arange(ne), order)
    S = np.tile(s, ne)
    W = np.tile(w, ne) * np.repeat(gen.lengths, order)
    r0 = gen.nodes[gen.elem_n0]
    r1 = gen.nodes[gen.elem_n1]
    RHO = np.repeat(r0[:, 0], order) + S * np.repeat(r1[:, 0] - r0[:, 0], order)
    Z = np.repeat(r0[:, 1], order) + S * np.repeat(r1[:, 1] - r0[:, 1], order)
    TR = np.repeat(gen.trho, order)
    TZ = np.repeat(gen.tz, order)
    L = np.repeat(gen.lengths, order)
    T0 = 1.0 - S
    T1 = S
    # d/dt [rho(t) T(t)]  with rho linear on the element:
    #   for T = 1-s:  d/ds[rho(s)(1-s)]/L ;  for T = s: d/ds[rho(s)s]/L
    drho_ds = np.repeat(r1[:, 0] - r0[:, 0], order)
    dRT0 = (drho_ds * (1.0 - S) - RHO) / L
    dRT1 = (drho_ds * S + RHO) / L
    return GaussData(E, S, W, RHO, Z, TR, TZ, T0, T1, dRT0, dRT1)


# ─────────────────────────────────────────────────────────────────────────────
# Modal kernels
# ─────────────────────────────────────────────────────────────────────────────

def modal_kernels_fft(rho_p, z_p, rho_q, z_q, k, m_max: int, n_xi: int = 0):
    """
    G_m for m = 0..m_max+1 at point pairs via uniform xi sampling + FFT.

    Inputs are broadcastable arrays of pair coordinates.  Returns complex
    array [..., m_max+2] (extra order for the Gc/Gs neighbor relations;
    negative m follow from G_{-m} = G_m).

    Accuracy: the integrand is periodic and smooth when the pair is not
    near-singular; trapezoid/FFT is then spectrally accurate.  Callers must
    route near pairs to modal_kernels_near.
    """

    rho_p = np.asarray(rho_p, dtype=float)
    rho_q = np.asarray(rho_q, dtype=float)
    z_p = np.asarray(z_p, dtype=float)
    z_q = np.asarray(z_q, dtype=float)
    if n_xi <= 0:
        # resolve both the mode count and the kernel oscillation k*2*sqrt(rr')
        osc = float(np.max(2.0 * abs(k) * np.sqrt(np.maximum(rho_p * rho_q, 0.0)))) if rho_p.size else 0.0
        n_xi = int(2 ** math.ceil(math.log2(max(64, 4 * (m_max + 2), 6 * (osc + 4)))))
    xi = 2.0 * np.pi * np.arange(n_xi) / n_xi - np.pi   # uniform on [-pi, pi)
    sin2 = np.sin(0.5 * xi) ** 2
    d2 = (rho_p - rho_q) ** 2 + (z_p - z_q) ** 2
    R = np.sqrt(d2[..., None] + 4.0 * (rho_p * rho_q)[..., None] * sin2)
    # Coincident points appear only in near-classified pairs, whose table
    # entries the caller zeroes; clamp R so they produce garbage quietly
    # instead of warnings.
    R = np.maximum(R, 1e-300)
    g = np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R)
    # G_m = int g e^{-jm xi} dxi  ~ (2pi/N) sum g(xi_l) e^{-jm xi_l}
    # with xi_l = -pi + 2pi l / N:  e^{-jm xi_l} = e^{+jm pi} e^{-j 2pi m l/N}
    spec = np.fft.fft(g, axis=-1)                        # sum g_l e^{-j2pi m l/N}
    m = np.arange(m_max + 2)
    phase = np.exp(1j * np.pi * m) * (2.0 * np.pi / n_xi)
    return spec[..., : m_max + 2] * phase


def modal_kernels_near(rho_p, z_p, rho_q, z_q, k, m_max: int, order: int = 48):
    """
    G_m for m = 0..m_max+1 at near-singular point pairs.

    Substitution xi = 2 asin(s), then s = s_scale * sinh(v): concentrates
    quadrature at xi = 0 where R -> d.  Handles d down to ~1e-12 * rho.
    Inputs are 1-D arrays of pair coordinates (n_pairs,).
    Returns [n_pairs, m_max+2].
    """

    rho_p = np.atleast_1d(np.asarray(rho_p, dtype=float))
    rho_q = np.atleast_1d(np.asarray(rho_q, dtype=float))
    z_p = np.atleast_1d(np.asarray(z_p, dtype=float))
    z_q = np.atleast_1d(np.asarray(z_q, dtype=float))
    n = rho_p.size
    d2 = (rho_p - rho_q) ** 2 + (z_p - z_q) ** 2
    rr4 = 4.0 * rho_p * rho_q

    out = np.zeros((n, m_max + 2), dtype=np.complex128)
    m = np.arange(m_max + 2)

    # Pairs with rho_p*rho_q ~ 0 (a point on the axis): R is xi-independent.
    on_axis = rr4 <= 1e-30
    if np.any(on_axis):
        R0 = np.sqrt(d2[on_axis])
        g0 = np.exp(-1j * complex(k) * R0) / (4.0 * np.pi * np.maximum(R0, 1e-300))
        # int e^{-jm xi} dxi = 2pi delta_m0
        out[on_axis, 0] = 2.0 * np.pi * g0

    idx = np.flatnonzero(~on_axis)
    if idx.size == 0:
        return out

    d = np.sqrt(np.maximum(d2[idx], 1e-300))
    a = np.sqrt(rr4[idx])                    # R^2 = d^2 + a^2 s^2, s = sin(xi/2)

    # Two-piece quadrature (both pieces evaluated for every pair; the core
    # collapses to zero width when the pair is not actually singular):
    #   core:  s in [0, s0], s0 = min(1, 20 d/a), sinh-graded toward s = 0 —
    #          resolves the 1/R spike;
    #   tail:  xi in [xi0, pi], plain Gauss with order scaled to the
    #          cos(m xi) and e^{-jkR} oscillation — the single sinh map
    #          starves this region and capped accuracy at ~1e-2.
    # Core must stay a NARROW singular patch: if it grows to cover the whole
    # range, the fixed-order sinh grid cannot resolve the cos(m xi)/e^{-jkR}
    # oscillation (that job belongs to the oscillation-scaled tail).
    s0 = np.minimum(0.25, 20.0 * d / a)
    xg, wg = np.polynomial.legendre.leggauss(order)
    u01 = 0.5 * (xg + 1.0)
    w01 = 0.5 * wg

    # ── core ──
    vmax = np.arcsinh((a / d) * s0)
    v = u01[None, :] * vmax[:, None]
    wv = w01[None, :] * vmax[:, None]
    s = (d / a)[:, None] * np.sinh(v)
    s = np.minimum(s, 1.0)
    xi = 2.0 * np.arcsin(s)
    R = np.sqrt(d[:, None] ** 2 + (a[:, None] * s) ** 2)
    ds_dv = (d / a)[:, None] * np.cosh(v)
    dxi_dv = 2.0 * ds_dv / np.sqrt(np.maximum(1.0 - s ** 2, 1e-15))
    g = np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R)
    w_all = wv * dxi_dv
    cosmx = np.cos(xi[:, :, None] * m[None, None, :])
    gw = g * w_all
    acc = 2.0 * (np.einsum("pv,pvm->pm", gw.real, cosmx)
                 + 1j * np.einsum("pv,pvm->pm", gw.imag, cosmx))

    # ── tail ──
    osc = float(np.max(abs(complex(k)) * a)) / math.pi + (m_max + 2)
    n_tail = int(min(1024, max(64, math.ceil(4.0 * osc))))
    xt, wt = np.polynomial.legendre.leggauss(n_tail)
    u01t = 0.5 * (xt + 1.0)
    w01t = 0.5 * wt
    xi0 = 2.0 * np.arcsin(s0)
    span = np.pi - xi0
    xi_t = xi0[:, None] + u01t[None, :] * span[:, None]
    w_t = w01t[None, :] * span[:, None]
    st = np.sin(0.5 * xi_t)
    Rt = np.sqrt(d[:, None] ** 2 + (a[:, None] * st) ** 2)
    gt = np.exp(-1j * complex(k) * Rt) / (4.0 * np.pi * Rt)
    cosmt = np.cos(xi_t[:, :, None] * m[None, None, :])
    gtw = gt * w_t
    acc += 2.0 * (np.einsum("pv,pvm->pm", gtw.real, cosmt)
                  + 1j * np.einsum("pv,pvm->pm", gtw.imag, cosmt))

    out[idx, :] = acc
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MFIE kernels.
#
# The MFIE K-term integrand  -p(R) [ (W_u.Rvec)(n.f_v) - (W_u.f_v)(n.Rvec) ]
# with p(R) = (1+jkR) e^{-jkR} / (4 pi R^3) is only weakly singular AS A
# WHOLE: its separate pieces diverge like 1/R^3.  Tabulating separate modal
# kernels and composing them would therefore lose all precision near the
# diagonal (catastrophic cancellation), so the full bracket is built INSIDE
# the azimuthal integrand and modes are projected from the assembled
# function.  Component pairs uv in {tt, tf, ft, ff}; the bracket has mixed
# parity in xi, so tables span m = -m_max..+m_max (centered index m + m_max).
# ─────────────────────────────────────────────────────────────────────────────

def _mfie_brackets(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k, xi):
    """The four MFIE bracket functions at azimuth offsets xi.

    Point arrays have shape S; xi has shape X; returns four arrays S+X.
    Test point at phi = 0; source at phi' = -xi.  n_hat = (-tz, 0, tr)
    (outward per the generatrix convention)."""

    cx, sx = np.cos(xi), np.sin(xi)
    Rx = rho_p[..., None] - rho_q[..., None] * cx
    Ry = rho_q[..., None] * sx
    Rz = (z_p - z_q)[..., None] + 0.0 * cx
    R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)
    R = np.maximum(R, 1e-300)
    p = (1.0 + 1j * complex(k) * R) * np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R ** 3)

    WtR = tr_p[..., None] * Rx + tz_p[..., None] * Rz
    WfR = Ry
    nR = -tz_p[..., None] * Rx + tr_p[..., None] * Rz
    n_tq = -(tz_p * tr_q)[..., None] * cx + (tr_p * tz_q)[..., None]
    n_fq = -tz_p[..., None] * sx
    Wt_tq = (tr_p * tr_q)[..., None] * cx + (tz_p * tz_q)[..., None]
    Wt_fq = tr_p[..., None] * sx
    Wf_tq = -tr_q[..., None] * sx
    Wf_fq = cx + 0.0 * Rx

    Ftt = -p * (WtR * n_tq - Wt_tq * nR)
    Ftf = -p * (WtR * n_fq - Wt_fq * nR)
    Fft = -p * (WfR * n_tq - Wf_tq * nR)
    Fff = -p * (WfR * n_fq - Wf_fq * nR)
    return Ftt, Ftf, Fft, Fff


def mfie_kernels_fft(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k,
                     m_max: int, n_xi: int = 0):
    """Modal MFIE kernels K_uv[..., m + m_max] for m = -m_max..m_max
    (far point pairs; FFT over uniform xi)."""

    rho_p = np.asarray(rho_p, dtype=float)
    if n_xi <= 0:
        # 1/R^3 has broader spectral content than 1/R: denser grid than the
        # single-layer FFT path.
        osc = float(np.max(2.0 * abs(k) * np.sqrt(np.maximum(rho_p * rho_q, 0.0)))) if rho_p.size else 0.0
        n_xi = int(2 ** math.ceil(math.log2(max(128, 6 * (m_max + 2), 8 * (osc + 4)))))
    xi = 2.0 * np.pi * np.arange(n_xi) / n_xi - np.pi
    Fs = _mfie_brackets(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k, xi)
    m = np.arange(-m_max, m_max + 1)
    out = []
    for F in Fs:
        spec = np.fft.fft(F, axis=-1)
        # int F e^{-jm xi} dxi with xi_l = -pi + 2pi l/N:
        #   m >= 0: bin m;  m < 0: bin N+m.  Common phase e^{jm pi} 2pi/N.
        bins = np.where(m >= 0, m, n_xi + m)
        phase = np.exp(1j * np.pi * m) * (2.0 * np.pi / n_xi)
        out.append(spec[..., bins] * phase)
    return tuple(out)


def _project_pm_brackets(Fp, Fm, w_pos, xi_pos, m) -> List[np.ndarray]:
    """Project half-range +-xi bracket samples onto modes:

        proj_m = int_0^pi [F(+xi) e^{-jm xi} + F(-xi) e^{+jm xi}] dxi
               = int_0^pi [S cos(m xi) - j D sin(m xi)] dxi,
        S = F(+xi) + F(-xi),   D = F(+xi) - F(-xi)   (weights folded in).

    Splitting into real cos/sin einsums costs ~4x fewer flops than the
    complex-exponential form and shares the trig tables across brackets.
    Returns one [n_pairs, len(m)] array per bracket."""

    arg = xi_pos[:, :, None] * m[None, None, :]
    cosm = np.cos(arg)
    sinm = np.sin(arg)
    outs = []
    for Fpos, Fneg in zip(Fp, Fm):
        S = (Fpos + Fneg) * w_pos
        D = (Fpos - Fneg) * w_pos
        pc = np.einsum("pv,pvm->pm", S.real, cosm) + \
             1j * np.einsum("pv,pvm->pm", S.imag, cosm)
        ps = np.einsum("pv,pvm->pm", D.real, sinm) + \
             1j * np.einsum("pv,pvm->pm", D.imag, sinm)
        outs.append(pc - 1j * ps)
    return outs


def mfie_kernels_near(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k,
                      m_max: int, order: int = 48):
    """Modal MFIE kernels for near point pairs (1-D pair lists) via the
    same capped sinh core + oscillation tail as modal_kernels_near, mirrored
    to negative xi (the brackets have mixed parity).  Returns four arrays
    [n_pairs, 2*m_max+1]."""

    rho_p = np.atleast_1d(np.asarray(rho_p, dtype=float))
    rho_q = np.atleast_1d(np.asarray(rho_q, dtype=float))
    z_p = np.atleast_1d(np.asarray(z_p, dtype=float))
    z_q = np.atleast_1d(np.asarray(z_q, dtype=float))
    tr_p = np.broadcast_to(np.asarray(tr_p, dtype=float), rho_p.shape)
    tz_p = np.broadcast_to(np.asarray(tz_p, dtype=float), rho_p.shape)
    tr_q = np.broadcast_to(np.asarray(tr_q, dtype=float), rho_q.shape)
    tz_q = np.broadcast_to(np.asarray(tz_q, dtype=float), rho_q.shape)

    d2 = (rho_p - rho_q) ** 2 + (z_p - z_q) ** 2
    rr4 = 4.0 * rho_p * rho_q
    d = np.sqrt(np.maximum(d2, 1e-300))
    a = np.sqrt(np.maximum(rr4, 1e-300))
    s0 = np.minimum(0.25, 20.0 * d / np.maximum(a, 1e-300))
    s0 = np.where(rr4 <= 1e-30, 1.0, s0)   # axis pairs: no singular core structure

    xg, wg = np.polynomial.legendre.leggauss(order)
    u01 = 0.5 * (xg + 1.0)
    w01 = 0.5 * wg
    # core in s = sin(xi/2), sinh-graded
    vmax = np.arcsinh((a / d) * s0)
    v = u01[None, :] * vmax[:, None]
    wv = w01[None, :] * vmax[:, None]
    s = np.minimum((d / a)[:, None] * np.sinh(v), 1.0)
    xi_c = 2.0 * np.arcsin(s)
    ds_dv = (d / a)[:, None] * np.cosh(v)
    w_c = wv * 2.0 * ds_dv / np.sqrt(np.maximum(1.0 - s ** 2, 1e-15))
    # tail
    osc = float(np.max(abs(complex(k)) * a)) / math.pi + (m_max + 2)
    n_tail = int(min(1024, max(64, math.ceil(4.0 * osc))))
    xt, wt = np.polynomial.legendre.leggauss(n_tail)
    xi0 = 2.0 * np.arcsin(np.minimum(s0, 1.0))
    span = np.pi - xi0
    xi_t = xi0[:, None] + 0.5 * (xt + 1.0)[None, :] * span[:, None]
    w_t = 0.5 * wt[None, :] * span[:, None]

    xi_pos = np.concatenate([xi_c, xi_t], axis=1)      # [n_pairs, nq]
    w_pos = np.concatenate([w_c, w_t], axis=1)
    m = np.arange(-m_max, m_max + 1)
    outs = []

    # Per-pair xi grids: pointwise broadcast form of _mfie_brackets.
    def brackets_grid(xi):
        cx, sx = np.cos(xi), np.sin(xi)
        Rx = rho_p[:, None] - rho_q[:, None] * cx
        Ry = rho_q[:, None] * sx
        Rz = (z_p - z_q)[:, None] * np.ones_like(cx)
        R = np.maximum(np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2), 1e-300)
        p = (1.0 + 1j * complex(k) * R) * np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R ** 3)
        WtR = tr_p[:, None] * Rx + tz_p[:, None] * Rz
        WfR = Ry
        nR = -tz_p[:, None] * Rx + tr_p[:, None] * Rz
        n_tq = -(tz_p * tr_q)[:, None] * cx + (tr_p * tz_q)[:, None] * np.ones_like(cx)
        n_fq = -tz_p[:, None] * sx
        Wt_tq = (tr_p * tr_q)[:, None] * cx + (tz_p * tz_q)[:, None] * np.ones_like(cx)
        Wt_fq = tr_p[:, None] * sx
        Wf_tq = -tr_q[:, None] * sx
        Wf_fq = cx * np.ones_like(Rx)
        return (-p * (WtR * n_tq - Wt_tq * nR), -p * (WtR * n_fq - Wt_fq * nR),
                -p * (WfR * n_tq - Wf_tq * nR), -p * (WfR * n_fq - Wf_fq * nR))

    Fp = brackets_grid(xi_pos)
    Fm = brackets_grid(-xi_pos)
    outs = _project_pm_brackets(Fp, Fm, w_pos, xi_pos, m)
    return tuple(outs)


def mfie_for_mode(K: np.ndarray, m: int, m_max: int) -> np.ndarray:
    """Extract mode m (centered table, index m + m_max)."""
    return K[..., m + m_max]


# ─────────────────────────────────────────────────────────────────────────────
# IBC kernels: the magnetic-current operator of the IBC-EFIE.
#
# Eliminating M = -Z_s n_hat' x J' gives the extra operator
#     + p(R) [ (W_u . n_hat_q)(Rvec . f_v) - (W_u . f_v)(Rvec . n_hat_q) ]
# (source-point normal, Z_s applied at the source during assembly).  Same
# only-weakly-singular-as-a-whole structure as the MFIE bracket, so the same
# assembled-integrand strategy applies.
# ─────────────────────────────────────────────────────────────────────────────

def _ibc_brackets_grid(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k, xi):
    """Four IBC bracket functions on per-pair xi grids ([n_pairs, n_xi])."""

    cx, sx = np.cos(xi), np.sin(xi)
    Rx = rho_p[:, None] - rho_q[:, None] * cx
    Ry = rho_q[:, None] * sx
    Rz = (z_p - z_q)[:, None] * np.ones_like(cx)
    R = np.maximum(np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2), 1e-300)
    p = (1.0 + 1j * complex(k) * R) * np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R ** 3)

    # source normal (outward): n_q = (-tz_q, 0, tr_q) rotated to phi' = -xi
    # dot table (test at phi = 0):
    Wt_nq = -(tr_p * tz_q)[:, None] * cx + (tz_p * tr_q)[:, None] * np.ones_like(cx)
    Wf_nq = tz_q[:, None] * sx
    R_tq = tr_q[:, None] * (rho_p[:, None] * cx - rho_q[:, None]) + tz_q[:, None] * Rz
    R_fq = rho_p[:, None] * sx
    R_nq = -tz_q[:, None] * (rho_p[:, None] * cx - rho_q[:, None]) + tr_q[:, None] * Rz
    Wt_tq = (tr_p * tr_q)[:, None] * cx + (tz_p * tz_q)[:, None] * np.ones_like(cx)
    Wt_fq = tr_p[:, None] * sx
    Wf_tq = -tr_q[:, None] * sx
    Wf_fq = cx * np.ones_like(Rx)

    Btt = p * (Wt_nq * R_tq - Wt_tq * R_nq)
    Btf = p * (Wt_nq * R_fq - Wt_fq * R_nq)
    Bft = p * (Wf_nq * R_tq - Wf_tq * R_nq)
    Bff = p * (Wf_nq * R_fq - Wf_fq * R_nq)
    return Btt, Btf, Bft, Bff


def ibc_kernels_fft(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k,
                    m_max: int, n_xi: int = 0):
    """Modal IBC kernels [Pp, Pq, 2*m_max+1] via FFT (far point pairs).
    Test/source point arrays are [Pp]/[Pq] vectors; all pair combinations
    are formed here."""

    Pp = len(np.atleast_1d(rho_p))
    Pq = len(np.atleast_1d(rho_q))
    if n_xi <= 0:
        osc = float(np.max(2.0 * abs(k) * np.sqrt(np.maximum(
            np.outer(rho_p, rho_q), 0.0))))
        n_xi = int(2 ** math.ceil(math.log2(max(128, 6 * (m_max + 2), 8 * (osc + 4)))))
    xi = 2.0 * np.pi * np.arange(n_xi) / n_xi - np.pi
    one = np.ones((Pp, Pq))
    pr = (np.asarray(rho_p)[:, None] * one).ravel()
    pz = (np.asarray(z_p)[:, None] * one).ravel()
    ptr = (np.asarray(tr_p)[:, None] * one).ravel()
    ptz = (np.asarray(tz_p)[:, None] * one).ravel()
    qr = (np.asarray(rho_q)[None, :] * one).ravel()
    qz = (np.asarray(z_q)[None, :] * one).ravel()
    qtr = (np.asarray(tr_q)[None, :] * one).ravel()
    qtz = (np.asarray(tz_q)[None, :] * one).ravel()
    Fs = _ibc_brackets_grid(pr, pz, ptr, ptz, qr, qz, qtr, qtz, k,
                            np.broadcast_to(xi, (len(pr), n_xi)))
    m = np.arange(-m_max, m_max + 1)
    bins = np.where(m >= 0, m, n_xi + m)
    phase = np.exp(1j * np.pi * m) * (2.0 * np.pi / n_xi)
    out = []
    for F in Fs:
        spec = np.fft.fft(F, axis=-1)
        out.append((spec[:, bins] * phase).reshape(Pp, Pq, -1))
    return tuple(out)


def ibc_kernels_near(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k,
                     m_max: int, order: int = 48):
    """Modal IBC kernels for near point-pair lists [n_pairs, 2*m_max+1],
    same two-piece grid as mfie_kernels_near."""

    rho_p = np.atleast_1d(np.asarray(rho_p, dtype=float))
    rho_q = np.atleast_1d(np.asarray(rho_q, dtype=float))
    z_p = np.atleast_1d(np.asarray(z_p, dtype=float))
    z_q = np.atleast_1d(np.asarray(z_q, dtype=float))
    tr_p = np.broadcast_to(np.asarray(tr_p, dtype=float), rho_p.shape)
    tz_p = np.broadcast_to(np.asarray(tz_p, dtype=float), rho_p.shape)
    tr_q = np.broadcast_to(np.asarray(tr_q, dtype=float), rho_q.shape)
    tz_q = np.broadcast_to(np.asarray(tz_q, dtype=float), rho_q.shape)

    d2 = (rho_p - rho_q) ** 2 + (z_p - z_q) ** 2
    rr4 = 4.0 * rho_p * rho_q
    d = np.sqrt(np.maximum(d2, 1e-300))
    a = np.sqrt(np.maximum(rr4, 1e-300))
    s0 = np.minimum(0.25, 20.0 * d / np.maximum(a, 1e-300))
    s0 = np.where(rr4 <= 1e-30, 1.0, s0)

    xg, wg = np.polynomial.legendre.leggauss(order)
    u01 = 0.5 * (xg + 1.0)
    w01 = 0.5 * wg
    vmax = np.arcsinh((a / d) * s0)
    v = u01[None, :] * vmax[:, None]
    wv = w01[None, :] * vmax[:, None]
    s = np.minimum((d / a)[:, None] * np.sinh(v), 1.0)
    xi_c = 2.0 * np.arcsin(s)
    ds_dv = (d / a)[:, None] * np.cosh(v)
    w_c = wv * 2.0 * ds_dv / np.sqrt(np.maximum(1.0 - s ** 2, 1e-15))
    osc = float(np.max(abs(complex(k)) * a)) / math.pi + (m_max + 2)
    n_tail = int(min(1024, max(64, math.ceil(4.0 * osc))))
    xt, wt = np.polynomial.legendre.leggauss(n_tail)
    xi0 = 2.0 * np.arcsin(np.minimum(s0, 1.0))
    span = np.pi - xi0
    xi_t = xi0[:, None] + 0.5 * (xt + 1.0)[None, :] * span[:, None]
    w_t = 0.5 * wt[None, :] * span[:, None]
    xi_pos = np.concatenate([xi_c, xi_t], axis=1)
    w_pos = np.concatenate([w_c, w_t], axis=1)
    m = np.arange(-m_max, m_max + 1)

    Fp = _ibc_brackets_grid(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k, xi_pos)
    Fm = _ibc_brackets_grid(rho_p, z_p, tr_p, tz_p, rho_q, z_q, tr_q, tz_q, k, -xi_pos)
    outs = _project_pm_brackets(Fp, Fm, w_pos, xi_pos, m)
    return tuple(outs)


def gc_gs_from_g(G: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gc_m and Gs_m from the table G[..., 0..m_max+1] (m >= 0 entries;
    negative orders via G_{-n} = G_n).

    Gc_m = (G_{m-1} + G_{m+1})/2       Gs_m = (G_{m-1} - G_{m+1})/(2j)
    """

    gm_m1 = G[..., abs(m - 1)]
    gm_p1 = G[..., m + 1]
    return 0.5 * (gm_m1 + gm_p1), (gm_m1 - gm_p1) / 2j


def kernels_for_mode(G: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(G_m, Gc_m, Gs_m) for any integer m (negative handled by symmetry:
    G_{-m} = G_m, Gc_{-m} = Gc_m, Gs_{-m} = -Gs_m)."""

    am = abs(m)
    gc, gs = gc_gs_from_g(G, am)
    if m < 0:
        gs = -gs
    return G[..., am], gc, gs
