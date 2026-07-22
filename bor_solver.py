"""
BoR-MoM solver: PEC EFIE/CFIE + IBC (phases 1-2), PMCHWT dielectrics and
coated PEC (phase 3).

Mixed-potential EFIE, Galerkin-tested per azimuthal mode m (see
BOR_CONVENTIONS.md and BOR_SOLVER_PLAN.md):

  Z I = V,   Z = j k eta0 * 2pi * [ vector-potential - (1/k^2) scalar ] terms

with triangle bases T_i(t) for both J_t and J_phi along the generatrix and
modal kernels G_m / Gc_m / Gs_m from bor_kernels.  Surface divergences use
(1/rho) d(rho T)/dt and (jm/rho) T, so only G itself is ever needed (no
kernel gradients).

Per-mode blocks (p = observation point, q = source point):

  Z^tt  =  C [ II rho rho' T T (t_rho t_rho' Gc + t_z t_z' G)
               - (1/k^2) II (rho T)' (rho' T)' G ]
  Z^tf  =  C [ II rho rho' T T t_rho(p) Gs      - (jm/k^2) II (rho T)' T G ]
  Z^ft  =  C [ -II rho rho' T T t_rho(q) Gs     + (jm/k^2) II T (rho' T)' G ]
  Z^ff  =  C [ II rho rho' T T Gc               - (m^2/k^2) II T T G ]
  C = j k eta0 2pi

Axis conditions at rho = 0 endpoints: J_t end bases retained only for
|m| = 1 (current flowing over the pole), J_phi end bases never (the exact
pole relation J_phi = -+ j J_t is approximated by J_phi(axis) = 0; the
rho -> 0 Jacobian suppresses the residual error, validated by the sphere
gate at all aspects).

Excitation: plane wave from direction (sin th, 0, cos th), phase
e^{+jk d.r} (matches the 2D solver), theta-pol (VV) / phi-pol (HH).
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import special as sp

from bor_kernels import (
    C0, ETA0, Generatrix, GaussData, gauss_on_generatrix,
    modal_kernels_fft, modal_kernels_near, kernels_for_mode,
    mfie_kernels_fft, mfie_kernels_near, mfie_for_mode,
    ibc_kernels_fft, ibc_kernels_near,
)


# ─────────────────────────────────────────────────────────────────────────────
# Refined near-pair Galerkin integration (log singularity along diagonal /
# shared corner) via quadtree grading toward the singular set.
# ─────────────────────────────────────────────────────────────────────────────

def _graded_cells(kind: str, depth: int = 4) -> List[Tuple[float, float, float, float]]:
    """Cells (s0, s1, sp0, sp1) covering [0,1]^2 refined toward the singular
    set: kind = 'diag' (s == s'), 'corner00', 'corner01', 'corner10', 'corner11'
    where cornerAB means singular at s = A, s' = B."""

    cells = []

    def touches(kind, s0, s1, p0, p1):
        if kind == "diag":
            return not (s1 <= p0 or p1 <= s0)
        a = 0.0 if kind[6] == "0" else 1.0
        b = 0.0 if kind[7] == "0" else 1.0
        return (s0 <= a <= s1) and (p0 <= b <= p1)

    def recurse(s0, s1, p0, p1, d):
        if not touches(kind, s0, s1, p0, p1) or d >= depth:
            cells.append((s0, s1, p0, p1))
            return
        sm, pm = 0.5 * (s0 + s1), 0.5 * (p0 + p1)
        recurse(s0, sm, p0, pm, d + 1)
        recurse(s0, sm, pm, p1, d + 1)
        recurse(sm, s1, p0, pm, d + 1)
        recurse(sm, s1, pm, p1, d + 1)

    recurse(0.0, 1.0, 0.0, 1.0, 0)
    return cells


_CELL_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _cell_points(kind: str, gorder: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(s_points, sp_points, weights) for the graded cell set of `kind`."""

    key = f"{kind}:{gorder}"
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    xg, wg = np.polynomial.legendre.leggauss(gorder)
    u = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    # Source axis uses order+1 so test/source nodes NEVER coincide exactly:
    # a coincident pair (R = 0) underflows the 1/R^3 MFIE kernel into NaN
    # and adds ln(eps)-noise to the EFIE log cells.
    xq, wq_ = np.polynomial.legendre.leggauss(gorder + 1)
    uq = 0.5 * (xq + 1.0)
    wq = 0.5 * wq_
    S, SP, W = [], [], []
    for (s0, s1, p0, p1) in _graded_cells(kind):
        hs, hp = s1 - s0, p1 - p0
        ss = s0 + u * hs
        pp = p0 + uq * hp
        SS, PP = np.meshgrid(ss, pp, indexing="ij")
        WW = np.outer(w * hs, wq * hp)
        S.append(SS.ravel()); SP.append(PP.ravel()); W.append(WW.ravel())
    out = (np.concatenate(S), np.concatenate(SP), np.concatenate(W))
    _CELL_CACHE[key] = out
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Point-level geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _points_on_element(gen: Generatrix, e: int, s: np.ndarray):
    n0, n1 = gen.elem_n0[e], gen.elem_n1[e]
    r0, r1 = gen.nodes[n0], gen.nodes[n1]
    rho = r0[0] + s * (r1[0] - r0[0])
    z = r0[1] + s * (r1[1] - r0[1])
    L = gen.lengths[e]
    T0, T1 = 1.0 - s, s
    drho = r1[0] - r0[0]
    dRT0 = (drho * (1.0 - s) - rho) / L
    dRT1 = (drho * s + rho) / L
    return rho, z, gen.trho[e], gen.tz[e], T0, T1, dRT0, dRT1, L


def _pair_blocks(m: int, k: float,
                 rho_p, tr_p, tz_p, T_p, D_p, w_p,
                 rho_q, tr_q, tz_q, T_q, D_q, w_q,
                 G, Gc, Gs):
    """
    The four per-mode Galerkin blocks for a set of weighted point pairs.

    T_p/D_p: [n_bases_p, n_pts] shape and (rho T)' matrices for the test side
    (likewise source side).  G/Gc/Gs: [n_pts_p, n_pts_q] kernels.  Weights
    include dt measures.  Returns (ztt, ztf, zft, zff) WITHOUT the C factor.
    """

    wp = w_p; wq = w_q
    rrw = (rho_p * wp)[:, None] * (rho_q * wq)[None, :]
    K_tt_vec = rrw * ((tr_p[:, None] * tr_q[None, :]) * Gc + (tz_p[:, None] * tz_q[None, :]) * G)
    K_sc = (wp[:, None] * wq[None, :]) * G
    K_tf_vec = rrw * (tr_p[:, None] * Gs)
    K_ft_vec = -rrw * (tr_q[None, :] * Gs)
    K_ff_vec = rrw * Gc

    ztt = T_p @ K_tt_vec @ T_q.T - (1.0 / k ** 2) * (D_p @ K_sc @ D_q.T)
    ztf = T_p @ K_tf_vec @ T_q.T - (1j * m / k ** 2) * (D_p @ K_sc @ T_q.T)
    zft = T_p @ K_ft_vec @ T_q.T + (1j * m / k ** 2) * (T_p @ K_sc @ D_q.T)
    zff = T_p @ K_ff_vec @ T_q.T - (m ** 2 / k ** 2) * (T_p @ K_sc @ T_q.T)
    return ztt, ztf, zft, zff


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

def _causal_medium(eps_r: complex, mu_r: complex) -> Tuple[complex, complex]:
    """(m, eta_r) for a homogeneous medium: refractive index with Im(m) <= 0
    (causal decay, same branch as mie_sphere/_causal_medium_index) and the
    relative impedance eta_r = mu_r / m, which guarantees k*eta = w mu mu0
    and k/eta = w eps eps0 exactly for whichever branch m took."""

    m = np.sqrt(complex(eps_r) * complex(mu_r))
    if m.imag > 0.0:
        m = -m
    if m.real < 0.0:
        m = -m
    return m, complex(mu_r) / m


class BorPecSolver:
    """Single-surface BoR operator factory + PEC/IBC solver.

    With medium=(eps_r, mu_r) the EFIE (T) and rotated-PV (P) operators are
    assembled in that homogeneous medium (complex k, medium eta) — the
    building blocks of the phase-3 PMCHWT systems.  Excitation and far-field
    methods always refer to the EXTERIOR (air) and are only meaningful on an
    instance with medium=None."""

    def __init__(self, points, freq_hz: float, gauss_order: int = 4,
                 near_depth: int = 4, medium=None, single_tables: bool = False):
        # single_tables stores the FAR kernel tables (the memory bound at
        # scale) as complex64; all near/self quadrature and the assembled
        # mode systems stay double.  Validated at <= 0.005 dB vs double.
        self._table_dtype = np.complex64 if single_tables else np.complex128
        self.gen = Generatrix(np.asarray(points, dtype=float))
        k0 = 2.0 * math.pi * freq_hz / C0
        if medium is None:
            self.k = k0
            self.eta = ETA0
        else:
            m_idx, eta_r = _causal_medium(*medium)
            self.k = k0 * m_idx
            self.eta = ETA0 * eta_r
        self.freq_hz = freq_hz
        self.g = gauss_on_generatrix(self.gen, gauss_order)
        self.gauss_order = gauss_order
        self.near_depth = near_depth
        self.Nn = self.gen.n_nodes
        self._build_point_matrices()
        self._G_table = None
        self._stream = None
        self._near_cache: Dict[int, Dict[Tuple[int, int], Tuple]] = {}

    def enable_streaming(self, m_max: int, efie: bool = True,
                         mfie: bool = False,
                         ibc_zs_pt: Optional[np.ndarray] = None,
                         single_blocks: bool = False,
                         tile_budget_gb: float = 1.0,
                         workers: int = 1,
                         mode_block: Optional[int] = None) -> None:
        """Phase-7b: build per-mode nodal far blocks instead of the
        [P, P, modes] Gauss-point tables (see bor_streaming).  Must be
        called before any assemble_* call; the near/self machinery is
        unaffected.  With IBC, the source Z_s is baked into the blocks, so
        assemble_ibc_extra must be called with the same zs_pt."""

        from bor_streaming import StreamingFarBlocks
        self._stream = StreamingFarBlocks(
            self, m_max, efie=efie, mfie=mfie, ibc_zs_pt=ibc_zs_pt,
            dtype=np.complex64 if single_blocks else np.complex128,
            tile_budget_gb=tile_budget_gb, workers=workers,
            mode_block=mode_block)

    # ── base-point kernel table (far pairs only; near pairs zeroed) ──
    def _build_point_matrices(self):
        g = self.g
        P = len(g.rho)
        self.P = P
        Nn = self.Nn
        T = np.zeros((Nn, P)); D = np.zeros((Nn, P))
        for p in range(P):
            e = g.elem[p]
            T[e, p] = g.T0[p];     D[e, p] = g.dRT0[p]
            T[e + 1, p] = g.T1[p]; D[e + 1, p] = g.dRT1[p]
        self.B_T, self.B_D = T, D
        # element adjacency classes for pair routing
        self.elem_of_pt = g.elem

    def _kernel_tables(self, m_max: int):
        """G_m table [P, P, m_max+2] at base Gauss points; same/adjacent
        element point-pairs zeroed (their Galerkin blocks are added by the
        refined near-pair path)."""

        if self._G_table is not None and self._G_table.shape[-1] >= m_max + 2:
            return self._G_table
        g = self.g
        RP = g.rho[:, None] * np.ones(self.P)[None, :]
        RQ = g.rho[None, :] * np.ones(self.P)[:, None]
        ZP = g.z[:, None] * np.ones(self.P)[None, :]
        ZQ = g.z[None, :] * np.ones(self.P)[:, None]
        ediff = np.abs(g.elem[:, None].astype(int) - g.elem[None, :].astype(int))
        near_mask = ediff <= 1
        G = modal_kernels_fft(RP, ZP, RQ, ZQ, self.k, m_max)
        # far table must not contain near-singular FFT garbage
        # (also excludes exact-coincidence pairs)
        near_flat = np.flatnonzero(near_mask.ravel())
        Gf = G.reshape(-1, G.shape[-1])
        Gf[near_flat, :] = 0.0
        self._G_table = Gf.reshape(self.P, self.P, -1).astype(
            self._table_dtype, copy=False)
        self._m_max_table = m_max
        return self._G_table

    # ── near element pairs: refined kernels cached per (e, f) ──
    def _near_pair_data(self, e: int, f: int, m_max: int):
        key = (e, f)
        cache = self._near_cache.setdefault(m_max, {})
        if key in cache:
            return cache[key]
        if e == f:
            kind = "diag"
        else:
            # shared node: e,f adjacent (f = e+1 or e-1)
            kind = "corner10" if f == e + 1 else "corner01"
            # element e local s=1 touches f local s'=0 for f=e+1
        s, sp, w = _cell_points(kind)
        rho_p, z_p, tr_p, tz_p, T0p, T1p, D0p, D1p, Lp = _points_on_element(self.gen, e, s)
        rho_q, z_q, tr_q, tz_q, T0q, T1q, D0q, D1q, Lq = _points_on_element(self.gen, f, sp)
        Gm = modal_kernels_near(rho_p, z_p, rho_q, z_q, self.k, m_max)
        data = (s, sp, w * Lp * Lq,
                rho_p, tr_p, tz_p, np.vstack([T0p, T1p]), np.vstack([D0p, D1p]),
                rho_q, tr_q, tz_q, np.vstack([T0q, T1q]), np.vstack([D0q, D1q]),
                Gm)
        cache[key] = data
        return data

    # ── full node-based Z for mode m ──
    def assemble_mode(self, m: int, m_max: int) -> np.ndarray:
        k = self.k
        g = self.g
        if self._stream is not None:
            ztt, ztf, zft, zff = self._stream.efie_blocks(m)
        else:
            Gtab = self._kernel_tables(m_max)
            G, Gc, Gs = kernels_for_mode(Gtab, m)
            ztt, ztf, zft, zff = _pair_blocks(
                m, k,
                g.rho, g.trho, g.tz, self.B_T, self.B_D, g.w,
                g.rho, g.trho, g.tz, self.B_T, self.B_D, g.w,
                G, Gc, Gs,
            )

        # near element pairs (self + adjacent): refined path
        ne = self.gen.n_elems
        for e in range(ne):
            for f in (e - 1, e, e + 1):
                if f < 0 or f >= ne:
                    continue
                (s, sp, w, rho_p, tr_p, tz_p, Tp, Dp,
                 rho_q, tr_q, tz_q, Tq, Dq, Gm) = self._near_pair_data(e, f, m_max)
                Gn, Gcn, Gsn = kernels_for_mode(Gm, m)
                # point-pair kernels are 1-D lists here: build diag-style
                # products via elementwise weighting (pairs are matched 1:1).
                wp = np.sqrt(np.abs(w))  # split weight symmetrically
                # For matched point-pair lists the "matrix" contraction
                # degenerates to sums over pairs:
                rr = rho_p * rho_q * w
                ktt = rr * ((tr_p * tr_q) * Gcn + (tz_p * tz_q) * Gn)
                ksc = w * Gn
                ktf = rr * (tr_p * Gsn)
                kft = -rr * (tr_q * Gsn)
                kff = rr * Gcn
                btt = np.einsum("ip,p,jp->ij", Tp, ktt, Tq) - (1.0 / k ** 2) * np.einsum("ip,p,jp->ij", Dp, ksc, Dq)
                btf = np.einsum("ip,p,jp->ij", Tp, ktf, Tq) - (1j * m / k ** 2) * np.einsum("ip,p,jp->ij", Dp, ksc, Tq)
                bft = np.einsum("ip,p,jp->ij", Tp, kft, Tq) + (1j * m / k ** 2) * np.einsum("ip,p,jp->ij", Tp, ksc, Dq)
                bff = np.einsum("ip,p,jp->ij", Tp, kff, Tq) - (m ** 2 / k ** 2) * np.einsum("ip,p,jp->ij", Tp, ksc, Tq)
                rows = np.array([e, e + 1]); cols = np.array([f, f + 1])
                ztt[np.ix_(rows, cols)] += btt
                ztf[np.ix_(rows, cols)] += btf
                zft[np.ix_(rows, cols)] += bft
                zff[np.ix_(rows, cols)] += bff

        C = 1j * k * self.eta * 2.0 * np.pi
        Nn = self.Nn
        Z = np.empty((2 * Nn, 2 * Nn), dtype=np.complex128)
        Z[:Nn, :Nn] = C * ztt
        Z[:Nn, Nn:] = C * ztf
        Z[Nn:, :Nn] = C * zft
        Z[Nn:, Nn:] = C * zff
        return Z

    # ── MFIE machinery (Phase 2) ──
    def _mfie_tables(self, m_max: int):
        """Four modal MFIE kernel tables [P, P, 2*m_max+1] at base Gauss
        points; near element-pair entries zeroed (refined path adds them)."""

        if getattr(self, "_K_tables", None) is not None:
            return self._K_tables
        g = self.g
        P = self.P
        one = np.ones((P, P))
        args = (g.rho[:, None] * one, g.z[:, None] * one,
                g.trho[:, None] * one, g.tz[:, None] * one,
                g.rho[None, :] * one, g.z[None, :] * one,
                g.trho[None, :] * one, g.tz[None, :] * one)
        K = mfie_kernels_fft(*args, self.k, m_max)
        ediff = np.abs(g.elem[:, None].astype(int) - g.elem[None, :].astype(int))
        near_flat = np.flatnonzero((ediff <= 1).ravel())
        K = list(K)
        for i in range(4):
            Kf = K[i].reshape(-1, K[i].shape[-1])
            Kf[near_flat, :] = 0.0
            K[i] = Kf.reshape(P, P, -1).astype(self._table_dtype, copy=False)
        self._K_tables = tuple(K)
        return self._K_tables

    def _near_mfie_data(self, e: int, f: int, m_max: int):
        cache = self._near_cache.setdefault(("mfie", m_max), {})
        if (e, f) in cache:
            return cache[(e, f)]
        kind = "diag" if e == f else ("corner10" if f == e + 1 else "corner01")
        s, sp, w = _cell_points(kind)
        rho_p, z_p, tr_p, tz_p, T0p, T1p, _, _, Lp = _points_on_element(self.gen, e, s)
        rho_q, z_q, tr_q, tz_q, T0q, T1q, _, _, Lq = _points_on_element(self.gen, f, sp)
        tr_pa = np.full_like(rho_p, tr_p); tz_pa = np.full_like(rho_p, tz_p)
        tr_qa = np.full_like(rho_q, tr_q); tz_qa = np.full_like(rho_q, tz_q)
        Kn = mfie_kernels_near(rho_p, z_p, tr_pa, tz_pa, rho_q, z_q, tr_qa, tz_qa,
                               self.k, m_max)
        data = (w * Lp * Lq, rho_p, np.vstack([T0p, T1p]),
                rho_q, np.vstack([T0q, T1q]), Kn)
        cache[(e, f)] = data
        return data

    def mass_blocks(self, weight=None) -> np.ndarray:
        """2pi * Int w(t) rho T_i T_j dt  (node-based [Nn, Nn]); weight is a
        per-Gauss-point array (default 1) — used for the MFIE J/2 term and
        the IBC Z_s term (with weight = Z_s at the Gauss points)."""

        g = self.g
        wgt = np.ones(self.P) if weight is None else np.asarray(weight)
        K = (g.w * g.rho * wgt)
        return 2.0 * np.pi * (self.B_T * K[None, :]) @ self.B_T.T

    def assemble_mfie_mode(self, m: int, m_max: int) -> np.ndarray:
        """Z_MFIE = (1/2) M - K  (node-based [2Nn, 2Nn]), where K is the
        Galerkin contraction of the modal MFIE brackets."""

        g = self.g
        if self._stream is not None and self._stream.K is not None:
            blocks = list(self._stream.bracket_blocks("mfie", m))
        else:
            Kt = self._mfie_tables(m_max)
            blocks = []
            wrho = g.w * g.rho
            for uv in range(4):
                Km = mfie_for_mode(Kt[uv], m, m_max)
                blocks.append(2.0 * np.pi * (self.B_T * wrho[None, :]) @ Km @ (self.B_T * wrho[None, :]).T)
        ktt, ktf, kft, kff = blocks

        ne = self.gen.n_elems
        for e in range(ne):
            for f in (e - 1, e, e + 1):
                if f < 0 or f >= ne:
                    continue
                w, rho_p, Tp, rho_q, Tq, Kn = self._near_mfie_data(e, f, m_max)
                rr = rho_p * rho_q * w
                rows = np.array([e, e + 1]); cols = np.array([f, f + 1])
                for uv, tgt in enumerate((ktt, ktf, kft, kff)):
                    Km = mfie_for_mode(Kn[uv], m, m_max)
                    blk = 2.0 * np.pi * np.einsum("ip,p,jp->ij", Tp, rr * Km, Tq)
                    tgt[np.ix_(rows, cols)] += blk

        Nn = self.Nn
        M = self.mass_blocks()
        Z = np.zeros((2 * Nn, 2 * Nn), dtype=np.complex128)
        Z[:Nn, :Nn] = 0.5 * M - ktt
        Z[:Nn, Nn:] = -ktf
        Z[Nn:, :Nn] = -kft
        Z[Nn:, Nn:] = 0.5 * M - kff
        return Z

    def rhs_mfie_mode(self, m: int, theta_inc_deg: float, pol: str) -> np.ndarray:
        """<W, n_hat x H_inc> for the plane wave (see phase-2 derivation)."""

        g = self.g
        k = self.k
        th = math.radians(theta_inc_deg)
        st, ct = math.sin(th), math.cos(th)
        u = k * g.rho * st
        P = np.exp(1j * k * ct * g.z)
        jm = lambda n: (1j) ** n * sp.jv(n, u)
        Jm = jm(m); Jm_m1 = jm(m - 1); Jm_p1 = jm(m + 1)
        Ic = math.pi * (Jm_m1 + Jm_p1)
        Is = (math.pi / 1j) * (Jm_m1 - Jm_p1)
        I1 = 2.0 * math.pi * Jm
        if pol.upper() in ("VV", "THETA", "TM"):
            # H_inc = -(1/eta0) y_hat e^{jk d.r}
            et = Ic / ETA0
            ef = -(g.trho * Is) / ETA0
        else:
            # H_inc = +(1/eta0) e_theta e^{jk d.r}
            et = (ct * Is) / ETA0
            ef = (ct * g.trho * Ic - st * g.tz * I1) / ETA0
        vt = self.B_T @ (g.w * g.rho * P * et)
        vf = self.B_T @ (g.w * g.rho * P * ef)
        return np.concatenate([vt, vf])

    # ── IBC operator (Phase 2): 0.5 Z_s mass + magnetic-current K' term ──
    def _ibc_tables(self, m_max: int):
        if getattr(self, "_KI_tables", None) is not None:
            return self._KI_tables
        g = self.g
        K = ibc_kernels_fft(g.rho, g.z, g.trho, g.tz, g.rho, g.z, g.trho, g.tz,
                            self.k, m_max)
        ediff = np.abs(g.elem[:, None].astype(int) - g.elem[None, :].astype(int))
        near_flat = np.flatnonzero((ediff <= 1).ravel())
        K = list(K)
        for i in range(4):
            Kf = K[i].reshape(-1, K[i].shape[-1])
            Kf[near_flat, :] = 0.0
            K[i] = Kf.reshape(self.P, self.P, -1).astype(self._table_dtype,
                                                         copy=False)
        self._KI_tables = tuple(K)
        return self._KI_tables

    def _near_ibc_data(self, e: int, f: int, m_max: int):
        cache = self._near_cache.setdefault(("ibc", m_max), {})
        if (e, f) in cache:
            return cache[(e, f)]
        kind = "diag" if e == f else ("corner10" if f == e + 1 else "corner01")
        s, sp, w = _cell_points(kind)
        rho_p, z_p, tr_p, tz_p, T0p, T1p, _, _, Lp = _points_on_element(self.gen, e, s)
        rho_q, z_q, tr_q, tz_q, T0q, T1q, _, _, Lq = _points_on_element(self.gen, f, sp)
        Kn = ibc_kernels_near(rho_p, z_p, np.full_like(rho_p, tr_p), np.full_like(rho_p, tz_p),
                              rho_q, z_q, np.full_like(rho_q, tr_q), np.full_like(rho_q, tz_q),
                              self.k, m_max)
        data = (w * Lp * Lq, rho_p, np.vstack([T0p, T1p]),
                rho_q, np.vstack([T0q, T1q]), Kn)
        cache[(e, f)] = data
        return data

    def _rot_pv_blocks(self, m: int, m_max: int, src_wpt=None, src_welem=None):
        """Galerkin contraction of the rotated-PV brackets
        B_uv = p(R) W_u . [Rvec x (n_hat_q x f_v)] with an optional source
        weight (Z_s for the IBC path; unit for PMCHWT).  Returns the four
        node-based blocks (Btt, Btf, Bft, Bff)."""

        g = self.g
        if (self._stream is not None and self._stream.B is not None
                and src_wpt is not None):
            # streaming blocks have the solve's Z_s baked into the source side
            blocks = list(self._stream.bracket_blocks("ibc", m))
        else:
            Kt = self._ibc_tables(m_max)
            wrho_p = g.w * g.rho
            wrho_q = g.w * g.rho if src_wpt is None else g.w * g.rho * src_wpt
            blocks = []
            for uv in range(4):
                Km = mfie_for_mode(Kt[uv], m, m_max)
                blocks.append(2.0 * np.pi * (self.B_T * wrho_p[None, :]) @ Km @ (self.B_T * wrho_q[None, :]).T)

        ne = self.gen.n_elems
        for e in range(ne):
            for f in (e - 1, e, e + 1):
                if f < 0 or f >= ne:
                    continue
                welem = 1.0 if src_welem is None else src_welem[f]
                if abs(welem) == 0.0:
                    continue
                w, rho_p, Tp, rho_q, Tq, Kn = self._near_ibc_data(e, f, m_max)
                rr = rho_p * rho_q * w * welem
                rows = np.array([e, e + 1]); cols = np.array([f, f + 1])
                for uv, tgt in enumerate(blocks):
                    Km = mfie_for_mode(Kn[uv], m, m_max)
                    blk = 2.0 * np.pi * np.einsum("ip,p,jp->ij", Tp, rr * Km, Tq)
                    tgt[np.ix_(rows, cols)] += blk
        return tuple(blocks)

    def assemble_ibc_extra(self, m: int, m_max: int, zs_pt: np.ndarray,
                           zs_elem: np.ndarray) -> np.ndarray:
        """
        IBC-EFIE additions from eliminating M = -Z_s n_hat x J:

            Z_extra = (1/2) <W, Z_s J> + <W, PV curl Int G M>

        The K' term applies Z_s at the SOURCE point.
        """

        ktt, ktf, kft, kff = self._rot_pv_blocks(m, m_max, zs_pt, zs_elem)
        Nn = self.Nn
        Mz = self.mass_blocks(weight=zs_pt)
        Z = np.zeros((2 * Nn, 2 * Nn), dtype=np.complex128)
        Z[:Nn, :Nn] = 0.5 * Mz + ktt
        Z[:Nn, Nn:] = ktf
        Z[Nn:, :Nn] = kft
        Z[Nn:, Nn:] = 0.5 * Mz + kff
        return Z

    def assemble_pmchwt_P(self, m: int, m_max: int) -> np.ndarray:
        """
        The PMCHWT rotated-PV operator P (node-based [2Nn, 2Nn]) acting on a
        magnetic current expanded in the SAME (t, phi) triangle bases:

            (P M)_tested = <W, PV Int p(R) Rvec x M dS'>
                         = <W, E_PV(M)> in this medium (E_s(M) = -curl Int G M)

        The IBC brackets B_uv are built for the rotated source n_hat_q x f_v
        (n x t = phi, n x phi = -t), so columns remap:  P[:, Mt] = -B[:, f_phi],
        P[:, Mphi] = +B[:, f_t].  The same bilinear form gives the H-side
        operator: <W, H_PV(J)> = -(P J).
        """

        Btt, Btf, Bft, Bff = self._rot_pv_blocks(m, m_max)
        Nn = self.Nn
        P = np.empty((2 * Nn, 2 * Nn), dtype=np.complex128)
        P[:Nn, :Nn] = -Btf
        P[:Nn, Nn:] = Btt
        P[Nn:, :Nn] = -Bff
        P[Nn:, Nn:] = Bft
        return P

    # ── cache warm-up (thread safety for parallel mode assembly) ──
    def prepare_operators(self, m_max: int, efie: bool = True,
                          mfie: bool = False, ibc: bool = False,
                          workers: int = 1) -> None:
        """Build every kernel table and near-pair cache this solver will need
        up front, so parallel per-mode assembly only READS shared state.
        The near-pair builds are independent per element pair and run on
        `workers` threads (each pair writes its own cache key)."""

        ne = self.gen.n_elems
        pairs = [(e, f) for e in range(ne) for f in (e - 1, e, e + 1)
                 if 0 <= f < ne]
        streaming = self._stream is not None   # far blocks already built
        jobs = []
        if efie:
            if not streaming:
                self._kernel_tables(m_max)
            jobs += [(self._near_pair_data, p) for p in pairs]
        if mfie:
            if not streaming:
                self._mfie_tables(m_max)
            jobs += [(self._near_mfie_data, p) for p in pairs]
        if ibc:
            if not (streaming and self._stream.B is not None):
                self._ibc_tables(m_max)
            jobs += [(self._near_ibc_data, p) for p in pairs]
        if max(1, int(workers)) <= 1:
            for fn, (e, f) in jobs:
                fn(e, f, m_max)
        else:
            with ThreadPoolExecutor(max_workers=int(workers)) as ex:
                list(ex.map(lambda job: job[0](job[1][0], job[1][1], m_max), jobs))

    # ── active-basis mask per mode ──
    def basis_mask(self, m: int) -> np.ndarray:
        Nn = self.Nn
        t_act = np.ones(Nn, dtype=bool)
        f_act = np.ones(Nn, dtype=bool)
        for end in (0, Nn - 1):
            if self.gen.node_on_axis(end):
                t_act[end] = (abs(m) == 1)
                f_act[end] = False
            else:
                t_act[end] = False   # open edge: J_t vanishes
                f_act[end] = True
        return np.concatenate([t_act, f_act])

    # ── excitation ──
    def rhs_mode(self, m: int, theta_inc_deg: float, pol: str) -> np.ndarray:
        g = self.g
        k = self.k
        th = math.radians(theta_inc_deg)
        st, ct = math.sin(th), math.cos(th)
        u = k * g.rho * st
        P = np.exp(1j * k * ct * g.z)
        jm = lambda n: (1j) ** n * sp.jv(n, u)
        Jm = jm(m); Jm_m1 = jm(m - 1); Jm_p1 = jm(m + 1)
        Ic = math.pi * (Jm_m1 + Jm_p1)
        Is = (math.pi / 1j) * (Jm_m1 - Jm_p1)
        I1 = 2.0 * math.pi * Jm
        if pol.upper() in ("VV", "THETA", "TM"):
            et = ct * g.trho * Ic - st * g.tz * I1
            ef = -ct * Is
        else:
            et = g.trho * Is
            ef = Ic
        vt = self.B_T @ (g.w * g.rho * P * et)
        vf = self.B_T @ (g.w * g.rho * P * ef)
        return np.concatenate([vt, vf])

    def rhs_h_mode(self, m: int, theta_inc_deg: float, pol: str) -> np.ndarray:
        """UNROTATED <W, H_inc> (PMCHWT H-row; the MFIE rhs is the rotated
        <W, n_hat x H_inc>).  The incident pair is (E, H):
            VV: E = e_theta P,  H = -(1/eta0) y_hat P
            HH: E = y_hat P,    H = +(1/eta0) e_theta P
        so <W, H_inc> reuses rhs_mode with the polarizations swapped."""

        if pol.upper() in ("VV", "THETA", "TM"):
            return -self.rhs_mode(m, theta_inc_deg, "HH") / ETA0
        return self.rhs_mode(m, theta_inc_deg, "VV") / ETA0

    # ── far field for one mode's solution ──
    def farfield_mode(self, m: int, sol: np.ndarray, theta_s_deg: float,
                      zs_pt: Optional[np.ndarray] = None,
                      msol: Optional[np.ndarray] = None) -> Tuple[complex, complex]:
        """
        Modal far-field (F_theta, F_phi).  For IBC surfaces the eliminated
        magnetic current M = -Z_s n_hat x J still RADIATES:
            M_t = Z_s J_phi,  M_phi = -Z_s J_t
            F_theta^M = -(jk/4pi) Int M . phi_hat_s e^{jk r_hat . r'}
            F_phi^M   = +(jk/4pi) Int M . theta_hat_s e^{...}
        (Weston's Z_s = eta0 null is exactly the J/M far-field cancellation —
        omitting this term leaves the operator right but the RCS wrong.)
        """

        g = self.g
        k = self.k
        Nn = self.Nn
        th = math.radians(theta_s_deg)
        st, ct = math.sin(th), math.cos(th)
        u = k * g.rho * st
        P = np.exp(1j * k * ct * g.z)
        jm = lambda n: (1j) ** n * sp.jv(n, u)
        Jm = jm(m); Jm_m1 = jm(m + 1); Jp = jm(m - 1)
        Icos = math.pi * (Jm_m1 + Jp)
        Isin = (math.pi / 1j) * (Jm_m1 - Jp)
        I1 = 2.0 * math.pi * Jm
        Jt = self.B_T.T @ sol[:Nn]     # current values at Gauss points
        Jf = self.B_T.T @ sol[Nn:]
        common = g.w * g.rho * P

        def proj_theta(Xt, Xf):
            return np.sum(common * (Xt * (ct * g.trho * Icos - st * g.tz * I1) + Xf * (-ct * Isin)))

        def proj_phi(Xt, Xf):
            return np.sum(common * (Xt * g.trho * Isin + Xf * Icos))

        pref_j = -1j * k * ETA0 / (4.0 * math.pi)
        f_theta = pref_j * proj_theta(Jt, Jf)
        f_phi = pref_j * proj_phi(Jt, Jf)
        Mt = Mf = None
        if zs_pt is not None:
            Mt = zs_pt * Jf
            Mf = -zs_pt * Jt
        elif msol is not None:
            Mt = self.B_T.T @ msol[:Nn]
            Mf = self.B_T.T @ msol[Nn:]
        if Mt is not None:
            pref_m = 1j * k / (4.0 * math.pi)
            f_theta += -pref_m * proj_phi(Mt, Mf)
            f_phi += pref_m * proj_theta(Mt, Mf)
        return f_theta, f_phi


def _mode_sweep(n_dofs: int, thetas, pols, m_max: int, mode_tol: float,
                assemble: Callable, rhs: Callable, farfield: Callable,
                prepare: Optional[Callable] = None, workers: int = 1,
                progress: Optional[Callable] = None,
                check_abort: Optional[Callable] = None,
                monitor_cond: bool = False):
    """
    Shared adaptive azimuthal-mode loop for every BoR formulation.

    assemble(m) -> (A_masked, mask); rhs(m, theta, pol) -> V (full, unmasked);
    farfield(m, full_sol, theta, pol) -> complex modal far-field contribution.

    mask=None means the closures own the reduction: rhs returns the REDUCED
    right-hand side and farfield receives the REDUCED solution vector (used
    by the junction solver, whose constraint reduction A_red = Q^T A Q is
    not expressible as a boolean mask).

    Each mode's system is factored ONCE: every (theta, pol) is a stacked RHS
    column of a single np.linalg.solve — an aspect sweep at fixed frequency
    costs one assembly + one LU per mode.  Modes are independent, so waves of
    `workers` modes run on threads (BLAS releases the GIL); call prepare
    first so kernel/near caches are read-only during the parallel section.
    Accumulation and the 2-quiet-modes truncation test remain in strict mode
    order, so results are identical to the serial loop.
    """

    thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
    pols = list(pols)
    F = np.zeros((len(pols), len(thetas)), dtype=np.complex128)
    if prepare is not None:
        prepare(m_max)
    workers = max(1, int(workers))

    def solve_am(am: int):
        dF = np.zeros_like(F)
        res = 0.0
        cond = 0.0
        for m in ([0] if am == 0 else [am, -am]):
            A, mask = assemble(m)
            cols = [rhs(m, th, pol) if mask is None else rhs(m, th, pol)[mask]
                    for th in thetas for pol in pols]
            B = np.stack(cols, axis=1)
            X = np.linalg.solve(A, B)
            bn = float(np.linalg.norm(B))
            if bn > 0.0:
                res = max(res, float(np.linalg.norm(A @ X - B)) / bn)
            if monitor_cond:
                cond = max(cond, float(np.linalg.cond(A)))
            ci = 0
            for it, th in enumerate(thetas):
                for ip, pol in enumerate(pols):
                    if mask is None:
                        sol = X[:, ci]
                    else:
                        sol = np.zeros(n_dofs, dtype=np.complex128)
                        sol[mask] = X[:, ci]
                    ci += 1
                    dF[ip, it] += farfield(m, sol, th, pol)
        return dF, res, cond

    modes_used = 0
    quiet = 0
    am = 0
    max_res = 0.0
    conds: List[float] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while am <= m_max and quiet < 2:
            if check_abort is not None:
                check_abort()
            wave = list(range(am, min(am + workers, m_max + 1)))
            for w_am, (dF, res, cond) in zip(wave, ex.map(solve_am, wave)):
                F += dF
                max_res = max(max_res, res)
                if monitor_cond:
                    conds.append(cond)
                modes_used = w_am
                scale = max(float(np.max(np.abs(F))), 1e-30)
                if float(np.max(np.abs(dF))) < mode_tol * scale:
                    quiet += 1
                    if quiet >= 2:
                        break
                else:
                    quiet = 0
            am = wave[-1] + 1
            if progress is not None:
                progress(modes_used, m_max)
    stats = {"linear_residual": max_res}
    if monitor_cond and conds:
        stats["max_cond"] = max(conds)
        stats["median_cond"] = float(np.median(conds))
    return F, modes_used, stats


def estimate_bor_table_gb(n_elems: int, m_max: int, formulation: str = "cfie",
                          has_ibc: bool = False, gauss_order: int = 4,
                          single_tables: bool = False) -> float:
    """Persistent far-table memory (GB) — the scale bound of the current
    all-modes-at-once FFT assembly (see the phase-7 streaming notes)."""

    P = float(gauss_order * n_elems)
    per = 8.0 if single_tables else 16.0
    total = P * P * (m_max + 2) * per
    if formulation in ("cfie", "mfie"):
        total += 4.0 * P * P * (2 * m_max + 1) * per
    if has_ibc:
        total += 4.0 * P * P * (2 * m_max + 1) * per
    return total / 1e9


def solve_bor(points, freq_hz: float, thetas_deg, formulation: str = "efie",
              cfie_alpha: float = 0.5, zs=None, n_modes: Optional[int] = None,
              gauss_order: int = 4, mode_tol: float = 1e-6, workers: int = 1,
              progress: Optional[Callable] = None,
              check_abort: Optional[Callable] = None,
              table_precision: str = "auto", assembly: str = "auto",
              stream_budget_gb: float = 8.0) -> Dict:
    """
    Monostatic RCS of a closed BoR at aspect angles thetas_deg (from +z).

    formulation: 'efie' (open shells / small bodies), 'cfie' (closed PEC —
    interior-resonance free), 'mfie' (diagnostics only).
    zs: surface impedance — None (PEC), a complex scalar, or a per-ELEMENT
    complex array (tapered IBC).  IBC uses the EFIE form E_tan = Z_s J;
    lossy Z_s also damps interior resonances.  CFIE with nonzero zs is not
    supported in phase 2 (raises).
    Returns dict with sigma_vv, sigma_hh (m^2) per angle.
    """

    form = formulation.strip().lower()
    solver = BorPecSolver(points, freq_hz, gauss_order=gauss_order)
    k = solver.k
    thetas = np.atleast_1d(np.asarray(thetas_deg, dtype=float))
    rho_max = float(np.max(solver.gen.nodes[:, 0]))
    st_max = float(np.max(np.abs(np.sin(np.radians(thetas)))))
    if n_modes is None:
        n_modes = int(math.ceil(k * rho_max * max(st_max, 0.05))) + 12
    m_max = n_modes

    # Per-Gauss-point / per-element surface impedance.
    zs_pt = None
    zs_elem = None
    if zs is not None:
        zs_arr = np.asarray(zs, dtype=complex)
        if zs_arr.ndim == 0:
            zs_elem = np.full(solver.gen.n_elems, complex(zs_arr))
        else:
            if len(zs_arr) != solver.gen.n_elems:
                raise ValueError("zs array must have one entry per generatrix element.")
            zs_elem = zs_arr.astype(complex)
        zs_pt = zs_elem[solver.g.elem]
        if np.all(np.abs(zs_pt) == 0.0):
            zs_pt = None
        elif form == "cfie":
            raise ValueError("CFIE with surface impedance is not supported in phase 2; "
                             "use formulation='efie' (lossy Z_s suppresses resonances).")

    alpha = float(cfie_alpha) if form == "cfie" else (1.0 if form == "efie" else 0.0)
    n_dofs = 2 * solver.Nn

    # ── far-assembly strategy and memory budget ──
    from bor_streaming import estimate_streaming_gb
    tp = str(table_precision).strip().lower()
    if tp not in ("auto", "single", "double"):
        raise ValueError("table_precision must be 'auto', 'single', or 'double'.")
    asm = str(assembly).strip().lower()
    if asm not in ("auto", "tables", "streaming"):
        raise ValueError("assembly must be 'auto', 'tables', or 'streaming'.")
    est_double = estimate_bor_table_gb(solver.gen.n_elems, m_max, form,
                                       zs_pt is not None, gauss_order, False)
    # auto: switch to the phase-7b streaming path early — the TABLE builders
    # sample [P, P, n_xi] in one shot, so their construction PEAK is several
    # times the stored table size (a 7 GB table can thrash a 32 GB machine
    # while building); streamed nodal blocks are 16x smaller and build in
    # bounded tiles.
    use_streaming = asm == "streaming" or (asm == "auto" and est_double > 2.0)
    if use_streaming:
        est_full = estimate_streaming_gb(solver.gen.n_elems, m_max, form,
                                         zs_pt is not None, False)
    else:
        est_full = est_double
    use_single = tp == "single" or (tp == "auto" and est_full > 4.0)
    if use_single and not use_streaming:
        solver._table_dtype = np.complex64
    est = est_full / (2.0 if use_single else 1.0)
    # phase-7d mode-block re-sweeps: when even the streamed blocks exceed
    # the budget, hold only an aligned range of modes and re-run the
    # (native, threaded) sampling sweep as the mode loop advances.
    mode_block = None
    est_held = est
    if use_streaming and est > float(stream_budget_gb):
        per_mode = est / (m_max + 1)
        mode_block = max(1, int(float(stream_budget_gb) / per_mode))
        est_held = est * mode_block / (m_max + 1)
    if est_held > 32.0:
        raise MemoryError(
            f"Estimated far-assembly memory {est_held:.1f} GB "
            f"({solver.gen.n_elems} elements, {m_max} modes, "
            f"{'streaming' if use_streaming else 'tables'}, "
            f"{'single' if use_single else 'double'} precision) exceeds the "
            "32 GB gate. Reduce the mesh/mode count"
            + ("" if use_streaming else
               ", or use assembly='streaming' (phase 7b)") + ".")
    table_note = (f"{'Streamed far blocks' if use_streaming else 'Far kernel tables'} "
                  f"stored in single precision ({est:.1f} GB; double would "
                  f"need {est_full:.1f} GB)."
                  if use_single and tp == "auto" else None)

    def prepare(mm):
        if use_streaming and solver._stream is None:
            solver.enable_streaming(
                mm, efie=alpha > 0.0, mfie=alpha < 1.0,
                ibc_zs_pt=zs_pt if (zs_pt is not None and alpha > 0.0) else None,
                single_blocks=use_single, workers=workers,
                mode_block=mode_block)
        solver.prepare_operators(mm, efie=alpha > 0.0, mfie=alpha < 1.0,
                                 ibc=zs_pt is not None and alpha > 0.0,
                                 workers=workers)

    def assemble(m):
        Z = np.zeros((n_dofs, n_dofs), dtype=np.complex128)
        if alpha > 0.0:
            Z += alpha * solver.assemble_mode(m, m_max)
            if zs_pt is not None:
                Z += alpha * solver.assemble_ibc_extra(m, m_max, zs_pt, zs_elem)
        if alpha < 1.0:
            Z += (1.0 - alpha) * ETA0 * solver.assemble_mfie_mode(m, m_max)
        mask = solver.basis_mask(m)
        return Z[np.ix_(mask, mask)], mask

    def rhs(m, th, pol):
        V = np.zeros(n_dofs, dtype=np.complex128)
        if alpha > 0.0:
            V += alpha * solver.rhs_mode(m, th, pol)
        if alpha < 1.0:
            V += (1.0 - alpha) * ETA0 * solver.rhs_mfie_mode(m, th, pol)
        return V

    def farfield(m, full, th, pol):
        fth, fph = solver.farfield_mode(m, full, th, zs_pt=zs_pt)
        return fth if pol == "VV" else fph

    # Interior-resonance guard: a closed body on the plain EFIE with a
    # LOSSLESS (purely reactive) surface impedance has nothing damping the
    # cavity resonances (lossy Z_s damps them; closed PEC bodies use CFIE).
    # A single-frequency spike test is unreliable (the resonant mode's
    # conditioning rises only ~2x over background at practical mesh), so the
    # guard is honest instead: warn about the CONFIGURATION, report the
    # per-mode conditioning so frequency sweeps can be inspected, and warn
    # hard only when conditioning is numerically dangerous.
    solve_warnings: List[str] = []
    if table_note:
        solve_warnings.append(table_note)
    closed = solver.gen.node_on_axis(0) and solver.gen.node_on_axis(solver.Nn - 1)
    lossless_ibc = (zs_pt is not None and
                    float(np.max(np.abs(np.real(zs_pt)))) <
                    1e-3 * float(np.max(np.abs(zs_pt))))
    monitor = form == "efie" and closed and lossless_ibc
    if monitor:
        solve_warnings.append(
            "Closed body with a lossless (purely reactive) surface impedance "
            "on the EFIE: interior cavity resonances are undamped, and "
            "results at resonant frequencies may be silently corrupted. "
            "Inspect max_cond across the frequency sweep for spikes, add "
            "loss to Z_s, or avoid closed-body reactive IBC.")

    F, modes_used, stats = _mode_sweep(n_dofs, thetas, ("VV", "HH"), m_max,
                                       mode_tol, assemble, rhs, farfield,
                                       prepare=prepare, workers=workers,
                                       progress=progress,
                                       check_abort=check_abort,
                                       monitor_cond=monitor)
    if stats.get("max_cond", 0.0) > 1e5:
        solve_warnings.append(
            f"Mode-system conditioning reached {stats['max_cond']:.1e}: an "
            "interior resonance is likely at or near this frequency; the "
            "result is untrustworthy there.")
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(n_dofs),
        "formulation": form,
        "assembly": "streaming" if use_streaming else "tables",
        "table_precision": "single" if use_single else "double",
        "stream_mode_block": (solver._stream.mode_block
                              if solver._stream is not None else None),
        "stream_sweeps": (solver._stream.n_sweeps
                          if solver._stream is not None else 0),
        "warnings": solve_warnings,
        **stats,
    }


def solve_bor_pec(points, freq_hz: float, thetas_deg, n_modes: Optional[int] = None,
                  gauss_order: int = 4, mode_tol: float = 1e-6) -> Dict:
    """Phase-1 entry point: PEC EFIE (kept for the phase-1 gate battery)."""

    return solve_bor(points, freq_hz, thetas_deg, formulation="efie",
                     n_modes=n_modes, gauss_order=gauss_order, mode_tol=mode_tol)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: PMCHWT (homogeneous dielectric body)
# ─────────────────────────────────────────────────────────────────────────────

def solve_bor_dielectric(points, freq_hz: float, thetas_deg, eps_r: complex,
                         mu_r: complex = 1.0, n_modes: Optional[int] = None,
                         gauss_order: int = 4, mode_tol: float = 1e-6,
                         workers: int = 1, progress: Optional[Callable] = None,
                         check_abort: Optional[Callable] = None) -> Dict:
    """
    Monostatic RCS of a closed homogeneous penetrable BoR via per-mode PMCHWT.

    Unknowns per mode: J (2Nn) and M' = M/eta0 (2Nn).  Combining the
    exterior representation's interior null-field limit with the interior
    representation's exterior limit cancels every identity/jump term and
    leaves (T = EFIE operator in its medium, P = rotated-PV operator):

        [ T_e + T_i              -eta0 (P_e + P_i)      ] [J ]   [  <W, E_inc>     ]
        [ eta0 (P_e + P_i)    T_e + (eta0^2/eta_i^2) T_i ] [M'] = [ eta0 <W, H_inc> ]

    (the eta0 scalings symmetrize the block magnitudes).  The exterior far
    field radiates BOTH J and M in air.
    """

    se = BorPecSolver(points, freq_hz, gauss_order=gauss_order)
    si = BorPecSolver(points, freq_hz, gauss_order=gauss_order,
                      medium=(eps_r, mu_r))
    k = se.k
    thetas = np.atleast_1d(np.asarray(thetas_deg, dtype=float))
    rho_max = float(np.max(se.gen.nodes[:, 0]))
    st_max = float(np.max(np.abs(np.sin(np.radians(thetas)))))
    if n_modes is None:
        n_modes = int(math.ceil(k * rho_max * max(st_max, 0.05))) + 12
    m_max = n_modes
    Nn = se.Nn
    eta_ratio2 = (ETA0 / si.eta) ** 2

    def prepare(mm):
        se.prepare_operators(mm, efie=True, ibc=True, workers=workers)
        si.prepare_operators(mm, efie=True, ibc=True, workers=workers)

    def assemble(m):
        T_e = se.assemble_mode(m, m_max)
        T_i = si.assemble_mode(m, m_max)
        P_sum = ETA0 * (se.assemble_pmchwt_P(m, m_max) + si.assemble_pmchwt_P(m, m_max))
        A = np.empty((4 * Nn, 4 * Nn), dtype=np.complex128)
        A[: 2 * Nn, : 2 * Nn] = T_e + T_i
        A[: 2 * Nn, 2 * Nn:] = -P_sum
        A[2 * Nn:, : 2 * Nn] = P_sum
        A[2 * Nn:, 2 * Nn:] = T_e + eta_ratio2 * T_i
        mask = np.tile(se.basis_mask(m), 2)
        return A[np.ix_(mask, mask)], mask

    def rhs(m, th, pol):
        return np.concatenate([se.rhs_mode(m, th, pol),
                               ETA0 * se.rhs_h_mode(m, th, pol)])

    def farfield(m, full, th, pol):
        fth, fph = se.farfield_mode(m, full[: 2 * Nn], th,
                                    msol=ETA0 * full[2 * Nn:])
        return fth if pol == "VV" else fph

    F, modes_used, stats = _mode_sweep(4 * Nn, thetas, ("VV", "HH"), m_max,
                                       mode_tol, assemble, rhs, farfield,
                                       prepare=prepare, workers=workers,
                                       progress=progress,
                                       check_abort=check_abort)
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(4 * Nn),
        "formulation": "pmchwt",
        "eps_r": complex(eps_r),
        "mu_r": complex(mu_r),
        **stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: cross-surface operators + coated-PEC multi-region solver
# ─────────────────────────────────────────────────────────────────────────────

def _segment_distance(p0, p1, q0, q1) -> float:
    """Min distance between two non-intersecting 2D segments (attained at an
    endpoint of one of them)."""

    def pt_seg(c, a, b):
        ab = b - a
        t = float(np.dot(c - a, ab) / max(np.dot(ab, ab), 1e-300))
        t = min(1.0, max(0.0, t))
        return float(np.hypot(*(c - (a + t * ab))))

    return min(pt_seg(p0, q0, q1), pt_seg(p1, q0, q1),
               pt_seg(q0, p0, p1), pt_seg(q1, p0, p1))


class BorCrossOperators:
    """T (EFIE) and P (rotated-PV) Galerkin blocks between two DIFFERENT
    generatrices in one homogeneous medium: test bases on solver sp, source
    bases on solver sq (both BorPecSolver instances with the same medium).

    Pairs closer than near_factor * max(element lengths) are re-integrated
    on a dense tensor Gauss grid with the adaptive near kernels — this is
    what keeps thin coatings accurate.  The surfaces may TOUCH at shared
    endpoints (coating-termination junctions): element pairs sharing such a
    point are log-singular in the Galerkin sense and are routed to the same
    graded corner-cell quadrature the same-surface assembly uses for
    adjacent elements.  Overlapping/crossing interiors remain an error."""

    def __init__(self, sp: BorPecSolver, sq: BorPecSolver,
                 near_factor: float = 2.0, near_order: int = 12):
        if not np.isclose(complex(sp.k), complex(sq.k)):
            raise ValueError("Cross operators need both solvers in the same medium.")
        self.sp, self.sq = sp, sq
        self.k, self.eta = sp.k, sp.eta
        self.near_order = near_order
        # element-pair classification by segment distance / shared endpoints
        gp, gq = sp.gen, sq.gen
        diag = max(float(np.max(gp.nodes)) - float(np.min(gp.nodes)),
                   float(np.max(gq.nodes)) - float(np.min(gq.nodes)), 1e-9)
        touch_tol = 1e-9 * diag
        self.near_pairs = []
        self.pair_kind: Dict[Tuple[int, int], Optional[str]] = {}
        for e in range(gp.n_elems):
            p_ends = (gp.nodes[gp.elem_n0[e]], gp.nodes[gp.elem_n1[e]])
            for f in range(gq.n_elems):
                q_ends = (gq.nodes[gq.elem_n0[f]], gq.nodes[gq.elem_n1[f]])
                shared = [(a, b) for a in (0, 1) for b in (0, 1)
                          if float(np.hypot(*(p_ends[a] - q_ends[b]))) <= touch_tol]
                d = _segment_distance(p_ends[0], p_ends[1], q_ends[0], q_ends[1])
                if len(shared) > 1:
                    raise ValueError("Cross-operator surfaces share a whole "
                                     "element (overlapping geometry).")
                if shared:
                    if _JN_DEBUG.get("skip_corners"):
                        continue
                    a, b = shared[0]
                    self.near_pairs.append((e, f))
                    self.pair_kind[(e, f)] = f"corner{a}{b}"
                elif d <= touch_tol:
                    raise ValueError("Cross-operator surfaces cross or touch "
                                     "away from a shared junction endpoint.")
                elif d < near_factor * max(gp.lengths[e], gq.lengths[f]):
                    self.near_pairs.append((e, f))
                    self.pair_kind[(e, f)] = None
        self.near_set = set(self.near_pairs)
        self._G = None
        self._B = None
        self._cache: Dict = {}

    def _tables(self, m_max: int):
        if self._G is not None and self._G.shape[-1] >= m_max + 2:
            return self._G, self._B
        gp, gq = self.sp.g, self.sq.g
        Pp, Pq = len(gp.rho), len(gq.rho)
        one = np.ones((Pp, Pq))
        G = modal_kernels_fft(gp.rho[:, None] * one, gp.z[:, None] * one,
                              gq.rho[None, :] * one, gq.z[None, :] * one,
                              self.k, m_max)
        B = ibc_kernels_fft(gp.rho, gp.z, gp.trho, gp.tz,
                            gq.rho, gq.z, gq.trho, gq.tz, self.k, m_max)
        if self.near_pairs:
            near_mask = np.zeros((Pp, Pq), dtype=bool)
            for (e, f) in self.near_pairs:
                near_mask[np.ix_(gp.elem == e, gq.elem == f)] = True
            G[near_mask, :] = 0.0
            B = list(B)
            for i in range(4):
                B[i][near_mask, :] = 0.0
            B = tuple(B)
        self._G, self._B = G, B
        return G, B

    def _near_data(self, e: int, f: int, m_max: int):
        key = (e, f)
        cache = self._cache.setdefault(m_max, {})
        if key in cache:
            return cache[key]
        kind = self.pair_kind.get(key)
        if kind is not None:
            # shared junction endpoint: graded corner cells (log singularity)
            s, sp_, W = _cell_points(kind)
        else:
            n = self.near_order
            xg, wg = np.polynomial.legendre.leggauss(n)
            u = 0.5 * (xg + 1.0); w1 = 0.5 * wg
            S, SP = np.meshgrid(u, u, indexing="ij")
            W = np.outer(w1, w1).ravel()
            s, sp_ = S.ravel(), SP.ravel()
        rho_p, z_p, tr_p, tz_p, T0p, T1p, D0p, D1p, Lp = _points_on_element(self.sp.gen, e, s)
        rho_q, z_q, tr_q, tz_q, T0q, T1q, D0q, D1q, Lq = _points_on_element(self.sq.gen, f, sp_)
        Gm = modal_kernels_near(rho_p, z_p, rho_q, z_q, self.k, m_max)
        Bn = ibc_kernels_near(rho_p, z_p, np.full_like(rho_p, tr_p), np.full_like(rho_p, tz_p),
                              rho_q, z_q, np.full_like(rho_q, tr_q), np.full_like(rho_q, tz_q),
                              self.k, m_max)
        data = (W * Lp * Lq,
                rho_p, tr_p, tz_p, np.vstack([T0p, T1p]), np.vstack([D0p, D1p]),
                rho_q, tr_q, tz_q, np.vstack([T0q, T1q]), np.vstack([D0q, D1q]),
                Gm, Bn)
        cache[key] = data
        return data

    def assemble_T(self, m: int, m_max: int) -> np.ndarray:
        """Cross EFIE operator [2Np, 2Nq] (same normalization as
        BorPecSolver.assemble_mode, C = j k eta 2pi of this medium)."""

        k = self.k
        gp, gq = self.sp.g, self.sq.g
        G, _ = self._tables(m_max)
        Gm, Gc, Gs = kernels_for_mode(G, m)
        ztt, ztf, zft, zff = _pair_blocks(
            m, k,
            gp.rho, gp.trho, gp.tz, self.sp.B_T, self.sp.B_D, gp.w,
            gq.rho, gq.trho, gq.tz, self.sq.B_T, self.sq.B_D, gq.w,
            Gm, Gc, Gs,
        )
        for (e, f) in self.near_pairs:
            (w, rho_p, tr_p, tz_p, Tp, Dp,
             rho_q, tr_q, tz_q, Tq, Dq, Gtab, _) = self._near_data(e, f, m_max)
            Gn, Gcn, Gsn = kernels_for_mode(Gtab, m)
            rr = rho_p * rho_q * w
            ktt = rr * ((tr_p * tr_q) * Gcn + (tz_p * tz_q) * Gn)
            ksc = w * Gn
            ktf = rr * (tr_p * Gsn)
            kft = -rr * (tr_q * Gsn)
            kff = rr * Gcn
            btt = np.einsum("ip,p,jp->ij", Tp, ktt, Tq) - (1.0 / k ** 2) * np.einsum("ip,p,jp->ij", Dp, ksc, Dq)
            btf = np.einsum("ip,p,jp->ij", Tp, ktf, Tq) - (1j * m / k ** 2) * np.einsum("ip,p,jp->ij", Dp, ksc, Tq)
            bft = np.einsum("ip,p,jp->ij", Tp, kft, Tq) + (1j * m / k ** 2) * np.einsum("ip,p,jp->ij", Tp, ksc, Dq)
            bff = np.einsum("ip,p,jp->ij", Tp, kff, Tq) - (m ** 2 / k ** 2) * np.einsum("ip,p,jp->ij", Tp, ksc, Tq)
            rows = np.array([e, e + 1]); cols = np.array([f, f + 1])
            ztt[np.ix_(rows, cols)] += btt
            ztf[np.ix_(rows, cols)] += btf
            zft[np.ix_(rows, cols)] += bft
            zff[np.ix_(rows, cols)] += bff

        C = 1j * k * self.eta * 2.0 * np.pi
        Np, Nq = self.sp.Nn, self.sq.Nn
        Z = np.empty((2 * Np, 2 * Nq), dtype=np.complex128)
        Z[:Np, :Nq] = C * ztt
        Z[:Np, Nq:] = C * ztf
        Z[Np:, :Nq] = C * zft
        Z[Np:, Nq:] = C * zff
        return Z

    def assemble_P(self, m: int, m_max: int) -> np.ndarray:
        """Cross rotated-PV operator [2Np, 2Nq] (see assemble_pmchwt_P)."""

        gp, gq = self.sp.g, self.sq.g
        _, Bt = self._tables(m_max)
        wrho_p = gp.w * gp.rho
        wrho_q = gq.w * gq.rho
        blocks = []
        for uv in range(4):
            Km = mfie_for_mode(Bt[uv], m, m_max)
            blocks.append(2.0 * np.pi * (self.sp.B_T * wrho_p[None, :]) @ Km @ (self.sq.B_T * wrho_q[None, :]).T)
        for (e, f) in self.near_pairs:
            (w, rho_p, _, _, Tp, _, rho_q, _, _, Tq, _, _, Bn) = self._near_data(e, f, m_max)
            rr = rho_p * rho_q * w
            rows = np.array([e, e + 1]); cols = np.array([f, f + 1])
            for uv, tgt in enumerate(blocks):
                Km = mfie_for_mode(Bn[uv], m, m_max)
                tgt[np.ix_(rows, cols)] += 2.0 * np.pi * np.einsum("ip,p,jp->ij", Tp, rr * Km, Tq)
        Btt, Btf, Bft, Bff = blocks
        Np, Nq = self.sp.Nn, self.sq.Nn
        P = np.empty((2 * Np, 2 * Nq), dtype=np.complex128)
        P[:Np, :Nq] = -Btf
        P[:Np, Nq:] = Btt
        P[Np:, :Nq] = -Bff
        P[Np:, Nq:] = Bft
        return P

    def prepare(self, m_max: int) -> None:
        """Warm every table/near cache (see BorPecSolver.prepare_operators)."""
        self._tables(m_max)
        for e, f in self.near_pairs:
            self._near_data(e, f, m_max)


def solve_bor_coated_pec(points_outer, points_core, freq_hz: float, thetas_deg,
                         eps_r: complex, mu_r: complex = 1.0,
                         n_modes: Optional[int] = None, gauss_order: int = 4,
                         mode_tol: float = 1e-6, near_factor: float = 2.0,
                         near_order: int = 12, workers: int = 1,
                         progress: Optional[Callable] = None,
                         check_abort: Optional[Callable] = None) -> Dict:
    """
    Monostatic RCS of a PEC core (generatrix points_core) fully covered by a
    homogeneous coating with outer surface points_outer (both closed, both
    traversed +z end to -z end so left-of-travel normals face their exterior).

    Unknowns per mode: J_o, M'_o = M_o/eta0 on the outer interface, J_c on
    the core.  PMCHWT rows on the outer interface pick up cross terms from
    J_c radiating in the layer; the core row is the EFIE in the layer:

      [ T_e+T_L          -eta0(P_e+P_L)          -T_L^oc      ] [J_o ]   [ V_E      ]
      [ eta0(P_e+P_L)    T_e+(eta0/eta_L)^2 T_L  -eta0 P_L^oc ] [M'_o] = [ eta0 V_H ]
      [ T_L^co           -eta0 P_L^co            -T_L^cc      ] [J_c ]   [ 0        ]

    Only (J_o, M_o) radiate in air.
    """

    se = BorPecSolver(points_outer, freq_hz, gauss_order=gauss_order)
    sLo = BorPecSolver(points_outer, freq_hz, gauss_order=gauss_order,
                       medium=(eps_r, mu_r))
    sLc = BorPecSolver(points_core, freq_hz, gauss_order=gauss_order,
                       medium=(eps_r, mu_r))
    Xoc = BorCrossOperators(sLo, sLc, near_factor=near_factor, near_order=near_order)
    Xco = BorCrossOperators(sLc, sLo, near_factor=near_factor, near_order=near_order)
    k = se.k
    thetas = np.atleast_1d(np.asarray(thetas_deg, dtype=float))
    rho_max = float(np.max(se.gen.nodes[:, 0]))
    st_max = float(np.max(np.abs(np.sin(np.radians(thetas)))))
    if n_modes is None:
        n_modes = int(math.ceil(k * rho_max * max(st_max, 0.05))) + 12
    m_max = n_modes
    No, Nc = se.Nn, sLc.Nn
    eta_ratio2 = (ETA0 / sLo.eta) ** 2
    ntot = 4 * No + 2 * Nc

    iJ = slice(0, 2 * No); iM = slice(2 * No, 4 * No); iC = slice(4 * No, ntot)

    def prepare(mm):
        se.prepare_operators(mm, efie=True, ibc=True, workers=workers)
        sLo.prepare_operators(mm, efie=True, ibc=True, workers=workers)
        sLc.prepare_operators(mm, efie=True, workers=workers)
        Xoc.prepare(mm)
        Xco.prepare(mm)

    def assemble(m):
        T_e = se.assemble_mode(m, m_max)
        T_Lo = sLo.assemble_mode(m, m_max)
        P_sum = ETA0 * (se.assemble_pmchwt_P(m, m_max) + sLo.assemble_pmchwt_P(m, m_max))
        A = np.zeros((ntot, ntot), dtype=np.complex128)
        A[iJ, iJ] = T_e + T_Lo
        A[iJ, iM] = -P_sum
        A[iJ, iC] = -Xoc.assemble_T(m, m_max)
        A[iM, iJ] = P_sum
        A[iM, iM] = T_e + eta_ratio2 * T_Lo
        A[iM, iC] = -ETA0 * Xoc.assemble_P(m, m_max)
        A[iC, iJ] = Xco.assemble_T(m, m_max)
        A[iC, iM] = -ETA0 * Xco.assemble_P(m, m_max)
        A[iC, iC] = -sLc.assemble_mode(m, m_max)
        mask_o = se.basis_mask(m)
        mask = np.concatenate([mask_o, mask_o, sLc.basis_mask(m)])
        return A[np.ix_(mask, mask)], mask

    def rhs(m, th, pol):
        V = np.zeros(ntot, dtype=np.complex128)
        V[iJ] = se.rhs_mode(m, th, pol)
        V[iM] = ETA0 * se.rhs_h_mode(m, th, pol)
        return V

    def farfield(m, full, th, pol):
        fth, fph = se.farfield_mode(m, full[iJ], th, msol=ETA0 * full[iM])
        return fth if pol == "VV" else fph

    F, modes_used, stats = _mode_sweep(ntot, thetas, ("VV", "HH"), m_max,
                                       mode_tol, assemble, rhs, farfield,
                                       prepare=prepare, workers=workers,
                                       progress=progress,
                                       check_abort=check_abort)
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(ntot),
        "formulation": "pmchwt-coated",
        "eps_r": complex(eps_r),
        "mu_r": complex(mu_r),
        **stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Partial coatings: coating terminating ON the PEC surface (junctions).
# ─────────────────────────────────────────────────────────────────────────────

_JN_DEBUG: Dict[str, object] = {}   # debug-only switches for the gate harness

def solve_bor_partial_coating(points_interface, points_covered, bare_pieces,
                              freq_hz: float, thetas_deg, eps_r: complex,
                              mu_r: complex = 1.0, bare_zs=None,
                              n_modes: Optional[int] = None,
                              gauss_order: int = 4, mode_tol: float = 1e-6,
                              near_factor: float = 2.0, near_order: int = 12,
                              workers: int = 1, progress: Optional[Callable] = None,
                              check_abort: Optional[Callable] = None) -> Dict:
    """
    Monostatic RCS of a PEC body PARTIALLY covered by a homogeneous coating:
    the dielectric interface S_d (points_interface) terminates on the PEC
    surface at junction circles where air, coating, and conductor meet.

    points_covered is the coated part of the core, bare_pieces the list of
    uncovered PEC generatrix pieces (0, 1, or 2 — cap or band coatings).
    All pieces are drawn in the global +z -> -z traversal with left-of-travel
    normals facing away from the surface they bound (exterior/air for S_d
    and the bare pieces, into the coating for the covered core).

    Formulation: exterior region bounded by S_d + bare pieces (currents J_d,
    M_d, J_1p in air), layer region bounded by S_d + covered core (currents
    -J_d, -M_d, J_2 in the coating medium).  PMCHWT rows on S_d, EFIE rows
    on each PEC piece; all with the same phase-3 operator blocks.  Junction
    conditions (per junction circle A):

      * through-current continuity ties the chain-end coefficients:
        J_1p(A) = (+t, +phi) J_d(A)  (air chain runs S_d and the bare piece
        in the same global traversal), and
        J_2(A)  = (+t, -phi) J_d(A)  (the layer chain traverses S_d REVERSED:
        the -J_d current in the flipped frame has +t and -phi components);
      * M_t(A) = 0 (tangential E along the junction circle vanishes on the
        conductor), while M_phi(A) stays free with its natural half-triangle
        end basis (it carries the normal-E wedge behavior).

    The constraints enter Galerkin-style: A_red = Q^T A_full Q — the tied
    row is the sum of the piece rows, exactly the classical BoR junction
    treatment (Putnam / Medgyesi-Mitschang).

    bare_zs (optional): per-piece Leontovich surface impedance — a list
    matching bare_pieces of None / complex scalar / per-element complex
    arrays (tapers).  The eliminated magnetic current M_1 = -Z_s n_hat x J_1
    keeps the piece's own operator on the validated Gauss-point IBC path
    (assemble_ibc_extra) and radiates onto OTHER surfaces through the
    existing cross T/P operators via the nodal column map
    (M_1t, M_1phi) = (+Z_s J_1phi, -Z_s J_1t).  At a junction adjoining an
    impedance piece the conductor no longer shorts E_phi, so the M_t(A) = 0
    condition becomes the tie M_d,t(A) = Z_s(A) J_phi(A).
    """

    sd_e = BorPecSolver(points_interface, freq_hz, gauss_order=gauss_order)
    sd_L = BorPecSolver(points_interface, freq_hz, gauss_order=gauss_order,
                        medium=(eps_r, mu_r))
    s2_L = BorPecSolver(points_covered, freq_hz, gauss_order=gauss_order,
                        medium=(eps_r, mu_r))
    bares = [BorPecSolver(p, freq_hz, gauss_order=gauss_order)
             for p in bare_pieces]

    # ── per-piece surface impedance (None = PEC) ──
    if bare_zs is None:
        bare_zs = [None] * len(bares)
    if len(bare_zs) != len(bares):
        raise ValueError("bare_zs must have one entry per bare piece.")
    zs_elems: List[Optional[np.ndarray]] = []
    zs_ptss: List[Optional[np.ndarray]] = []
    S_maps: List[Optional[np.ndarray]] = []
    for b, zs in zip(bares, bare_zs):
        ne = b.gen.n_elems
        zs_arr = None
        if zs is not None:
            za = np.asarray(zs, dtype=complex)
            zs_arr = np.full(ne, complex(za)) if za.ndim == 0 else za.astype(complex)
            if len(zs_arr) != ne:
                raise ValueError("Per-element bare_zs array length must match "
                                 "the piece's element count.")
            if not np.any(np.abs(zs_arr) > 0.0):
                zs_arr = None
        zs_elems.append(zs_arr)
        if zs_arr is None:
            zs_ptss.append(None)
            S_maps.append(None)
        else:
            zs_ptss.append(zs_arr[b.g.elem])
            zn = np.empty(b.Nn, dtype=complex)
            zn[0] = zs_arr[0]
            zn[-1] = zs_arr[-1]
            zn[1:-1] = 0.5 * (zs_arr[:-1] + zs_arr[1:])
            S = np.zeros((2 * b.Nn, 2 * b.Nn), dtype=complex)
            S[:b.Nn, b.Nn:] = np.diag(zn)        # M_1t   = +Z_s J_1phi
            S[b.Nn:, :b.Nn] = -np.diag(zn)       # M_1phi = -Z_s J_1t
            S_maps.append(S)

    all_nodes = np.vstack([sd_e.gen.nodes, s2_L.gen.nodes] +
                          [b.gen.nodes for b in bares])
    diag = max(float(np.ptp(all_nodes[:, 0])) + float(np.ptp(all_nodes[:, 1])), 1e-9)
    jn_tol = 1e-8 * diag

    # ── junction detection: cluster non-axis chain endpoints ──
    def endpoints(solver):
        gen = solver.gen
        out = []
        for node in (0, gen.n_nodes - 1):
            if gen.node_on_axis(node):
                out.append((node, None))
            else:
                out.append((node, gen.nodes[node]))
        return out

    junctions: List[Dict] = []   # {pos, d_node, c_node, bare: (piece, node)}

    def register(kind_key, piece_idx, node, pos):
        for jn in junctions:
            if float(np.hypot(*(jn["pos"] - pos))) <= jn_tol:
                if kind_key in jn:
                    raise ValueError("Two same-role chain ends meet at one junction.")
                jn[kind_key] = (piece_idx, node) if kind_key == "bare" else node
                return
        junctions.append({"pos": pos,
                          kind_key: (piece_idx, node) if kind_key == "bare" else node})

    for node, pos in endpoints(sd_e):
        if pos is not None:
            register("d_node", None, node, pos)
    for node, pos in endpoints(s2_L):
        if pos is not None:
            register("c_node", None, node, pos)
    for bi, b in enumerate(bares):
        for node, pos in endpoints(b):
            if pos is not None:
                register("bare", bi, node, pos)
    for jn in junctions:
        if "d_node" not in jn or "c_node" not in jn or "bare" not in jn:
            raise ValueError(
                "Every off-axis chain endpoint must be a coating-termination "
                "junction where the interface, the covered core, and exactly "
                f"one bare piece meet; found an incomplete junction at "
                f"(rho, z) = ({jn['pos'][0]:.6g}, {jn['pos'][1]:.6g}).")
        bi, bn = jn["bare"]
        za = zs_elems[bi]
        jn["zs"] = complex(za[0 if bn == 0 else -1]) if za is not None else 0.0

    solve_warnings: List[str] = []
    for jn in junctions:
        if abs(jn["zs"]) > 0.02 * ETA0:
            solve_warnings.append(
                f"Surface impedance is {abs(jn['zs']):.1f} ohm at the coating "
                f"junction (rho, z) = ({jn['pos'][0]:.4g}, {jn['pos'][1]:.4g}): "
                "an abrupt Z_s step AT a coating edge is an ill-defined "
                "sheet-model limit (E_phi is discontinuous along the junction "
                "line) and the solution does not mesh-converge there — expect "
                "~0.5 dB accuracy.  Taper Z_s toward zero at the junction "
                "(the physical edge treatment) for converged results.")

    # ── cross operators (touching allowed at the junctions) ──
    xkw = dict(near_factor=near_factor, near_order=near_order)
    X_d2 = BorCrossOperators(sd_L, s2_L, **xkw)
    X_2d = BorCrossOperators(s2_L, sd_L, **xkw)
    X_d1 = [BorCrossOperators(sd_e, b, **xkw) for b in bares]
    X_1d = [BorCrossOperators(b, sd_e, **xkw) for b in bares]
    X_11 = {(i, j): BorCrossOperators(bares[i], bares[j], **xkw)
            for i in range(len(bares)) for j in range(len(bares)) if i != j}

    k = sd_e.k
    thetas = np.atleast_1d(np.asarray(thetas_deg, dtype=float))
    rho_max = max([float(np.max(sd_e.gen.nodes[:, 0])),
                   float(np.max(s2_L.gen.nodes[:, 0]))] +
                  [float(np.max(b.gen.nodes[:, 0])) for b in bares])
    st_max = float(np.max(np.abs(np.sin(np.radians(thetas)))))
    if n_modes is None:
        n_modes = int(math.ceil(k * rho_max * max(st_max, 0.05))) + 12
    m_max = n_modes
    eta_ratio2 = (ETA0 / sd_L.eta) ** 2

    Nd, N2 = sd_e.Nn, s2_L.Nn
    N1 = [b.Nn for b in bares]
    off_Jd, off_M, off_J2 = 0, 2 * Nd, 4 * Nd
    off_J1 = []
    acc = 4 * Nd + 2 * N2
    for n in N1:
        off_J1.append(acc)
        acc += 2 * n
    n_full = acc

    def prepare(mm):
        sd_e.prepare_operators(mm, efie=True, ibc=True, workers=workers)
        sd_L.prepare_operators(mm, efie=True, ibc=True, workers=workers)
        s2_L.prepare_operators(mm, efie=True, workers=workers)
        for bi, b in enumerate(bares):
            b.prepare_operators(mm, efie=True, ibc=zs_elems[bi] is not None,
                                workers=workers)
        for X in [X_d2, X_2d] + X_d1 + X_1d + list(X_11.values()):
            X.prepare(mm)

    # ── junction-aware constraint matrix Q(m) ──
    _Q_cache: Dict[int, np.ndarray] = {}

    def build_Q(m):
        Q = _Q_cache.get(m)
        if Q is not None:
            return Q
        d_jn_nodes = {jn["d_node"] for jn in junctions}
        c_jn_nodes = {jn["c_node"] for jn in junctions}
        b_jn_nodes = {(jn["bare"][0], jn["bare"][1]) for jn in junctions}
        # M_t(A) = 0 is kept at IBC junctions too.  Alternatives tried and
        # rejected against the mixed-impedance sphere cross-check: a pointwise
        # tie M_t(A) = Z_s(A) J_phi(A) fails at +7.6 dB (the wedge-line field
        # limits are singular; the Leontovich relation holds on the impedance
        # sheet, not on the interface's approach to the line), and a free
        # half-basis end DOF fails at 5.5 dB non-convergent (the lone M_t end
        # DOF has no jump partner controlling it).  The E_phi != 0 wall value
        # that M_t(A) = 0 suppresses is finite, so the error is a converging
        # one-node approximation like the corner treatment.

        def surf_mask(solver, jn_nodes, is_m):
            """(t_active, phi_active) with axis rules; junction nodes stay
            active here (masters/M_phi) unless excluded below."""
            Nn = solver.Nn
            t_act = np.ones(Nn, dtype=bool)
            f_act = np.ones(Nn, dtype=bool)
            for end in (0, Nn - 1):
                if solver.gen.node_on_axis(end):
                    t_act[end] = (abs(m) == 1)
                    f_act[end] = False
                elif end in jn_nodes:
                    if is_m:
                        t_act[end] = False   # M_t = 0 at the conductor line
                else:
                    t_act[end] = False       # true open edge (not expected)
        # NOTE: masks for slave nodes are applied by the tie logic below.
            return t_act, f_act

        # column ownership: -1 = masked, -2 = slave, >=0 reduced index
        col = np.full(n_full, -1, dtype=int)
        red = 0

        def assign(offset, acts):
            nonlocal red
            t_act, f_act = acts
            Nn = len(t_act)
            for i in range(Nn):
                if t_act[i]:
                    col[offset + i] = red; red += 1
            for i in range(Nn):
                if f_act[i]:
                    col[offset + Nn + i] = red; red += 1

        assign(off_Jd, surf_mask(sd_e, d_jn_nodes, False))
        assign(off_M, surf_mask(sd_e, d_jn_nodes, True))
        t2, f2 = surf_mask(s2_L, c_jn_nodes, False)
        for jn in junctions:
            t2[jn["c_node"]] = False; f2[jn["c_node"]] = False
        assign(off_J2, (t2, f2))
        b_acts = []
        for bi, b in enumerate(bares):
            tb, fb = surf_mask(b, {n for (p, n) in b_jn_nodes if p == bi}, False)
            for jn in junctions:
                if jn["bare"][0] == bi:
                    tb[jn["bare"][1]] = False; fb[jn["bare"][1]] = False
            b_acts.append((tb, fb))
            assign(off_J1[bi], (tb, fb))

        Q = np.zeros((n_full, red), dtype=np.complex128)
        active = col >= 0
        Q[np.flatnonzero(active), col[active]] = 1.0
        # ties: slaves follow the interface's junction-end coefficients
        tie_mode = _JN_DEBUG.get("tie_mode", "full")
        extra = []
        for jn in junctions:
            dn = jn["d_node"]
            cn = jn["c_node"]
            bi, bn = jn["bare"]
            master_t = col[off_Jd + dn]
            master_f = col[off_Jd + Nd + dn]
            if master_t >= 0:
                if tie_mode == "none":
                    extra += [off_J2 + cn, off_J1[bi] + bn]
                else:
                    Q[off_J2 + cn, master_t] = 1.0
                    Q[off_J1[bi] + bn, master_t] = 1.0
            if master_f >= 0:
                if tie_mode in ("none", "tonly"):
                    extra += [off_J2 + N2 + cn, off_J1[bi] + N1[bi] + bn]
                else:
                    Q[off_J2 + N2 + cn, master_f] = -1.0
                    Q[off_J1[bi] + N1[bi] + bn, master_f] = 1.0
        if extra:
            Qx = np.zeros((n_full, red + len(extra)), dtype=np.complex128)
            Qx[:, :red] = Q
            for i, row in enumerate(extra):
                Qx[row, red + i] = 1.0
            Q = Qx
        _Q_cache[m] = Q
        return Q

    def assemble(m):
        A = np.zeros((n_full, n_full), dtype=np.complex128)
        sl_Jd = slice(off_Jd, off_Jd + 2 * Nd)
        sl_M = slice(off_M, off_M + 2 * Nd)
        sl_J2 = slice(off_J2, off_J2 + 2 * N2)
        T_e = sd_e.assemble_mode(m, m_max)
        T_L = sd_L.assemble_mode(m, m_max)
        P_sum = ETA0 * (sd_e.assemble_pmchwt_P(m, m_max) + sd_L.assemble_pmchwt_P(m, m_max))
        iE = sl_Jd; iH = sl_M            # row blocks share the column layout
        A[iE, sl_Jd] = T_e + T_L
        A[iE, sl_M] = -P_sum
        A[iE, sl_J2] = -X_d2.assemble_T(m, m_max)
        A[iH, sl_Jd] = P_sum
        A[iH, sl_M] = T_e + eta_ratio2 * T_L
        A[iH, sl_J2] = -ETA0 * X_d2.assemble_P(m, m_max)
        # The covered-core row is -LayEq (NOT +LayEq as in the junction-free
        # coated solver): every row block must carry the same region-equation
        # orientation — rows here are -(AirEq) on air-bounded surfaces and
        # -(AirEq - LayEq) on the interface — or the Q^T junction fold sums
        # the layer-region equation with the wrong sign and the through-DOF
        # rows are inconsistent (caught by the eps=1 cap gate at +9 dB).
        A[sl_J2, sl_Jd] = -X_2d.assemble_T(m, m_max)
        A[sl_J2, sl_M] = ETA0 * X_2d.assemble_P(m, m_max)
        A[sl_J2, sl_J2] = s2_L.assemble_mode(m, m_max)
        for bi, b in enumerate(bares):
            sl_b = slice(off_J1[bi], off_J1[bi] + 2 * N1[bi])
            Sb = S_maps[bi]
            # eliminated M_1 = -Z_s n x J_1 radiates onto the other surfaces
            # through the SAME cross operators via the nodal column map Sb
            A[iE, sl_b] = X_d1[bi].assemble_T(m, m_max)
            A[iH, sl_b] = ETA0 * X_d1[bi].assemble_P(m, m_max)
            if Sb is not None and not _JN_DEBUG.get("no_cross_m1"):
                A[iE, sl_b] += -X_d1[bi].assemble_P(m, m_max) @ Sb
                A[iH, sl_b] += (1.0 / ETA0) * (X_d1[bi].assemble_T(m, m_max) @ Sb)
            A[sl_b, sl_Jd] = X_1d[bi].assemble_T(m, m_max)
            A[sl_b, sl_M] = -ETA0 * X_1d[bi].assemble_P(m, m_max)
            A[sl_b, sl_b] = b.assemble_mode(m, m_max)
            if zs_elems[bi] is not None:
                A[sl_b, sl_b] += b.assemble_ibc_extra(m, m_max, zs_ptss[bi],
                                                      zs_elems[bi])
            for bj in range(len(bares)):
                if bj != bi:
                    sl_bj = slice(off_J1[bj], off_J1[bj] + 2 * N1[bj])
                    A[sl_b, sl_bj] = X_11[(bi, bj)].assemble_T(m, m_max)
                    if S_maps[bj] is not None:
                        A[sl_b, sl_bj] += -X_11[(bi, bj)].assemble_P(m, m_max) @ S_maps[bj]
        Q = build_Q(m)
        return Q.T @ A @ Q, None

    def rhs(m, th, pol):
        V = np.zeros(n_full, dtype=np.complex128)
        V[off_Jd:off_Jd + 2 * Nd] = sd_e.rhs_mode(m, th, pol)
        V[off_M:off_M + 2 * Nd] = ETA0 * sd_e.rhs_h_mode(m, th, pol)
        for bi, b in enumerate(bares):
            V[off_J1[bi]:off_J1[bi] + 2 * N1[bi]] = b.rhs_mode(m, th, pol)
        return build_Q(m).T @ V

    def farfield(m, x_red, th, pol):
        x = build_Q(m) @ x_red
        if "capture" in _JN_DEBUG:
            _JN_DEBUG["capture"].append((m, th, pol, x.copy()))
        fth, fph = sd_e.farfield_mode(m, x[off_Jd:off_Jd + 2 * Nd], th,
                                      msol=ETA0 * x[off_M:off_M + 2 * Nd])
        for bi, b in enumerate(bares):
            ft, fp = b.farfield_mode(m, x[off_J1[bi]:off_J1[bi] + 2 * N1[bi]],
                                     th, zs_pt=zs_ptss[bi])
            fth += ft; fph += fp
        return fth if pol == "VV" else fph

    F, modes_used, stats = _mode_sweep(n_full, thetas, ("VV", "HH"), m_max,
                                       mode_tol, assemble, rhs, farfield,
                                       prepare=prepare, workers=workers,
                                       progress=progress,
                                       check_abort=check_abort)
    return {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(n_full),
        "n_junctions": len(junctions),
        "formulation": "pmchwt-partial-coating",
        "eps_r": complex(eps_r),
        "mu_r": complex(mu_r),
        "warnings": solve_warnings,
        **stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Generic multi-region assembly (phase 6: multi-layer coatings).
#
# The phase-5 lesson mechanized: every block and every junction tie follows
# from the REGION PRESCRIPTION.  Each region r contributes, for surfaces
# s, s' on its boundary with orientation weights sigma_rs (+1 if r is the
# surface's reference/exterior side, -1 otherwise):
#
#   A[J_s,  J_s' ] += sigma_rs sigma_rs' T_r^{ss'}
#   A[J_s,  M'_s'] += -eta0 sigma_rs sigma_rs' P_r^{ss'}
#   A[M'_s, J_s' ] += eta0 sigma_rs sigma_rs' P_r^{ss'}
#   A[M'_s, M'_s'] += (eta0/eta_r)^2 sigma_rs sigma_rs' T_r^{ss'}
#
# (M' = M/eta0; conductors carry no M rows/cols).  PMCHWT emerges on every
# interface, the phase-3/5 systems are special cases (including the phase-5
# row-orientation fix, which this construction produces automatically), and
# junction ties follow the sigma/traversal rule below.
# ─────────────────────────────────────────────────────────────────────────────

class _MultiRegionBor:
    """surfaces: list of (points, is_conductor).  regions: list of dicts
    {"medium": None|(eps, mu), "bounds": [(surf_idx, sigma), ...],
    "exterior": bool}.  Interfaces bounding the exterior region must carry
    sigma = +1 there (the far field then sums their (J, M) directly)."""

    def __init__(self, surfaces, regions, freq_hz: float, gauss_order: int = 4,
                 near_factor: float = 2.0, near_order: int = 12):
        self.regions = regions
        self.n_surf = len(surfaces)
        self.is_cond = [bool(c) for (_, c) in surfaces]
        self.adj: List[List[int]] = [[] for _ in surfaces]   # regions per surface
        self.sigma: Dict[Tuple[int, int], int] = {}
        for ri, reg in enumerate(regions):
            for (si, sg) in reg["bounds"]:
                self.adj[si].append(ri)
                self.sigma[(ri, si)] = int(sg)
        self.ext_region = next(ri for ri, r in enumerate(regions) if r.get("exterior"))
        for (si, sg) in regions[self.ext_region]["bounds"]:
            if sg != +1:
                raise ValueError("Exterior-bounding surfaces must have sigma=+1.")
        # per-(surface, region) solvers
        self.solv: Dict[Tuple[int, int], BorPecSolver] = {}
        for si, (pts, _) in enumerate(surfaces):
            for ri in self.adj[si]:
                self.solv[(si, ri)] = BorPecSolver(
                    pts, freq_hz, gauss_order=gauss_order,
                    medium=regions[ri]["medium"])
        # cross operators per region and ordered surface pair
        self.X: Dict[Tuple[int, int, int], BorCrossOperators] = {}
        for ri, reg in enumerate(regions):
            ids = [si for (si, _) in reg["bounds"]]
            for si in ids:
                for sj in ids:
                    if si != sj:
                        self.X[(ri, si, sj)] = BorCrossOperators(
                            self.solv[(si, ri)], self.solv[(sj, ri)],
                            near_factor=near_factor, near_order=near_order)
        # DOF layout: per surface J (2Nn) then M' (2Nn, interfaces only)
        self.Nn = [self.solv[(si, self.adj[si][0])].Nn for si in range(self.n_surf)]
        self.off_J: List[int] = []
        self.off_M: List[Optional[int]] = []
        acc = 0
        for si in range(self.n_surf):
            self.off_J.append(acc)
            acc += 2 * self.Nn[si]
            if self.is_cond[si]:
                self.off_M.append(None)
            else:
                self.off_M.append(acc)
                acc += 2 * self.Nn[si]
        self.n_full = acc

        # ── junction detection over off-axis endpoints ──
        all_pts = np.vstack([self.solv[(si, self.adj[si][0])].gen.nodes
                             for si in range(self.n_surf)])
        diag = max(float(np.ptp(all_pts[:, 0])) + float(np.ptp(all_pts[:, 1])), 1e-9)
        jn_tol = 1e-8 * diag
        self.junctions: List[List[Tuple[int, int]]] = []   # [(surf, node), ...]
        for si in range(self.n_surf):
            gen = self.solv[(si, self.adj[si][0])].gen
            for node in (0, gen.n_nodes - 1):
                if gen.node_on_axis(node):
                    continue
                pos = gen.nodes[node]
                for jn in self.junctions:
                    s0, n0 = jn[0]
                    p0 = self.solv[(s0, self.adj[s0][0])].gen.nodes[n0]
                    if float(np.hypot(*(p0 - pos))) <= jn_tol:
                        jn.append((si, node))
                        break
                else:
                    self.junctions.append([(si, node)])
        for jn in self.junctions:
            if len(jn) < 2:
                si, node = jn[0]
                raise ValueError(f"Surface {si} has an off-axis free endpoint "
                                 "that is not part of a junction.")
            # master = first INTERFACE at the junction (an interface carries
            # the M master DOFs; a pure-conductor junction is just a corner
            # and any member may lead); every slave must share a region.
            mi = next((idx for idx, (si, _) in enumerate(jn)
                       if not self.is_cond[si]), 0)
            jn[0], jn[mi] = jn[mi], jn[0]
            mstr = jn[0]
            for (si, node) in jn[1:]:
                if not set(self.adj[si]) & set(self.adj[mstr[0]]):
                    raise ValueError("Junction surface shares no region with "
                                     "the junction master.")
        self._Q_cache: Dict[int, np.ndarray] = {}

    # ── operator plumbing ──
    def prepare(self, m_max: int, workers: int = 1) -> None:
        for (si, ri), s in self.solv.items():
            s.prepare_operators(m_max, efie=True, ibc=not self.is_cond[si],
                                workers=workers)
        for X in self.X.values():
            X.prepare(m_max)

    def _dir(self, si: int, node: int) -> int:
        """+1 if the drawn tangent points INTO the junction node (chain end)."""
        return +1 if node != 0 else -1

    def build_Q(self, m: int) -> np.ndarray:
        Q = self._Q_cache.get(m)
        if Q is not None:
            return Q
        jn_nodes = {(si, node) for jn in self.junctions for (si, node) in jn}
        slave = {(si, node) for jn in self.junctions for (si, node) in jn[1:]}
        # phase-5 condition at CONDUCTOR junctions: E_phi shorts on the metal
        # line, so M_t = 0 for every interface end there (the master keeps
        # its free M_phi half-basis; slaves get only the phi tie back)
        m_t_masked = {(si, node) for jn in self.junctions
                      if any(self.is_cond[sj] for (sj, _) in jn)
                      for (si, node) in jn if not self.is_cond[si]}

        col = np.full(self.n_full, -1, dtype=int)
        red = 0

        def assign(offset, si, is_m):
            nonlocal red
            Nn = self.Nn[si]
            gen = self.solv[(si, self.adj[si][0])].gen
            t_act = np.ones(Nn, dtype=bool)
            f_act = np.ones(Nn, dtype=bool)
            for end in (0, Nn - 1):
                if gen.node_on_axis(end):
                    t_act[end] = (abs(m) == 1)
                    f_act[end] = False
                elif (si, end) in slave:
                    t_act[end] = False
                    f_act[end] = False
                elif (si, end) in jn_nodes:
                    if is_m and (si, end) in m_t_masked:
                        t_act[end] = False   # M_t = 0 at the conductor line
                else:
                    t_act[end] = False       # open edge (not expected)
            for i in range(Nn):
                if t_act[i]:
                    col[offset + i] = red; red += 1
            for i in range(Nn):
                if f_act[i]:
                    col[offset + Nn + i] = red; red += 1

        for si in range(self.n_surf):
            assign(self.off_J[si], si, False)
            if self.off_M[si] is not None:
                assign(self.off_M[si], si, True)

        Q = np.zeros((self.n_full, red), dtype=np.complex128)
        active = col >= 0
        Q[np.flatnonzero(active), col[active]] = 1.0
        # ties: sigma/traversal rule via a region shared with the master.
        #   t:   sigma_rm dir_m J_m,t = -sigma_rs dir_s J_s,t
        #   phi: sigma_rm J_m,phi = sigma_rs J_s,phi
        # (the phi relations around a 3-region cycle are inconsistent — the
        # projection H.t_hat differs per surface — so each slave ties via a
        # region it shares with the MASTER and the remaining pairwise
        # relation is left to the Galerkin system.)
        for jn in self.junctions:
            sm, nm = jn[0]
            dir_m = self._dir(sm, nm)
            for (ss, ns) in jn[1:]:
                r = next(iter(set(self.adj[ss]) & set(self.adj[sm])))
                sg_m, sg_s = self.sigma[(r, sm)], self.sigma[(r, ss)]
                ct = -(sg_m * dir_m) / (sg_s * self._dir(ss, ns))
                cf = sg_m / sg_s
                mt = col[self.off_J[sm] + nm]
                mf = col[self.off_J[sm] + self.Nn[sm] + nm]
                if mt >= 0:
                    Q[self.off_J[ss] + ns, mt] = ct
                if mf >= 0:
                    Q[self.off_J[ss] + self.Nn[ss] + ns, mf] = cf
                if self.off_M[sm] is not None and self.off_M[ss] is not None:
                    mmt = col[self.off_M[sm] + nm]
                    mmf = col[self.off_M[sm] + self.Nn[sm] + nm]
                    if mmt >= 0:
                        Q[self.off_M[ss] + ns, mmt] = ct
                    if mmf >= 0:
                        Q[self.off_M[ss] + self.Nn[ss] + ns, mmf] = cf
        self._Q_cache[m] = Q
        return Q

    def assemble(self, m: int, m_max: int):
        A = np.zeros((self.n_full, self.n_full), dtype=np.complex128)
        for ri, reg in enumerate(self.regions):
            eta_r = self.solv[(reg["bounds"][0][0], ri)].eta
            eta2 = (ETA0 / eta_r) ** 2
            for (si, sg_i) in reg["bounds"]:
                for (sj, sg_j) in reg["bounds"]:
                    ss = sg_i * sg_j
                    if si == sj:
                        T = self.solv[(si, ri)].assemble_mode(m, m_max)
                        P = self.solv[(si, ri)].assemble_pmchwt_P(m, m_max) \
                            if not self.is_cond[si] else None
                    else:
                        T = self.X[(ri, si, sj)].assemble_T(m, m_max)
                        P = self.X[(ri, si, sj)].assemble_P(m, m_max)
                    slJ_i = slice(self.off_J[si], self.off_J[si] + 2 * self.Nn[si])
                    slJ_j = slice(self.off_J[sj], self.off_J[sj] + 2 * self.Nn[sj])
                    A[slJ_i, slJ_j] += ss * T
                    if self.off_M[sj] is not None:
                        slM_j = slice(self.off_M[sj], self.off_M[sj] + 2 * self.Nn[sj])
                        A[slJ_i, slM_j] += -ETA0 * ss * P
                    if self.off_M[si] is not None:
                        slM_i = slice(self.off_M[si], self.off_M[si] + 2 * self.Nn[si])
                        A[slM_i, slJ_j] += ETA0 * ss * P
                        if self.off_M[sj] is not None:
                            A[slM_i, slM_j] += eta2 * ss * T
        Q = self.build_Q(m)
        return Q.T @ A @ Q, None

    def rhs(self, m: int, th: float, pol: str) -> np.ndarray:
        V = np.zeros(self.n_full, dtype=np.complex128)
        for (si, _) in self.regions[self.ext_region]["bounds"]:
            s = self.solv[(si, self.ext_region)]
            V[self.off_J[si]:self.off_J[si] + 2 * self.Nn[si]] = s.rhs_mode(m, th, pol)
            if self.off_M[si] is not None:
                V[self.off_M[si]:self.off_M[si] + 2 * self.Nn[si]] = \
                    ETA0 * s.rhs_h_mode(m, th, pol)
        return self.build_Q(m).T @ V

    def farfield(self, m: int, x_red: np.ndarray, th: float, pol: str) -> complex:
        x = self.build_Q(m) @ x_red
        fth = fph = 0.0
        for (si, _) in self.regions[self.ext_region]["bounds"]:
            s = self.solv[(si, self.ext_region)]
            J = x[self.off_J[si]:self.off_J[si] + 2 * self.Nn[si]]
            msol = (ETA0 * x[self.off_M[si]:self.off_M[si] + 2 * self.Nn[si]]
                    if self.off_M[si] is not None else None)
            ft, fp = s.farfield_mode(m, J, th, msol=msol)
            fth += ft; fph += fp
        return fth if pol == "VV" else fph

    def rho_max(self) -> float:
        return max(float(np.max(self.solv[(si, self.adj[si][0])].gen.nodes[:, 0]))
                   for si in range(self.n_surf))


def _solve_multiregion(sys_: _MultiRegionBor, freq_hz, thetas_deg, n_modes,
                       mode_tol, workers, progress, check_abort,
                       formulation: str, extra: Dict) -> Dict:
    thetas = np.atleast_1d(np.asarray(thetas_deg, dtype=float))
    k = 2.0 * math.pi * freq_hz / C0
    st_max = float(np.max(np.abs(np.sin(np.radians(thetas)))))
    if n_modes is None:
        n_modes = int(math.ceil(k * sys_.rho_max() * max(st_max, 0.05))) + 12
    m_max = n_modes
    F, modes_used, stats = _mode_sweep(
        sys_.n_full, thetas, ("VV", "HH"), m_max, mode_tol,
        lambda m: sys_.assemble(m, m_max), sys_.rhs, sys_.farfield,
        prepare=lambda mm: sys_.prepare(mm, workers=workers),
        workers=workers, progress=progress, check_abort=check_abort)
    extra = {**extra, **stats}
    out = {
        "theta_deg": thetas.tolist(),
        "sigma_vv": (4.0 * math.pi * np.abs(F[0]) ** 2).tolist(),
        "sigma_hh": (4.0 * math.pi * np.abs(F[1]) ** 2).tolist(),
        "amp_vv": F[0].tolist(),
        "amp_hh": F[1].tolist(),
        "modes_used": modes_used,
        "n_unknowns": int(sys_.n_full),
        "n_junctions": len(sys_.junctions),
        "formulation": formulation,
    }
    out.update(extra)
    return out


def solve_bor_coated2_pec(points_outer, points_mid, points_core,
                          freq_hz: float, thetas_deg,
                          eps_inner: complex, mu_inner: complex,
                          eps_outer: complex, mu_outer: complex,
                          n_modes: Optional[int] = None, gauss_order: int = 4,
                          mode_tol: float = 1e-6, near_factor: float = 2.0,
                          near_order: int = 12, workers: int = 1,
                          progress: Optional[Callable] = None,
                          check_abort: Optional[Callable] = None) -> Dict:
    """PEC core under TWO full coating layers (all three generatrices closed
    axis-to-axis, +z -> -z, normals toward the exterior side)."""

    sys_ = _MultiRegionBor(
        surfaces=[(points_outer, False), (points_mid, False), (points_core, True)],
        regions=[
            {"medium": None, "bounds": [(0, +1)], "exterior": True},
            {"medium": (eps_outer, mu_outer), "bounds": [(0, -1), (1, +1)]},
            {"medium": (eps_inner, mu_inner), "bounds": [(1, -1), (2, +1)]},
        ],
        freq_hz=freq_hz, gauss_order=gauss_order,
        near_factor=near_factor, near_order=near_order)
    return _solve_multiregion(sys_, freq_hz, thetas_deg, n_modes, mode_tol,
                              workers, progress, check_abort,
                              "pmchwt-coated-2layer",
                              {"eps_inner": complex(eps_inner),
                               "eps_outer": complex(eps_outer)})


def solve_bor_coated_n_pec(interface_points, points_core, freq_hz: float,
                           thetas_deg, eps_list, mu_list,
                           n_modes: Optional[int] = None, gauss_order: int = 4,
                           mode_tol: float = 1e-6, near_factor: float = 2.0,
                           near_order: int = 12, workers: int = 1,
                           progress: Optional[Callable] = None,
                           check_abort: Optional[Callable] = None) -> Dict:
    """PEC core under N full coating layers.  interface_points is the list
    of interface generatrices OUTERMOST FIRST; eps_list/mu_list are per
    layer INNERMOST FIRST (matching mie_sphere.sigma_multilayer_pec_sphere)."""

    N = len(interface_points)
    if len(eps_list) != N or len(mu_list) != N:
        raise ValueError("One (eps, mu) per layer, innermost first.")
    surfaces = [(p, False) for p in interface_points] + [(points_core, True)]
    regions = [{"medium": None, "bounds": [(0, +1)], "exterior": True}]
    for i in range(N):
        # surface i separates layer (N - i) outside from layer (N - i - 1)
        # inside; layer j uses eps_list[j - 1] (innermost first).
        lay = N - i          # region below surface i is layer `lay`
        inner_bound = (i + 1, +1) if i == N - 1 else (i + 1, +1)
        regions.append({"medium": (eps_list[lay - 1], mu_list[lay - 1]),
                        "bounds": [(i, -1), inner_bound]})
    sys_ = _MultiRegionBor(surfaces=surfaces, regions=regions, freq_hz=freq_hz,
                           gauss_order=gauss_order, near_factor=near_factor,
                           near_order=near_order)
    return _solve_multiregion(sys_, freq_hz, thetas_deg, n_modes, mode_tol,
                              workers, progress, check_abort,
                              f"pmchwt-coated-{N}layer",
                              {"eps_layers": [complex(e) for e in eps_list]})


def solve_bor_coating_patch(points_patch, points_mid_covered, points_mid_bare,
                            points_core, freq_hz: float, thetas_deg,
                            eps_inner: complex, mu_inner: complex,
                            eps_patch: complex, mu_patch: complex,
                            n_modes: Optional[int] = None, gauss_order: int = 4,
                            mode_tol: float = 1e-6, near_factor: float = 2.0,
                            near_order: int = 12, workers: int = 1,
                            progress: Optional[Callable] = None,
                            check_abort: Optional[Callable] = None) -> Dict:
    """A second-layer coating PATCH terminating on a fully coated PEC body:
    the patch's outer interface (points_patch) meets the inner coating's
    interface at dielectric triple junctions (air / patch / inner coating —
    no conductor on the junction line).  points_mid_covered is the part of
    the inner interface under the patch, points_mid_bare the exposed
    part(s) (a list); the PEC core stays fully covered by the inner layer."""

    bare_list = (points_mid_bare if isinstance(points_mid_bare, (list, tuple))
                 else [points_mid_bare])
    surfaces = [(points_patch, False), (points_mid_covered, False)]
    surfaces += [(p, False) for p in bare_list]
    surfaces.append((points_core, True))
    core_idx = len(surfaces) - 1
    bare_idx = list(range(2, 2 + len(bare_list)))
    regions = [
        {"medium": None, "exterior": True,
         "bounds": [(0, +1)] + [(bi, +1) for bi in bare_idx]},
        {"medium": (eps_patch, mu_patch),
         "bounds": [(0, -1), (1, +1)]},
        {"medium": (eps_inner, mu_inner),
         "bounds": [(1, -1)] + [(bi, -1) for bi in bare_idx] + [(core_idx, +1)]},
    ]
    sys_ = _MultiRegionBor(surfaces=surfaces, regions=regions, freq_hz=freq_hz,
                           gauss_order=gauss_order, near_factor=near_factor,
                           near_order=near_order)
    return _solve_multiregion(sys_, freq_hz, thetas_deg, n_modes, mode_tol,
                              workers, progress, check_abort,
                              "pmchwt-coating-patch",
                              {"eps_inner": complex(eps_inner),
                               "eps_patch": complex(eps_patch)})


# ─────────────────────────────────────────────────────────────────────────────
# Canonical generatrices for gates
# ─────────────────────────────────────────────────────────────────────────────

def sphere_generatrix(a: float, n: int) -> np.ndarray:
    """North pole (+z) to south pole: outward left-normals per convention."""
    th = np.linspace(0.0, math.pi, n + 1)
    return np.column_stack([a * np.sin(th), a * np.cos(th)])


def cylinder_generatrix(a: float, L: float, n_rad: int, n_len: int) -> np.ndarray:
    """Closed cylinder: top cap center -> rim -> side -> bottom rim -> center."""
    top = np.column_stack([np.linspace(0.0, a, n_rad + 1), np.full(n_rad + 1, L / 2)])
    side = np.column_stack([np.full(n_len - 1, a), np.linspace(L / 2, -L / 2, n_len + 1)[1:-1]])
    bot = np.column_stack([np.linspace(a, 0.0, n_rad + 1), np.full(n_rad + 1, -L / 2)])
    return np.vstack([top, side, bot])
