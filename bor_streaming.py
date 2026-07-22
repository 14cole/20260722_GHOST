"""
Phase-7b streaming far assembly for the BoR solver.

The table path stores modal kernels at GAUSS-POINT pairs — [P, P, modes]
with P = gauss_order * N_t — which is the memory bound at scale.  This
module keeps the FFT-over-azimuth amortization (one xi sweep yields every
mode) but contracts BOTH Galerkin sides immediately, tile by tile, so the
persistent storage is per-mode NODAL blocks:

    EFIE   4 * (m_max + 1) * Nn^2   (m >= 0; ztt/zff even in m, ztf/zft odd)
    MFIE   4 * (2 m_max + 1) * Nn^2 (brackets have mixed parity)
    IBC    4 * (2 m_max + 1) * Nn^2 (source Z_s baked into the contraction)

— a 16x reduction versus the tables (32x with single-precision blocks).

For each tile of test elements the azimuthal integrand is sampled on the
same uniform xi grid the table path uses ([rows, P, n_xi]), FFT'd, near
(adjacent-element) pair entries are zeroed exactly as the table path zeroes
them, and the modal kernels are contracted through the reference shape
functions:

    out[e + a, f + b] += sum_gh L[a, e, g] K[e, g, f, h] R[b, f, h]

with L/R the per-point nodal weights (shape value x rho w x tangent
component, or the (rho T)' divergence weights).  Because tiles use the same
xi grid, the same FFT, and the same Galerkin points as the table path, the
streamed blocks match the table-path contraction to float roundoff — the
equivalence gate in tests/validate_bor_streaming.py checks exactly that.

The near/self machinery (graded cells, adaptive kernels) is untouched: the
solver adds its near corrections on top of these far blocks as before.
"""

import ctypes
import math
import os
import platform
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from bor_kernels import _mfie_brackets, _ibc_brackets_grid


# ── phase-7c native sampling kernel (ctypes; NumPy fallback if absent) ──

def _load_native():
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (f"bor_stream_kernel.{sysname}-{machine}", "bor_stream_kernel"):
        path = os.path.join(here, base + ".so")
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        dp = ctypes.POINTER(ctypes.c_double)
        ci = ctypes.c_int
        cd = ctypes.c_double
        lib.sample_g.argtypes = [ci, ci, ci, dp, dp, dp, dp, cd, dp, dp]
        lib.sample_g.restype = None
        bracket_args = [ci, ci, ci] + [dp] * 8 + [cd, dp, dp] + [dp] * 4
        lib.sample_mfie.argtypes = bracket_args
        lib.sample_mfie.restype = None
        lib.sample_ibc.argtypes = bracket_args
        lib.sample_ibc.restype = None
        return lib
    return None


_NATIVE = _load_native()
_FALLBACK_NOTICE_SHOWN = False


def _notice_numpy_fallback():
    """One-time stderr notice when the streaming build runs on the NumPy
    sampler.  Results are bit-equivalent; assembly is ~2-8x slower.  Also
    diagnoses the found-but-wrong-platform case (e.g. a Mac .so copied to a
    Linux cluster), which the loader correctly refuses to load."""
    global _FALLBACK_NOTICE_SHOWN
    if _FALLBACK_NOTICE_SHOWN:
        return
    _FALLBACK_NOTICE_SHOWN = True
    here = os.path.dirname(os.path.abspath(__file__))
    tag = f"{platform.system().lower()}-{platform.machine().lower()}"
    others = [f for f in sorted(os.listdir(here))
              if f.startswith("bor_stream_kernel.") and f.endswith(".so")
              and tag not in f]
    hint = (f" (found {', '.join(others)} — built for a DIFFERENT platform, "
            "so it was correctly skipped)" if others else "")
    print(
        "bor_streaming: native sampling kernel not available for this "
        f"platform{hint}; using the NumPy fallback (bit-equivalent, ~2-8x "
        "slower assembly). Compile it on THIS machine with:\n"
        "  cc -O3 -shared -fPIC -o "
        f"bor_stream_kernel.{tag}.so bor_stream_kernel.c -lm",
        file=sys.stderr, flush=True)


def _dp(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _n_xi_efie(k: complex, rho_max: float, m_max: int) -> int:
    osc = 2.0 * abs(k) * rho_max
    return int(2 ** math.ceil(math.log2(max(64, 4 * (m_max + 2), 6 * (osc + 4)))))


def _n_xi_bracket(k: complex, rho_max: float, m_max: int) -> int:
    osc = 2.0 * abs(k) * rho_max
    return int(2 ** math.ceil(math.log2(max(128, 6 * (m_max + 2), 8 * (osc + 4)))))


class StreamingFarBlocks:
    """Per-mode nodal far blocks for one BorPecSolver surface.

    efie=True builds the four EFIE blocks (without the C = jk eta 2pi
    factor, matching _pair_blocks); mfie=True the four MFIE bracket blocks
    (WITH the 2pi Galerkin factor, matching assemble_mfie_mode's far
    contraction); ibc_zs_pt (per-Gauss-point Z_s) the IBC bracket blocks
    with the source weight baked in (matching _rot_pv_blocks)."""

    def __init__(self, solver, m_max: int, efie: bool = True,
                 mfie: bool = False, ibc_zs_pt: Optional[np.ndarray] = None,
                 dtype=np.complex128, tile_budget_gb: float = 1.0,
                 workers: int = 1, mode_block: Optional[int] = None):
        self.solver = solver
        self.m_max = int(m_max)
        self.dtype = dtype
        g = solver.g
        gen = solver.gen
        self.Nn = solver.Nn
        ne = gen.n_elems
        self.go = solver.gauss_order
        P = solver.P
        k = solver.k
        Nn, go, mm = self.Nn, self.go, self.m_max

        # ── per-point nodal weight vectors [2, P] ──
        wrho = g.w * g.rho
        self._lv = {
            "r": np.stack([g.T0 * wrho * g.trho, g.T1 * wrho * g.trho]),
            "z": np.stack([g.T0 * wrho * g.tz, g.T1 * wrho * g.tz]),
            "1": np.stack([g.T0 * wrho, g.T1 * wrho]),
            "s": np.stack([g.T0 * g.w, g.T1 * g.w]),
            "d": np.stack([g.dRT0 * g.w, g.dRT1 * g.w]),
        }
        rv_ibc = None
        if ibc_zs_pt is not None:
            rv_ibc = np.stack([g.T0 * wrho * ibc_zs_pt, g.T1 * wrho * ibc_zs_pt])

        # ── configuration (block storage is allocated per MODE RANGE) ──
        # EFIE: 9 order-primitives (the Gc/Gs neighbor relations and the
        # mode-dependent scalars commute with contraction, so per-mode blocks
        # are combined on retrieval): rr, zz, r1, 1r, 11, dd, ds, sd, ss.
        self._efie = efie
        self._mfie = mfie
        self._has_ibc = ibc_zs_pt is not None
        self._rv_ibc = rv_ibc
        self.k = complex(k)
        workers = max(1, int(workers))
        self._workers = workers
        # phase-7d mode-block re-sweeps: hold blocks only for an aligned
        # range of modes; re-run the sampling sweep when the mode loop
        # advances past it (memory / n_ranges at sampling x n_ranges).
        # Ranges are multiples of the engine's worker count so a thread
        # wave never straddles a rebuild.
        Bm = int(mode_block) if mode_block else mm + 1
        Bm = max(Bm, workers)
        Bm = ((Bm + workers - 1) // workers) * workers
        self.mode_block = min(Bm, mm + 1)
        self.n_sweeps = 0

        rho_max = float(np.max(gen.nodes[:, 0]))
        # xi grids sized for the FULL mode set (identical to the table path
        # for every range, preserving the bit-level equivalence gates)
        self._nx_e = _n_xi_efie(k, rho_max, mm)
        self._nx_b = _n_xi_bracket(k, rho_max, mm)
        nx_worst = max(self._nx_e,
                       self._nx_b if (mfie or self._has_ibc) else 0)
        # tile size from the [rows, P, n_xi] sampling footprint (~11 arrays
        # of that shape live at once in the bracket samplers; each worker
        # thread holds its own tile)
        rows_max = max(go, int(tile_budget_gb * 1e9 /
                               (P * nx_worst * 176.0 * workers)))
        self._te = max(1, rows_max // go)

        # native (ctypes) sampling kernel: real-k only (the air region —
        # exactly what the solve_bor streaming path serves)
        self._native = (_NATIVE if (_NATIVE is not None and
                                    abs(complex(k).imag) == 0.0) else None)
        if _NATIVE is None:
            _notice_numpy_fallback()
        self._q = tuple(np.ascontiguousarray(v) for v in
                        (g.rho, g.z, g.trho, g.tz))
        self._acc_lock = threading.Lock()
        self._range_lock = threading.RLock()
        self.Z = self.K = self.B = None
        self.lo, self.hi = 1, 0          # empty range
        self._ord_lo = 0
        self._sidx: Dict[int, int] = {}
        self._ensure(0)

    # ── mode-range machinery ──
    def _ensure(self, am: int) -> None:
        if self.lo <= am <= self.hi:
            return
        with self._range_lock:
            if self.lo <= am <= self.hi:
                return
            lo = (am // self.mode_block) * self.mode_block
            hi = min(lo + self.mode_block - 1, self.m_max)
            self._build_range(lo, hi)

    def _build_range(self, lo: int, hi: int) -> None:
        Nn, go, mm = self.Nn, self.go, self.m_max
        ne = self.solver.gen.n_elems
        k = self.solver.k
        ord_lo = max(0, lo - 1)
        n_ord = hi + 2 - ord_lo
        if self._efie:
            self.Z = np.zeros((9, n_ord, Nn, Nn), dtype=self.dtype)
        ms = [m for m in range(-hi, hi + 1) if lo <= abs(m) <= hi]
        self._sidx = {m: i for i, m in enumerate(ms)}
        if self._mfie:
            self.K = np.zeros((4, len(ms), Nn, Nn), dtype=self.dtype)
        if self._has_ibc:
            self.B = np.zeros((4, len(ms), Nn, Nn), dtype=self.dtype)
        self._ord_lo = ord_lo

        orders = np.arange(ord_lo, hi + 2)
        ph_e = np.exp(1j * np.pi * orders) * (2.0 * np.pi / self._nx_e)
        msarr = np.asarray(ms)
        bins_b = np.where(msarr >= 0, msarr, self._nx_b + msarr)
        ph_b = np.exp(1j * np.pi * msarr) * (2.0 * np.pi / self._nx_b)

        def do_tile(e0):
            e1 = min(e0 + self._te, ne)
            rows = slice(e0 * go, e1 * go)
            re = e1 - e0
            if self._efie:
                Gn = self._sample_G(rows, k, self._nx_e, ph_e, ord_lo, hi)
                self._zero_near(Gn, e0, e1)
                self._accumulate_efie(Gn, rows, e0, re, k)
            if self._mfie:
                Fs = self._sample_brackets("mfie", rows, re, k, self._nx_b)
                self._accumulate_brackets(Fs, self.K, self._lv["1"],
                                          self._lv["1"], rows, e0, re,
                                          bins_b, ph_b)
            if self._has_ibc:
                Fs = self._sample_brackets("ibc", rows, re, k, self._nx_b)
                self._accumulate_brackets(Fs, self.B, self._lv["1"],
                                          self._rv_ibc, rows, e0, re,
                                          bins_b, ph_b)

        starts = list(range(0, ne, self._te))
        if self._workers <= 1:
            for e0 in starts:
                do_tile(e0)
        else:
            with ThreadPoolExecutor(max_workers=self._workers) as ex:
                list(ex.map(do_tile, starts))
        self.lo, self.hi = lo, hi
        self.n_sweeps += 1

    def _sample_brackets(self, which: str, rows, re: int, k, nx_b: int):
        """The four bracket integrand tiles [rows, P, nx_b] (native kernel
        when available, NumPy fallback otherwise)."""
        g = self.solver.g
        P = self.solver.P
        go = self.go
        nr = re * go
        xi = 2.0 * np.pi * np.arange(nx_b) / nx_b - np.pi
        if self._native is not None:
            rho_q, z_q, tr_q, tz_q = self._q
            rp = np.ascontiguousarray(g.rho[rows])
            zp = np.ascontiguousarray(g.z[rows])
            trp = np.ascontiguousarray(g.trho[rows])
            tzp = np.ascontiguousarray(g.tz[rows])
            cx = np.ascontiguousarray(np.cos(xi))
            sx = np.ascontiguousarray(np.sin(xi))
            Fs = tuple(np.empty((nr, P, nx_b), dtype=np.complex128)
                       for _ in range(4))
            fn = self._native.sample_mfie if which == "mfie" else self._native.sample_ibc
            fn(nr, P, nx_b, _dp(rp), _dp(zp), _dp(trp), _dp(tzp),
               _dp(rho_q), _dp(z_q), _dp(tr_q), _dp(tz_q),
               float(np.real(k)), _dp(cx), _dp(sx),
               _dp(Fs[0]), _dp(Fs[1]), _dp(Fs[2]), _dp(Fs[3]))
            return Fs
        if which == "mfie":
            return _mfie_brackets(
                g.rho[rows][:, None] * np.ones(P)[None, :],
                g.z[rows][:, None] * np.ones(P)[None, :],
                g.trho[rows][:, None] * np.ones(P)[None, :],
                g.tz[rows][:, None] * np.ones(P)[None, :],
                g.rho[None, :] * np.ones(nr)[:, None],
                g.z[None, :] * np.ones(nr)[:, None],
                g.trho[None, :] * np.ones(nr)[:, None],
                g.tz[None, :] * np.ones(nr)[:, None],
                k, xi)
        one = np.ones((nr, P))
        Fs = _ibc_brackets_grid(
            (g.rho[rows][:, None] * one).ravel(),
            (g.z[rows][:, None] * one).ravel(),
            (g.trho[rows][:, None] * one).ravel(),
            (g.tz[rows][:, None] * one).ravel(),
            (g.rho[None, :] * one).ravel(),
            (g.z[None, :] * one).ravel(),
            (g.trho[None, :] * one).ravel(),
            (g.tz[None, :] * one).ravel(),
            k, np.broadcast_to(xi, (nr * P, nx_b)))
        return tuple(F.reshape(nr, P, nx_b) for F in Fs)

    # ── sampling / masking ──
    def _sample_G(self, rows, k, n_xi, phase, ord_lo, hi):
        g = self.solver.g
        xi = 2.0 * np.pi * np.arange(n_xi) / n_xi - np.pi
        if self._native is not None:
            rp = np.ascontiguousarray(g.rho[rows])
            zp = np.ascontiguousarray(g.z[rows])
            sin2 = np.ascontiguousarray(np.sin(0.5 * xi) ** 2)
            gk = np.empty((len(rp), self.solver.P, n_xi), dtype=np.complex128)
            self._native.sample_g(len(rp), self.solver.P, n_xi,
                                  _dp(rp), _dp(zp), _dp(self._q[0]),
                                  _dp(self._q[1]), float(np.real(k)),
                                  _dp(sin2), _dp(gk))
        else:
            sin2 = np.sin(0.5 * xi) ** 2
            d2 = (g.rho[rows][:, None] - g.rho[None, :]) ** 2 + \
                 (g.z[rows][:, None] - g.z[None, :]) ** 2
            rr4 = 4.0 * g.rho[rows][:, None] * g.rho[None, :]
            R = np.sqrt(d2[..., None] + rr4[..., None] * sin2)
            R = np.maximum(R, 1e-300)
            gk = np.exp(-1j * complex(k) * R) / (4.0 * np.pi * R)
        spec = np.fft.fft(gk, axis=-1)
        return spec[..., ord_lo:hi + 2] * phase

    def _zero_near(self, Kt, e0, e1):
        ne = self.solver.gen.n_elems
        go = self.go
        for e in range(e0, e1):
            c0 = max(0, e - 1) * go
            c1 = min(ne, e + 2) * go
            Kt[(e - e0) * go:(e - e0 + 1) * go, c0:c1] = 0.0

    # ── contraction: ALL modes/orders in one einsum per weight pair ──
    def _contract_all(self, Kn, lv_rows, rv, e0, re, out):
        """out[n_modes, Nn, Nn] += nodal contraction of Kn [rows, P, n_modes]
        with per-point left weights lv_rows [2, rows], right weights rv [2, P]."""
        go = self.go
        ce = self.solver.gen.n_elems
        Kr = Kn.reshape(re, go, ce, go, -1)
        L = lv_rows.reshape(2, re, go)
        R = rv.reshape(2, ce, go)
        M = np.einsum("aeg,egfhm,bfh->maebf", L, Kr, R, optimize=True)
        if out.dtype != M.dtype:
            M = M.astype(out.dtype)
        # adjacent tiles share a boundary node row: serialize the adds
        with self._acc_lock:
            out[:, e0:e0 + re, 0:ce] += M[:, 0, :, 0, :]
            out[:, e0:e0 + re, 1:ce + 1] += M[:, 0, :, 1, :]
            out[:, e0 + 1:e0 + re + 1, 0:ce] += M[:, 1, :, 0, :]
            out[:, e0 + 1:e0 + re + 1, 1:ce + 1] += M[:, 1, :, 1, :]

    _EFIE_COMBOS = (("r", "r"), ("z", "z"), ("r", "1"), ("1", "r"),
                    ("1", "1"), ("d", "d"), ("d", "s"), ("s", "d"), ("s", "s"))

    def _accumulate_efie(self, Gn, rows, e0, re, k):
        for ci, (lx, rx) in enumerate(self._EFIE_COMBOS):
            self._contract_all(Gn, self._lv[lx][:, rows], self._lv[rx],
                               e0, re, self.Z[ci])

    def _accumulate_brackets(self, Fs, store, lv_full, rv, rows, e0, re,
                             bins, phase):
        lv_rows = lv_full[:, rows]
        for uv, F in enumerate(Fs):
            spec = np.fft.fft(F, axis=-1)
            Km_all = spec[..., bins] * (2.0 * np.pi * phase)
            self._zero_near(Km_all, e0, e0 + re)
            self._contract_all(Km_all, lv_rows, rv, e0, re, store[uv])

    # ── per-mode block retrieval (complex128 copies; near loops add onto them) ──
    def efie_blocks(self, m: int):
        # primitive order: 0 rr, 1 zz, 2 r1, 3 1r, 4 11, 5 dd, 6 ds, 7 sd, 8 ss
        am = abs(m)
        with self._range_lock:
            self._ensure(am)
            k = self.k
            o = self._ord_lo
            lo, hi = abs(am - 1), am + 1

            def g(ci, n):
                return self.Z[ci, n - o].astype(np.complex128)

            ztt = 0.5 * (g(0, lo) + g(0, hi)) + g(1, am) - (1.0 / k ** 2) * g(5, am)
            ztf = (g(2, lo) - g(2, hi)) / 2j - (1j * am / k ** 2) * g(6, am)
            zft = -(g(3, lo) - g(3, hi)) / 2j + (1j * am / k ** 2) * g(7, am)
            zff = 0.5 * (g(4, lo) + g(4, hi)) - (am ** 2 / k ** 2) * g(8, am)
        if m < 0:
            ztf = -ztf
            zft = -zft
        return ztt, ztf, zft, zff

    def bracket_blocks(self, which: str, m: int):
        with self._range_lock:
            self._ensure(abs(m))
            store = self.K if which == "mfie" else self.B
            mi = self._sidx[m]
            return tuple(store[uv, mi].astype(np.complex128) for uv in range(4))

    def memory_gb(self) -> float:
        total = 0
        for arr in (self.Z, self.K, self.B):
            if arr is not None:
                total += arr.nbytes
        return total / 1e9


def estimate_streaming_gb(n_elems: int, m_max: int, formulation: str = "cfie",
                          has_ibc: bool = False,
                          single_blocks: bool = False) -> float:
    """Persistent per-mode nodal block memory (GB) for the streaming path."""
    Nn = float(n_elems + 1)
    per = 8.0 if single_blocks else 16.0
    total = 4.0 * Nn * Nn * (m_max + 1) * per
    if formulation in ("cfie", "mfie"):
        total += 4.0 * Nn * Nn * (2 * m_max + 1) * per
    if has_ibc:
        total += 4.0 * Nn * Nn * (2 * m_max + 1) * per
    return total / 1e9
