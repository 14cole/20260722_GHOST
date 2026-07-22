"""
2D Helmholtz Fast Multipole Method for boundary element acceleration.

Replaces O(N^2) dense assembly/matvec with O(N log N) via:
  1. Adaptive quadtree over panel centers
  2. Multipole expansions (P2M -> M2M -> M2L -> L2L -> L2P)
  3. Direct near-field (reuses existing near-singular quadrature)

Usage:
    from fmm_helmholtz_2d import FMMOperator
    op = FMMOperator(mesh, k, obs_normal_deriv=False, n_digits=6)
    y = op.matvec(x)
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.special import jv as _jv, hankel2 as _h2

_NATIVE_WARNED = False

def _warn_no_native():
    """Warn once that the compiled near-field extension is unavailable.

    For real k the near-field is assembled by the C extension (fmm_near). When
    it is missing or built for the wrong platform, assembly silently falls back
    to a pure-Python path that is ~100x slower and can look like a hang on
    larger meshes. Surface that instead of degrading quietly.
    """
    global _NATIVE_WARNED
    if _NATIVE_WARNED:
        return
    _NATIVE_WARNED = True
    import warnings, platform
    warnings.warn(
        "fmm_helmholtz_2d: compiled near-field extension (fmm_near) not found "
        f"for {platform.system().lower()}-{platform.machine().lower()}; using the "
        "pure-Python near-field, which is ~100x slower. Build it with:\n"
        "    cc -O3 -shared -fPIC -o fmm_near.$(python3 -c "
        "'import platform;print(f\"{platform.system().lower()}-{platform.machine().lower()}\")').so "
        "fmm_near.c -lm",
        RuntimeWarning, stacklevel=2,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Quadtree
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Box:
    box_id: int
    center: np.ndarray
    half_size: float
    level: int
    parent: int
    children: List[int]
    panel_ids: List[int]
    is_leaf: bool

class QuadTree:
    def __init__(self, centers: np.ndarray, max_leaf: int = 40):
        self.centers = np.asarray(centers, dtype=float)
        self.max_leaf = max_leaf
        self.boxes: List[Box] = []
        self.n_levels = 0
        self._build()

    def _build(self):
        N = len(self.centers)
        if N == 0: return
        lo = self.centers.min(axis=0); hi = self.centers.max(axis=0)
        mid = 0.5 * (lo + hi)
        half = max(0.5 * (hi[0]-lo[0]), 0.5*(hi[1]-lo[1]), 1e-12) * 1.01
        self.boxes.append(Box(0, mid, half, 0, -1, [], list(range(N)), True))
        self._subdivide(0)
        self.n_levels = max(b.level for b in self.boxes) + 1

    def _subdivide(self, box_id: int):
        box = self.boxes[box_id]
        if len(box.panel_ids) <= self.max_leaf or box.level >= 20: return
        box.is_leaf = False
        cx, cy = box.center; h = box.half_size * 0.5
        offsets = [(-1,-1),(1,-1),(-1,1),(1,1)]
        child_panels = [[] for _ in range(4)]
        for pid in box.panel_ids:
            px, py = self.centers[pid]
            child_panels[(0 if px < cx else 1) + 2*(0 if py < cy else 1)].append(pid)
        for q in range(4):
            dx, dy = offsets[q]
            cid = len(self.boxes)
            self.boxes.append(Box(cid, np.array([cx+dx*h, cy+dy*h]), h,
                                  box.level+1, box_id, [], child_panels[q], True))
            box.children.append(cid)
        box.panel_ids = []
        for cid in box.children: self._subdivide(cid)

    def get_leaves(self): return [b.box_id for b in self.boxes if b.is_leaf]
    def get_level_boxes(self, level): return [b.box_id for b in self.boxes if b.level == level]

def _build_lists(tree: QuadTree):
    """Build near and interaction (M2L) lists with guaranteed complete partition."""
    boxes = tree.boxes
    def _near(b1, b2):
        dx = abs(b1.center[0]-b2.center[0]); dy = abs(b1.center[1]-b2.center[1])
        return dx <= (b1.half_size+b2.half_size)*1.01 and dy <= (b1.half_size+b2.half_size)*1.01

    # Precompute per-level box lists and per-box near neighbors within each level.
    level_boxes = [tree.get_level_boxes(l) for l in range(tree.n_levels)]
    near_per_box: Dict[int, List[int]] = {}
    for lev in range(tree.n_levels):
        lb = level_boxes[lev]
        for bid in lb:
            nlist = []
            for other in lb:
                if _near(boxes[bid], boxes[other]):
                    nlist.append(other)
            near_per_box[bid] = nlist

    leaves = tree.get_leaves()
    near = {lid: list(near_per_box.get(lid, [])) for lid in leaves}

    interact: Dict[int, List[int]] = {b.box_id: [] for b in boxes}
    # A box's interaction list is: children of its parent's near neighbors,
    # restricted to those that are well-separated from the box itself.
    for lev in range(2, tree.n_levels):
        for bid in level_boxes[lev]:
            box = boxes[bid]
            for pn_id in near_per_box.get(box.parent, []):
                pn = boxes[pn_id]
                for cn_id in pn.children:
                    cn = boxes[cn_id]
                    if not _near(box, cn):
                        interact[bid].append(cn_id)

    # ── Completeness fix: ensure every leaf pair is covered ─────────────
    # Check which leaf pairs are handled by M2L at some ancestor level.
    # Leaf pair (li, lj) is covered if:
    #   1. lj is in near[li], OR
    #   2. Some ancestor of lj is in interact[some ancestor of li]
    # Uncovered pairs get added to the near list.
    leaf_set = set(leaves)

    def _get_ancestors(bid):
        """Return list of (box_id, level) from bid up to root."""
        path = []
        while bid >= 0:
            path.append(bid)
            bid = boxes[bid].parent
        return path

    ancestor_cache = {lid: _get_ancestors(lid) for lid in leaves}

    # Precompute the set of leaves in each box's subtree (bottom-up). A box that
    # appears in an interaction list "covers" — via M2L — every leaf beneath it.
    leaves_under: Dict[int, set] = {}
    for lev in range(tree.n_levels - 1, -1, -1):
        for bid in level_boxes[lev]:
            box = boxes[bid]
            if box.is_leaf:
                leaves_under[bid] = {bid}
            else:
                acc: set = set()
                for cid in box.children:
                    acc.update(leaves_under.get(cid, ()))
                leaves_under[bid] = acc

    # A leaf pair (li, lj) is M2L-covered iff some ancestor aj of lj sits in
    # interact[ai] for some ancestor ai of li — equivalently, lj lies under one
    # of the boxes in the union of li's ancestors' interaction lists. Any leaf
    # that is neither adjacent (already in near[li]) nor M2L-covered is added to
    # near[li] so the near + M2L partition stays complete. This is the set-based
    # equivalent of the previous O(L^2 * depth^2) all-pairs ancestor scan, which
    # dominated list construction once the tree was allowed to refine.
    for li in leaves:
        m2l_boxes: set = set()
        for ai in ancestor_cache[li]:
            m2l_boxes.update(interact.get(ai, ()))
        covered: set = set()
        for src in m2l_boxes:
            covered.update(leaves_under.get(src, ()))
        additions = leaf_set.difference(near[li]).difference(covered)
        if additions:
            near[li].extend(sorted(additions))

    return near, interact

# ═══════════════════════════════════════════════════════════════════════════════
# Translation operators (fully vectorized)
# ═══════════════════════════════════════════════════════════════════════════════

def _trunc_order(k, diam, n_digits=6, domain_diam=None):
    """
    Multipole/local truncation order for a box of side `diam`.

    The linear term is the box bandwidth k*diam.  The excess must grow with
    the electrical size of the WHOLE domain, not just the box: local
    expansions at deep levels accumulate content translated down from the
    top of the hierarchy, so a fixed "+n_digits" excess loses accuracy as
    k*D grows (measured against dense assembly: 7e-7 at kD=20 but 3e-3 at
    kD=244 with the old flat +8 excess).  The calibrated excess-bandwidth
    term 1.3 * n_digits^(2/3) * (k*D)^(1/3) holds the S and K' matvec
    error near the n_digits target through kD ~ 250 (see review notes).
    """

    kd = abs(k) * (domain_diam if domain_diam is not None else diam)
    excess = 1.3 * (float(n_digits) ** (2.0 / 3.0)) * (max(kd, 1.0) ** (1.0 / 3.0))
    p = int(math.ceil(abs(k) * diam + excess)) + 2
    return max(p, n_digits + 4, 6)

def _translation_matrix_J(k, d_vec, p):
    """Build J-based translation matrix T[n,m] = J_{idx}(kr) * exp(phase)."""
    rho = math.sqrt(d_vec[0]**2 + d_vec[1]**2)
    if rho < 1e-15: return np.eye(2*p+1, dtype=np.complex128)
    alpha = math.atan2(d_vec[1], d_vec[0])
    ns = np.arange(-p, p+1)
    idx = ns[:, None] - ns[None, :]  # n - m for M2M, or m - n for L2L
    J_vals = _jv(idx.ravel(), k * rho).reshape(2*p+1, 2*p+1)
    return J_vals * np.exp(-1j * idx * alpha)  # M2M sign

def _translation_matrix_H(k, d_vec, p):
    """Build H-based translation matrix for M2L."""
    rho = math.sqrt(d_vec[0]**2 + d_vec[1]**2)
    alpha = math.atan2(d_vec[1], d_vec[0])
    ns = np.arange(-p, p+1)
    idx = ns[None, :] - ns[:, None]  # m - n
    H_vals = _h2(idx.ravel(), k * rho).reshape(2*p+1, 2*p+1)
    return H_vals * np.exp(1j * idx * alpha)

def _p2m(sources, strengths, center, k, p):
    d = sources - center; r = np.sqrt(d[:,0]**2 + d[:,1]**2); theta = np.arctan2(d[:,1], d[:,0])
    ns = np.arange(-p, p+1)
    Jn = _jv(ns[:, None], (k * r)[None, :])  # (2p+1, N_src)
    phase = np.exp(-1j * ns[:, None] * theta[None, :])
    return np.sum(strengths[None, :] * Jn * phase, axis=1)

def _m2m(O_child, child_c, parent_c, k, p):
    T = _translation_matrix_J(k, child_c - parent_c, p)
    return T @ O_child

def _m2l(O_src, src_c, tgt_c, k, p):
    T = _translation_matrix_H(k, tgt_c - src_c, p)
    return T @ O_src

def _l2l(L_parent, parent_c, child_c, k, p):
    d = child_c - parent_c
    rho = math.sqrt(d[0]**2 + d[1]**2)
    if rho < 1e-15: return L_parent.copy()
    alpha = math.atan2(d[1], d[0])
    ns = np.arange(-p, p+1)
    idx = ns[None, :] - ns[:, None]  # m - n
    J_vals = _jv(idx.ravel(), k * rho).reshape(2*p+1, 2*p+1)
    T = J_vals * np.exp(1j * idx * alpha)
    return T @ L_parent

def _l2p_slp(L, targets, center, k, p):
    d = targets - center; r = np.sqrt(d[:,0]**2 + d[:,1]**2); theta = np.arctan2(d[:,1], d[:,0])
    ns = np.arange(-p, p+1)
    Jn = _jv(ns[:, None], (k * r)[None, :])
    phase = np.exp(1j * ns[:, None] * theta[None, :])
    return 0.25j * np.sum(L[:, None] * Jn * phase, axis=0)

def _l2p_dlp_normal(L, targets, normals, center, k, p):
    """Evaluate normal derivative of local expansion (K' kernel)."""
    d = targets - center; rho = np.sqrt(d[:,0]**2 + d[:,1]**2)
    rho_safe = np.maximum(rho, 1e-15); theta = np.arctan2(d[:,1], d[:,0])
    cos_t = np.cos(theta); sin_t = np.sin(theta)
    ns = np.arange(-p, p+1)
    kr = k * rho_safe
    Jn = _jv(ns[:, None], kr[None, :])         # (2p+1, N)
    Jn_m1 = _jv(ns[:, None] - 1, kr[None, :])
    Jn_p1 = _jv(ns[:, None] + 1, kr[None, :])
    dJn = 0.5 * (Jn_m1 - Jn_p1)
    e_int = np.exp(1j * ns[:, None] * theta[None, :])
    grad_x = k * dJn * cos_t[None, :] * e_int - (1j * ns[:, None] / rho_safe[None, :]) * sin_t[None, :] * Jn * e_int
    grad_y = k * dJn * sin_t[None, :] * e_int + (1j * ns[:, None] / rho_safe[None, :]) * cos_t[None, :] * Jn * e_int
    return 0.25j * np.sum(L[:, None] * (normals[:,0][None,:]*grad_x + normals[:,1][None,:]*grad_y), axis=0)

# ═══════════════════════════════════════════════════════════════════════════════
# FMM Operator
# ═══════════════════════════════════════════════════════════════════════════════

class FMMOperator:
    """
    FMM-accelerated matvec for 2D Helmholtz SLP or K'.

    Parameters
    ----------
    mesh : LinearMesh
    k : complex — wavenumber
    obs_normal_deriv : bool — False=SLP, True=K'
    source_element_mask : optional bool array
    n_digits : int — FMM accuracy digits (default 6)
    max_leaf : int — max panels per tree leaf
    quad_order : int — far-field quadrature order
    near_quad_order : int — near-field quadrature order
    """
    def __init__(self, mesh, k, obs_normal_deriv=False, source_element_mask=None,
                 n_digits=6, max_leaf=40, quad_order=4, near_quad_order=8,
                 _shared_tree=None, _shared_lists=None, _shared_geom=None):
        self.mesh = mesh
        self.k = complex(k)
        self.obs_nd = obs_normal_deriv
        self.n_digits = n_digits
        self.nnodes = len(mesh.nodes)
        self.nelems = len(list(mesh.elements))
        self.src_mask = np.ones(self.nelems, dtype=bool) if source_element_mask is None \
            else np.asarray(source_element_mask, dtype=bool)

        # Reuse shared geometry arrays if provided.
        if _shared_geom is not None:
            self.elements = _shared_geom['elements']
            self.centers = _shared_geom['centers']
            self.lengths = _shared_geom['lengths']
            self.normals = _shared_geom['normals']
            self.p0s = _shared_geom['p0s']
            self.segs = _shared_geom['segs']
            self.node_ids = _shared_geom['node_ids']
        else:
            elements = list(mesh.elements)
            self.elements = elements
            self.centers = np.array([e.center for e in elements])
            self.lengths = np.array([e.length for e in elements])
            self.normals = np.array([e.normal for e in elements])
            self.p0s = np.array([e.p0 for e in elements])
            self.segs = np.array([e.p1 - e.p0 for e in elements])
            self.node_ids = np.array([e.node_ids for e in elements], dtype=int)

        from rcs_solver import _get_quadrature, _linear_shape_values, _sk_blocks_near_linear
        qt, qw = _get_quadrature(max(2, quad_order))
        self.qt = np.array(qt, dtype=float)
        self.qw = np.array(qw, dtype=float)
        self.phi = np.array([_linear_shape_values(float(t)) for t in qt])
        self._sk_near = _sk_blocks_near_linear
        self.nq_order = near_quad_order

        # Reuse shared tree/lists or build new ones.
        if _shared_tree is not None and _shared_lists is not None:
            self.tree = _shared_tree
            self.near_list, self.interact_list = _shared_lists
        else:
            self.tree = QuadTree(self.centers, max_leaf)
            self.near_list, self.interact_list = _build_lists(self.tree)

        # Truncation order per level (depends on k, so always recomputed).
        # The excess term is tied to the ROOT electrical size (see
        # _trunc_order): deep-level expansions carry content translated down
        # from the top of the hierarchy.
        root_boxes = self.tree.get_level_boxes(0)
        root_diam = 2 * self.tree.boxes[root_boxes[0]].half_size if root_boxes else 0.0
        self.p_level = []
        for lev in range(self.tree.n_levels):
            lboxes = self.tree.get_level_boxes(lev)
            diam = 2*self.tree.boxes[lboxes[0]].half_size if lboxes else 0
            self.p_level.append(_trunc_order(self.k, diam, n_digits, domain_diam=root_diam))

        # Precompute all source quadrature points per leaf.
        self._precompute_quad_points()

        # Pre-assemble sparse near-field matrix.
        self._build_near_matrix()

    def _precompute_quad_points(self):
        """Precompute quadrature point positions for each element."""
        Q = len(self.qt)
        self.all_qpts = self.p0s[:, None, :] + self.qt[None, :, None] * self.segs[:, None, :]
        self.wl_phi = self.qw[None, :, None] * self.lengths[:, None, None] * self.phi[None, :, :]

    def _build_near_matrix(self):
        """Pre-assemble near-field matrix as a sparse CSR. Uses C extension for real-k regular pairs."""
        from rcs_solver import _near_singular_scheme, _get_quadrature, EPS
        from scipy.sparse import coo_matrix
        N = self.nnodes
        # Accumulate into COO triples; convert to CSR at the end.
        # Memory is O(nnz) rather than O(N^2), so large meshes are feasible.
        self._near_rows: List[np.ndarray] = []
        self._near_cols: List[np.ndarray] = []
        self._near_vals: List[np.ndarray] = []

        # Collect all unique near pairs, classify as special or regular.
        computed = set()
        special = []     # self + touching → Python recursive quadrature
        regular = {}     # quadrature_order → [(obs_idx, src_idx), ...]
        for leaf_id in self.tree.get_leaves():
            leaf = self.tree.boxes[leaf_id]
            if not leaf.panel_ids: continue
            near_src_set = set()
            for nlid in self.near_list.get(leaf_id, []):
                for pid in self.tree.boxes[nlid].panel_ids:
                    if self.src_mask[pid]: near_src_set.add(pid)
            for oi in leaf.panel_ids:
                nids_o = set(self.node_ids[oi])
                for si in near_src_set:
                    if (oi, si) in computed: continue
                    computed.add((oi, si))
                    if oi == si or bool(nids_o & set(self.node_ids[si])):
                        special.append((oi, si))
                    else:
                        dist = float(np.linalg.norm(self.centers[oi] - self.centers[si]))
                        scale = max(self.lengths[oi], self.lengths[si], 1e-15)
                        adapt_order, _ = _near_singular_scheme(dist, scale)
                        q = max(self.nq_order, min(16, max(5, int(adapt_order))))
                        regular.setdefault(q, []).append((oi, si))

        # Special pairs: Python (self/touching need Duffy/recursive).
        for oi, si in special:
            obs = self.elements[oi]; src = self.elements[si]
            oids = np.array(obs.node_ids, dtype=int)
            sids = np.array(src.node_ids, dtype=int)
            s_blk, k_blk = self._sk_near(obs, src, self.k, self.obs_nd,
                                          self.nq_order, self.nq_order)
            blk = k_blk if self.obs_nd else s_blk
            # Broadcast 2x2 block into flat row/col/val triples.
            rr, cc = np.meshgrid(oids, sids, indexing='ij')
            self._near_rows.append(rr.ravel())
            self._near_cols.append(cc.ravel())
            self._near_vals.append(np.asarray(blk, dtype=np.complex128).ravel())

        # Regular pairs: try Cython/C extension for real k, fall back to Python.
        # Guard against tiny / zero k.real where a relative tolerance collapses.
        k_is_real = abs(complex(self.k).imag) < 1e-12 * max(abs(complex(self.k)), 1.0)
        native_mod = self._load_native() if k_is_real else None
        if k_is_real and native_mod is None:
            _warn_no_native()

        for q_order, pairs in regular.items():
            if native_mod is not None:
                self._batch_near_native(native_mod, pairs, q_order)
            else:
                self._batch_near_python(pairs, q_order)

        self._finalize_near_matrix()

    @staticmethod
    def _load_native():
        """Try to load the compiled Cython extension, then fall back to ctypes C .so.

        Candidate .so/.dll names are tried platform/arch-specific first
        (e.g. ``fmm_near.darwin-arm64.so``) so a binary committed for one
        platform — such as the HPC Linux x86-64 build named ``fmm_near.so`` —
        does not shadow the matching local build. A ctypes library is accepted
        only if it actually exports ``compute_sk_blocks_batch_q``; a
        wrong-architecture file is rejected at load (raises OSError) and a file
        missing the symbol is skipped, rather than silently forcing the ~100x
        slower Python near-field path at call time.
        """
        # Try Cython first (preferred — no manual compile step)
        try:
            import fmm_near_cy
            return ("cython", fmm_near_cy)
        except ImportError:
            pass

        # Fall back to ctypes .so/.dll. Prefer a platform/arch-tagged build,
        # then the generic name.
        import ctypes, os, platform
        sysname = platform.system().lower()    # 'darwin', 'linux', 'windows'
        machine = platform.machine().lower()   # 'arm64', 'x86_64', 'amd64', ...
        here = os.path.dirname(__file__)
        bases = [f'fmm_near.{sysname}-{machine}', 'fmm_near']
        exts = ['.so', '.dylib', '.dll']
        for base in bases:
            for ext in exts:
                for path in (os.path.join(here, base + ext), os.path.join('.', base + ext)):
                    if not os.path.isfile(path):
                        continue
                    try:
                        lib = ctypes.CDLL(path)
                    except OSError:
                        continue
                    if hasattr(lib, 'compute_sk_blocks_batch_q'):
                        return ("ctypes", lib)
        return None

    def _batch_near_native(self, native_mod, pairs, q_order):
        """Batch near-field using Cython or C extension (real k only)."""
        from rcs_solver import _get_quadrature
        P = len(pairs)
        obs_idx = np.array([p[0] for p in pairs], dtype=int)
        src_idx = np.array([p[1] for p in pairs], dtype=int)

        qt, qw = _get_quadrature(q_order)
        qt_arr = np.array(qt, dtype=np.float64)
        qw_arr = np.array(qw, dtype=np.float64)

        obs_p0 = self.p0s[obs_idx].astype(np.float64)
        obs_seg = self.segs[obs_idx].astype(np.float64)
        obs_n = self.normals[obs_idx].astype(np.float64)
        obs_L = self.lengths[obs_idx].astype(np.float64)
        src_p0 = self.p0s[src_idx].astype(np.float64)
        src_seg = self.segs[src_idx].astype(np.float64)
        src_n = self.normals[src_idx].astype(np.float64)
        src_L = self.lengths[src_idx].astype(np.float64)

        kind, mod = native_mod
        if kind == "cython":
            s_blocks, k_blocks = mod.compute_sk_blocks_batch(
                qt_arr, qw_arr,
                obs_p0, obs_seg, obs_n, obs_L,
                src_p0, src_seg, src_n, src_L,
                float(self.k.real), 1 if self.obs_nd else 0,
            )
            blocks = k_blocks if self.obs_nd else s_blocks
        else:
            # ctypes path
            import ctypes
            nq = len(qt_arr)
            s_re = np.zeros(4*P, dtype=np.float64)
            s_im = np.zeros(4*P, dtype=np.float64)
            k_re = np.zeros(4*P, dtype=np.float64)
            k_im = np.zeros(4*P, dtype=np.float64)
            lib = mod
            lib.compute_sk_blocks_batch_q(
                ctypes.c_int(P), ctypes.c_int(nq),
                qt_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                qw_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                obs_p0.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                obs_seg.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                obs_n.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                obs_L.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                src_p0.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                src_seg.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                src_n.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                src_L.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_double(float(self.k.real)),
                ctypes.c_int(1 if self.obs_nd else 0),
                s_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                s_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                k_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                k_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            blk_re = k_re if self.obs_nd else s_re
            blk_im = k_im if self.obs_nd else s_im
            blocks = (blk_re + 1j * blk_im).reshape(P, 2, 2)

        obs_nids = self.node_ids[obs_idx]
        src_nids = self.node_ids[src_idx]
        for a in range(2):
            for b in range(2):
                self._near_rows.append(obs_nids[:, a].astype(np.int64, copy=False))
                self._near_cols.append(src_nids[:, b].astype(np.int64, copy=False))
                self._near_vals.append(blocks[:, a, b].astype(np.complex128, copy=False))

    def _batch_near_python(self, pairs, q_order):
        """Vectorized batch near-field for complex k."""
        from rcs_solver import _get_quadrature, _linear_shape_values
        from scipy.special import hankel2 as _h2_scipy

        P = len(pairs)
        if P == 0: return
        obs_idx = np.array([p[0] for p in pairs], dtype=int)
        src_idx = np.array([p[1] for p in pairs], dtype=int)
        k = self.k

        obs_p0 = self.p0s[obs_idx]; obs_seg = self.segs[obs_idx]
        obs_n = self.normals[obs_idx]; obs_L = self.lengths[obs_idx]
        src_p0 = self.p0s[src_idx]; src_seg = self.segs[src_idx]
        src_n = self.normals[src_idx]; src_L = self.lengths[src_idx]
        obs_nids = self.node_ids[obs_idx]; src_nids = self.node_ids[src_idx]

        qt, qw = _get_quadrature(q_order)
        phi_vals = np.array([_linear_shape_values(float(t)) for t in qt])
        Q = len(qt)

        blocks = np.zeros((P, 2, 2), dtype=np.complex128)

        for qi in range(Q):
            to = float(qt[qi]); wo = float(qw[qi])
            phi_o = phi_vals[qi]
            r_obs = obs_p0 + to * obs_seg

            for qj in range(Q):
                ts = float(qt[qj]); ws = float(qw[qj])
                phi_s = phi_vals[qj]
                r_src = src_p0 + ts * src_seg

                diff = r_obs - r_src
                dist = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
                dist_safe = np.maximum(dist, 1e-15)
                kr = k * dist_safe

                h0 = _h2_scipy(0, kr)
                h1 = _h2_scipy(1, kr)
                g = 0.25j * h0

                if self.obs_nd:
                    proj = (diff[:, 0]*obs_n[:, 0] + diff[:, 1]*obs_n[:, 1]) / dist_safe
                    dk = (-0.25j * k) * h1 * proj
                else:
                    proj = (src_n[:, 0]*diff[:, 0] + src_n[:, 1]*diff[:, 1]) / dist_safe
                    dk = (0.25j * k) * h1 * proj

                kernel = dk if self.obs_nd else g
                w = wo * ws
                scale = obs_L * src_L * w

                for a in range(2):
                    for b in range(2):
                        blocks[:, a, b] += phi_o[a] * phi_s[b] * scale * kernel

        for a in range(2):
            for b in range(2):
                self._near_rows.append(obs_nids[:, a].astype(np.int64, copy=False))
                self._near_cols.append(src_nids[:, b].astype(np.int64, copy=False))
                self._near_vals.append(blocks[:, a, b].astype(np.complex128, copy=False))

    def _finalize_near_matrix(self):
        """Convert accumulated COO triples into a CSR sparse matrix."""
        from scipy.sparse import coo_matrix
        N = self.nnodes
        if not self._near_vals:
            self._near_mat = coo_matrix((N, N), dtype=np.complex128).tocsr()
        else:
            rows = np.concatenate(self._near_rows)
            cols = np.concatenate(self._near_cols)
            vals = np.concatenate(self._near_vals)
            # coo_matrix sums duplicate entries on conversion to CSR, matching
            # np.add.at semantics on the old dense path.
            self._near_mat = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
        # Drop the triple buffers; they can be large.
        self._near_rows = []
        self._near_cols = []
        self._near_vals = []

    def matvec(self, x):
        x = np.asarray(x, dtype=np.complex128).ravel()
        result = self._near_mat @ x  # near field via cached matrix
        self._far_field_fmm(x, result)
        return result

    def _far_field_fmm(self, x, result):
        tree = self.tree; boxes = tree.boxes; k = self.k
        if tree.n_levels < 2: return

        # ── P2M ─────────────────────────────────────────────────────────
        multipole: Dict[int, np.ndarray] = {}
        for lid in tree.get_leaves():
            leaf = boxes[lid]; p = self.p_level[leaf.level]
            src_pids = [pid for pid in leaf.panel_ids if self.src_mask[pid]]
            if not src_pids:
                multipole[lid] = np.zeros(2*p+1, dtype=np.complex128); continue
            # Batch all source quad points for this leaf.
            pts_list = []; str_list = []
            for pid in src_pids:
                nids = self.node_ids[pid]
                x_local = x[nids]  # (2,)
                # Source strength at each quad point = w * L * (phi . x_local)
                strengths = np.sum(self.wl_phi[pid] * x_local[None, None, :], axis=2).ravel()  # (Q,)
                pts_list.append(self.all_qpts[pid])  # (Q, 2)
                str_list.append(strengths)
            pts = np.vstack(pts_list); strs = np.concatenate(str_list)
            multipole[lid] = _p2m(pts, strs, leaf.center, k, p)

        # ── M2M upward ──────────────────────────────────────────────────
        for lev in range(tree.n_levels - 1, 0, -1):
            for bid in tree.get_level_boxes(lev):
                box = boxes[bid]
                if box.parent < 0: continue
                O = multipole.get(bid)
                if O is None or np.all(np.abs(O) < 1e-30): continue
                pid = box.parent; p_c = self.p_level[box.level]; p_p = self.p_level[boxes[pid].level]
                p_use = min(p_c, p_p)
                # Truncate child expansion if needed.
                if len(O) != 2*p_use+1:
                    O2 = np.zeros(2*p_use+1, dtype=np.complex128)
                    pc = (len(O)-1)//2
                    lo = max(-p_use, -pc); hi = min(p_use, pc)
                    O2[lo+p_use:hi+p_use+1] = O[lo+pc:hi+pc+1]
                    O = O2
                O_sh = _m2m(O, box.center, boxes[pid].center, k, p_use)
                if pid not in multipole:
                    multipole[pid] = np.zeros(2*p_p+1, dtype=np.complex128)
                pp = (len(multipole[pid])-1)//2
                lo = max(-p_use, -pp); hi = min(p_use, pp)
                multipole[pid][lo+pp:hi+pp+1] += O_sh[lo+p_use:hi+p_use+1]

        # ── M2L ─────────────────────────────────────────────────────────
        local: Dict[int, np.ndarray] = {}
        for lev in range(2, tree.n_levels):
            for bid in tree.get_level_boxes(lev):
                box = boxes[bid]; p_lev = self.p_level[box.level]
                L = local.get(bid, np.zeros(2*p_lev+1, dtype=np.complex128))
                for sbid in self.interact_list.get(bid, []):
                    O = multipole.get(sbid)
                    if O is None or np.all(np.abs(O) < 1e-30): continue
                    ps = (len(O)-1)//2; p_use = min(ps, p_lev)
                    if ps != p_use:
                        O2 = np.zeros(2*p_use+1, dtype=np.complex128)
                        lo = max(-p_use, -ps); hi = min(p_use, ps)
                        O2[lo+p_use:hi+p_use+1] = O[lo+ps:hi+ps+1]
                        O = O2
                    L_c = _m2l(O, boxes[sbid].center, box.center, k, p_use)
                    lo = max(-p_use, -p_lev); hi = min(p_use, p_lev)
                    L[lo+p_lev:hi+p_lev+1] += L_c[lo+p_use:hi+p_use+1]
                local[bid] = L

        # ── L2L downward ────────────────────────────────────────────────
        for lev in range(2, tree.n_levels):
            for bid in tree.get_level_boxes(lev):
                box = boxes[bid]
                if box.parent < 0: continue
                Lp = local.get(box.parent)
                if Lp is None or np.all(np.abs(Lp) < 1e-30): continue
                pp = (len(Lp)-1)//2; pc = self.p_level[box.level]; p_use = min(pp, pc)
                if pp != p_use:
                    Lp2 = np.zeros(2*p_use+1, dtype=np.complex128)
                    lo = max(-p_use, -pp); hi = min(p_use, pp)
                    Lp2[lo+p_use:hi+p_use+1] = Lp[lo+pp:hi+pp+1]
                    Lp = Lp2
                Lsh = _l2l(Lp, boxes[box.parent].center, box.center, k, p_use)
                if bid not in local:
                    local[bid] = np.zeros(2*pc+1, dtype=np.complex128)
                lo = max(-p_use, -pc); hi = min(p_use, pc)
                local[bid][lo+pc:hi+pc+1] += Lsh[lo+p_use:hi+p_use+1]

        # ── L2P ─────────────────────────────────────────────────────────
        for lid in tree.get_leaves():
            leaf = boxes[lid]
            L = local.get(lid)
            if L is None or np.all(np.abs(L) < 1e-30): continue
            if not leaf.panel_ids: continue
            p_lev = (len(L)-1)//2

            # Batch all observer quad points for this leaf.
            pids = leaf.panel_ids
            Q = len(self.qt)
            all_tgts = self.all_qpts[pids].reshape(-1, 2)  # (n_panels*Q, 2)

            if self.obs_nd:
                # K': need normals at each quad point (same for all Q in one panel)
                all_normals = np.repeat(self.normals[pids], Q, axis=0)
                field = _l2p_dlp_normal(L, all_tgts, all_normals, leaf.center, k, p_lev)
            else:
                field = _l2p_slp(L, all_tgts, leaf.center, k, p_lev)

            field = field.reshape(len(pids), Q)  # (n_panels, Q)

            # Accumulate into result: result[node_a] += Σ_q w*L*phi_a * field
            # Vectorized accumulation over panel endpoints (single einsum pass).
            pids_arr = np.asarray(pids, dtype=int)
            # wl_phi[pid, q, a] * field[pi, q] -> contribs[pi, a]
            contribs = np.einsum('pqa,pq->pa', self.wl_phi[pids_arr], field)
            leaf_nids = self.node_ids[pids_arr]  # (n_panels, 2)
            for a in range(2):
                np.add.at(result, leaf_nids[:, a], contribs[:, a])


def fmm_assemble_matvec(mesh, k, obs_normal_deriv, source_element_mask=None,
                         n_digits=6, max_leaf=40):
    """Create an FMM matvec callable: f(x) -> A @ x."""
    op = FMMOperator(mesh, k, obs_normal_deriv, source_element_mask, n_digits, max_leaf)
    return op.matvec
