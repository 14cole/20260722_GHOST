# BoR Solver Conventions (Phase 0)

Pinned before any solver code exists. Everything in the BoR effort — the
sphere references, the modal kernels, the far fields — uses these
definitions. They deliberately match the 2D solver where the concepts
overlap.

## Time convention and waves

- Time factor **e^{+jωt}** (same as the 2D solver).
- Outgoing spherical waves use the **spherical Hankel function of the
  second kind** h_n^{(2)}(kr) (2D analog: H_0^{(2)}).
- Lossy media have **Im(ε_r) ≤ 0, Im(μ_r) ≤ 0**; refractive index
  m = sqrt(ε_r μ_r) taken with Im(m) ≤ 0 (causal decay), matching
  `_causal_medium_index` in the 2D solver.
- Free space: k0 = 2πf/c, η0 = 376.730313668 Ω.

## Geometry and angles

- BoR axis = **z axis**. Generatrix drawn in the (ρ, z) half-plane, ρ ≥ 0.
- **Aspect angle θ** is measured **from +z (nose-on = 0°)**; broadside =
  90°. Monostatic sweeps run θ ∈ [0°, 180°].
- Azimuthal modes e^{jmφ}; plane-wave incidence in the φ = 0 plane WLOG.

## Polarization

- **VV (θ-pol)**: incident E in the plane of incidence (contains the z
  axis). **HH (φ-pol)**: incident E perpendicular to it.
- In the principal plane, BoR symmetry decouples VV and HH; cross-pol
  terms appear only in off-plane bistatic cuts.

## RCS normalization

- σ = 4π r² |E_s|² / |E_i|², units m²; reported as dBsm = 10 log10(σ/1 m²).
- No 2D-style per-length quantities anywhere in the BoR stack.

## Riccati–Bessel functions (used throughout the sphere references)

- ψ_n(z) = z j_n(z) (regular), ζ_n(z) = z y_n(z),
  **ξ_n(z) = ψ_n(z) − j ζ_n(z) = z h_n^{(2)}(z)** (outgoing, e^{+jωt}).
- Derivatives via the recurrence R_n'(z) = z f_{n−1}(z) − n f_n(z).
- Computed from cylindrical J/Y of half-integer order (complex-capable).

## Boundary conditions (Debye-potential form, per mode n)

Radial function U(r) per region (ψ regular / ζ standing / ξ outgoing in
its region's k_i r), matched at each interface r = a:

- **TM (electric) modes**: U continuous and (k_i/ε_i) U' continuous.
- **TE (magnetic) modes**: U continuous and (k_i/μ_i) U' continuous.
- **PEC surface**: TM → U' = 0; TE → U = 0.
- **Leontovich (IBC) surface** (normal outward into air):
  TM → η0 U'/U = j Z_s k0-normalized form; implemented as the modal
  equation [ψ' + Aξ'] = (jZ_s/η0)[ψ + Aξ] (TM) and its dual with
  (jη0... ) for TE — signs are pinned by the Z_s → 0 PEC limit and the
  Rayleigh/GO anchors, not asserted from a text.

## Validation anchors (independent of any textbook coefficient table)

1. **Rayleigh dipole limit** (ka → 0):
   σ_back = 4π k⁴ a⁶ |(ε_r−1)/(ε_r+2) − (μ_r−1)/(μ_r+2)|²;
   PEC limit (ε→∞, μ→0): σ_back = 9π k⁴ a⁶.
2. **Geometric-optics limit** (ka → ∞): σ/πa² → 1 for PEC.
3. **Optical theorem**: lossless bodies must give σ_ext = σ_sca to
   near machine precision; lossy must give σ_ext > σ_sca.
4. **Degenerate-limit equalities**: Z_s→0 ⇒ PEC; coating ε=1 ⇒ bare PEC of
   the core radius; core radius→0 ⇒ homogeneous dielectric sphere.
5. Bistatic pattern at θ_bis = 180° ⇒ the monostatic sum.
