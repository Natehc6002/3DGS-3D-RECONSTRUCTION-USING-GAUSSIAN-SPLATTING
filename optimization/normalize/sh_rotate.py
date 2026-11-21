"""SH coefficient rotation for Brush Gaussian PLY files.

Brush evaluates view-dependent colour with the real spherical-harmonic
basis of Sloan, "Efficient Spherical Harmonic Evaluation", JCGT 2013,
as implemented in the Brush source itself:

    crates/brush-render/src/kernels/sh.rs
    (fetched from https://github.com/ArthurBrussee/brush, main,
    2026-11-21; constants below are Brush's f32 literals verbatim)

Per channel the local coefficient order is index 0 = DC (`f_dc_*`) and
1..15 = the `f_rest_*` block in PLY order (band 1: 3 coefficients,
band 2: 5, band 3: 7) - the channel-major layout documented in
optimization/README.md.

A rigid scene rotation R must satisfy, for every view direction d,

    colour'(d) = colour(R^T d)

because the Gaussian and the camera move with the scene. With the
orthonormal basis {B_j} the new coefficients are

    c'_j = sum_i M_ji c_i,
    M_ji = <B_j, B_i o R^T> / <B_j, B_j>

evaluated per degree block (the basis is block diagonal in l) by an
exact product quadrature (Gauss-Legendre in cos(theta) x periodic
midpoint in phi - machine precision for band<=3 products). Every
rotation is verified numerically before use (verify_rotation): colour
of sample directions before and after the rotation must agree to
SH_TOLERANCE, otherwise the caller must refuse to rotate.
"""
from __future__ import annotations

import numpy as np

# Brush f32 literals (crates/brush-render/src/kernels/sh.rs).
SH_C0 = 0.2820948
_F0A = 0.4886025        # band 1
_F1A = 0.54627424       # band 2 (xy / x2-y2 terms)
_F0B = -1.0925485       # band 2 (z * {y, x} terms)
_P6A, _P6B = 0.9461747, 0.31539157   # band 2 m=0
_F0C_A, _F0C_B = -2.285229, 0.4570458   # band 3 m=+-1 envelope
_F1B = 1.4453057        # band 3 m=+-2 envelope
_F2A = -0.5900436       # band 3 m=+-3 envelope
_P12A, _P12B = 1.8658817, 1.119529  # band 3 m=0

SH_TOLERANCE = 1e-5     # max |colour error| accepted after a rotation
QUADRATURE_N_U = 5      # Gauss-Legendre nodes in cos(theta) (exact <= deg 9)
QUADRATURE_N_PHI = 16   # periodic midpoint nodes in phi (exact |k| <= 15)
VERIFY_DIRS = 512       # random directions for the verification
VERIFY_SETS = 8         # random coefficient sets when none are given
VERIFY_SEED = 0


def sh_basis(dirs, degree):
    """Evaluate the Brush SH basis at unit directions.

    dirs: (n, 3) float64 rows. Returns (n, (degree+1)**2) with column j
    = basis function of local index j (0 = DC), matching the PLY layout.
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    cols = [np.full(x.shape, SH_C0, dtype=np.float64)]
    if degree >= 1:
        cols += [-_F0A * y, _F0A * z, -_F0A * x]
    if degree >= 2:
        z2 = z * z
        fc1 = x * x - y * y
        fs1 = 2.0 * x * y
        f0b = _F0B * z
        cols += [_F1A * fs1, f0b * y, _P6A * z2 - _P6B, f0b * x,
                 _F1A * fc1]
    if degree >= 3:
        z2 = z * z
        f0c = _F0C_A * z2 + _F0C_B
        f1b = _F1B * z
        fc1 = x * x - y * y
        fs1 = 2.0 * x * y
        fc2 = x * fc1 - y * fs1
        fs2 = x * fs1 + y * fc1
        cols += [_F2A * fs2, f1b * fs1, f0c * y,
                 z * (_P12A * z2 - _P12B), f0c * x, f1b * fc1,
                 _F2A * fc2]
    if degree >= 4:
        raise ValueError("SH degree > 3 is not a Brush export format")
    return np.stack(cols, axis=1)


def product_quadrature(n_u=QUADRATURE_N_U, n_phi=QUADRATURE_N_PHI):
    """Exact sphere quadrature for band<=3 products.

    Gauss-Legendre nodes in u = cos(theta) (the sin(theta) Jacobian
    turns every integrand here into a polynomial in u of degree <= 9)
    times a periodic midpoint rule in phi (exact for Fourier modes
    |k| <= 15). Returns (dirs, weights) with dirs unit-normalized.
    """
    u, w = np.polynomial.legendre.leggauss(n_u)
    phi = 2.0 * np.pi * (np.arange(n_phi, dtype=np.float64) + 0.5) / n_phi
    uu, pp = np.meshgrid(u, phi, indexing="ij")
    rho = np.sqrt(np.maximum(0.0, 1.0 - uu * uu))
    dirs = np.stack([rho * np.cos(pp), rho * np.sin(pp), uu],
                    axis=-1).reshape(-1, 3)
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    return dirs, weights


def random_unit_dirs(n, seed=VERIFY_SEED):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rotation_matrix(r, degree):
    """Coefficient matrix M with c' = M c giving colour'(d) = colour(r^T d).

    Per degree block l: M_ji = sum_k w_k B_j(d_k) B_i(r^T d_k) /
    sum_k w_k B_j(d_k)^2 over the exact product quadrature. DC (l=0) is
    exactly identity and is left untouched.
    """
    r = np.asarray(r, dtype=np.float64)
    nb = (degree + 1) ** 2
    if degree == 0:
        return np.eye(1)
    dirs, wts = product_quadrature()
    b = sh_basis(dirs, degree)
    b_rot = sh_basis(dirs @ r, degree)     # rows are r^T d
    bw = b * wts[:, None]
    m = np.eye(nb, dtype=np.float64)       # DC block stays identity
    for ell in range(1, degree + 1):
        sl = slice(ell * ell, (ell + 1) * (ell + 1))
        bl, br = b[:, sl], b_rot[:, sl]
        denom = np.einsum("kj,kj->j", bl, bw[:, sl])
        m[sl, sl] = (bl.T @ (br * wts[:, None])) / denom[:, None]
    return m


def verify_rotation(r, degree, coeffs=None, n_dirs=VERIFY_DIRS,
                    n_sets=VERIFY_SETS, seed=VERIFY_SEED):
    """Max |colour_after(d) - colour_before(r^T d)| over sample directions.

    colour_after(d) uses the rotated coefficients (c' = M c), the other
    side the original coefficients at the rotated direction - exactly
    the invariant a scene rotation must keep. coeffs: optional
    (n, nb) real per-channel coefficient rows (the file's own values);
    otherwise random sets are used. Returns (max_error, M).
    """
    m = rotation_matrix(r, degree)
    if degree == 0:
        return 0.0, m
    nb = (degree + 1) ** 2
    dirs = random_unit_dirs(n_dirs, seed)
    b = sh_basis(dirs, degree)
    b_rot = sh_basis(dirs @ np.asarray(r, dtype=np.float64), degree)
    if coeffs is None:
        rng = np.random.default_rng(seed + 1)
        coeffs = rng.normal(size=(n_sets, nb))
    else:
        coeffs = np.asarray(coeffs, dtype=np.float64)
        if coeffs.ndim == 1:
            coeffs = coeffs[None, :]
        if coeffs.shape[1] != nb:
            raise ValueError("coefficients do not match the SH degree")
        if len(coeffs) > n_sets * 4:
            step = max(1, len(coeffs) // (n_sets * 4))
            coeffs = coeffs[::step][:n_sets * 4]
    max_err = 0.0
    for c in coeffs:
        old = b_rot @ c                  # colour(r^T d), original c
        new = b @ (m @ c)                # colour(d), rotated c
        max_err = max(max_err, float(np.max(np.abs(new - old))))
    return max_err, m


def rotation_is_identity(r, atol=1e-12):
    return bool(np.allclose(np.asarray(r, dtype=np.float64),
                            np.eye(3), atol=atol))
