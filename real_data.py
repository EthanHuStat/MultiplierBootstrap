#!/usr/bin/env python3
"""
Standalone real-data spike-count comparison.

This file contains every project-specific function used by the analysis.
It does not import another user/project Python file.

Retained methods
----------------
    Proposed
    BA2017
    FGZ2022
    KML2023
    BN2002
    DO2019
    Onat2009
    DY2022       (calibrated local gap-ratio statistic)
    PY2014       (calibrated leading eigenvalue-gap rule)

External inputs
---------------
    data/GEUVADIS_matrix.npy
    data/1000G_EUR_matrix.npy
    data/ding_yang_null_calibration.csv
    one Pass-Yao calibration CSV in data/

The two matrices are loaded exactly as stored. No shared centering, scaling,
transposition, filtering, feature selection, or imputation is performed.
Method-specific matrix operations required by an estimator are retained.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import gamma as gamma_distribution
from scipy.stats import norm

STANDALONE_IMPLEMENTATION = True
Array = np.ndarray


def standalone_import_report() -> Dict[str, Any]:
    """Return the external modules imported by this standalone file."""
    return {
        "project_python_dependencies": [],
        "external_packages": ["numpy", "pandas", "scipy"],
        "standard_library": ["json", "time", "pathlib", "typing"],
    }


def as_observations_by_features(
    X: Array,
    features_axis: str = "columns",
    center: bool = True,
    scale: bool = False,
    dtype=np.float64,
) -> Array:
    X = np.asarray(X, dtype=dtype)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {X.shape}.")

    if features_axis == "rows":
        X = X.T.copy()
    elif features_axis == "columns":
        X = X.copy()
    else:
        raise ValueError("features_axis must be 'rows' or 'columns'.")

    if center:
        X = X - np.nanmean(X, axis=0, keepdims=True)

    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

    if scale:
        sd = X.std(axis=0, ddof=1, keepdims=True)
        sd[sd == 0] = 1.0
        X = X / sd

    return X


def sample_cov_eigenvalues(
    X: Array,
    assume_centered: bool = True,
    sort_desc: bool = True,
) -> Array:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {X.shape}.")
    if not assume_centered:
        X = X - X.mean(axis=0, keepdims=True)

    n, p = X.shape
    if p >= n:
        G = (X @ X.T) / n
        eigs = np.linalg.eigvalsh(G)
    else:
        S = (X.T @ X) / n
        eigs = np.linalg.eigvalsh(S)
    eigs = np.maximum(eigs, 0.0)
    if sort_desc:
        eigs = np.sort(eigs)[::-1]
    return eigs


def top_k_cov_eigs_from_matrix(A: Array, k: int) -> Array:
    d = int(A.shape[0])
    k = min(int(k), d)
    vals = eigh(
        A,
        subset_by_index=[d - k, d - 1],
        eigvals_only=True,
        check_finite=False,
        overwrite_a=True,
    )
    return np.sort(np.maximum(vals, 0.0))[::-1]


def top_k_weighted_cov_eigs(X: Array, w: Array, k: int) -> Array:
    n, p = X.shape
    Xw = X * np.sqrt(w)[:, None]
    if p <= n:
        M = (Xw.T @ Xw) / n
    else:
        M = (Xw @ Xw.T) / n
    return top_k_cov_eigs_from_matrix(M, k)


def _safe_kmax(eigs: Array, kmax: Optional[int]) -> int:
    if kmax is None:
        return len(eigs) - 1
    return int(max(1, min(int(kmax), len(eigs) - 1)))


def mp_upper_edge_from_diag(
    variances: Array,
    gamma: float,
    eps: float = 1e-10,
) -> float:
    t = np.asarray(variances, dtype=np.float64)
    t = t[np.isfinite(t) & (t > 0)]
    if len(t) == 0:
        return 0.0

    tmax = float(np.max(t))
    left = -1.0 / tmax + eps
    right = -eps

    def deriv(v: float) -> float:
        return 1.0 / (v * v) - gamma * np.mean((t * t) / ((1.0 + t * v) ** 2))

    def xmap(v: float) -> float:
        return -1.0 / v + gamma * np.mean(t / (1.0 + t * v))

    try:
        grid = np.linspace(left, right, 400)
        vals = np.array([deriv(v) for v in grid])
        ok = np.isfinite(vals)
        grid, vals = grid[ok], vals[ok]
        idx = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        if len(idx) == 0:
            raise ValueError("No sign change found.")

        roots = []
        for i in idx:
            try:
                roots.append(brentq(deriv, grid[i], grid[i + 1], maxiter=100))
            except ValueError:
                continue
        if not roots:
            raise ValueError("No roots found.")
        edges = [xmap(v) for v in roots if np.isfinite(xmap(v))]
        return float(np.max(edges))
    except Exception:
        sigma2 = float(np.mean(t))
        return sigma2 * (1.0 + np.sqrt(gamma)) ** 2


def _simulate_gamma_bulk_eigs_common_random_numbers(
    n: int,
    p: int,
    gamma_shape: float,
    U_tau: Array,
    Z_common: Array,
) -> Array:
    M = int(U_tau.shape[0])
    m = min(n, p)
    out = np.empty((M, m), dtype=np.float64)
    U = np.clip(np.asarray(U_tau, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    for b in range(M):
        tau = gamma_distribution.ppf(U[b], a=gamma_shape, scale=1.0)
        Xb = Z_common[b] * np.sqrt(tau)[None, :]
        out[b] = sample_cov_eigenvalues(Xb)[:m]
    return out


def _simulate_gamma_bulk_eigs(
    n: int,
    p: int,
    gamma_shape: float,
    M: int,
    rng: np.random.Generator,
) -> Array:
    m = min(n, p)
    out = np.empty((M, m), dtype=np.float64)
    for b in range(M):
        tau = rng.gamma(shape=gamma_shape, scale=1.0, size=p)
        Z = rng.standard_normal(size=(n, p))
        Xb = Z * np.sqrt(tau)[None, :]
        out[b] = sample_cov_eigenvalues(Xb)[:m]
    return out


def _central_indices(m: int, beta: float) -> Array:
    lo = int(np.floor(m * beta))
    hi = int(np.floor(m * (1.0 - beta)))
    lo = max(lo, 0)
    hi = min(hi, m - 1)
    if hi <= lo:
        raise ValueError("Invalid beta; central bulk window is empty.")
    return np.arange(lo, hi + 1)


def _ls_scale(y: Array, x: Array) -> float:
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 1.0
    return float(np.dot(x, y) / denom)


def bema_gamma(
    X: Array,
    features_axis: str = "columns",
    beta: float = 0.10,
    M_fit: int = 20,
    M_final: int = 200,
    q: float = 0.90,
    shape_bounds: Tuple[float, float] = (0.1, 50.0),
    center: bool = True,
    scale: bool = False,
    random_state: int = 0,
    kmax: Optional[int] = None,
    deterministic: bool = True,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    Xn = as_observations_by_features(X, features_axis, center, scale)
    n, p = Xn.shape
    eigs = sample_cov_eigenvalues(Xn)
    m = len(eigs)
    kmax = _safe_kmax(eigs, kmax)
    bulk_idx = _central_indices(m, beta)

    if deterministic:
        U_fit = rng.uniform(size=(M_fit, p))
        Z_fit = rng.standard_normal(size=(M_fit, n, p))

        def objective(shape: float) -> float:
            sim = _simulate_gamma_bulk_eigs_common_random_numbers(
                n, p, shape, U_fit, Z_fit
            )
            sim_mean = sim.mean(axis=0)
            a = _ls_scale(eigs[bulk_idx], sim_mean[bulk_idx])
            diff = a * sim_mean[bulk_idx] - eigs[bulk_idx]
            return float(np.sum(diff * diff))
    else:
        def objective(shape: float) -> float:
            sim = _simulate_gamma_bulk_eigs(n, p, shape, M_fit, rng)
            sim_mean = sim.mean(axis=0)
            a = _ls_scale(eigs[bulk_idx], sim_mean[bulk_idx])
            diff = a * sim_mean[bulk_idx] - eigs[bulk_idx]
            return float(np.sum(diff * diff))

    opt = minimize_scalar(
        objective,
        bounds=shape_bounds,
        method="bounded",
        options={"xatol": 1e-3},
    )
    shape_hat = float(opt.x)
    sim_final = _simulate_gamma_bulk_eigs(n, p, shape_hat, M_final, rng)

    mean_raw = sim_final.mean(axis=0)
    upper_raw = np.quantile(sim_final, q, axis=0)
    lower_raw = np.quantile(sim_final, 1.0 - q, axis=0)
    scale_hat = _ls_scale(eigs[bulk_idx], mean_raw[bulk_idx])

    bulk_mean = scale_hat * mean_raw
    bulk_upper = scale_hat * upper_raw
    bulk_lower = scale_hat * lower_raw
    edge = float(np.max(bulk_upper))
    khat = int(np.sum(eigs[:kmax] > edge))
    return {
        "method": "BEMA",
        "k_hat": khat,
        "edge": edge,
        "gamma_shape_hat": shape_hat,
        "scale_hat": float(scale_hat),
        "bulk_mean": bulk_mean,
        "bulk_upper": bulk_upper,
        "bulk_lower": bulk_lower,
        "eigenvalues": eigs,
        "n": int(n),
        "p": int(p),
        "beta": float(beta),
        "M_fit": int(M_fit),
        "M_final": int(M_final),
        "q": float(q),
        "deterministic_fit": bool(deterministic),
        "opt_success": bool(opt.success),
    }


def _corrcoef_from_data(X: Array, center: bool = True) -> Array:
    Xn = as_observations_by_features(X, "columns", center=center, scale=False)
    if not center:
        Xn = Xn - Xn.mean(axis=0, keepdims=True)
    sd = Xn.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = Xn / sd
    R = (Z.T @ Z) / max(Z.shape[0] - 1, 1)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def ekc_bva2017(
    X: Array,
    features_axis: str = "columns",
    center: bool = True,
    input_is_correlation: bool = False,
    N: Optional[int] = None,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    if input_is_correlation:
        if N is None:
            raise ValueError("N must be supplied when input_is_correlation=True.")
        R = X.copy()
        n_obs = int(N)
    else:
        Xn = as_observations_by_features(X, features_axis, center=center, scale=False)
        n_obs = int(Xn.shape[0])
        R = _corrcoef_from_data(Xn, center=center)

    J = int(R.shape[0])
    eigs = np.linalg.eigvalsh(R)
    eigs = np.sort(np.maximum(eigs, 0.0))[::-1]
    l_up = float((1.0 + np.sqrt(J / n_obs)) ** 2)

    prefix_before = np.r_[0.0, np.cumsum(eigs[:-1])]
    correction_factor = (J - prefix_before) / np.arange(J, 0, -1)
    refs_unrestricted = l_up * correction_factor
    refs = np.maximum(refs_unrestricted, 1.0)
    khat = int(np.sum(eigs > refs))
    return {
        "method": "EKC-BvA2017",
        "k_hat": khat,
        "eigenvalues": eigs,
        "references": refs,
        "references_unrestricted": refs_unrestricted,
        "n": int(n_obs),
        "p": int(J),
        "l_up": l_up,
        "input_is_correlation": bool(input_is_correlation),
    }


def bai_ng_icp1(
    X: Array,
    features_axis: str = "columns",
    center: bool = True,
    scale: bool = False,
    kmax: int = 20,
) -> Dict[str, Any]:
    Xn = as_observations_by_features(X, features_axis, center=center, scale=scale)
    n, p = Xn.shape
    eigs = sample_cov_eigenvalues(Xn)
    total = float(np.sum(eigs))
    kmax = _safe_kmax(eigs, kmax)
    penalty = ((n + p) / (n * p)) * np.log((n * p) / (n + p))

    ks = np.arange(0, kmax + 1)
    ic = np.empty_like(ks, dtype=float)
    Vk = np.empty_like(ks, dtype=float)
    for idx, k in enumerate(ks):
        residual = max(total - float(np.sum(eigs[:k])), 1e-12)
        Vk[idx] = residual / p
        ic[idx] = np.log(Vk[idx]) + k * penalty

    khat = int(ks[np.argmin(ic)])
    return {
        "method": "Bai-Ng ICp1",
        "k_hat": khat,
        "ks": ks,
        "ic": ic,
        "V": Vk,
        "eigenvalues": eigs,
        "n": int(n),
        "p": int(p),
        "penalty": float(penalty),
    }


def pass_yao_calibrated_raw_gap_from_row(
    eigenvalues: Array,
    calibration_row: pd.Series,
    kmax: int = 20,
) -> Dict[str, Any]:
    """Pass-Yao (2014) calibrated leading-gap estimator.

    The estimator reads a calibrated ``d_n`` value, forms the unscaled
    sample-eigenvalue gaps, and counts consecutive leading gaps satisfying
    ``gap[j] >= d_n``. It stops at the first leading gap below ``d_n``.
    """
    d_col = None
    for candidate in ("d_n", "dn", "threshold", "gap_threshold"):
        if candidate in calibration_row.index:
            d_col = candidate
            break

    if d_col is None:
        raise ValueError(
            "Pass-Yao calibration row does not contain "
            "d_n/dn/threshold/gap_threshold."
        )

    d_n = float(calibration_row[d_col])
    values = np.asarray(eigenvalues, dtype=float)
    gaps = np.asarray(values[:-1] - values[1:], dtype=float)
    kmax_effective = int(min(int(kmax), len(gaps)))

    k_hat = 0
    for j in range(kmax_effective):
        if gaps[j] >= d_n:
            k_hat = j + 1
        else:
            break

    return {
        "method": "PY2014",
        "k_hat": int(k_hat),
        "d_n": float(d_n),
        "gaps": gaps,
        "kmax_effective": int(kmax_effective),
        "variant": "calibrated_raw_leading_gap",
        "threshold_source": "selected_calibration_row",
        "calibration_row": calibration_row.to_dict(),
    }


ONATSKI_TABLE = {
    0.15: [2.75, 3.62, 4.15, 4.54, 4.89, 5.20, 5.45, 5.70],
    0.10: [3.33, 4.31, 4.91, 5.40, 5.77, 6.13, 6.42, 6.66],
    0.09: [3.50, 4.49, 5.13, 5.62, 6.03, 6.39, 6.67, 6.92],
    0.08: [3.69, 4.72, 5.37, 5.91, 6.31, 6.68, 6.95, 7.25],
    0.07: [3.92, 4.99, 5.66, 6.24, 6.62, 7.00, 7.32, 7.59],
    0.06: [4.20, 5.31, 6.03, 6.57, 7.00, 7.41, 7.74, 8.04],
    0.05: [4.52, 5.73, 6.46, 7.01, 7.50, 7.95, 8.29, 8.59],
    0.04: [5.02, 6.26, 6.97, 7.63, 8.16, 8.61, 9.06, 9.36],
    0.03: [5.62, 6.91, 7.79, 8.48, 9.06, 9.64, 10.11, 10.44],
    0.02: [6.55, 8.15, 9.06, 9.93, 10.47, 11.27, 11.75, 12.13],
    0.01: [8.74, 10.52, 11.67, 12.56, 13.42, 14.26, 14.88, 15.25],
}


def onatski_critical_value(alpha: float, h: int) -> float:
    h = int(h)
    if h < 1 or h > 8:
        raise ValueError("Onatski Table I supports h=1,...,8 only.")
    alpha = float(alpha)
    if alpha not in ONATSKI_TABLE:
        allowed = sorted(ONATSKI_TABLE.keys())
        raise ValueError(f"alpha={alpha} not in Onatski table. Allowed: {allowed}.")
    return float(ONATSKI_TABLE[alpha][h - 1])


def onatski_complex_split_eigenvalues(X: Array, center: bool = True) -> Array:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {X.shape}.")
    if center:
        X = X - X.mean(axis=0, keepdims=True)

    n, p = X.shape
    m = n // 2
    if m < 2:
        raise ValueError("Need at least four observations for complex split.")
    X1 = X[:m, :]
    X2 = X[m:2 * m, :]
    Y = X1 + 1j * X2

    if p <= m:
        S = (Y.conj().T @ Y) / m
        eigs = np.linalg.eigvalsh(S)
    else:
        G = (Y @ Y.conj().T) / m
        eigs = np.linalg.eigvalsh(G)
    return np.sort(np.maximum(eigs.real, 0.0))[::-1]


def onatski_gap_ratio_statistic(eigs: Array, k0: int, k1: int) -> Tuple[float, Array]:
    eigs = np.asarray(eigs, dtype=np.float64)
    k0 = int(k0)
    k1 = int(k1)
    if k1 <= k0:
        raise ValueError("Need k1 > k0.")
    if k1 - k0 > 8:
        raise ValueError("Use k1-k0 <= 8 for Onatski Table I.")
    if len(eigs) < k1 + 2:
        raise ValueError("Not enough eigenvalues for Onatski statistic.")

    ratios = []
    for i in range(k0 + 1, k1 + 1):
        num = eigs[i - 1] - eigs[i]
        den = eigs[i] - eigs[i + 1]
        ratios.append(num / max(float(den), 1e-300))
    ratios_arr = np.asarray(ratios, dtype=float)
    return float(np.max(ratios_arr)), ratios_arr


def onatski_gap_test(eigs: Array, k0: int, k1: int, alpha: float = 0.05) -> Dict[str, Any]:
    h = int(k1) - int(k0)
    R, ratios = onatski_gap_ratio_statistic(eigs, k0, k1)
    crit = onatski_critical_value(alpha, h)
    return {
        "reject": bool(R > crit),
        "R": float(R),
        "critical_value": float(crit),
        "ratios": ratios,
        "k0": int(k0),
        "k1": int(k1),
        "h": int(h),
        "alpha": float(alpha),
    }


def onatski_sequential_fixed_upper(
    X: Array,
    features_axis: str = "columns",
    center: bool = True,
    alpha: float = 0.05,
    k_upper: int = 8,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    if features_axis == "rows":
        X = X.T.copy()
    elif features_axis != "columns":
        raise ValueError("features_axis must be 'columns' or 'rows'.")

    eigs = onatski_complex_split_eigenvalues(X, center=center)
    effective_upper = min(int(k_upper), len(eigs) - 2, 8)
    tests = []

    for k0 in range(effective_upper):
        k1 = effective_upper
        test = onatski_gap_test(eigs, k0=k0, k1=k1, alpha=alpha)
        tests.append(test)
        if not test["reject"]:
            return {
                "method": "Onatski sequential fixed-upper",
                "k_hat": int(k0),
                "eigenvalues": eigs,
                "tests": tests,
                "alpha": float(alpha),
                "k_upper": int(effective_upper),
                "variant": "complex_split",
                "stop_R": float(test["R"]),
                "stop_critical_value": float(test["critical_value"]),
                "num_rejections_before_stop": int(len(tests) - 1),
            }

    return {
        "method": "Onatski sequential fixed-upper",
        "k_hat": int(effective_upper),
        "eigenvalues": eigs,
        "tests": tests,
        "alpha": float(alpha),
        "k_upper": int(effective_upper),
        "variant": "complex_split",
        "stop_R": float(tests[-1]["R"]) if tests else np.nan,
        "stop_critical_value": float(tests[-1]["critical_value"]) if tests else np.nan,
        "num_rejections_before_stop": int(len(tests)),
    }


def dpa(
    X: Array,
    features_axis: str = "columns",
    center: bool = True,
    scale: bool = False,
    kmax: Optional[int] = None,
    threshold_scale: float = 1.0,
) -> Dict[str, Any]:
    Xn = as_observations_by_features(X, features_axis, center=center, scale=scale)
    n, p = Xn.shape
    eigs = sample_cov_eigenvalues(Xn)
    kmax = _safe_kmax(eigs, kmax)
    variances = Xn.var(axis=0, ddof=0)
    edge = threshold_scale * mp_upper_edge_from_diag(variances, gamma=p / n)
    khat = int(np.sum(eigs[:kmax] > edge))
    return {
        "method": "DPA",
        "k_hat": khat,
        "edge": float(edge),
        "eigenvalues": eigs,
        "n": int(n),
        "p": int(p),
        "threshold_scale": float(threshold_scale),
    }


def act_adjusted_correlation_thresholding(
    X: Array,
    features_axis: str = "columns",
    center: bool = True,
    rmax: int = 20,
) -> Dict[str, Any]:
    """
    ACT: adjusted correlation thresholding of Fan, Guo and Zheng.

    This method is intentionally run on the sample correlation matrix, even
    when the other methods in this script are run on the sample covariance
    eigenvalues.  It first computes bias-corrected sample correlation
    eigenvalues and then thresholds them at

        1 + sqrt(p / (n - 1)).

    Parameters
    ----------
    X : ndarray
        Data matrix.
    features_axis : {"columns", "rows"}
        Whether variables are columns or rows.
    center : bool
        Whether to demean columns before computing the sample correlation.
    rmax : int
        Maximum number of factors/spikes to consider.
    """
    Xn = as_observations_by_features(
        X,
        features_axis=features_axis,
        center=center,
        scale=False,
    )
    n, p = Xn.shape
    if n <= 1 or p <= 2:
        return {
            "method": "ACT",
            "k_hat": 0,
            "corrected_eigs": np.array([], dtype=float),
            "threshold": np.nan,
            "n": int(n),
            "p": int(p),
        }

    # Sample covariance uses / n to match the rest of this script.  The
    # subsequent normalization removes this global factor in the correlation
    # matrix, so /n versus /(n-1) does not change R_hat.
    S = (Xn.T @ Xn) / float(n)
    d = np.sqrt(np.maximum(np.diag(S), 0.0))
    d[d <= 1e-14] = 1.0
    R = S / np.outer(d, d)
    R = 0.5 * (R + R.T)

    eigs = np.linalg.eigvalsh(R)[::-1]
    eigs = np.asarray(eigs, dtype=float)

    # Need lambda_{j+1}; keep j <= p-1 in 1-based indexing.
    rmax_eff = int(max(0, min(int(rmax), p - 1)))
    corrected = np.full(rmax_eff, np.nan, dtype=float)
    rho_vals = np.full(rmax_eff, np.nan, dtype=float)

    tiny = 1e-12
    for j0 in range(rmax_eff):
        # j is 1-based index used in the paper.
        j = j0 + 1
        z = float(eigs[j0])
        lam_next = float(eigs[j0 + 1])
        rho_j = float(p - j) / float(n - 1)
        rho_vals[j0] = rho_j

        tail = eigs[j0 + 1:]
        denom = tail - z
        denom[np.abs(denom) < tiny] = -tiny
        bulk_sum = float(np.sum(1.0 / denom))

        pseudo_point = (3.0 * z + lam_next) / 4.0
        pseudo_denom = pseudo_point - z
        if abs(pseudo_denom) < tiny:
            pseudo_denom = -tiny
        pseudo_term = 1.0 / pseudo_denom

        m_nj = (bulk_sum + pseudo_term) / float(p - j)
        underline_m = -(1.0 - rho_j) / z + rho_j * m_nj
        if abs(underline_m) > tiny and np.isfinite(underline_m):
            corrected[j0] = -1.0 / underline_m

    threshold = float(1.0 + np.sqrt(p / float(n - 1)))
    khat = int(np.sum(np.isfinite(corrected) & (corrected > threshold)))

    return {
        "method": "ACT",
        "k_hat": khat,
        "corrected_eigs": corrected,
        "threshold": threshold,
        "rho_j": rho_vals,
        "sample_corr_eigs": eigs[: min(len(eigs), rmax_eff + 2)],
        "n": int(n),
        "p": int(p),
    }


def draw_chisq_weights(n: int, N: int, rng: np.random.Generator) -> Array:
    return rng.chisquare(df=N, size=n) / N


# =============================================================================
# Calibrated PY2014 and DY2022-local implementations
# copied into this standalone file from the supplied method script
# =============================================================================

def _safe_loglog_n(n: int) -> float:
    return float(max(np.log(np.log(max(int(n), 4))), 1e-12))


def pass_yao_threshold(n: int, p: int, C: float = 4.0) -> Tuple[float, float]:
    """
    Distinct-spike paper simulation threshold.

    d_n = C sqrt(2 log log n) / (n^(2/3) beta),
    beta = (1+sqrt(c)) (1+sqrt(1/c))^(1/3), c=p/n.
    """
    c = float(p) / float(n)
    beta = (1.0 + np.sqrt(c)) * (1.0 + np.sqrt(1.0 / c)) ** (1.0 / 3.0)
    d_n = C * np.sqrt(2.0 * _safe_loglog_n(n)) / ((n ** (2.0 / 3.0)) * beta)
    return float(d_n), float(beta)


def pass_yao_two_gap_threshold_from_C(n: int, C: float) -> float:
    """
    Possibly-equal-spike paper threshold form.

    d_n = C n^(-2/3) sqrt(2 log log n).
    The calibrated implementation below estimates d_n directly and reports the
    implied C.
    """
    return float(C * np.sqrt(2.0 * _safe_loglog_n(n)) / (n ** (2.0 / 3.0)))


def pass_yao_two_gap_C_from_threshold(n: int, d_n: float) -> float:
    denom = np.sqrt(2.0 * _safe_loglog_n(n))
    return float(float(d_n) * (n ** (2.0 / 3.0)) / denom)


def _pass_yao_gap_rule_scaled(
    eigs_scaled: Array,
    n: int,
    p: int,
    smax: int = 20,
    C: float = 4.0,
    variant: str = "one_gap",
    d_n_override: Optional[float] = None,
) -> Tuple[int, float, float, Array]:
    eigs_scaled = np.sort(np.asarray(eigs_scaled, dtype=float))[::-1]
    gaps = eigs_scaled[:-1] - eigs_scaled[1:]

    if d_n_override is None:
        d_n, beta = pass_yao_threshold(n, p, C=C)
    else:
        d_n = float(d_n_override)
        _, beta = pass_yao_threshold(n, p, C=1.0)

    smax_eff = min(int(smax), len(gaps) - 1)

    if variant == "one_gap":
        for j in range(0, smax_eff + 1):
            if gaps[j] < d_n:
                return int(j), float(d_n), float(beta), gaps
        return int(smax_eff + 1), float(d_n), float(beta), gaps

    if variant == "two_gap":
        # Later equal/close-spike extension: stop only after two consecutive
        # small gaps.  j is the estimated number of spikes.
        max_j = min(int(smax), len(gaps) - 2)
        for j in range(0, max_j + 1):
            if gaps[j] < d_n and gaps[j + 1] < d_n:
                return int(j), float(d_n), float(beta), gaps
        return int(max_j + 1), float(d_n), float(beta), gaps

    raise ValueError("variant must be 'one_gap' or 'two_gap'.")


def pass_yao_unknown_sigma(
    X: Array,
    features_axis: str = "columns",
    center: bool = True,
    scale: bool = False,
    kmax: int = 20,
    smax: int = 20,
    C: float = 4.0,
    max_iter: int = 50,
    variant: str = "one_gap",
    d_n_override: Optional[float] = None,
    threshold_source: str = "formula",
) -> Dict[str, Any]:
    Xn = as_observations_by_features(X, features_axis, center=center, scale=scale)
    n, p = Xn.shape
    eigs = sample_cov_eigenvalues(Xn)
    smax = min(int(smax), int(kmax), len(eigs) - 3)

    # Estimate sigma^2 from all p sample covariance eigenvalues.  When p>n,
    # sample_cov_eigenvalues returns only the nonzero eigenvalues; the missing
    # p-n covariance eigenvalues are zero and must be included in the divisor.
    sigma2 = float(np.sum(eigs) / max(int(p), 1))
    q_old = -1
    converged = False
    d_n = np.nan
    beta = np.nan
    gaps = np.array([])
    it = -1

    for it in range(int(max_iter)):
        eigs_scaled = eigs / max(sigma2, 1e-12)
        q_new, d_n, beta, gaps = _pass_yao_gap_rule_scaled(
            eigs_scaled,
            n=n,
            p=p,
            smax=smax,
            C=C,
            variant=variant,
            d_n_override=d_n_override,
        )
        q_new = int(min(q_new, len(eigs) - 1))
        if q_new == q_old:
            converged = True
            break
        q_old = q_new
        denom = max(int(p) - q_new, 1)
        sigma2 = float(np.sum(eigs[q_new:]) / denom)

    method = "Pass-Yao-1gap" if variant == "one_gap" else "Pass-Yao-2gap"
    return {
        "method": method,
        "k_hat": int(max(q_old, 0)),
        "sigma2_hat": float(sigma2),
        "d_n": float(d_n),
        "beta": float(beta),
        "gaps_scaled": gaps,
        "eigenvalues": eigs,
        "n": int(n),
        "p": int(p),
        "smax": int(smax),
        "C": float(C),
        "variant": variant,
        "threshold_source": str(threshold_source),
        "converged": bool(converged),
        "iterations": int(it + 1),
    }






# =============================================================================
# Real-data configuration
# =============================================================================

WORKDIR = Path(".")
DATA_DIR = WORKDIR / "data"
OUT_DIR = WORKDIR / "real_data_analysis_comparison_outputs"

ALPHA = 0.01
B_FINAL = 500
KMAX = 20

BEMA_RANDOM_STATE = 20260618
BEMA_BULK_TRIM_ALPHA = 0.20
BEMA_Q = 0.99
BEMA_M_FIT = 20
BEMA_M_FINAL = 200
BEMA_DETERMINISTIC = True

ONATSKI_ALPHA = ALPHA
ONATSKI_K_UPPER = 8
DPA_THRESHOLD_SCALE = 1.05

PASS_YAO_SMAX = 20
PASS_YAO_MAX_ITER = 50
PASS_YAO_FALLBACK_C = 4.0
DY_R_STAR = 8

DATASETS: Dict[str, Dict[str, Any]] = {
    "GEUVADIS": {
        "path": DATA_DIR / "GEUVADIS_matrix.npy",
        "expected_label_spikes": 4,
        "r0": 7,
        "fixed_N": 3,
        "proposed_seed": 20260608,
    },
    "1000G_EUR": {
        "path": DATA_DIR / "1000G_EUR_matrix.npy",
        "expected_label_spikes": 4,
        "r0": 7,
        "fixed_N": 25,
        "proposed_seed": 20260608,
    },
}

# Calibration mapping copied from the uploaded comparison notebook:
# shape rule first, then dataset-name fallback, then observed dimensions.
CALIBRATION_KEY_BY_DATASET = {
    "GEUVADIS": {"n": 500, "p": 200},
    "1000G_EUR": {"n": 500, "p": 750},
}

CALIBRATION_KEY_BY_SHAPE = {
    (400, 200): {"n": 500, "p": 200},
    (400, 1500): {"n": 500, "p": 750},
}

DY_R_STAR_BY_DATASET = {
    "GEUVADIS": 8,
    "1000G_EUR": 8,
}

DING_YANG_CALIBRATION_CSV: Optional[Path] = (
    DATA_DIR / "ding_yang_null_calibration.csv"
)
PASS_YAO_CALIBRATION_CSV: Optional[Path] = (
    DATA_DIR / "pass_yao2_calibration.csv"
)

PASS_YAO_ROW_SELECTOR = {
    "GEUVADIS": 0,
    "1000G_EUR": 0,
}

METHOD_ORDER = [
    "Proposed",
    "BA2017",
    "FGZ2022",
    "KML2023",
    "BN2002",
    "DO2019",
    "Onat2009",
    "DY2022",
    "PY2014",
]


# =============================================================================
# Direct input loading
# =============================================================================

def load_analysis_matrix(path: Path) -> Array:
    """Load one n-by-p matrix exactly as stored."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    X = np.asarray(np.load(path), dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2D n-by-p matrix at {path}; got shape {X.shape}."
        )
    if not np.isfinite(X).all():
        raise ValueError(
            f"{path} contains NaN or infinite values. This workflow does "
            "not alter or impute the supplied matrix."
        )
    return X


def load_datasets() -> Dict[str, Dict[str, Any]]:
    objects: Dict[str, Dict[str, Any]] = {}
    for name, config in DATASETS.items():
        X = load_analysis_matrix(Path(config["path"]))
        objects[name] = {
            **config,
            "X": X,
            "n": int(X.shape[0]),
            "p": int(X.shape[1]),
        }
        print(
            f"{name}: loaded {config['path']} exactly as stored; "
            f"shape n x p = {X.shape}"
        )
    return objects


# =============================================================================
# Proposed fixed-N held-out multiplier bootstrap
# =============================================================================

def proposed_bootstrap_edge_test(
    X: Array,
    r0: int,
    N: int,
    B_final: int = B_FINAL,
    alpha: float = ALPHA,
    seed: int = 20260608,
    kmax: int = KMAX,
) -> Dict[str, Any]:
    """Run the fixed-N held-out bootstrap estimator."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")
    if int(B_final) < 3:
        raise ValueError("B_final must be at least 3.")
    if int(r0) < 0:
        raise ValueError("r0 must be nonnegative.")
    if int(N) <= 0:
        raise ValueError("N must be positive.")

    rng = np.random.default_rng(int(seed))
    sample_values = sample_cov_eigenvalues(
        X,
        assume_centered=True,
        sort_desc=True,
    )

    target_count = int(r0) + 1
    target_index = int(r0)
    if len(sample_values) < target_count:
        raise ValueError(
            f"Need at least {target_count} sample eigenvalues; "
            f"only {len(sample_values)} are available."
        )

    bootstrap_targets = np.empty(int(B_final), dtype=np.float64)
    for b in range(int(B_final)):
        weights = draw_chisq_weights(X.shape[0], int(N), rng)
        weighted_values = top_k_weighted_cov_eigs(
            X,
            weights,
            target_count,
        )
        bootstrap_targets[b] = float(weighted_values[target_index])

    bootstrap_train = bootstrap_targets[:-1]
    holdout_lambda = float(bootstrap_targets[-1])

    E_hat = float(sample_values[target_index])
    E0_hat = float(np.mean(bootstrap_train))
    se = float(np.std(bootstrap_train, ddof=1))
    bias_hat = float(E0_hat - E_hat)
    corrected_center = float(holdout_lambda - bias_hat)

    z_alpha = float(norm.ppf(1.0 - float(alpha) / 2.0))
    lower = float(corrected_center - z_alpha * se)
    upper = float(corrected_center + z_alpha * se)

    kmax_effective = min(int(kmax), len(sample_values))
    r_hat = int(np.sum(sample_values[:kmax_effective] > upper))

    return {
        "method": "Proposed",
        "k_hat": r_hat,
        "r_hat": r_hat,
        "r0": int(r0),
        "N": int(N),
        "B_final": int(B_final),
        "alpha": float(alpha),
        "seed": int(seed),
        "E_hat": E_hat,
        "E0_hat": E0_hat,
        "bias_hat": bias_hat,
        "holdout_lambda": holdout_lambda,
        "corrected_center": corrected_center,
        "se": se,
        "lower": lower,
        "upper": upper,
        "sample_eigenvalues": sample_values,
        "bootstrap_targets": bootstrap_targets,
    }


# =============================================================================
# External calibration files and notebook-matching row selection
# =============================================================================

def _calibration_candidates(kind: str) -> list[Path]:
    if kind == "dy":
        patterns = (
            "*ding*yang*calib*.csv",
            "*ding*yang*.csv",
            "*dy*calib*.csv",
            "*dy*null*.csv",
        )
    elif kind == "py":
        patterns = (
            "*pass*yao*calib*.csv",
            "*pass*yao*.csv",
            "*py*calib*.csv",
            "*py*.csv",
        )
    else:
        raise ValueError(f"Unknown calibration kind: {kind}")

    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(DATA_DIR.glob(pattern)))

    unique: list[Path] = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if path.is_file() and resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _resolve_one_calibration(
    explicit_path: Optional[Path],
    kind: str,
    display_name: str,
) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = _calibration_candidates(kind)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No {display_name} calibration CSV was found in {DATA_DIR}. "
            f"Set the explicit calibration path near the top of this file."
        )
    raise RuntimeError(
        f"Multiple {display_name} calibration CSV files were found: "
        f"{[str(path) for path in candidates]}. Set the explicit path near "
        "the top of this file."
    )


def resolve_calibration_files() -> Tuple[Path, Path]:
    """Resolve only external CSV inputs; no project Python file is imported."""
    dy_path = _resolve_one_calibration(
        DING_YANG_CALIBRATION_CSV,
        "dy",
        "Ding-Yang",
    )
    py_path = _resolve_one_calibration(
        PASS_YAO_CALIBRATION_CSV,
        "py",
        "Pass-Yao",
    )
    return dy_path, py_path


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    lookup = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def calibration_key_for_dataset(
    dataset_name: str,
    n: int,
    p: int,
) -> Tuple[int, int]:
    """
    Match the uploaded notebook:
      1. shape mapping;
      2. dataset-name mapping;
      3. observed dimensions.
    """
    shape_key = CALIBRATION_KEY_BY_SHAPE.get((int(n), int(p)))
    if shape_key is not None:
        return int(shape_key["n"]), int(shape_key["p"])

    dataset_key = CALIBRATION_KEY_BY_DATASET.get(dataset_name)
    if dataset_key is not None:
        return int(dataset_key["n"]), int(dataset_key["p"])

    return int(n), int(p)


def select_py_calibration(
    calibration: pd.DataFrame,
    dataset_name: str,
    n: int,
    p: int,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Match the uploaded notebook's Pass-Yao row selection exactly.

    It filters only by the mapped (n,p), then applies
    PASS_YAO_ROW_SELECTOR. It does not filter by alpha.
    """
    cal_n, cal_p = calibration_key_for_dataset(dataset_name, n, p)

    n_col = find_column(calibration, ("n",))
    p_col = find_column(calibration, ("p",))
    if n_col is None or p_col is None:
        raise ValueError(
            "Pass-Yao calibration CSV must contain n and p columns."
        )

    rows = calibration.loc[
        (calibration[n_col].astype(int) == int(cal_n))
        & (calibration[p_col].astype(int) == int(cal_p))
    ].copy()

    if rows.empty:
        raise ValueError(
            f"No Pass-Yao calibration rows match "
            f"(n,p)=({cal_n},{cal_p}) for dataset {dataset_name}."
        )

    selector = PASS_YAO_ROW_SELECTOR.get(dataset_name, 0)
    if isinstance(selector, int):
        if selector < 0 or selector >= len(rows):
            raise IndexError(
                f"PASS_YAO_ROW_SELECTOR[{dataset_name!r}]={selector} "
                f"but only {len(rows)} rows match."
            )
        row = rows.iloc[int(selector)].copy()
    else:
        filtered = rows.copy()
        for column, value in dict(selector).items():
            filtered = filtered.loc[filtered[column] == value]
        if filtered.empty:
            raise ValueError(
                f"Pass-Yao selector {selector} produced no matching rows."
            )
        row = filtered.iloc[0].copy()

    return row, {
        "calibration_n": int(cal_n),
        "calibration_p": int(cal_p),
        "calibration_selection": "notebook mapped dimension and row selector",
        "calibration_row_index": int(row.name)
        if isinstance(row.name, (int, np.integer))
        else str(row.name),
    }





# =============================================================================
# DY2022 calibrated direct-local rule
# =============================================================================

def select_dy_calibration_row_old_notebook(
    calibration: pd.DataFrame,
    dataset_name: str,
    n: int,
    p: int,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Apply the calibrated direct-local Ding-Yang rule.

    1. Map the observed dimension to the configured calibration dimension.
    2. Filter by n and p.
    3. Prefer alpha=0.01 rows when available.
    4. Select rows.iloc[0], without indexing the table by r0.
    """
    cal_n, cal_p = calibration_key_for_dataset(dataset_name, n, p)

    n_col = find_column(calibration, ("n",))
    p_col = find_column(calibration, ("p",))
    alpha_col = find_column(calibration, ("alpha",))

    if n_col is None or p_col is None:
        raise ValueError(
            "Ding-Yang calibration CSV must contain n and p columns."
        )

    rows = calibration.loc[
        (calibration[n_col].astype(int) == int(cal_n))
        & (calibration[p_col].astype(int) == int(cal_p))
    ].copy()

    if rows.empty:
        raise ValueError(
            f"No Ding-Yang calibration rows match dataset={dataset_name}, "
            f"observed (n,p)=({n},{p}), calibration "
            f"(n,p)=({cal_n},{cal_p})."
        )

    alpha_matched = False
    if alpha_col is not None:
        alpha_rows = rows.loc[
            np.isclose(rows[alpha_col].astype(float), float(ALPHA))
        ].copy()
        if not alpha_rows.empty:
            rows = alpha_rows
            alpha_matched = True

    row = rows.iloc[0].copy()
    row["observed_n"] = int(n)
    row["observed_p"] = int(p)
    row["calibration_n_used"] = int(cal_n)
    row["calibration_p_used"] = int(cal_p)

    return row, {
        "calibration_n": int(cal_n),
        "calibration_p": int(cal_p),
        "calibration_selection": (
            "mapped dimension; alpha matched; first row"
            if alpha_matched
            else "mapped dimension; first row"
        ),
        "dy_calibration_row_index": (
            int(row.name)
            if isinstance(row.name, (int, np.integer))
            else str(row.name)
        ),
    }


def ding_yang_local_old_notebook(
    eigenvalues: Array,
    calibration_row: pd.Series,
    default_h: int = 8,
) -> Dict[str, Any]:
    """
     ``ding_yang_direct_from_calibration(...)[\"local\"]`` from
    the selected calibration row.

    A single first-row ``critical_local`` is reused for all leading ratios:

        ratio[r0] = gap[r0] / gap[h],  r0 = 0, ..., h-1.

    The estimate is the number of consecutive leading ratios that exceed
    that one critical value.
    """
    if "critical_local" not in calibration_row.index:
        raise ValueError(
            "Ding-Yang calibration row must contain critical_local."
        )

    critical_local = float(calibration_row["critical_local"])
    h = (
        int(calibration_row["h"])
        if "h" in calibration_row.index
        else int(default_h)
    )

    eigs = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    gaps = eigs[:-1] - eigs[1:]

    if len(gaps) <= h:
        raise ValueError(
            f"Need at least h+1={h + 1} gaps, got {len(gaps)}."
        )

    edge_gap = float(gaps[h])
    if edge_gap <= 0:
        raise ValueError(
            "Nonpositive reference edge gap in Ding-Yang local statistic."
        )

    local_ratios = np.asarray(gaps[:h] / edge_gap, dtype=float)

    k_hat = 0
    for r0, ratio in enumerate(local_ratios):
        if float(ratio) > critical_local:
            k_hat = r0 + 1
        else:
            break

    return {
        "method": "DY2022",
        "variant": "old_notebook_direct_local",
        "k_hat": int(k_hat),
        "critical_local": critical_local,
        "h": int(h),
        "edge_gap": edge_gap,
        "local_ratios": local_ratios,
        "first_local_ratio": (
            float(local_ratios[0]) if len(local_ratios) else np.nan
        ),
        "first_test_reject": bool(
            len(local_ratios)
            and float(local_ratios[0]) > critical_local
        ),
        "gaps": np.asarray(gaps[: h + 1], dtype=float),
        "calibration_row": calibration_row.to_dict(),
    }


# =============================================================================
# Method execution and result tables
# =============================================================================

def extract_k_hat(result: Any) -> int:
    if isinstance(result, dict):
        for key in ("k_hat", "r_hat", "estimated_spikes", "khat"):
            if key in result:
                return int(result[key])
    if isinstance(result, (int, np.integer)):
        return int(result)
    if isinstance(result, (float, np.floating)) and np.isfinite(result):
        return int(result)
    raise ValueError(f"Could not extract an estimated spike count: {result!r}")


def success_row(
    dataset_name: str,
    dataset: Dict[str, Any],
    method: str,
    result: Any,
    elapsed: float,
    note: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    k_hat = extract_k_hat(result)
    row = {
        "dataset": dataset_name,
        "n": int(dataset["n"]),
        "p": int(dataset["p"]),
        "method": method,
        "estimated_spikes": int(k_hat),
        "expected_label_spikes": int(dataset["expected_label_spikes"]),
        "exact_match_expected": int(
            int(k_hat) == int(dataset["expected_label_spikes"])
        ),
        "time_seconds": float(elapsed),
        "note": note,
    }
    if extra:
        row.update(extra)
    return row


def error_row(
    dataset_name: str,
    dataset: Dict[str, Any],
    method: str,
    error: Exception,
) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "n": int(dataset["n"]),
        "p": int(dataset["p"]),
        "method": method,
        "estimated_spikes": np.nan,
        "expected_label_spikes": int(dataset["expected_label_spikes"]),
        "exact_match_expected": np.nan,
        "time_seconds": np.nan,
        "note": f"ERROR: {type(error).__name__}: {error}",
    }


def run_dataset(
    dataset_name: str,
    dataset: Dict[str, Any],
    ding_yang_calibration: pd.DataFrame,
    pass_yao_calibration: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    X = np.asarray(dataset["X"], dtype=np.float64)
    n, p = X.shape
    eigenvalues = sample_cov_eigenvalues(
        X,
        assume_centered=True,
        sort_desc=True,
    )

    rows = []
    diagnostics: Dict[str, Any] = {
        "sample_eigenvalues": eigenvalues[:50].tolist()
    }

    def execute(
        method_name: str,
        function,
        note: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            start = time.perf_counter()
            result = function()
            elapsed = time.perf_counter() - start
            rows.append(
                success_row(
                    dataset_name,
                    dataset,
                    method_name,
                    result,
                    elapsed,
                    note=note,
                    extra=extra,
                )
            )
            diagnostics[method_name] = result
        except Exception as error:
            rows.append(error_row(dataset_name, dataset, method_name, error))

    execute(
        "Proposed",
        lambda: proposed_bootstrap_edge_test(
            X=X,
            r0=int(dataset["r0"]),
            N=int(dataset["fixed_N"]),
            B_final=int(B_FINAL),
            alpha=float(ALPHA),
            seed=int(dataset["proposed_seed"]),
            kmax=int(KMAX),
        ),
        note="fixed-seed held-out multiplier bootstrap",
        extra={
            "random_seed": int(dataset["proposed_seed"]),
            "r0": int(dataset["r0"]),
            "fixed_N": int(dataset["fixed_N"]),
        },
    )

    execute(
        "BA2017",
        lambda: ekc_bva2017(
            X,
            features_axis="columns",
            center=False,
            input_is_correlation=False,
            N=n,
        ),
    )

    execute(
        "FGZ2022",
        lambda: act_adjusted_correlation_thresholding(
            X,
            features_axis="columns",
            center=False,
            rmax=int(KMAX),
        ),
    )

    execute(
        "KML2023",
        lambda: bema_gamma(
            X,
            features_axis="columns",
            beta=float(BEMA_BULK_TRIM_ALPHA),
            M_fit=int(BEMA_M_FIT),
            M_final=int(BEMA_M_FINAL),
            q=float(BEMA_Q),
            center=False,
            scale=False,
            random_state=int(BEMA_RANDOM_STATE),
            kmax=int(KMAX),
            deterministic=bool(BEMA_DETERMINISTIC),
        ),
        note=f"fixed random_state={BEMA_RANDOM_STATE}",
        extra={"random_seed": int(BEMA_RANDOM_STATE)},
    )

    execute(
        "BN2002",
        lambda: bai_ng_icp1(
            X,
            features_axis="columns",
            center=False,
            scale=False,
            kmax=int(KMAX),
        ),
    )

    execute(
        "DO2019",
        lambda: dpa(
            X,
            features_axis="columns",
            center=False,
            scale=False,
            kmax=int(KMAX),
            threshold_scale=float(DPA_THRESHOLD_SCALE),
        ),
    )

    execute(
        "Onat2009",
        lambda: onatski_sequential_fixed_upper(
            X,
            features_axis="columns",
            center=False,
            alpha=float(ONATSKI_ALPHA),
            k_upper=int(ONATSKI_K_UPPER),
        ),
    )

    try:
        dy_row, dy_meta = select_dy_calibration_row_old_notebook(
            ding_yang_calibration,
            dataset_name,
            n,
            p,
        )
        start = time.perf_counter()
        result = ding_yang_local_old_notebook(
            eigenvalues,
            dy_row,
            default_h=int(DY_R_STAR_BY_DATASET.get(dataset_name, 8)),
        )
        elapsed = time.perf_counter() - start
        rows.append(
            success_row(
                dataset_name,
                dataset,
                "DY2022",
                result,
                elapsed,
                note=(
                    "calibrated direct local gap-ratio rule; "
                    f"{dy_meta['calibration_selection']}"
                ),
                extra={
                    **dy_meta,
                    "dy_variant": result["variant"],
                    "dy_h": int(result["h"]),
                    "dy_critical_local": float(
                        result["critical_local"]
                    ),
                    "dy_first_local_ratio": float(
                        result["first_local_ratio"]
                    ),
                    "dy_first_test_reject": bool(
                        result["first_test_reject"]
                    ),
                },
            )
        )
        diagnostics["DY2022"] = result
    except Exception as error:
        rows.append(error_row(dataset_name, dataset, "DY2022", error))

    try:
        py_row, py_meta = select_py_calibration(
            pass_yao_calibration,
            dataset_name,
            n,
            p,
        )
        d_col = find_column(
            pd.DataFrame([py_row]),
            ("d_n", "dn", "threshold", "gap_threshold"),
        )
        if d_col is None:
            raise ValueError(
                "pass_yao2_calibration.csv must contain d_n."
            )
        d_n = float(py_row[d_col])
        start = time.perf_counter()
        result = pass_yao_calibrated_raw_gap_from_row(
            eigenvalues=eigenvalues,
            calibration_row=py_row,
            kmax=int(KMAX),
        )
        elapsed = time.perf_counter() - start
        rows.append(
            success_row(
                dataset_name,
                dataset,
                "PY2014",
                result,
                elapsed,
                note=(
                    "calibrated raw leading-gap rule; "
                    f"{py_meta['calibration_selection']}"
                ),
                extra={
                    **py_meta,
                    "pass_yao_d_n": d_n,
                    "pass_yao_variant": result["variant"],
                    "pass_yao_first_gap": (
                        float(result["gaps"][0])
                        if len(result["gaps"]) > 0
                        else np.nan
                    ),
                },
            )
        )
        diagnostics["PY2014"] = result
    except Exception as error:
        rows.append(error_row(dataset_name, dataset, "PY2014", error))

    return pd.DataFrame(rows), diagnostics


# =============================================================================
# Saving and entry point
# =============================================================================

def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def make_wide_table(results: pd.DataFrame) -> pd.DataFrame:
    ordered = results.copy()
    ordered["method"] = pd.Categorical(
        ordered["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    return (
        ordered
        .sort_values(["method", "dataset"])
        .pivot_table(
            index="method",
            columns="dataset",
            values="estimated_spikes",
            aggfunc="first",
            observed=False,
        )
        .reindex(METHOD_ORDER)
    )


def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dy_path, py_path = resolve_calibration_files()
    datasets = load_datasets()
    dy_calibration = pd.read_csv(dy_path)
    py_calibration = pd.read_csv(py_path)

    # print("Build:", BUILD_ID)
    # print("Implementation:", Path(__file__).resolve())
    # print("Ding-Yang calibration:", dy_path.resolve())
    # print("Pass-Yao calibration:", py_path.resolve())
    # print("Output directory:", OUT_DIR.resolve())

    result_frames = []
    diagnostics: Dict[str, Any] = {}
    eigenvalue_frames = []

    for dataset_name, dataset in datasets.items():
        # print("\n" + "=" * 88)
        print("Running:", dataset_name)
        frame, dataset_diagnostics = run_dataset(
            dataset_name,
            dataset,
            dy_calibration,
            py_calibration,
        )
        result_frames.append(frame)
        diagnostics[dataset_name] = dataset_diagnostics

        eigenvalues = sample_cov_eigenvalues(
            dataset["X"],
            assume_centered=True,
            sort_desc=True,
        )
        eig_frame = pd.DataFrame({
            "dataset": dataset_name,
            "rank": np.arange(1, min(50, len(eigenvalues)) + 1),
            "sample_eigenvalue": eigenvalues[:50],
        })
        eigenvalue_frames.append(eig_frame)
        eig_frame.to_csv(
            OUT_DIR / f"{dataset_name}_top_eigenvalues.csv",
            index=False,
        )

    results = pd.concat(result_frames, ignore_index=True)
    results["method"] = pd.Categorical(
        results["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    results = results.sort_values(["dataset", "method"]).reset_index(drop=True)
    wide = make_wide_table(results)

    results.to_csv(OUT_DIR / "real_data_comparison_long.csv", index=False)
    wide.to_csv(OUT_DIR / "real_data_comparison_wide.csv")
    pd.concat(eigenvalue_frames, ignore_index=True).to_csv(
        OUT_DIR / "top_eigenvalues_all_datasets.csv",
        index=False,
    )

    metadata = {
        "single_project_python_file": str(Path(__file__).resolve()),
        "data_loaded_without_shared_preprocessing": True,
        "method_order": METHOD_ORDER,
        "alpha": ALPHA,
        "B_final": B_FINAL,
        "ding_yang_calibration": dy_path,
        "pass_yao_calibration": py_path,
        "ding_yang_variant": "calibrated direct local",
        "pass_yao_variant": "unknown-sigma calibrated two-gap",
        "datasets": {
            name: {
                "path": dataset["path"],
                "shape": [dataset["n"], dataset["p"]],
                "r0": dataset["r0"],
                "fixed_N": dataset["fixed_N"],
                "proposed_seed": dataset["proposed_seed"],
            }
            for name, dataset in datasets.items()
        },
    }
    with open(OUT_DIR / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(metadata), handle, indent=2)
    with open(
        OUT_DIR / "method_diagnostics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(json_safe(diagnostics), handle, indent=2)

    # print("\nLong comparison:")
    # print(
    #     results[
    #         [
    #             "dataset",
    #             "method",
    #             "estimated_spikes",
    #             "expected_label_spikes",
    #             "exact_match_expected",
    #             "time_seconds",
    #             "note",
    #         ]
    #     ].to_string(index=False)
    # )
    # print("\nWide comparison:")
    # print(wide.to_string())
    # print("\nSaved outputs under:", OUT_DIR.resolve())
    return results, wide


if __name__ == "__main__":
    main()
