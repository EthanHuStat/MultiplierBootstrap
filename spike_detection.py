#!/usr/bin/env python3
"""
All-method spike-detection simulation with random to reproduce Figures 3, 4 in the main manuscript and S2-S7 in the supplementary material.

This script uses demeaned sample covariance matrices.  The population
covariance is

    Sigma = Q diag(lambda) Q.T,

where Q is a random block-Haar orthogonal matrix with block size 25 or 10.
The coordinate partition and spike columns are randomized in every replication,
so spike eigenvectors are not aligned with coordinate axes.

Methods included
----------------
    Proposed bootstrap method
    BEMA-Gamma
    EKC-BvA2017
    Bai-Ng ICp1
    Pass-Yao two-gap with calibrated threshold
    Onatski sequential fixed-upper test
    DPA
    ACT adjusted correlation thresholding (Fan-Guo-Zheng)
    Ding-Yang local gap-ratio sequential tests

Main design
-----------
    entry distributions: Gaussian first, then standardized t_10
    Pass-Yao and Ding-Yang calibrations: Gaussian only
    covariance matrix: demeaned sample covariance, X.T @ X / n
    rotation: random block-Haar, block sizes 25 and 10
    dimensions: active (500,200) and (500,750)
    epsilon grid: 0.1,...,1.0, 1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4.0,4.5,5.0
    tasks:
        easy_uniform_r5_equal_blockhaar:
            uniform bulk [0.75,1.25], r=5 equal spikes theta_c + eps
        hard_twomass_r5_equal_blockhaar:
            two-mass bulk {0.75,1.25}, r=5 equal spikes theta_c + eps

Run example
-----------
    mkdir -p logs
    nohup python3 spike_detection.py         >> logs/spike_detection.log 2>&1 &

Useful overrides
----------------
    N_REPS=50 B_BOOT=500 N_JOBS=10 BEMA_M_FIT=5 BEMA_M_FINAL=50         python3 spike_detection.py

    TASKS=easy_uniform_r5_equal_blockhaar         python3 Spike_detection.py

"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Limit BLAS threading before importing numpy/scipy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy import sparse as sp
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import gamma as gamma_distribution
from scipy.stats import norm


Array = np.ndarray


# ============================================================
# Configuration
# ============================================================

ENTRY_DISTS = ("gaussian", "t10")
CALIBRATION_DIST = "gaussian"
OUTDIR = os.environ.get(
    "OUTDIR",
    "spike_detection_results",
)

TASKS = (
    "easy_uniform_r5_equal_blockhaar",
    "hard_twomass_r5_equal_blockhaar",
)
TASKS_OVERRIDE = os.environ.get("TASKS", "").strip()
if TASKS_OVERRIDE:
    TASKS = tuple(x.strip() for x in TASKS_OVERRIDE.split(",") if x.strip())

# Active quick-test dimensions.  The other two are kept here so they can
# be uncommented for the final run.
DIMENSIONS = (
    (500, 200),
    (500, 750),
)
EPS_GRID_OVERRIDE = os.environ.get("EPSILON_GRID", "").strip()
if EPS_GRID_OVERRIDE:
    EPSILON_GRID = tuple(float(x.strip()) for x in EPS_GRID_OVERRIDE.split(","))
else:
    EPSILON_GRID = (
        tuple(round(float(x), 1) for x in np.arange(0.1, 2.0 + 1e-12, 0.1))
        + (2.5, 3, 3.5, 4.0, 4.5, 5.0)
    )

# Run both block-Haar sizes in the same experiment.
ROTATION_BLOCK_SIZES = (10,)

PRIMARY_N = {
    (500, 200): 4,
    (500, 750): 15,
    (750, 500): 9,
}
R_TRUE_BY_TASK = {
    "easy_uniform_r5_equal_blockhaar": 5,
    "hard_twomass_r5_equal_blockhaar": 5,
}
R0_GRID_BY_TASK = {
    "easy_uniform_r5_equal_blockhaar": (7,),
    "hard_twomass_r5_equal_blockhaar": (7,),
}


K_MAX = int(os.environ.get("K_MAX", "20"))
ALPHA = float(os.environ.get("ALPHA", "0.01"))
N_REPS = int(os.environ.get("N_REPS", "500"))
B_BOOT = int(os.environ.get("B_BOOT", "2000"))
N_JOBS = int(os.environ.get("N_JOBS", "50"))
CHUNKSIZE = int(os.environ.get("CHUNKSIZE", "1"))
MP_START_METHOD = os.environ.get("MP_START_METHOD", "fork")
SAVE_DETAILS = os.environ.get("SAVE_DETAILS", "1") != "0"
REUSE_RESULTS = os.environ.get("REUSE_RESULTS", "0") != "0"

# Comparison-method controls.  RUN_BEMA_GAMMA=1 is slow but included by default
# because the requested run is all methods.  Set RUN_BEMA_GAMMA=0 for a quick run.
RUN_COMPARISON_METHODS = os.environ.get("RUN_COMPARISON_METHODS", "1") != "0"
RUN_ACT = os.environ.get("RUN_ACT", "1") != "0"
RUN_BEMA_GAMMA = os.environ.get("RUN_BEMA_GAMMA", "1") != "0"
RUN_BEMA0 = os.environ.get("RUN_BEMA0", "0") != "0"
BEMA_M_FIT = int(os.environ.get("BEMA_M_FIT", "20"))
BEMA_M_FINAL = int(os.environ.get("BEMA_M_FINAL", "200"))
BEMA_DETERMINISTIC = os.environ.get("BEMA_DETERMINISTIC", "1") != "0"
# BEMA has two different tuning notions.  The bulk trim alpha is a
# fitting-window parameter; keep the paper default 0.20 (middle 60%).
# The tail beta is the size-like upper-tail probability; use 0.01 here.
BEMA_BULK_TRIM_ALPHA = float(os.environ.get("BEMA_BULK_TRIM_ALPHA", "0.20"))
BEMA_TAIL_BETA = float(os.environ.get("BEMA_TAIL_BETA", str(ALPHA)))
BEMA_GAMMA_Q = float(os.environ.get("BEMA_GAMMA_Q", str(1.0 - BEMA_TAIL_BETA)))
# For BEMA0, beta is the upper-tail Tracy-Widom level.  The alpha argument
# in bema0_standard is a bulk-trimming fraction, so keep it separate.
BEMA0_BULK_TRIM = float(os.environ.get("BEMA0_BULK_TRIM", "0.20"))
BEMA0_TW_ALPHA = float(os.environ.get("BEMA0_TW_ALPHA", str(ALPHA)))

PASS_YAO_SMAX = int(os.environ.get("PASS_YAO_SMAX", "20"))
RUN_PASS_YAO_1GAP = os.environ.get("RUN_PASS_YAO_1GAP", "1") != "0"
# One-gap uses the distinct-spike paper's simulation threshold.
# The default constant is C=6, as requested.
PASS_YAO_1GAP_C = float(
    os.environ.get("PASS_YAO_1GAP_C", os.environ.get("PASS_YAO_C", "6.0"))
)
# Two-gap calibrated threshold.  This finite-sample calibration is done
# once at the beginning for each (task,n,p), using the same no-spike bulk
# spectrum as that task.  We take the PASS_YAO2_CALIB_Q quantile of the
# leading scaled null gap and use it as d_n.
PASS_YAO2_CALIBRATE = os.environ.get("PASS_YAO2_CALIBRATE", "1") != "0"
PASS_YAO2_CALIB_REPS = int(os.environ.get("PASS_YAO2_CALIB_REPS", "2000"))
PASS_YAO2_CALIB_Q = float(os.environ.get("PASS_YAO2_CALIB_Q", str(1.0 - ALPHA)))
PASS_YAO2_CALIB_DIST = CALIBRATION_DIST
PASS_YAO2_CALIB_SEED = int(os.environ.get("PASS_YAO2_CALIB_SEED", "20260605"))
PASS_YAO2_REUSE_CALIB = os.environ.get("PASS_YAO2_REUSE_CALIB", "1") != "0"
PASS_YAO2_CALIB_CENTER = os.environ.get("PASS_YAO2_CALIB_CENTER", "1") != "0"
# Fallback only if PASS_YAO2_CALIBRATE=0.
PASS_YAO2_FALLBACK_C = float(os.environ.get("PASS_YAO2_FALLBACK_C", "4.0"))
# Additional uncalibrated 2014/two-gap method using the paper-formula
# threshold d_n = C n^(-2/3) sqrt(2 log log n).  It is included by
# default with C=6, as requested, and can be changed with
# PASS_YAO2_DEFAULT_C.
RUN_PASS_YAO2_DEFAULT = os.environ.get("RUN_PASS_YAO2_DEFAULT", "1") != "0"
PASS_YAO2_DEFAULT_C = float(os.environ.get("PASS_YAO2_DEFAULT_C", "6.0"))
PASS_YAO_MAX_ITER = int(os.environ.get("PASS_YAO_MAX_ITER", "50"))

ONATSKI_ALPHA = float(os.environ.get("ONATSKI_ALPHA", str(ALPHA)))
ONATSKI_K_UPPER = int(os.environ.get("ONATSKI_K_UPPER", "8"))

# Ding-Yang local gap-ratio test.  The null table is calibrated once at
# the beginning of the experiment, saved, and then reused inside all
# replications.  We run only DY-local for comparison by default.
RUN_DY_LOCAL = os.environ.get("RUN_DY_LOCAL", "1") != "0"
RUN_DY_MAX = False  # Max statistic disabled; use DY-local only.
DY_ALPHA = float(os.environ.get("DY_ALPHA", str(ALPHA)))
DY_R_STAR = int(os.environ.get("DY_R_STAR", str(ONATSKI_K_UPPER)))
DY_NULL_REPS = int(os.environ.get("DY_NULL_REPS", "2000"))
DY_NULL_SEED = int(os.environ.get("DY_NULL_SEED", "20260606"))
DY_NULL_CENTER = os.environ.get("DY_NULL_CENTER", "1") != "0"
DY_CALIB_DIST = CALIBRATION_DIST
DY_REUSE_CALIB = os.environ.get("DY_REUSE_CALIB", "1") != "0"

T_DF = 10.0
T_SCALE = np.sqrt((T_DF - 2.0) / T_DF)
BASE_SEED = int(os.environ.get("BASE_SEED", "20260604"))

GLOBAL_EDGE_INFO: Dict[Tuple[str, int, int], Dict[str, Any]] = {}


# ============================================================
# General helpers
# ============================================================

def stable_seed(*parts: Any) -> int:
    text = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


def eps_label(eps: float) -> str:
    return "eps" + str(float(eps)).replace(".", "p")


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


def top_k_sample_cov_eigs(X: Array, k: int) -> Array:
    n, p = X.shape
    if p <= n:
        M = (X.T @ X) / n
    else:
        M = (X @ X.T) / n
    return top_k_cov_eigs_from_matrix(M, k)


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


def _top_svd_rank1(X: Array) -> Tuple[float, Array, Array]:
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return float(s[0]), U[:, 0], Vt[0, :]


# ============================================================
# Edge helpers for data generation and proposed method
# ============================================================

def population_bulk_eigs_by_name(bulk_spectrum: str, p_bulk: int) -> Array:
    p_bulk = int(p_bulk)
    if bulk_spectrum == "uniform_0p75_1p25":
        if p_bulk == 1:
            return np.array([1.0], dtype=float)
        return np.linspace(0.75, 1.25, p_bulk, dtype=float)

    if bulk_spectrum == "two_mass_0p75_1p25":
        n1 = p_bulk // 2
        n2 = p_bulk - n1
        return np.concatenate([
            0.75 * np.ones(n1, dtype=float),
            1.25 * np.ones(n2, dtype=float),
        ])

    raise ValueError(f"Unknown bulk_spectrum: {bulk_spectrum}")


def f_edge(x: float, eigs: Array, phi: float) -> float:
    return float(-1.0 / x + phi * np.mean(1.0 / (x + 1.0 / eigs)))


def df_edge(x: float, eigs: Array, phi: float) -> float:
    return float(
        1.0 / (x * x)
        - phi * np.mean(1.0 / ((x + 1.0 / eigs) ** 2))
    )


def d2f_edge(x: float, eigs: Array, phi: float) -> float:
    return float(
        -2.0 / (x ** 3)
        + 2.0 * phi * np.mean(1.0 / ((x + 1.0 / eigs) ** 3))
    )


def deformed_mp_upper_edge(eigs: Array, phi: float) -> Tuple[float, float, float, float]:
    eigs = np.asarray(eigs, dtype=float)
    sigma_max = float(np.max(eigs))
    left = -1.0 / sigma_max + 1e-10
    right = -1e-10

    try:
        bcrit = brentq(
            lambda x: df_edge(x, eigs, phi),
            left,
            right,
            maxiter=500,
            xtol=1e-13,
            rtol=1e-13,
        )
    except ValueError:
        grid = np.linspace(left, right, 20000)
        vals = np.array([df_edge(x, eigs, phi) for x in grid])
        idx = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        if len(idx) == 0:
            raise RuntimeError(
                f"Could not locate edge root for phi={phi}, sigma_max={sigma_max}."
            )
        i = int(idx[-1])
        bcrit = brentq(
            lambda x: df_edge(x, eigs, phi),
            grid[i],
            grid[i + 1],
            maxiter=500,
            xtol=1e-13,
            rtol=1e-13,
        )

    edge = f_edge(bcrit, eigs, phi)
    theta_threshold = -1.0 / bcrit
    tw_b_coeff = (d2f_edge(bcrit, eigs, phi) / 2.0) ** (1.0 / 3.0)
    return float(edge), float(bcrit), float(theta_threshold), float(tw_b_coeff)


# ============================================================
# Marchenko-Pastur helpers for BEMA and DPA
# ============================================================

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


def mp_pdf_grid(gamma: float, num: int = 20000) -> Tuple[Array, Array]:
    g = float(gamma)
    a = (1.0 - np.sqrt(g)) ** 2
    b = (1.0 + np.sqrt(g)) ** 2
    x = np.linspace(max(a, 1e-12), b, num)
    dens = np.sqrt(np.maximum((b - x) * (x - a), 0.0)) / (2.0 * np.pi * g * x)
    area = np.trapz(dens, x)
    if area > 0:
        dens = dens / area
    return x, dens


def mp_quantile(prob: Union[Array, float], gamma: float) -> Array:
    probs = np.asarray(prob, dtype=np.float64)
    x, dens = mp_pdf_grid(gamma)
    cdf = np.cumsum((dens[:-1] + dens[1:]) * np.diff(x) / 2.0)
    cdf = np.r_[0.0, cdf]
    cdf = cdf / cdf[-1]
    return np.interp(probs, cdf, x)


def tracy_widom1_quantile(prob: float) -> float:
    try:
        from scipy.stats import tracywidom  # type: ignore
        return float(tracywidom.ppf(prob, beta=1))
    except Exception:
        ps = np.array([0.80, 0.85, 0.90, 0.95, 0.975, 0.99])
        qs = np.array([-0.165, 0.103, 0.450, 0.979, 1.454, 2.023])
        return float(np.interp(prob, ps, qs))


# ============================================================
# Comparison methods
# ============================================================

def bema0_standard(
    X: Array,
    features_axis: str = "columns",
    alpha: float = 0.20,
    beta: float = 0.10,
    center: bool = True,
    scale: bool = False,
    kmax: Optional[int] = None,
) -> Dict[str, Any]:
    Xn = as_observations_by_features(X, features_axis, center, scale)
    n, p = Xn.shape
    eigs = sample_cov_eigenvalues(Xn)
    m = len(eigs)
    kmax = _safe_kmax(eigs, kmax)

    lo = max(0, int(np.floor(alpha * m)))
    hi = min(m - 1, int(np.floor((1.0 - alpha) * m)))
    idx = np.arange(lo, hi + 1)
    gamma = p / n

    ranks = idx + 1
    probs = 1.0 - (ranks - 0.5) / m
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    qmp = mp_quantile(probs, gamma)
    sigma2 = float(np.dot(qmp, eigs[idx]) / max(np.dot(qmp, qmp), 1e-12))

    tw = tracy_widom1_quantile(1.0 - beta)
    edge = sigma2 * (
        (1.0 + np.sqrt(gamma)) ** 2
        + tw
        * (n ** (-2.0 / 3.0))
        * (gamma ** (-1.0 / 6.0))
        * ((1.0 + np.sqrt(gamma)) ** (4.0 / 3.0))
    )
    khat = int(np.sum(eigs[:kmax] > edge))
    return {
        "method": "BEMA0",
        "k_hat": khat,
        "edge": float(edge),
        "sigma2_hat": float(sigma2),
        "eigenvalues": eigs,
        "n": int(n),
        "p": int(p),
        "alpha": float(alpha),
        "beta": float(beta),
    }


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



def ding_yang_gap_statistics_from_eigs(
    eigs: Array,
    r0: int,
    r_star: int,
) -> Dict[str, float]:
    """Compute Ding-Yang local and max gap-ratio statistics.

    Eigenvalues can be from Y Y.T or from a scaled covariance matrix; the
    ratios are invariant to multiplying all eigenvalues by the same constant.
    Indices below use Python zero-based indexing:

        local numerator   = lambda_{r0+1} - lambda_{r0+2}
        local denominator = lambda_{r_star+1} - lambda_{r_star+2}

    The max statistic is the Onatski-style maximum adjacent gap ratio over
    indices after r0 up to r_star.
    """
    eigs = np.asarray(eigs, dtype=np.float64)
    r0 = int(r0)
    r_star = int(r_star)
    if r0 < 0 or r0 >= r_star:
        raise ValueError("Require 0 <= r0 < r_star.")
    if len(eigs) < r_star + 2:
        raise ValueError("Need at least r_star + 2 eigenvalues.")

    gaps = eigs[:-1] - eigs[1:]
    tiny = 1e-300

    local_num = float(gaps[r0])
    local_den = float(gaps[r_star])
    local = local_num / max(local_den, tiny)

    ratios = []
    for i in range(r0 + 1, r_star + 1):
        # Paper index i is one-based and r0 < i <= r_star.
        # zero-based gap index is i - 1.
        num = float(gaps[i - 1])
        den = float(gaps[i])
        ratios.append(num / max(den, tiny))
    max_stat = float(np.max(ratios)) if ratios else np.nan

    return {
        "local": float(local),
        "max": float(max_stat),
        "local_num": float(local_num),
        "local_den": float(local_den),
    }


def ding_yang_sequential_gap_test(
    X: Array,
    features_axis: str = "columns",
    center: bool = True,
    r_star: int = 8,
    alpha: float = 0.01,
    statistic: str = "local",
    crit_by_r0: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Sequential Ding-Yang rank estimator using precomputed critical values.

    Tests H0: r = r0 versus H1: r > r0 for r0 = 0, ..., r_star - 1.
    The returned k_hat is the first r0 not rejected.  The intended use here is
    statistic='local'; statistic='max' is implemented for completeness.
    """
    if statistic not in {"local", "max"}:
        raise ValueError("statistic must be 'local' or 'max'.")
    if crit_by_r0 is None:
        raise ValueError(
            "Ding-Yang critical values were not supplied. Run the null "
            "calibration at the beginning of the experiment."
        )

    Xn = as_observations_by_features(
        X,
        features_axis=features_axis,
        center=center,
        scale=False,
    )
    n, p = Xn.shape
    eigs = sample_cov_eigenvalues(Xn)
    if len(eigs) < int(r_star) + 2:
        raise ValueError("Need at least r_star + 2 nonzero sample eigenvalues.")

    tests = []
    for r0 in range(int(r_star)):
        stats = ding_yang_gap_statistics_from_eigs(eigs, r0=r0, r_star=r_star)
        stat_value = float(stats[statistic])
        crit = float(crit_by_r0[int(r0)])
        reject = bool(stat_value > crit)
        tests.append({
            "r0": int(r0),
            "statistic": statistic,
            "stat": stat_value,
            "critical_value": crit,
            "reject": reject,
        })
        if not reject:
            return {
                "method": f"Ding-Yang-{statistic}",
                "k_hat": int(r0),
                "r_star": int(r_star),
                "alpha": float(alpha),
                "statistic": statistic,
                "tests": tests,
                "stop_stat": stat_value,
                "stop_critical_value": crit,
                "num_rejections_before_stop": int(len(tests) - 1),
                "eigenvalues": eigs,
                "n": int(n),
                "p": int(p),
            }

    last = tests[-1]
    return {
        "method": f"Ding-Yang-{statistic}",
        "k_hat": int(r_star),
        "r_star": int(r_star),
        "alpha": float(alpha),
        "statistic": statistic,
        "tests": tests,
        "stop_stat": float(last["stat"]),
        "stop_critical_value": float(last["critical_value"]),
        "num_rejections_before_stop": int(len(tests)),
        "eigenvalues": eigs,
        "n": int(n),
        "p": int(p),
    }


def _dy_null_one_rep_all_h(args: Tuple[int, int, int, bool, int]) -> Tuple[Array, Array]:
    """One Gaussian no-spike Wishart draw for the Ding-Yang calibration."""
    n, p, r_star, center, seed = args
    rng = np.random.default_rng(int(seed))
    X = rng.standard_normal(size=(int(n), int(p)))
    if bool(center):
        X = X - X.mean(axis=0, keepdims=True)
    eigs = sample_cov_eigenvalues(X)
    gaps = eigs[:-1] - eigs[1:]
    tiny = 1e-300
    local_stats = np.empty(int(r_star), dtype=np.float64)
    max_stats = np.empty(int(r_star), dtype=np.float64)
    for h in range(1, int(r_star) + 1):
        # Null G2 for h = r_star - r0:
        # (mu_1 - mu_2) / (mu_{h+1} - mu_{h+2}).
        local_stats[h - 1] = float(gaps[0]) / max(float(gaps[h]), tiny)
        ratios = gaps[:h] / np.maximum(gaps[1:h + 1], tiny)
        max_stats[h - 1] = float(np.max(ratios))
    return local_stats, max_stats

def dpa(
    X: Array,
    features_axis: str = "columns",
    center: bool = True,
    scale: bool = False,
    kmax: Optional[int] = None,
    threshold_scale: float = 1.05,
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


def ddpa(
    X: Array,
    features_axis: str = "columns",
    center: bool = True,
    scale: bool = False,
    kmax: int = 20,
    threshold_scale: float = 1.05,
) -> Dict[str, Any]:
    Xr = as_observations_by_features(X, features_axis, center=center, scale=scale)
    n, p = Xr.shape
    selected_edges = []
    selected_eigs = []

    for _ in range(int(kmax)):
        eigs = sample_cov_eigenvalues(Xr)
        top = float(eigs[0])
        variances = Xr.var(axis=0, ddof=0)
        edge = threshold_scale * mp_upper_edge_from_diag(variances, gamma=p / n)
        selected_edges.append(edge)
        selected_eigs.append(top)
        if not (top > edge):
            break
        s, u, v = _top_svd_rank1(Xr)
        Xr = Xr - s * np.outer(u, v)

    khat = int(sum(np.asarray(selected_eigs) > np.asarray(selected_edges)))
    return {
        "method": "DDPA",
        "k_hat": khat,
        "edges": np.asarray(selected_edges),
        "top_residual_eigenvalues": np.asarray(selected_eigs),
        "n": int(n),
        "p": int(p),
        "threshold_scale": float(threshold_scale),
    }


def _ddpa_plus_alg3_keep(lam: Array, gamma: float) -> Dict[str, Any]:
    """
    Algorithm-3 style DDPA+ criterion based on empirical singular-value
    transforms.  lam are eigenvalues of X'X/n or XX'/n, sorted decreasing.
    """
    lam = np.sort(np.asarray(lam, dtype=float))[::-1]
    r = int(len(lam))
    if r < 3 or lam[0] <= 0:
        return {"keep": False, "reason": "too_few_eigenvalues"}

    lambda1 = float(lam[0])
    tail = lam[1:]
    denom = tail - lambda1
    if np.any(np.abs(denom) < 1e-14):
        denom = np.where(np.abs(denom) < 1e-14, -1e-14, denom)

    m = float(np.mean(1.0 / denom))
    v = float(gamma * m - (1.0 - gamma) / lambda1)
    D = float(lambda1 * m * v)
    ell = float(1.0 / D) if np.isfinite(D) and abs(D) > 1e-14 else np.nan

    mprime = float(np.mean(1.0 / (denom ** 2)))
    vprime = float(gamma * mprime + (1.0 - gamma) / (lambda1 ** 2))
    Dprime = float(m * v + lambda1 * (m * vprime + mprime * v))

    cr2 = float(m / (Dprime * ell)) if np.isfinite(Dprime * ell) and abs(Dprime * ell) > 1e-14 else np.nan
    cl2 = float(v / (Dprime * ell)) if np.isfinite(Dprime * ell) and abs(Dprime * ell) > 1e-14 else np.nan

    # Numerical guard: the theoretical quantities are squared cosines.
    if not (np.isfinite(ell) and np.isfinite(cr2) and np.isfinite(cl2)):
        keep = False
    elif ell <= 0 or cr2 <= 0 or cl2 <= 0:
        keep = False
    else:
        cr2_clip = float(np.clip(cr2, 0.0, 1.0))
        cl2_clip = float(np.clip(cl2, 0.0, 1.0))
        rhs = float(4.0 * (ell ** 2) * cr2_clip * cl2_clip)
        keep = bool(lambda1 < rhs)

    return {
        "keep": bool(keep),
        "lambda1": lambda1,
        "m": m,
        "v": v,
        "D": D,
        "ell": ell,
        "mprime": mprime,
        "vprime": vprime,
        "Dprime": Dprime,
        "cr2": cr2,
        "cl2": cl2,
        "criterion_rhs": float(rhs) if "rhs" in locals() else np.nan,
    }


def ddpa_plus_alg3(
    X: Array,
    features_axis: str = "columns",
    center: bool = True,
    scale: bool = False,
    kmax: int = 20,
) -> Dict[str, Any]:
    Xr = as_observations_by_features(X, features_axis, center=center, scale=scale)
    n, p = Xr.shape
    gamma = float(p) / float(n)
    diagnostics = []

    for _ in range(int(kmax)):
        eigs = sample_cov_eigenvalues(Xr)
        crit = _ddpa_plus_alg3_keep(eigs, gamma=gamma)
        diagnostics.append(crit)
        if not crit.get("keep", False):
            break
        s, u, v = _top_svd_rank1(Xr)
        Xr = Xr - s * np.outer(u, v)

    khat = int(sum(bool(d.get("keep", False)) for d in diagnostics))
    return {
        "method": "DDPA+ Algorithm3",
        "k_hat": khat,
        "diagnostics": diagnostics,
        "n": int(n),
        "p": int(p),
        "gamma": float(gamma),
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


def estimate_all_comparison_methods(
    X: Array,
    features_axis: str = "columns",
    kmax: int = 20,
    center: bool = True,
    scale_covariance_methods: bool = False,
    random_state: int = 0,
    pass_yao2_d_n: Optional[float] = None,
    pass_yao2_C: Optional[float] = None,
    pass_yao2_threshold_source: str = "two_gap_null_gap_calibrated",
    dy_local_crit_by_r0: Optional[Dict[int, float]] = None,
    dy_max_crit_by_r0: Optional[Dict[int, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    if RUN_ACT:
        results["ACT"] = act_adjusted_correlation_thresholding(
            X,
            features_axis=features_axis,
            center=center,
            rmax=kmax,
        )

    if RUN_BEMA_GAMMA:
        results["BEMA"] = bema_gamma(
            X,
            features_axis=features_axis,
            center=center,
            scale=scale_covariance_methods,
            beta=BEMA_BULK_TRIM_ALPHA,
            M_fit=BEMA_M_FIT,
            M_final=BEMA_M_FINAL,
            q=BEMA_GAMMA_Q,
            random_state=random_state,
            kmax=kmax,
            deterministic=BEMA_DETERMINISTIC,
        )

    # if RUN_BEMA0:
    #     results["BEMA0"] = bema0_standard(
    #         X,
    #         features_axis=features_axis,
    #         alpha=BEMA0_BULK_TRIM,
    #         beta=BEMA0_TW_ALPHA,
    #         center=center,
    #         scale=scale_covariance_methods,
    #         kmax=kmax,
    #     )

    results["EKC"] = ekc_bva2017(
        X,
        features_axis=features_axis,
        center=center,
    )
    results["Bai-Ng"] = bai_ng_icp1(
        X,
        features_axis=features_axis,
        center=center,
        scale=scale_covariance_methods,
        kmax=kmax,
    )
    # if RUN_PASS_YAO_1GAP:
    #     results["Pass-Yao-1gap"] = pass_yao_unknown_sigma(
    #         X,
    #         features_axis=features_axis,
    #         center=center,
    #         scale=scale_covariance_methods,
    #         kmax=kmax,
    #         smax=PASS_YAO_SMAX,
    #         C=PASS_YAO_1GAP_C,
    #         max_iter=PASS_YAO_MAX_ITER,
    #         variant="one_gap",
    #         threshold_source="distinct_spike_formula",
    #     )
    # if RUN_PASS_YAO2_DEFAULT:
    #     # Uncalibrated 2014/two-gap threshold: d_n = C n^(-2/3)
    #     # sqrt(2 log log n).  This is kept next to the calibrated version
    #     # for sensitivity checks.
    #     n_obs = X.shape[0] if features_axis == "columns" else X.shape[1]
    #     pass_yao2_default_d_n = pass_yao_two_gap_threshold_from_C(
    #         int(n_obs), PASS_YAO2_DEFAULT_C
    #     )
    #     results["Pass-Yao-2gap-default"] = pass_yao_unknown_sigma(
    #         X,
    #         features_axis=features_axis,
    #         center=center,
    #         scale=scale_covariance_methods,
    #         kmax=kmax,
    #         smax=PASS_YAO_SMAX,
    #         C=PASS_YAO2_DEFAULT_C,
    #         max_iter=PASS_YAO_MAX_ITER,
    #         variant="two_gap",
    #         d_n_override=float(pass_yao2_default_d_n),
    #         threshold_source="two_gap_formula_default",
    #     )

    if pass_yao2_d_n is None:
        n_obs = X.shape[0] if features_axis == "columns" else X.shape[1]
        pass_yao2_d_n = pass_yao_two_gap_threshold_from_C(
            int(n_obs), PASS_YAO2_FALLBACK_C
        )
        pass_yao2_C = PASS_YAO2_FALLBACK_C
        pass_yao2_threshold_source = "two_gap_formula_fallback"
    results["Pass-Yao-2gap"] = pass_yao_unknown_sigma(
        X,
        features_axis=features_axis,
        center=center,
        scale=scale_covariance_methods,
        kmax=kmax,
        smax=PASS_YAO_SMAX,
        C=float(pass_yao2_C) if pass_yao2_C is not None else PASS_YAO2_FALLBACK_C,
        max_iter=PASS_YAO_MAX_ITER,
        variant="two_gap",
        d_n_override=float(pass_yao2_d_n),
        threshold_source=pass_yao2_threshold_source,
    )
    results["Onatski"] = onatski_sequential_fixed_upper(
        X,
        features_axis=features_axis,
        center=center,
        alpha=ONATSKI_ALPHA,
        k_upper=ONATSKI_K_UPPER,
    )
    if RUN_DY_LOCAL:
        results["Ding-Yang-local"] = ding_yang_sequential_gap_test(
            X,
            features_axis=features_axis,
            center=center,
            r_star=DY_R_STAR,
            alpha=DY_ALPHA,
            statistic="local",
            crit_by_r0=dy_local_crit_by_r0,
        )
    results["DPA"] = dpa(
        X,
        features_axis=features_axis,
        center=center,
        scale=scale_covariance_methods,
        kmax=kmax,
    )
    # results["DDPA"] = ddpa(
    #     X,
    #     features_axis=features_axis,
    #     center=center,
    #     scale=scale_covariance_methods,
    #     kmax=kmax,
    #     threshold_scale=1.0,
    # )
    # results["DDPA+"] = ddpa_plus_alg3(
    #     X,
    #     features_axis=features_axis,
    #     center=center,
    #     scale=scale_covariance_methods,
    #     kmax=kmax,
    # )
    return results


def results_to_table(results: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    return {name: int(res["k_hat"]) for name, res in results.items()}


# ============================================================
# Simulation design
# ============================================================

@dataclass
class Block:
    task_name: str
    n: int
    p: int
    epsilon: float
    rotation_block_size: int

    @property
    def true_r(self) -> int:
        return int(R_TRUE_BY_TASK[self.task_name])

    @property
    def r0_grid(self) -> Tuple[int, ...]:
        return tuple(int(x) for x in R0_GRID_BY_TASK[self.task_name])

    @property
    def N(self) -> int:
        return int(PRIMARY_N[(self.n, self.p)])

    @property
    def bulk_spectrum(self) -> str:
        if self.task_name == "easy_uniform_r5_equal_blockhaar":
            return "uniform_0p75_1p25"
        if self.task_name == "hard_twomass_r5_equal_blockhaar":
            return "two_mass_0p75_1p25"
        raise ValueError(f"Unknown task_name: {self.task_name}")

    @property
    def name(self) -> str:
        return (
            f"{self.task_name}_block{self.rotation_block_size}"
            f"_n{self.n}_p{self.p}_{eps_label(self.epsilon)}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "n": int(self.n),
            "p": int(self.p),
            "epsilon": float(self.epsilon),
            "rotation_block_size": int(self.rotation_block_size),
        }


def make_blocks() -> list[Block]:
    blocks: list[Block] = []
    for task_name in TASKS:
        for block_size in ROTATION_BLOCK_SIZES:
            for n, p in DIMENSIONS:
                for eps in EPSILON_GRID:
                    blocks.append(
                        Block(task_name, int(n), int(p), float(eps), int(block_size))
                    )
    return blocks


def make_spikes(task_name: str, theta_c: float, epsilon: float) -> Array:
    eps = float(epsilon)
    if task_name in {
        "easy_uniform_r5_equal_blockhaar",
        "hard_twomass_r5_equal_blockhaar",
    }:
        return np.repeat(float(theta_c) + eps, 5).astype(float)
    raise ValueError(f"Unknown task_name: {task_name}")


def population_eigenvalues(block: Block, theta_c: float) -> Tuple[Array, Array]:
    spikes = make_spikes(block.task_name, theta_c, block.epsilon)
    bulk = population_bulk_eigs_by_name(block.bulk_spectrum, block.p - len(spikes))
    return np.concatenate([spikes, bulk]), spikes


def generate_entries(
    rng: np.random.Generator,
    n: int,
    p: int,
    entry_dist: str,
) -> Array:
    entry_dist = str(entry_dist)
    if entry_dist == "gaussian":
        return rng.standard_normal(size=(n, p))
    if entry_dist in {"t10", "t10_normalized", "student_t10"}:
        return rng.standard_t(df=T_DF, size=(n, p)) * T_SCALE
    raise ValueError(f"Unknown entry distribution: {entry_dist}")



def generate_entries_for_dist(
    rng: np.random.Generator,
    n: int,
    p: int,
    dist: str,
) -> Array:
    """Generate null entries for a requested calibration distribution."""
    dist = str(dist)
    if dist in {"gaussian", "normal", "wishart"}:
        return rng.standard_normal(size=(n, p))
    if dist in {"t10", "t10_normalized", "student_t10"}:
        return rng.standard_t(df=T_DF, size=(n, p)) * T_SCALE
    raise ValueError(f"Unknown calibration distribution: {dist}")


def _pass_yao2_calibration_one_gap(
    args: Tuple[str, int, int, str, str, bool, int],
) -> float:
    """One Monte Carlo draw for the PY two-gap calibration.

    The PY rule is applied to scaled eigenvalues lambda_j / sigma_hat^2, so
    the calibrated null gap is also computed on this scaled-eigenvalue scale.
    The no-spike covariance bulk is matched to the simulation task.
    """
    task_name, n, p, bulk_spectrum, dist, center, seed = args
    rng = np.random.default_rng(int(seed))
    Z = generate_entries_for_dist(rng, int(n), int(p), str(dist))
    bulk_eigs = population_bulk_eigs_by_name(str(bulk_spectrum), int(p))
    X = Z * np.sqrt(bulk_eigs)[None, :]
    if bool(center):
        X = X - X.mean(axis=0, keepdims=True)
    eigs = sample_cov_eigenvalues(X)
    if len(eigs) < 2:
        return np.nan
    # Unknown-sigma PY divides by sigma_hat^2.  Include zero covariance
    # eigenvalues implicitly when p > n by dividing the trace by p.
    sigma2_hat = float(np.sum(eigs) / max(int(p), 1))
    scaled = eigs / max(sigma2_hat, 1e-12)
    return float(scaled[0] - scaled[1])


def _pass_yao_setup_keys() -> Tuple[Tuple[str, int, int], ...]:
    return tuple(
        sorted({(str(task_name), int(n), int(p))
                for task_name in TASKS for n, p in DIMENSIONS})
    )


def _calibration_cache_matches(
    df: pd.DataFrame,
    setup_keys: Tuple[Tuple[str, int, int], ...],
) -> bool:
    required = {
        "task_name", "bulk_spectrum", "calib_dist",
        "calib_reps", "calib_quantile", "calib_center", "n", "p",
        "d_n", "C_calibrated",
    }
    if not required.issubset(df.columns):
        return False
    got = {
        (str(r.task_name), int(r.n), int(r.p))
        for r in df.itertuples(index=False)
    }
    if got != set(setup_keys):
        return False
    checks = [
        (df["calib_dist"].astype(str) == str(CALIBRATION_DIST)).all(),
        (df["calib_reps"].astype(int) == int(PASS_YAO2_CALIB_REPS)).all(),
        np.allclose(df["calib_quantile"].astype(float),
                    float(PASS_YAO2_CALIB_Q)),
        (df["calib_center"].astype(bool) == bool(PASS_YAO2_CALIB_CENTER)).all(),
    ]
    return bool(all(checks))


def _dy_calibration_cache_matches(df: pd.DataFrame) -> bool:
    required = {
        "n", "p", "r_star", "r0", "h", "alpha", "q", "reps",
        "seed", "center", "calib_dist", "critical_local", "critical_max",
    }
    if not required.issubset(set(df.columns)):
        return False
    dims = {(int(n), int(p)) for n, p in DIMENSIONS}
    got_dims = {(int(r.n), int(r.p)) for r in df.itertuples(index=False)}
    if got_dims != dims:
        return False
    checks = [
        (df["calib_dist"].astype(str) == str(DY_CALIB_DIST)).all(),
        (df["r_star"].astype(int) == int(DY_R_STAR)).all(),
        np.allclose(df["alpha"].astype(float), float(DY_ALPHA)),
        np.allclose(df["q"].astype(float), float(1.0 - DY_ALPHA)),
        (df["reps"].astype(int) == int(DY_NULL_REPS)).all(),
        (df["seed"].astype(int) == int(DY_NULL_SEED)).all(),
        (df["center"].astype(bool) == bool(DY_NULL_CENTER)).all(),
    ]
    expected_rows = len(dims) * int(DY_R_STAR)
    return bool(all(checks) and len(df) == expected_rows)


def precompute_ding_yang_calibration(
    logger: logging.Logger,
    outdir: Path,
) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    """Precompute the Ding-Yang null critical-value table once per run.

    The table depends only on (n, p, r_star, r0, alpha) and not on epsilon,
    spike strength, bulk type, or replication index.  We save it for
    reproducibility but avoid printing the table itself.
    """
    if not (RUN_DY_LOCAL or RUN_DY_MAX):
        return {}

    path = outdir / "ding_yang_null_calibration.csv"
    if DY_REUSE_CALIB and path.exists():
        cached = pd.read_csv(path)
        if _dy_calibration_cache_matches(cached):
            logger.info("Ding-Yang Gaussian null calibration finished.")
            out: Dict[Tuple[int, int, int], Dict[str, float]] = {}
            for r in cached.itertuples(index=False):
                out[(int(r.n), int(r.p), int(r.r0))] = {
                    "h": int(r.h),
                    "critical_local": float(r.critical_local),
                    "critical_max": float(r.critical_max),
                }
            return out

    rows = []
    out: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    q = float(1.0 - DY_ALPHA)
    ctx = mp.get_context(MP_START_METHOD)

    for n, p in DIMENSIONS:
        seeds = [
            stable_seed(DY_NULL_SEED, DY_CALIB_DIST, "ding_yang", n, p, b)
            for b in range(int(DY_NULL_REPS))
        ]
        args = [
            (int(n), int(p), int(DY_R_STAR), bool(DY_NULL_CENTER), int(seed))
            for seed in seeds
        ]
        if int(N_JOBS) > 1 and int(DY_NULL_REPS) > 1:
            with ctx.Pool(processes=int(N_JOBS)) as pool:
                vals = list(pool.imap_unordered(
                    _dy_null_one_rep_all_h,
                    args,
                    chunksize=max(1, int(CHUNKSIZE)),
                ))
        else:
            vals = [_dy_null_one_rep_all_h(arg) for arg in args]

        local_mat = np.vstack([v[0] for v in vals])
        max_mat = np.vstack([v[1] for v in vals])
        for r0 in range(int(DY_R_STAR)):
            h = int(DY_R_STAR) - int(r0)
            crit_local = float(np.quantile(local_mat[:, h - 1], q))
            crit_max = float(np.quantile(max_mat[:, h - 1], q))
            rows.append({
                "n": int(n),
                "p": int(p),
                "r_star": int(DY_R_STAR),
                "r0": int(r0),
                "h": int(h),
                "alpha": float(DY_ALPHA),
                "q": q,
                "reps": int(DY_NULL_REPS),
                "seed": int(DY_NULL_SEED),
                "center": bool(DY_NULL_CENTER),
                "calib_dist": str(DY_CALIB_DIST),
                "critical_local": crit_local,
                "mean_local": float(np.mean(local_mat[:, h - 1])),
                "critical_max": crit_max,
                "mean_max": float(np.mean(max_mat[:, h - 1])),
            })
            out[(int(n), int(p), int(r0))] = {
                "h": int(h),
                "critical_local": crit_local,
                "critical_max": crit_max,
            }

    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Ding-Yang Gaussian null calibration finished.")
    return out

def precompute_pass_yao2_calibration(
    logger: logging.Logger,
    outdir: Path,
) -> Dict[Tuple[str, int, int], Dict[str, float]]:
    """
    Calibrate the PY possibly-equal-spike two-gap threshold by no-spike MC.

    For each active (task,n,p), simulate the same no-spike population bulk as
    the task, compute the leading scaled null eigengap

        (lambda_1 / sigma_hat^2) - (lambda_2 / sigma_hat^2),

    take the PASS_YAO2_CALIB_Q quantile, and use that as d_n in the PY
    two-small-gaps rule.  This calibrated variant is a finite-sample benchmark;
    the two formula-based PY variants are still included separately.
    """
    setup_keys = _pass_yao_setup_keys()
    path = outdir / "pass_yao2_calibration.csv"

    if PASS_YAO2_REUSE_CALIB and path.exists():
        cached = pd.read_csv(path)
        if _calibration_cache_matches(cached, setup_keys):
            return {
                (str(r.task_name), int(r.n), int(r.p)): {
                    "d_n": float(r.d_n),
                    "C_calibrated": float(r.C_calibrated),
                    "gap_quantile": float(r.gap_quantile),
                }
                for r in cached.itertuples(index=False)
            }

    rows = []
    if not PASS_YAO2_CALIBRATE:
        logger.info(
            "PASS_YAO2_CALIBRATE=0; using fallback C=%.4f for PY two-gap.",
            float(PASS_YAO2_FALLBACK_C),
        )
        for task_name, n, p in setup_keys:
            dummy = Block(
                str(task_name), int(n), int(p), float(EPSILON_GRID[0]),
                int(ROTATION_BLOCK_SIZES[0]),
            )
            d_n = pass_yao_two_gap_threshold_from_C(n, PASS_YAO2_FALLBACK_C)
            rows.append({
                "entry_dist": CALIBRATION_DIST,
                "task_name": str(task_name),
                "bulk_spectrum": dummy.bulk_spectrum,
                "n": int(n),
                "p": int(p),
                "calib_dist": "none_formula_fallback",
                "calib_reps": 0,
                "calib_quantile": np.nan,
                "calib_center": bool(PASS_YAO2_CALIB_CENTER),
                "gap_quantile": float(d_n),
                "gap_mean": np.nan,
                "gap_median": np.nan,
                "gap_sd": np.nan,
                "d_n": float(d_n),
                "C_calibrated": float(PASS_YAO2_FALLBACK_C),
                "threshold_source": "two_gap_formula_fallback",
            })
    else:
        ctx = mp.get_context(MP_START_METHOD)
        for task_name, n, p in setup_keys:
            dummy = Block(
                str(task_name), int(n), int(p), float(EPSILON_GRID[0]),
                int(ROTATION_BLOCK_SIZES[0]),
            )
            bulk_spectrum = dummy.bulk_spectrum
            seeds = [
                stable_seed(
                    PASS_YAO2_CALIB_SEED, CALIBRATION_DIST,
                    task_name, bulk_spectrum, n, p, b,
                )
                for b in range(int(PASS_YAO2_CALIB_REPS))
            ]
            args = [
                (str(task_name), int(n), int(p), str(bulk_spectrum),
                 str(CALIBRATION_DIST), bool(PASS_YAO2_CALIB_CENTER),
                 int(seed))
                for seed in seeds
            ]
            if int(N_JOBS) > 1 and int(PASS_YAO2_CALIB_REPS) > 1:
                with ctx.Pool(processes=N_JOBS) as pool:
                    gaps = list(pool.imap_unordered(
                        _pass_yao2_calibration_one_gap,
                        args,
                        chunksize=CHUNKSIZE,
                    ))
                gaps = np.asarray(gaps, dtype=np.float64)
            else:
                gaps = np.asarray([
                    _pass_yao2_calibration_one_gap(arg) for arg in args
                ], dtype=np.float64)
            gaps = gaps[np.isfinite(gaps)]
            d_n = float(np.quantile(gaps, PASS_YAO2_CALIB_Q))
            C_cal = pass_yao_two_gap_C_from_threshold(n, d_n)
            rows.append({
                "entry_dist": CALIBRATION_DIST,
                "task_name": str(task_name),
                "bulk_spectrum": str(bulk_spectrum),
                "n": int(n),
                "p": int(p),
                "calib_dist": str(CALIBRATION_DIST),
                "calib_reps": int(len(gaps)),
                "calib_quantile": float(PASS_YAO2_CALIB_Q),
                "calib_center": bool(PASS_YAO2_CALIB_CENTER),
                "gap_quantile": float(d_n),
                "gap_mean": float(np.mean(gaps)),
                "gap_median": float(np.median(gaps)),
                "gap_sd": float(np.std(gaps, ddof=1)),
                "d_n": float(d_n),
                "C_calibrated": float(C_cal),
                "threshold_source": "two_gap_null_gap_calibrated",
            })
            # Only log the threshold, not the full calibration table.
            logger.info(
                "Pass-Yao-2gap Gaussian calibration | task=%s | n=%d p=%d | "
                "d_n=%.8f | C=%.4f | q=%.3f",
                str(task_name), int(n), int(p), float(d_n), float(C_cal),
                float(PASS_YAO2_CALIB_Q),
            )

    df = pd.DataFrame(rows).sort_values(
        ["task_name", "n", "p"]
    ).reset_index(drop=True)
    df.to_csv(path, index=False)
    return {
        (str(r.task_name), int(r.n), int(r.p)): {
            "d_n": float(r.d_n),
            "C_calibrated": float(r.C_calibrated),
            "gap_quantile": float(r.gap_quantile),
        }
        for r in df.itertuples(index=False)
    }



def haar_orthogonal_block(m: int, rng: np.random.Generator) -> Array:
    """Generate one dense m x m Haar orthogonal block."""
    Z = rng.standard_normal(size=(int(m), int(m)))
    Q, R = np.linalg.qr(Z)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs


def random_block_haar_sparse(
    p: int,
    r: int,
    block_size: int,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    """
    Random block-Haar orthogonal matrix as a sparse CSR matrix.

    Blocks are independent Haar matrices.  Rows are randomly permuted so the
    block supports are random coordinate subsets.  Columns are permuted so the
    first r columns, used as spike eigenvectors, are drawn across random block
    columns when possible.
    """
    p = int(p)
    r = int(r)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if r > p:
        raise ValueError("r cannot exceed p.")

    blocks: list[np.ndarray] = []
    block_cols: list[np.ndarray] = []
    start = 0
    while start < p:
        end = min(start + block_size, p)
        blocks.append(haar_orthogonal_block(end - start, rng))
        block_cols.append(np.arange(start, end, dtype=int))
        start = end

    Q = sp.block_diag(blocks, format="csr")

    # Randomize coordinate supports.
    row_perm = rng.permutation(p)
    Q = Q[row_perm, :]

    # Choose spike columns across blocks, then put them first.
    chosen: list[int] = []
    block_order = list(rng.permutation(len(block_cols)))
    while len(chosen) < r:
        progressed = False
        for bidx in block_order:
            candidates = [int(c) for c in block_cols[bidx] if int(c) not in chosen]
            if candidates:
                chosen.append(int(rng.choice(candidates)))
                progressed = True
                if len(chosen) == r:
                    break
        if not progressed:
            break

    remaining = [j for j in range(p) if j not in set(chosen)]
    col_perm = np.asarray(chosen + remaining, dtype=int)
    Q = Q[:, col_perm].tocsr()
    return Q


def generate_data(
    block: Block,
    theta_c: float,
    rng: np.random.Generator,
    entry_dist: str,
) -> Tuple[Array, Array, Array]:
    pop_eigs, spikes = population_eigenvalues(block, theta_c)
    Z = generate_entries(rng, block.n, block.p, entry_dist)

    # Work in the eigenbasis first, then apply the sparse block-Haar rotation.
    # This gives Cov(X_i) = Q diag(pop_eigs) Q.T while avoiding a dense p x p
    # multiplication.  The matrix is centered later before forming S.
    Y = Z * np.sqrt(pop_eigs)[None, :]
    Q = random_block_haar_sparse(
        p=block.p,
        r=block.true_r,
        block_size=block.rotation_block_size,
        rng=rng,
    )
    X = (Q @ Y.T).T
    return np.asarray(X, dtype=np.float64), pop_eigs, spikes


def draw_chisq_weights(n: int, N: int, rng: np.random.Generator) -> Array:
    return rng.chisquare(df=N, size=n) / N


# ============================================================
# Proposed method
# ============================================================

def proposed_detection_from_sample_eigs(
    X: Array,
    sample_vals: Array,
    r0_grid: Tuple[int, ...],
    N: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[int, Dict[str, Any]]:
    max_r0 = int(max(r0_grid))
    max_proxy_index = max_r0 + 1
    if len(sample_vals) < max_proxy_index:
        raise ValueError(f"Need at least {max_proxy_index} sample eigenvalues.")

    boot_edge_proxy = np.empty((B_BOOT, max_proxy_index), dtype=float)
    for b in range(B_BOOT):
        w = draw_chisq_weights(X.shape[0], N, rng)
        boot_edge_proxy[b, :] = top_k_weighted_cov_eigs(X, w, max_proxy_index)

    out: Dict[int, Dict[str, Any]] = {}
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    for r0 in r0_grid:
        ell = int(r0 + 1)
        idx = ell - 1
        boot_targets = boot_edge_proxy[:, idx]
        boot_train = boot_targets[:-1]
        boot_holdout = float(boot_targets[-1])

        E_hat = float(sample_vals[idx])
        E0_hat = float(np.mean(boot_train))
        boot_sd_train = float(np.std(boot_train, ddof=1))
        v0_hat = float(X.shape[0] * boot_sd_train ** 2)
        bias_hat = float(E0_hat - E_hat)
        center_corrected = float(boot_holdout - bias_hat)
        upper_threshold = float(center_corrected + z_alpha * boot_sd_train)
        r_hat = int(np.sum(sample_vals[:K_MAX] > upper_threshold))

        out[int(r0)] = {
            "r_hat": int(r_hat),
            "edge_proxy_index": int(ell),
            "E_hat": E_hat,
            "E0_hat": E0_hat,
            "bias_hat": bias_hat,
            "boot_holdout": boot_holdout,
            "boot_sd_train": boot_sd_train,
            "v0_hat": v0_hat,
            "center_corrected": center_corrected,
            "upper_threshold": upper_threshold,
        }
    return out


# ============================================================
# Logging and setup
# ============================================================

def setup_logger(outdir: Path) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("all_methods_gaussian_t10_blockhaar")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(outdir / "run.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def precompute_edge_info(logger: logging.Logger, outdir: Path) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    rows = []
    info: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    logger.info("Precomputing edge information for each (task,n,p).")

    for task_name in TASKS:
        for n, p in DIMENSIONS:
            dummy = Block(task_name, int(n), int(p), EPSILON_GRID[0], int(ROTATION_BLOCK_SIZES[0]))
            bulk_spectrum = dummy.bulk_spectrum
            phi = float(p / n)
            ref_eigs = population_bulk_eigs_by_name(bulk_spectrum, p)
            edge, bcrit, theta_c, tw_b_coeff = deformed_mp_upper_edge(ref_eigs, phi)
            key = (task_name, int(n), int(p))
            info[key] = {
                "edge": edge,
                "critical_b": bcrit,
                "theta_threshold": theta_c,
                "tw_b_coeff": tw_b_coeff,
                "phi_edge": phi,
                "p_edge_reference": int(p),
                "bulk_spectrum": bulk_spectrum,
            }
            rows.append({
                "entry_dist": "distribution_independent",
                "task_name": task_name,
                "n": int(n),
                "p": int(p),
                "bulk_spectrum": bulk_spectrum,
                "phi_edge": phi,
                "p_edge_reference": int(p),
                "critical_b": bcrit,
                "edge": edge,
                "theta_threshold": theta_c,
                "tw_b_coeff": tw_b_coeff,
            })
            logger.info(
                "Edge setup | task=%s | n=%d | p=%d | bulk=%s | phi=%.6f | "
                "edge=%.10f | theta_c=%.10f | b=%.10f",
                task_name, int(n), int(p), bulk_spectrum, phi, edge, theta_c, bcrit,
            )

    pd.DataFrame(rows).to_csv(outdir / "edge_info.csv", index=False)
    return info


# ============================================================
# One replication and block runner
# ============================================================

def run_one_rep(
    task: Tuple[Dict[str, Any], int, Dict[str, Any], str],
) -> Dict[str, Any]:
    block_dict, rep, edge_info, entry_dist = task
    block = Block(**block_dict)
    rep = int(rep)
    entry_dist = str(entry_dist)

    rng = np.random.default_rng(
        stable_seed(BASE_SEED, entry_dist, block.name, rep)
    )
    theta_c = float(edge_info["theta_threshold"])
    edge = float(edge_info["edge"])
    X, pop_eigs, spikes = generate_data(block, theta_c, rng, entry_dist)

    # Center once for covariance-scale methods and proposed method.
    Xc = X - X.mean(axis=0, keepdims=True)
    sample_vals = top_k_sample_cov_eigs(Xc, K_MAX)

    row: Dict[str, Any] = {
        "entry_dist": entry_dist,
        "task_name": block.task_name,
        "n": int(block.n),
        "p": int(block.p),
        "rotation_block_size": int(block.rotation_block_size),
        "bulk_spectrum": block.bulk_spectrum,
        "epsilon": float(block.epsilon),
        "N": int(block.N),
        "rep": int(rep),
        "r_true": int(block.true_r),
        "edge": edge,
        "theta_c": theta_c,
        "critical_b": float(edge_info["critical_b"]),
        "spike_1": float(spikes[0]) if len(spikes) >= 1 else np.nan,
        "spike_2": float(spikes[1]) if len(spikes) >= 2 else np.nan,
        "spike_3": float(spikes[2]) if len(spikes) >= 3 else np.nan,
        "spike_4": float(spikes[3]) if len(spikes) >= 4 else np.nan,
        "spike_5": float(spikes[4]) if len(spikes) >= 5 else np.nan,
        "lambda1_hat": float(sample_vals[0]) if len(sample_vals) >= 1 else np.nan,
        "lambda2_hat": float(sample_vals[1]) if len(sample_vals) >= 2 else np.nan,
        "lambda3_hat": float(sample_vals[2]) if len(sample_vals) >= 3 else np.nan,
        "lambda4_hat": float(sample_vals[3]) if len(sample_vals) >= 4 else np.nan,
        "lambda5_hat": float(sample_vals[4]) if len(sample_vals) >= 5 else np.nan,
    }

    proposed = proposed_detection_from_sample_eigs(
        X=Xc,
        sample_vals=sample_vals,
        r0_grid=block.r0_grid,
        N=block.N,
        alpha=ALPHA,
        rng=rng,
    )
    for r0, res in proposed.items():
        prefix = f"proposed_r0_{int(r0)}"
        row[prefix + "_rhat"] = int(res["r_hat"])
        row[prefix + "_correct"] = float(res["r_hat"] == block.true_r)
        row[prefix + "_under_detect"] = float(res["r_hat"] < block.true_r)
        row[prefix + "_over_detect"] = float(res["r_hat"] > block.true_r)
        for key in (
            "edge_proxy_index", "E_hat", "E0_hat", "bias_hat", "boot_holdout",
            "boot_sd_train", "v0_hat", "center_corrected", "upper_threshold",
        ):
            row[prefix + "_" + key] = res[key]

    if RUN_COMPARISON_METHODS:
        results = estimate_all_comparison_methods(
            Xc,
            features_axis="columns",
            kmax=K_MAX,
            center=False,
            scale_covariance_methods=False,
            random_state=stable_seed(BASE_SEED, "methods", entry_dist, block.name, rep),
            pass_yao2_d_n=edge_info.get("pass_yao2_d_n"),
            pass_yao2_C=edge_info.get("pass_yao2_C_calibrated"),
            pass_yao2_threshold_source=edge_info.get(
                "pass_yao2_threshold_source", "two_gap_null_gap_calibrated"
            ),
            dy_local_crit_by_r0=edge_info.get("dy_local_crit_by_r0"),
            dy_max_crit_by_r0=edge_info.get("dy_max_crit_by_r0"),
        )
        table = results_to_table(results)
        name_map = {
            "ACT": "ACT",
            "BEMA": "BEMA",
            "BEMA0": "BEMA0",
            "EKC": "EKC",
            "Bai-Ng": "Bai_Ng",
            "Pass-Yao-1gap": "Pass_Yao_1gap",
            "Pass-Yao-2gap": "Pass_Yao_2gap",
            "Pass-Yao-2gap-default": "Pass_Yao_2gap_default",
            "Onatski": "Onatski",
            "Ding-Yang-local": "Ding_Yang_local",
            "Ding-Yang-max": "Ding_Yang_max",
            "DPA": "DPA",
            "DDPA": "DDPA",
            "DDPA+": "DDPA_plus",
        }
        for name, khat in table.items():
            prefix = name_map.get(name, name.replace("-", "_").replace("+", "plus"))
            row[prefix + "_rhat"] = int(khat)
            row[prefix + "_correct"] = float(int(khat) == block.true_r)
            row[prefix + "_under_detect"] = float(int(khat) < block.true_r)
            row[prefix + "_over_detect"] = float(int(khat) > block.true_r)

        # Save useful diagnostics for methods with tuning/test statistics.
        for key, prefix in (
            ("Pass-Yao-1gap", "Pass_Yao_1gap"),
            ("Pass-Yao-2gap", "Pass_Yao_2gap"),
            ("Pass-Yao-2gap-default", "Pass_Yao_2gap_default"),
        ):
            if key in results:
                row[prefix + "_sigma2_hat"] = float(results[key]["sigma2_hat"])
                row[prefix + "_d_n"] = float(results[key]["d_n"])
                row[prefix + "_C"] = float(results[key].get("C", np.nan))
                row[prefix + "_threshold_source"] = str(
                    results[key].get("threshold_source", "")
                )
                row[prefix + "_converged"] = bool(results[key]["converged"])
        if "ACT" in results:
            row["ACT_threshold"] = float(results["ACT"].get("threshold", np.nan))
            corr_eigs = results["ACT"].get("corrected_eigs", [])
            for j0, val in enumerate(np.asarray(corr_eigs)[:10], start=1):
                row[f"ACT_corrected_lambda_{j0}"] = float(val)

        if "Onatski" in results:
            row["Onatski_stop_R"] = float(results["Onatski"].get("stop_R", np.nan))
            row["Onatski_stop_critical_value"] = float(
                results["Onatski"].get("stop_critical_value", np.nan)
            )
            row["Onatski_num_rejections_before_stop"] = int(
                results["Onatski"].get("num_rejections_before_stop", 0)
            )

        if "Ding-Yang-local" in results:
            row["Ding_Yang_local_stop_stat"] = float(
                results["Ding-Yang-local"].get("stop_stat", np.nan)
            )
            row["Ding_Yang_local_stop_critical_value"] = float(
                results["Ding-Yang-local"].get("stop_critical_value", np.nan)
            )
            row["Ding_Yang_local_num_rejections_before_stop"] = int(
                results["Ding-Yang-local"].get("num_rejections_before_stop", 0)
            )

        if "Ding-Yang-max" in results:
            row["Ding_Yang_max_stop_stat"] = float(
                results["Ding-Yang-max"].get("stop_stat", np.nan)
            )
            row["Ding_Yang_max_stop_critical_value"] = float(
                results["Ding-Yang-max"].get("stop_critical_value", np.nan)
            )
            row["Ding_Yang_max_num_rejections_before_stop"] = int(
                results["Ding-Yang-max"].get("num_rejections_before_stop", 0)
            )

    return row


def summarize_details(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["entry_dist", "task_name", "rotation_block_size", "n", "p", "epsilon"]
    for _, sub in details.groupby(group_cols):
        row: Dict[str, Any] = {
            "entry_dist": str(sub["entry_dist"].iloc[0]),
            "task_name": str(sub["task_name"].iloc[0]),
            "rotation_block_size": int(sub["rotation_block_size"].iloc[0]),
            "n": int(sub["n"].iloc[0]),
            "p": int(sub["p"].iloc[0]),
            "bulk_spectrum": str(sub["bulk_spectrum"].iloc[0]),
            "epsilon": float(sub["epsilon"].iloc[0]),
            "N": int(sub["N"].iloc[0]),
            "r_true": int(sub["r_true"].iloc[0]),
            "edge": float(sub["edge"].iloc[0]),
            "theta_c": float(sub["theta_c"].iloc[0]),
            "spike_1": float(sub["spike_1"].iloc[0]),
            "spike_2": float(sub["spike_2"].iloc[0]),
            "spike_3": float(sub["spike_3"].iloc[0]),
            "n_reps": int(len(sub)),
            "B": int(B_BOOT),
            "alpha": float(ALPHA),
        }
        for col in sorted(c for c in sub.columns if c.endswith("_correct")):
            row[col.replace("_correct", "_accuracy")] = float(sub[col].mean())
        for col in sorted(c for c in sub.columns if c.endswith("_rhat")):
            row[col.replace("_rhat", "_mean_rhat")] = float(sub[col].mean())
        proposed_threshold_cols = sorted(
            c for c in sub.columns
            if c.startswith("proposed_r0_") and c.endswith("_upper_threshold")
        )
        for c in proposed_threshold_cols:
            suffix = "_upper_threshold"
            prefix = c[:-len(suffix)]
            row[prefix + "_mean_upper_threshold"] = float(sub[c].mean())
            row[prefix + "_sd_upper_threshold"] = float(sub[c].std(ddof=1))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["entry_dist", "task_name", "rotation_block_size", "n", "p", "epsilon"])
        .reset_index(drop=True)
    )


def make_accuracy_long(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    id_cols = [
        "entry_dist", "task_name", "rotation_block_size", "n", "p",
        "bulk_spectrum", "epsilon", "r_true", "N", "n_reps",
        "B", "alpha",
    ]
    acc_cols = [c for c in summary.columns if c.endswith("_accuracy")]
    long = summary[id_cols + acc_cols].melt(
        id_vars=id_cols,
        value_vars=acc_cols,
        var_name="method",
        value_name="accuracy",
    )
    long["method"] = long["method"].str.replace("_accuracy", "", regex=False)
    return long


def log_setup_detection_rates(
    logger: logging.Logger,
    block: Block,
    summary: pd.DataFrame,
    entry_dist: str,
) -> None:
    if summary.empty:
        logger.info("[%s] empty summary", block.name)
        return
    s = summary.iloc[0]
    parts = []
    for col in [c for c in summary.columns if c.endswith("_accuracy")]:
        if col not in s.index or pd.isna(s[col]):
            continue

        label = col.replace("_accuracy", "")
        mean_rhat_col = label + "_mean_rhat"
        accuracy = float(s[col])

        if mean_rhat_col in s.index and pd.notna(s[mean_rhat_col]):
            mean_rhat = float(s[mean_rhat_col])
            parts.append(
                f"{label}: accuracy={accuracy:.4f}, mean_rhat={mean_rhat:.4f}"
            )
        else:
            parts.append(f"{label}: accuracy={accuracy:.4f}")

    logger.info(
        "[%s | %s | block=%d | n=%d p=%d | eps=%.1f] %s",
        str(entry_dist),
        block.task_name,
        int(block.rotation_block_size),
        int(block.n),
        int(block.p),
        float(block.epsilon),
        " | ".join(parts),
    )


def read_completed_setup(block_dir: Path) -> bool:
    return REUSE_RESULTS and (block_dir / "summary.csv").exists()


def run_block(
    block: Block,
    logger: logging.Logger,
    outdir: Path,
    entry_dist: str,
    edge_info_map: Dict[Tuple[str, int, int], Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    block_dir = outdir / block.name
    block_dir.mkdir(parents=True, exist_ok=True)
    summary_path = block_dir / "summary.csv"
    details_path = block_dir / "details.csv"

    edge_info = edge_info_map[(block.task_name, block.n, block.p)]
    theta_c = float(edge_info["theta_threshold"])
    spikes = make_spikes(block.task_name, theta_c, block.epsilon)

    block_info = pd.DataFrame([{
        "block_name": block.name,
        "entry_dist": entry_dist,
        "task_name": block.task_name,
        "n": int(block.n),
        "p": int(block.p),
        "rotation_block_size": int(block.rotation_block_size),
        "bulk_spectrum": block.bulk_spectrum,
        "epsilon": float(block.epsilon),
        "r_true": int(block.true_r),
        "spike_1": float(spikes[0]) if len(spikes) >= 1 else np.nan,
        "spike_2": float(spikes[1]) if len(spikes) >= 2 else np.nan,
        "spike_3": float(spikes[2]) if len(spikes) >= 3 else np.nan,
        "theta_c": theta_c,
        "edge": float(edge_info["edge"]),
        "critical_b": float(edge_info["critical_b"]),
        "N": int(block.N),
        "r0_grid": ",".join(str(x) for x in block.r0_grid),
        "n_reps": int(N_REPS),
        "B": int(B_BOOT),
        "alpha": float(ALPHA),
        "K_MAX": int(K_MAX),
        "Pass_Yao_1gap_C": float(PASS_YAO_1GAP_C),
        "Pass_Yao_2gap_d_n": float(edge_info.get("pass_yao2_d_n", np.nan)),
        "Pass_Yao_2gap_C_calibrated": float(
            edge_info.get("pass_yao2_C_calibrated", np.nan)
        ),
        "Pass_Yao_2gap_threshold_source": str(
            edge_info.get("pass_yao2_threshold_source", "")
        ),
        "Pass_Yao_2gap_calibration_dist": str(CALIBRATION_DIST),
        "Pass_Yao_2gap_default_C": float(PASS_YAO2_DEFAULT_C),
        "Pass_Yao_2gap_default_d_n": float(
            pass_yao_two_gap_threshold_from_C(block.n, PASS_YAO2_DEFAULT_C)
        ),
        "Pass_Yao_2gap_default_threshold_source": "two_gap_formula_default",
        "RUN_PASS_YAO2_DEFAULT": bool(RUN_PASS_YAO2_DEFAULT),
        "ONATSKI_ALPHA": float(ONATSKI_ALPHA),
        "ONATSKI_K_UPPER": int(ONATSKI_K_UPPER),
        "DY_ALPHA": float(DY_ALPHA),
        "DY_R_STAR": int(DY_R_STAR),
        "DY_NULL_REPS": int(DY_NULL_REPS),
        "DY_calibration_dist": str(DY_CALIB_DIST),
        "RUN_DY_LOCAL": bool(RUN_DY_LOCAL),
        "RUN_DY_MAX": bool(RUN_DY_MAX),
        "RUN_BEMA_GAMMA": bool(RUN_BEMA_GAMMA),
        "RUN_ACT": bool(RUN_ACT),
        "RUN_BEMA0": bool(RUN_BEMA0),
        "comparison_methods": "Proposed,ACT,BEMA,BEMA0,EKC,Bai-Ng,"
            "Pass-Yao-1gap,Pass-Yao-2gap-calibrated,"
            "Pass-Yao-2gap-default-C6,Onatski,Ding-Yang-local,"
            "DPA,DDPA,DDPA+",
    }])
    block_info.to_csv(block_dir / "block_info.csv", index=False)

    if read_completed_setup(block_dir):
        summary = pd.read_csv(summary_path)
        log_setup_detection_rates(logger, block, summary, entry_dist)
        details = pd.read_csv(details_path) if details_path.exists() else pd.DataFrame()
        return details, summary


    tasks = [
        (block.to_dict(), int(rep), edge_info, str(entry_dist))
        for rep in range(1, N_REPS + 1)
    ]
    rows = []
    ctx = mp.get_context(MP_START_METHOD)
    with ctx.Pool(processes=N_JOBS) as pool:
        for row in pool.imap_unordered(run_one_rep, tasks, chunksize=CHUNKSIZE):
            rows.append(row)

    details = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)
    summary = summarize_details(details)
    if SAVE_DETAILS:
        details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    log_setup_detection_rates(logger, block, summary, entry_dist)
    return details, summary


# ============================================================
# Main
# ============================================================

def attach_calibrations_to_edge_info(
    edge_info_map: Dict[Tuple[str, int, int], Dict[str, Any]],
    pass_yao2_calib: Dict[Tuple[str, int, int], Dict[str, float]],
    dy_calib: Dict[Tuple[int, int, int], Dict[str, float]],
) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    """Attach the Gaussian calibration tables to every experiment setup."""
    for key, info in edge_info_map.items():
        task_key, n_key, p_key = key
        cal = pass_yao2_calib[(str(task_key), int(n_key), int(p_key))]
        info["pass_yao2_d_n"] = float(cal["d_n"])
        info["pass_yao2_C_calibrated"] = float(cal["C_calibrated"])
        info["pass_yao2_threshold_source"] = (
            "two_gap_null_gap_calibrated_gaussian"
            if PASS_YAO2_CALIBRATE
            else "two_gap_formula_fallback"
        )
        info["pass_yao2_calibration_dist"] = str(CALIBRATION_DIST)
        info["dy_calibration_dist"] = str(DY_CALIB_DIST)

        if dy_calib:
            info["dy_local_crit_by_r0"] = {
                int(r0): float(v["critical_local"])
                for (nn, pp, r0), v in dy_calib.items()
                if int(nn) == int(n_key) and int(pp) == int(p_key)
            }
            info["dy_max_crit_by_r0"] = {
                int(r0): float(v["critical_max"])
                for (nn, pp, r0), v in dy_calib.items()
                if int(nn) == int(n_key) and int(pp) == int(p_key)
            }
    return edge_info_map


def save_combined_outputs(
    outdir: Path,
    all_details: list[pd.DataFrame],
    all_summaries: list[pd.DataFrame],
    partial: bool,
) -> None:
    suffix = "_partial" if partial else ""

    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_summary.to_csv(
            outdir / f"combined_summary{suffix}.csv",
            index=False,
        )
        accuracy_long = make_accuracy_long(combined_summary)
        accuracy_long.to_csv(
            outdir / f"accuracy_long{suffix}.csv",
            index=False,
        )
        if not accuracy_long.empty:
            pivot = accuracy_long.pivot_table(
                index=[
                    "entry_dist", "task_name", "rotation_block_size",
                    "n", "p", "epsilon",
                ],
                columns="method",
                values="accuracy",
            ).reset_index()
            pivot.to_csv(
                outdir / f"accuracy_pivot{suffix}.csv",
                index=False,
            )

    if SAVE_DETAILS and all_details:
        pd.concat(all_details, ignore_index=True).to_csv(
            outdir / f"combined_details{suffix}.csv",
            index=False,
        )


def run_for_entry_dist(
    entry_dist: str,
    logger: logging.Logger,
    root_outdir: Path,
    edge_info_map: Dict[Tuple[str, int, int], Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all setups for one distribution in its own output directory."""
    entry_dist = str(entry_dist)
    dist_outdir = root_outdir / entry_dist
    dist_outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Entry distribution started: %s", entry_dist)
    logger.info("Output subdirectory: %s", dist_outdir)

    all_details: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    blocks = make_blocks()
    logger.info(
        "Number of setups for %s: %d",
        entry_dist,
        len(blocks),
    )

    for block in blocks:
        details, summary = run_block(
            block,
            logger,
            dist_outdir,
            entry_dist,
            edge_info_map,
        )
        if not details.empty:
            all_details.append(details)
        if not summary.empty:
            all_summaries.append(summary)

        save_combined_outputs(
            dist_outdir,
            all_details,
            all_summaries,
            partial=True,
        )

    save_combined_outputs(
        dist_outdir,
        all_details,
        all_summaries,
        partial=False,
    )

    combined_details = (
        pd.concat(all_details, ignore_index=True)
        if all_details else pd.DataFrame()
    )
    combined_summary = (
        pd.concat(all_summaries, ignore_index=True)
        if all_summaries else pd.DataFrame()
    )

    logger.info("Entry distribution finished: %s", entry_dist)
    return combined_details, combined_summary


def main() -> None:
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir)

    logger.info("Run started")
    logger.info("ENTRY_DISTS = %s", ENTRY_DISTS)
    logger.info("Run order: gaussian first, then standardized t10.")
    logger.info(
        "Pass-Yao and Ding-Yang calibration distribution = %s only",
        CALIBRATION_DIST,
    )
    logger.info("OUTDIR = %s", OUTDIR)
    logger.info("TASKS = %s", TASKS)
    logger.info("DIMENSIONS = %s", DIMENSIONS)
    logger.info("EPSILON_GRID = %s", EPSILON_GRID)
    logger.info("PRIMARY_N = %s", PRIMARY_N)
    logger.info("R0_GRID_BY_TASK = %s", R0_GRID_BY_TASK)
    logger.info(
        "N_REPS = %d | B_BOOT = %d | ALPHA = %.3f",
        N_REPS,
        B_BOOT,
        ALPHA,
    )
    logger.info(
        "N_JOBS = %d | MP_START_METHOD = %s",
        N_JOBS,
        MP_START_METHOD,
    )
    logger.info("ROTATION_BLOCK_SIZES = %s", ROTATION_BLOCK_SIZES)
    logger.info("REUSE_RESULTS = %s", REUSE_RESULTS)
    logger.info(
        "RUN_BEMA_GAMMA = %s | RUN_BEMA0 = %s",
        RUN_BEMA_GAMMA,
        RUN_BEMA0,
    )
    logger.info(
        "BEMA_M_FIT = %d | BEMA_M_FINAL = %d | "
        "BEMA_BULK_TRIM_ALPHA = %.3f | BEMA_TAIL_BETA = %.3f | "
        "BEMA_GAMMA_Q = %.3f | BEMA0_TW_ALPHA = %.3f",
        BEMA_M_FIT,
        BEMA_M_FINAL,
        BEMA_BULK_TRIM_ALPHA,
        BEMA_TAIL_BETA,
        BEMA_GAMMA_Q,
        BEMA0_TW_ALPHA,
    )
    logger.info(
        "PASS_YAO_1GAP_C = %.3f | PASS_YAO_SMAX = %d",
        PASS_YAO_1GAP_C,
        PASS_YAO_SMAX,
    )
    logger.info(
        "PASS_YAO2_CALIBRATE = %s | reps=%d | q=%.3f | "
        "calibration_dist=%s | reuse=%s",
        PASS_YAO2_CALIBRATE,
        PASS_YAO2_CALIB_REPS,
        PASS_YAO2_CALIB_Q,
        CALIBRATION_DIST,
        PASS_YAO2_REUSE_CALIB,
    )
    logger.info(
        "RUN_PASS_YAO2_DEFAULT = %s | PASS_YAO2_DEFAULT_C = %.3f",
        RUN_PASS_YAO2_DEFAULT,
        PASS_YAO2_DEFAULT_C,
    )
    logger.info(
        "ONATSKI_ALPHA = %.3f | ONATSKI_K_UPPER = %d",
        ONATSKI_ALPHA,
        ONATSKI_K_UPPER,
    )
    logger.info(
        "DY_ALPHA = %.3f | DY_R_STAR = %d | DY_NULL_REPS = %d | "
        "calibration_dist=%s | RUN_DY_LOCAL = %s | RUN_DY_MAX = %s",
        DY_ALPHA,
        DY_R_STAR,
        DY_NULL_REPS,
        DY_CALIB_DIST,
        RUN_DY_LOCAL,
        RUN_DY_MAX,
    )

    # Calibrate once, using Gaussian null draws only.  The resulting tables
    # are then reused unchanged in both the Gaussian and t10 experiments.
    calibration_outdir = outdir / "calibration_gaussian"
    calibration_outdir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Gaussian-only calibration stage.")
    pass_yao2_calib = precompute_pass_yao2_calibration(
        logger,
        calibration_outdir,
    )
    dy_calib = precompute_ding_yang_calibration(
        logger,
        calibration_outdir,
    )
    edge_info_map = precompute_edge_info(logger, calibration_outdir)
    edge_info_map = attach_calibrations_to_edge_info(
        edge_info_map,
        pass_yao2_calib,
        dy_calib,
    )
    logger.info(
        "Gaussian-only calibration stage finished; tables will be reused "
        "for every entry distribution."
    )

    all_distribution_details: list[pd.DataFrame] = []
    all_distribution_summaries: list[pd.DataFrame] = []

    for index, entry_dist in enumerate(ENTRY_DISTS, start=1):
        logger.info(
            "Starting distribution %d/%d: %s",
            index,
            len(ENTRY_DISTS),
            entry_dist,
        )
        details, summary = run_for_entry_dist(
            entry_dist,
            logger,
            outdir,
            edge_info_map,
        )
        if not details.empty:
            all_distribution_details.append(details)
        if not summary.empty:
            all_distribution_summaries.append(summary)

        save_combined_outputs(
            outdir,
            all_distribution_details,
            all_distribution_summaries,
            partial=True,
        )
        logger.info(
            "Finished distribution %d/%d: %s",
            index,
            len(ENTRY_DISTS),
            entry_dist,
        )

    save_combined_outputs(
        outdir,
        all_distribution_details,
        all_distribution_summaries,
        partial=False,
    )

    logger.info(
        "All distribution runs finished successfully in order: %s",
        " -> ".join(ENTRY_DISTS),
    )
    logger.info(
        "Pass-Yao and Ding-Yang calibrations used Gaussian draws only."
    )


if __name__ == "__main__":
    main()
