#!/usr/bin/env python3
"""
Coverage for bulk spectra using Gaussian and t10 entries to reproduce Table 1 and Table S1.

The distributions are run sequentially in this fixed order:
    1. gaussian
    2. t10

Gaussian entries have variance one. The t10 entries are rescaled to
variance one.

Setups
------
(n, p):
    (500, 200), N = 4
    (500, 750), N = 15
    (750, 500), N = 9

Bulk spectra:
    identity_0p9
    uniform_0p75_1p25
    two_mass_0p75_1p25

Model types:
    nonspike:
        target = lambda_1
    one_spike_6:
        population spikes = (6)
        target = lambda_2
    two_spike_7_6:
        population spikes = (7, 6)
        target = lambda_3

Edge convention
---------------
For all model types, including spiked models, use c = p/n.

For all spectra, the edge is computed from the manuscript equation

    f(x) = -1/x + (p/n) * mean[1 / (x + sigma^{-1})],

using the p-coordinate reference bulk spectrum. The right edge is f(b),
where b is the largest critical point f'(b)=0.

Bootstrap
---------
Multiplier weights:
    w_i ~ chi^2_N / N

CI:
    Uses the updated held-out bootstrap center and bias correction:

        E_hat      = target sample eigenvalue mu_{r+1}(Q)
        E0_hat     = mean of the first B-1 bootstrap target eigenvalues
        lambda_ho  = the B-th bootstrap target eigenvalue
        center     = lambda_ho - E0_hat + E_hat
        se         = sd of the first B-1 bootstrap target eigenvalues

        CI(alpha) = center +/- z_{alpha/2} * se,
        computed for all alpha values in ALPHA_GRID from the same
        bootstrap replicates.  Changing alpha only changes z_{alpha/2}
        and the nominal target coverage 1-alpha.

Defaults
--------
B_BOOT = 2000
N_REPS = 1000
N_JOBS = 50

Run
---
mkdir -p logs

nohup python3 accuracy.py \
  >> logs/accuracy.log 2>&1 &

The Gaussian run finishes before the t10 run begins. Results are saved in:

    Table1_results/gaussian/
    Table1_results/t10/

A cross-distribution summary is saved as:

    Table1_results/combined_summary_all_distributions.csv

Monitor
-------
tail -f logs/accuracy.log
"""

import os

# Avoid BLAS oversubscription in multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import logging
import multiprocessing as mp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.stats import norm


# ============================================================
# Configuration
# ============================================================

ENTRY_DISTS = ("gaussian", "t10")


class RunConfig(object):
    def __init__(
        self,
        outdir="accuracy_results",
        n_reps=1000,
        b_boot=2000,
        alpha=0.05,
        n_jobs=50,
        mp_start_method="fork",
        chunksize=1,
        plot_dpi=180,
        entry_dist="gaussian",
    ):
        self.outdir = str(outdir)
        self.n_reps = int(n_reps)
        self.b_boot = int(b_boot)
        self.alpha = float(alpha)
        self.n_jobs = int(n_jobs)
        self.mp_start_method = str(mp_start_method)
        self.chunksize = int(chunksize)
        self.plot_dpi = int(plot_dpi)
        self.entry_dist = str(entry_dist)
        if self.entry_dist not in ENTRY_DISTS:
            raise ValueError(
                "Unknown entry_dist: {}. Expected one of {}.".format(
                    self.entry_dist,
                    ENTRY_DISTS,
                )
            )

    @property
    def z_alpha(self):
        return float(norm.ppf(1.0 - self.alpha / 2.0))

    @property
    def target_coverage(self):
        return float(1.0 - self.alpha)

    def to_dict(self):
        return {
            "outdir": self.outdir,
            "n_reps": self.n_reps,
            "b_boot": self.b_boot,
            "alpha": self.alpha,
            "n_jobs": self.n_jobs,
            "mp_start_method": self.mp_start_method,
            "chunksize": self.chunksize,
            "plot_dpi": self.plot_dpi,
            "entry_dist": self.entry_dist,
        }


CFG = RunConfig()

# Evaluate both nominal levels using the same generated data and the
# same bootstrap eigenvalues.  Alpha only changes z_{alpha/2} and
# the target coverage line 1-alpha; it does not change the multipliers,
# E0_hat, the held-out center, or the bootstrap standard error.
ALPHA_GRID = (0.05, 0.10)


def z_for_alpha(alpha):
    return float(norm.ppf(1.0 - float(alpha) / 2.0))


def target_coverage_for_alpha(alpha):
    return float(1.0 - float(alpha))


def alpha_tag(alpha):
    return "alpha_{:.2f}".format(float(alpha)).replace(".", "p")


PRIMARY_N = {
    (500, 200): 4,
    (500, 750): 15,
    (750, 500): 9,
}

# Only run the primary N for each dimension.
NP_GRID = {dim: (int(N),) for dim, N in PRIMARY_N.items()}


BULK_SPECTRA = [
    # "identity",
    "identity_0p9",
    "uniform_0p75_1p25",
    "two_mass_0p75_1p25",
]

MODEL_TYPES = [
    "nonspike",
    "one_spike_6",
    "two_spike_7_6"
]


# ============================================================
# Config classes
# ============================================================

class ComboConfig(object):
    def __init__(self, n, p, bulk_spectrum, model_type, entry_dist):
        self.n = int(n)
        self.p = int(p)
        self.bulk_spectrum = str(bulk_spectrum)
        self.model_type = str(model_type)
        self.entry_dist = str(entry_dist)
        if self.entry_dist not in ENTRY_DISTS:
            raise ValueError(
                "Unknown entry_dist: {}. Expected one of {}.".format(
                    self.entry_dist,
                    ENTRY_DISTS,
                )
            )

    @property
    def r(self):
        if self.model_type == "nonspike":
            return 0
        if self.model_type == "one_spike_6":
            return 1
        if self.model_type == "two_spike_7_6":
            return 2
        raise ValueError("Unknown model_type: {}".format(self.model_type))

    @property
    def p_bulk(self):
        return self.p - self.r

    @property
    def phi_edge(self):
        # Requested convention: always use p/n, including spiked models.
        return float(self.p / self.n)

    @property
    def target_index(self):
        # 1-indexed target eigenvalue.
        return self.r + 1

    @property
    def k_needed(self):
        # Need one extra eigenvalue for gap diagnostics.
        return self.r + 2

    @property
    def spikes(self):
        if self.model_type == "nonspike":
            return tuple()
        if self.model_type == "one_spike_6":
            return (6.0,)
        if self.model_type == "two_spike_7_6":
            return (7.0, 6.0)
        raise ValueError("Unknown model_type: {}".format(self.model_type))

    @property
    def spike_1(self):
        return self.spikes[0] if len(self.spikes) >= 1 else np.nan

    @property
    def spike_2(self):
        return self.spikes[1] if len(self.spikes) >= 2 else np.nan

    @property
    def spike_label(self):
        if len(self.spikes) == 0:
            return "none"
        if len(self.spikes) == 1:
            return "{:.4g}".format(self.spikes[0])
        return "({:.4g},{:.4g})".format(self.spikes[0], self.spikes[1])

    @property
    def name(self):
        return "n{}_p{}_{}_{}_{}".format(
            self.n,
            self.p,
            self.bulk_spectrum,
            self.entry_dist,
            self.model_type,
        )

    def to_dict(self):
        return {
            "n": self.n,
            "p": self.p,
            "bulk_spectrum": self.bulk_spectrum,
            "model_type": self.model_type,
            "entry_dist": self.entry_dist,
        }


def make_combos(entry_dist):
    combos = []
    for n, p in NP_GRID.keys():
        for spectrum in BULK_SPECTRA:
            for model_type in MODEL_TYPES:
                combos.append(
                    ComboConfig(
                        n=n,
                        p=p,
                        bulk_spectrum=spectrum,
                        model_type=model_type,
                        entry_dist=entry_dist,
                    )
                )
    return combos


def get_n_grid(combo):
    key = (combo.n, combo.p)
    if key not in NP_GRID:
        raise KeyError("Missing N grid for {}".format(key))
    return tuple(int(x) for x in NP_GRID[key])


# ============================================================
# Logging
# ============================================================

def setup_logger(outdir, entry_dist):
    outdir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("nonspike_coverage_" + entry_dist)
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


# ============================================================
# Population spectra and edges
# ============================================================

def population_bulk_eigs_by_name(bulk_spectrum, p_bulk):
    """
    Population bulk eigenvalues.

    For edge computation, this is called with p_bulk=p so that the edge uses
    the p-coordinate reference spectrum and coefficient p/n.

    For data generation in spiked models, this is called with p_bulk=p-r.
    """
    p_bulk = int(p_bulk)

    if bulk_spectrum == "identity":
        return np.ones(p_bulk, dtype=float)

    if bulk_spectrum == "identity_0p9":
        return 0.9 * np.ones(p_bulk, dtype=float)

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

    raise ValueError("Unknown bulk spectrum: {}".format(bulk_spectrum))


def bulk_population_eigs(combo):
    return population_bulk_eigs_by_name(combo.bulk_spectrum, combo.p_bulk)


def full_population_eigs(combo, bulk_eigs):
    if combo.r == 0:
        return bulk_eigs

    spikes = np.array(combo.spikes, dtype=float)
    return np.concatenate([spikes, bulk_eigs])


def f_edge(x, eigs, phi):
    """
    Manuscript convention:

        f(x) = -1/x + phi * mean[1/(x + sigma^{-1})].

    Here phi is always p/n.
    """
    return float(-1.0 / x + phi * np.mean(1.0 / (x + 1.0 / eigs)))


def df_edge(x, eigs, phi):
    return float(
        1.0 / (x * x)
        - phi * np.mean(1.0 / ((x + 1.0 / eigs) ** 2))
    )


def d2f_edge(x, eigs, phi):
    return float(
        -2.0 / (x ** 3)
        + 2.0 * phi * np.mean(1.0 / ((x + 1.0 / eigs) ** 3))
    )


def deformed_mp_upper_edge(eigs, phi):
    """
    Rightmost edge using the manuscript convention.

    The largest critical point b solves f'(b)=0 on (-1/sigma_max, 0),
    and the right edge is E=f(b).
    """
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
                "Could not locate edge root for phi={}, sigma_max={}".format(
                    phi,
                    sigma_max,
                )
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


def precompute_edge_info(logger, outdir):
    rows = []
    edge_info = {}

    logger.info("Precomputing edge information once per (n,p,spectrum).")

    for n, p in NP_GRID.keys():
        phi = float(p / n)
        for spectrum in BULK_SPECTRA:
            # Use p-coordinate reference spectrum for the edge.
            ref_eigs = population_bulk_eigs_by_name(spectrum, p)
            edge, bcrit, theta_threshold, tw_b_coeff = deformed_mp_upper_edge(
                ref_eigs,
                phi,
            )

            key = (int(n), int(p), str(spectrum))
            edge_info[key] = {
                "edge": edge,
                "critical_b": bcrit,
                "theta_threshold": theta_threshold,
                "tw_b_coeff": tw_b_coeff,
                "phi_edge": phi,
                "p_edge_reference": int(p),
            }

            rows.append({
                "n": int(n),
                "p": int(p),
                "bulk_spectrum": str(spectrum),
                "phi_edge": phi,
                "p_edge_reference": int(p),
                "critical_b": bcrit,
                "edge": edge,
                "theta_threshold": theta_threshold,
                "tw_b_coeff": tw_b_coeff,
            })

            logger.info(
                "Edge setup | n=%d | p=%d | spectrum=%s | phi=p/n=%.6f | "
                "b=%.10f | edge=%.10f | theta_threshold=%.10f | "
                "TW_b_coeff=%.10f",
                int(n),
                int(p),
                str(spectrum),
                phi,
                bcrit,
                edge,
                theta_threshold,
                tw_b_coeff,
            )

    edge_df = pd.DataFrame(rows)
    edge_df.to_csv(outdir / "edge_info.csv", index=False)

    return edge_info


# ============================================================
# Data generation and eigenvalue computation
# ============================================================

def sample_entries(n, p, rng, entry_dist):
    if entry_dist == "gaussian":
        return rng.standard_normal(size=(n, p))

    if entry_dist == "t10":
        df = 10.0
        return rng.standard_t(df=df, size=(n, p)) * np.sqrt((df - 2.0) / df)

    raise ValueError("Unknown entry_dist: {}".format(entry_dist))


def generate_data(combo, full_eigs, rng):
    Z = sample_entries(combo.n, combo.p, rng, combo.entry_dist)
    return Z * np.sqrt(full_eigs)[None, :]


def top_k_cov_eigs_from_matrix(A, k):
    d = A.shape[0]
    k = min(int(k), d)

    vals = eigh(
        A,
        subset_by_index=[d - k, d - 1],
        eigvals_only=True,
        check_finite=False,
        overwrite_a=True,
    )
    return np.sort(vals)[::-1]


def top_k_sample_cov_eigs(X, k):
    n, p = X.shape
    if p <= n:
        M = (X.T @ X) / n
    else:
        M = (X @ X.T) / n
    return top_k_cov_eigs_from_matrix(M, k)


def top_k_weighted_cov_eigs(X, w, k):
    n, p = X.shape
    Xw = X * np.sqrt(w)[:, None]
    if p <= n:
        M = (Xw.T @ Xw) / n
    else:
        M = (Xw @ Xw.T) / n
    return top_k_cov_eigs_from_matrix(M, k)


def draw_chisq_weights(n, N, rng):
    return rng.chisquare(df=N, size=n) / N


# ============================================================
# One repetition
# ============================================================

def run_one_rep(task):
    """
    One repetition for one fixed combo and one fixed N.

    task = (combo_dict, N, rep, edge, full_eigs, cfg_dict)
    """
    combo_dict, N, rep, edge, full_eigs, cfg_dict = task
    combo = ComboConfig(**combo_dict)
    cfg = RunConfig(**cfg_dict)

    if cfg.b_boot < 3:
        raise ValueError(
            "The updated framework needs b_boot >= 3: "
            "B-1 bootstrap replicates estimate E0/variance and "
            "one held-out replicate gives the center."
        )

    rng = np.random.default_rng(None)

    X = generate_data(combo, full_eigs, rng)
    vals = top_k_sample_cov_eigs(X, combo.k_needed)

    target_idx = combo.target_index - 1
    target_lambda = float(vals[target_idx])

    lambda1 = float(vals[0]) if len(vals) >= 1 else np.nan
    lambda2 = float(vals[1]) if len(vals) >= 2 else np.nan
    lambda3 = float(vals[2]) if len(vals) >= 3 else np.nan
    lambda4 = float(vals[3]) if len(vals) >= 4 else np.nan

    gap12 = float(lambda1 - lambda2) if len(vals) >= 2 else np.nan
    gap23 = float(lambda2 - lambda3) if len(vals) >= 3 else np.nan
    gap34 = float(lambda3 - lambda4) if len(vals) >= 4 else np.nan

    boot_targets = np.empty(cfg.b_boot, dtype=float)
    boot_spike_bulk_gaps = np.empty(cfg.b_boot, dtype=float) if combo.r > 0 else None

    for b in range(cfg.b_boot):
        w = draw_chisq_weights(combo.n, N, rng)
        boot_vals = top_k_weighted_cov_eigs(X, w, combo.target_index)

        boot_targets[b] = boot_vals[target_idx]

        if combo.r == 1:
            boot_spike_bulk_gaps[b] = boot_vals[0] - boot_vals[1]
        elif combo.r == 2:
            boot_spike_bulk_gaps[b] = boot_vals[1] - boot_vals[2]

    # --------------------------------------------------------
    # Updated bootstrap framework
    # --------------------------------------------------------
    # Use the first B-1 bootstrap replicates to estimate the
    # conditional bootstrap center E0 and the bootstrap variance.
    # Use the last replicate as a held-out randomized center.
    #
    # E_hat is the original sample eigenvalue mu_{r+1}(Q), which
    # estimates the deterministic edge E under the null.
    #
    # The bias estimate is
    #     Delta0_hat = E0_hat - E_hat,
    # and the corrected center is
    #     center_corrected = lambda_holdout - Delta0_hat
    #                      = lambda_holdout - E0_hat + E_hat.
    # --------------------------------------------------------
    boot_train = boot_targets[:-1]
    boot_holdout = float(boot_targets[-1])

    E_hat = float(target_lambda)
    E0_hat = float(np.mean(boot_train))
    boot_sd_train = float(np.std(boot_train, ddof=1))
    v0_hat = float(combo.n * boot_sd_train ** 2)

    bias_hat = float(E0_hat - E_hat)
    center_corrected = float(boot_holdout - bias_hat)
    center_corrected_minus_edge = float(center_corrected - edge)

    old_boot_mean = float(np.mean(boot_targets))
    old_boot_sd = float(np.std(boot_targets, ddof=1))

    if combo.r == 1:
        spike_bulk_gap = gap12
    elif combo.r == 2:
        spike_bulk_gap = gap23
    else:
        spike_bulk_gap = np.nan

    base_row = {
        "combo_name": combo.name,
        "model_type": combo.model_type,
        "entry_dist": combo.entry_dist,
        "n": int(combo.n),
        "p": int(combo.p),
        "p_bulk": int(combo.p_bulk),
        "phi_edge": float(combo.phi_edge),
        "bulk_spectrum": combo.bulk_spectrum,
        "r": int(combo.r),
        "spike_1": float(combo.spike_1),
        "spike_2": float(combo.spike_2),
        "spike_label": combo.spike_label,
        "N": int(N),
        "rep": int(rep),
        "target_index": int(combo.target_index),
        "edge": float(edge),
        "target_lambda": target_lambda,
        "target_minus_edge": float(target_lambda - edge),
        "required_right_halfwidth": float(max(edge - target_lambda, 0.0)),
        "lambda1_hat": lambda1,
        "lambda2_hat": lambda2,
        "lambda3_hat": lambda3,
        "lambda4_hat": lambda4,
        "gap12": gap12,
        "gap23": gap23,
        "gap34": gap34,
        "spike_bulk_gap": spike_bulk_gap,
        "E_hat": E_hat,
        "E0_hat": E0_hat,
        "bias_hat": bias_hat,
        "boot_holdout_target": boot_holdout,
        "center_corrected": center_corrected,
        "center_corrected_minus_edge": center_corrected_minus_edge,
        "boot_sd_train": boot_sd_train,
        "v0_hat": v0_hat,
        "boot_mean_target": E0_hat,
        "boot_sd_target": boot_sd_train,
        "old_boot_mean_target": old_boot_mean,
        "old_boot_sd_target": old_boot_sd,
        "B": int(cfg.b_boot),
        "B_train": int(cfg.b_boot - 1),
        "B_holdout": 1,
    }

    if combo.r > 0:
        base_row.update({
            "boot_spike_bulk_gap_mean": float(np.mean(boot_spike_bulk_gaps)),
            "boot_spike_bulk_gap_min": float(np.min(boot_spike_bulk_gaps)),
            "boot_spike_bulk_gap_q05": float(np.quantile(boot_spike_bulk_gaps, 0.05)),
            "boot_spike_bulk_gap_q50": float(np.quantile(boot_spike_bulk_gaps, 0.50)),
        })
    else:
        base_row.update({
            "boot_spike_bulk_gap_mean": np.nan,
            "boot_spike_bulk_gap_min": np.nan,
            "boot_spike_bulk_gap_q05": np.nan,
            "boot_spike_bulk_gap_q50": np.nan,
        })

    rows = []
    for alpha in ALPHA_GRID:
        z_alpha = z_for_alpha(alpha)
        target_coverage = target_coverage_for_alpha(alpha)

        ci_lower = float(center_corrected - z_alpha * boot_sd_train)
        ci_upper = float(center_corrected + z_alpha * boot_sd_train)
        ci_width = float(ci_upper - ci_lower)
        covered = float(ci_lower <= edge <= ci_upper)

        # Old interval, saved only for diagnostics/comparison.
        # The main columns ci_lower/ci_upper/covered use the new method.
        old_ci_lower = float(target_lambda - z_alpha * old_boot_sd)
        old_ci_upper = float(target_lambda + z_alpha * old_boot_sd)
        old_ci_width = float(old_ci_upper - old_ci_lower)
        old_covered = float(old_ci_lower <= edge <= old_ci_upper)

        row = dict(base_row)
        row.update({
            "alpha": float(alpha),
            "alpha_tag": alpha_tag(alpha),
            "z_alpha": float(z_alpha),
            "target_coverage": float(target_coverage),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_width,
            "covered": covered,
            "old_ci_lower": old_ci_lower,
            "old_ci_upper": old_ci_upper,
            "old_ci_width": old_ci_width,
            "old_covered": old_covered,
        })
        rows.append(row)

    return rows


# ============================================================
# Summaries and plotting
# ============================================================

def summarize_details(details, cfg):
    rows = []

    for (alpha, N), sub in details.groupby(["alpha", "N"]):
        coverage = float(sub["covered"].mean())
        old_coverage = float(sub["old_covered"].mean())
        target_coverage = float(sub["target_coverage"].iloc[0])

        rows.append({
            "combo_name": str(sub["combo_name"].iloc[0]),
            "model_type": str(sub["model_type"].iloc[0]),
            "entry_dist": str(sub["entry_dist"].iloc[0]),
            "n": int(sub["n"].iloc[0]),
            "p": int(sub["p"].iloc[0]),
            "p_bulk": int(sub["p_bulk"].iloc[0]),
            "phi_edge": float(sub["phi_edge"].iloc[0]),
            "bulk_spectrum": str(sub["bulk_spectrum"].iloc[0]),
            "r": int(sub["r"].iloc[0]),
            "spike_1": float(sub["spike_1"].iloc[0]),
            "spike_2": float(sub["spike_2"].iloc[0]),
            "spike_label": str(sub["spike_label"].iloc[0]),
            "alpha": float(alpha),
            "alpha_tag": str(sub["alpha_tag"].iloc[0]),
            "z_alpha": float(sub["z_alpha"].iloc[0]),
            "target_coverage": target_coverage,
            "N": int(N),
            "target_index": int(sub["target_index"].iloc[0]),
            "edge": float(sub["edge"].iloc[0]),
            "coverage": coverage,
            "mc_se_coverage": float(
                np.sqrt(coverage * (1.0 - coverage) / len(sub))
            ),
            "old_coverage": old_coverage,
            "old_mc_se_coverage": float(
                np.sqrt(old_coverage * (1.0 - old_coverage) / len(sub))
            ),
            "mean_target_lambda": float(sub["target_lambda"].mean()),
            "sd_target_lambda": float(sub["target_lambda"].std(ddof=1)),
            "mean_target_minus_edge": float(sub["target_minus_edge"].mean()),
            "q05_target_minus_edge": float(sub["target_minus_edge"].quantile(0.05)),
            "q95_target_minus_edge": float(sub["target_minus_edge"].quantile(0.95)),
            "mean_required_right_halfwidth": float(sub["required_right_halfwidth"].mean()),
            "q90_required_right_halfwidth": float(
                sub["required_right_halfwidth"].quantile(0.90)
            ),
            "q95_required_right_halfwidth": float(
                sub["required_right_halfwidth"].quantile(0.95)
            ),
            "q975_required_right_halfwidth": float(
                sub["required_right_halfwidth"].quantile(0.975)
            ),
            "mean_E_hat": float(sub["E_hat"].mean()),
            "mean_E0_hat": float(sub["E0_hat"].mean()),
            "mean_bias_hat": float(sub["bias_hat"].mean()),
            "sd_bias_hat": float(sub["bias_hat"].std(ddof=1)),
            "mean_boot_holdout_target": float(sub["boot_holdout_target"].mean()),
            "mean_center_corrected": float(sub["center_corrected"].mean()),
            "sd_center_corrected": float(sub["center_corrected"].std(ddof=1)),
            "mean_center_corrected_minus_edge": float(
                sub["center_corrected_minus_edge"].mean()
            ),
            "q05_center_corrected_minus_edge": float(
                sub["center_corrected_minus_edge"].quantile(0.05)
            ),
            "q95_center_corrected_minus_edge": float(
                sub["center_corrected_minus_edge"].quantile(0.95)
            ),
            "mean_boot_mean_target": float(sub["boot_mean_target"].mean()),
            "mean_boot_sd": float(sub["boot_sd_target"].mean()),
            "sd_boot_sd": float(sub["boot_sd_target"].std(ddof=1)),
            "mean_boot_sd_train": float(sub["boot_sd_train"].mean()),
            "mean_v0_hat": float(sub["v0_hat"].mean()),
            "mean_ci_width": float(sub["ci_width"].mean()),
            "sd_ci_width": float(sub["ci_width"].std(ddof=1)),
            "old_mean_boot_sd": float(sub["old_boot_sd_target"].mean()),
            "old_mean_ci_width": float(sub["old_ci_width"].mean()),
            "mean_gap12": float(sub["gap12"].mean()),
            "q05_gap12": float(sub["gap12"].quantile(0.05)),
            "mean_gap23": float(sub["gap23"].mean()),
            "q05_gap23": float(sub["gap23"].quantile(0.05)),
            "mean_gap34": float(sub["gap34"].mean()),
            "q05_gap34": float(sub["gap34"].quantile(0.05)),
            "mean_spike_bulk_gap": float(sub["spike_bulk_gap"].mean()),
            "q05_spike_bulk_gap": float(sub["spike_bulk_gap"].quantile(0.05)),
            "mean_boot_spike_bulk_gap_q05": float(
                sub["boot_spike_bulk_gap_q05"].mean()
            ),
            "mean_boot_spike_bulk_gap_min": float(
                sub["boot_spike_bulk_gap_min"].mean()
            ),
            "n_reps": int(len(sub)),
            "B": int(sub["B"].iloc[0]),
        })

    return pd.DataFrame(rows).sort_values(["alpha", "N"]).reset_index(drop=True)


def save_combo_plots(combo_dir, summary, cfg):
    combo_name = str(summary["combo_name"].iloc[0])

    # Coverage vs N, one curve for each alpha.
    plt.figure(figsize=(8.8, 5.2))
    for alpha, ss in summary.groupby("alpha"):
        ss = ss.sort_values("N")
        plt.plot(
            ss["N"],
            ss["coverage"],
            marker="o",
            label="alpha={:.2f}, target={:.2f}".format(
                float(alpha), float(ss["target_coverage"].iloc[0])
            ),
        )
        plt.axhline(
            float(ss["target_coverage"].iloc[0]),
            linestyle="--",
            linewidth=1.0,
        )
    plt.xlabel("N")
    plt.ylabel("Coverage")
    plt.title("{}: coverage vs N".format(combo_name))
    plt.legend()
    plt.tight_layout()
    plt.savefig(combo_dir / "coverage_vs_N.png", dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close()

    # Coverage minus target is useful when comparing alpha=0.05 and 0.10.
    plt.figure(figsize=(8.8, 5.2))
    for alpha, ss in summary.groupby("alpha"):
        ss = ss.sort_values("N")
        plt.plot(
            ss["N"],
            ss["coverage"] - ss["target_coverage"],
            marker="o",
            label="alpha={:.2f}".format(float(alpha)),
        )
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("N")
    plt.ylabel("Coverage - target coverage")
    plt.title("{}: coverage error vs N".format(combo_name))
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        combo_dir / "coverage_error_vs_N.png",
        dpi=cfg.plot_dpi,
        bbox_inches="tight",
    )
    plt.close()

    # CI width differs by alpha only through z_{alpha/2}.
    plt.figure(figsize=(8.8, 5.2))
    for alpha, ss in summary.groupby("alpha"):
        ss = ss.sort_values("N")
        plt.plot(
            ss["N"],
            ss["mean_ci_width"],
            marker="o",
            label="alpha={:.2f}".format(float(alpha)),
        )
    plt.xlabel("N")
    plt.ylabel("Mean CI width")
    plt.title("{}: CI width vs N".format(combo_name))
    plt.legend()
    plt.tight_layout()
    plt.savefig(combo_dir / "ci_width_vs_N.png", dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8.8, 5.2))
    ss0 = summary.sort_values("N").drop_duplicates("N")
    plt.plot(ss0["N"], ss0["mean_boot_sd"], marker="o")
    plt.xlabel("N")
    plt.ylabel("Mean bootstrap SD")
    plt.title("{}: bootstrap SD vs N".format(combo_name))
    plt.tight_layout()
    plt.savefig(combo_dir / "boot_sd_vs_N.png", dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8.8, 5.2))
    plt.plot(ss0["N"], ss0["mean_target_minus_edge"], marker="o")
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("N")
    plt.ylabel("Mean target lambda - edge")
    plt.title("{}: target location vs N".format(combo_name))
    plt.tight_layout()
    plt.savefig(
        combo_dir / "target_minus_edge_vs_N.png",
        dpi=cfg.plot_dpi,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8.8, 5.2))
    plt.plot(ss0["N"], ss0["mean_center_corrected_minus_edge"], marker="o")
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("N")
    plt.ylabel("Mean corrected center - edge")
    plt.title("{}: corrected center location vs N".format(combo_name))
    plt.tight_layout()
    plt.savefig(
        combo_dir / "center_corrected_minus_edge_vs_N.png",
        dpi=cfg.plot_dpi,
        bbox_inches="tight",
    )
    plt.close()

    if int(summary["r"].iloc[0]) > 0:
        plt.figure(figsize=(8.8, 5.2))
        plt.plot(ss0["N"], ss0["mean_spike_bulk_gap"], marker="o",
                 label="mean spike-bulk gap")
        plt.plot(ss0["N"], ss0["q05_spike_bulk_gap"], marker="s",
                 label="q05 spike-bulk gap")
        plt.axhline(0.0, linestyle="--", linewidth=1.2)
        plt.xlabel("N")
        plt.ylabel("Gap")
        plt.title("{}: spike-bulk gap diagnostics".format(combo_name))
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            combo_dir / "gap_diagnostics_vs_N.png",
            dpi=cfg.plot_dpi,
            bbox_inches="tight",
        )
        plt.close()


def save_combined_plots(outdir, combined_summary, cfg):
    entry_dist = str(combined_summary["entry_dist"].iloc[0])
    for alpha, sub_alpha in combined_summary.groupby("alpha"):
        alpha_name = alpha_tag(alpha)
        target_coverage = float(sub_alpha["target_coverage"].iloc[0])

        for (n, p), sub_np in sub_alpha.groupby(["n", "p"]):
            plt.figure(figsize=(11.0, 6.4))

            for (spectrum, model_type), ss in sub_np.groupby(["bulk_spectrum", "model_type"]):
                ss = ss.sort_values("N")
                label = "{}, {}".format(spectrum, model_type)
                plt.plot(ss["N"], ss["coverage"], marker="o", linewidth=1.4, label=label)

            plt.axhline(
                target_coverage,
                linestyle="--",
                linewidth=1.4,
                label="target={:.2f}".format(target_coverage),
            )
            plt.xlabel("N")
            plt.ylabel("Coverage")
            plt.title("All coverage curves: {} | alpha={:.2f} | n={}, p={}".format(
                entry_dist, float(alpha), n, p
            ))
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(
                outdir / "combined_coverage_{}_n{}_p{}.png".format(alpha_name, n, p),
                dpi=cfg.plot_dpi,
                bbox_inches="tight",
            )
            plt.close()

        for (n, p, spectrum), sub in sub_alpha.groupby(["n", "p", "bulk_spectrum"]):
            plt.figure(figsize=(9.0, 5.4))

            for model_type, ss in sub.groupby("model_type"):
                ss = ss.sort_values("N")
                plt.plot(ss["N"], ss["coverage"], marker="o", linewidth=1.8,
                         label=model_type)

            plt.axhline(
                target_coverage,
                linestyle="--",
                linewidth=1.4,
                label="target={:.2f}".format(target_coverage),
            )
            plt.xlabel("N")
            plt.ylabel("Coverage")
            plt.title("Coverage: {} | alpha={:.2f} | n={}, p={}, {}".format(
                entry_dist, float(alpha), n, p, spectrum
            ))
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                outdir / "coverage_compare_{}_n{}_p{}_{}.png".format(
                    alpha_name,
                    n,
                    p,
                    spectrum,
                ),
                dpi=cfg.plot_dpi,
                bbox_inches="tight",
            )
            plt.close()

            plt.figure(figsize=(9.0, 5.4))

            for model_type, ss in sub.groupby("model_type"):
                ss = ss.sort_values("N")
                plt.plot(ss["N"], ss["mean_ci_width"], marker="o", linewidth=1.8,
                         label=model_type)

            plt.xlabel("N")
            plt.ylabel("Mean CI width")
            plt.title("CI width: {} | alpha={:.2f} | n={}, p={}, {}".format(
                entry_dist, float(alpha), n, p, spectrum
            ))
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                outdir / "ci_width_compare_{}_n{}_p{}_{}.png".format(
                    alpha_name,
                    n,
                    p,
                    spectrum,
                ),
                dpi=cfg.plot_dpi,
                bbox_inches="tight",
            )
            plt.close()

    # One direct comparison of the two alpha levels for each (n,p,spectrum,model).
    for (n, p, spectrum, model_type), sub in combined_summary.groupby(
        ["n", "p", "bulk_spectrum", "model_type"]
    ):
        plt.figure(figsize=(8.8, 5.2))
        for alpha, ss in sub.groupby("alpha"):
            ss = ss.sort_values("N")
            plt.plot(
                ss["N"],
                ss["coverage"] - ss["target_coverage"],
                marker="o",
                linewidth=1.6,
                label="alpha={:.2f}".format(float(alpha)),
            )
        plt.axhline(0.0, linestyle="--", linewidth=1.2)
        plt.xlabel("N")
        plt.ylabel("Coverage - target coverage")
        plt.title("Coverage error: n={}, p={}, {}, {}".format(
            n, p, spectrum, model_type
        ))
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            outdir / "coverage_error_compare_alpha_n{}_p{}_{}_{}.png".format(
                n,
                p,
                spectrum,
                model_type,
            ),
            dpi=cfg.plot_dpi,
            bbox_inches="tight",
        )
        plt.close()


# ============================================================
# Main experiment
# ============================================================

def run_combo(combo, cfg, logger, outdir, edge_info):
    combo_dir = outdir / combo.name
    combo_dir.mkdir(parents=True, exist_ok=True)

    N_grid = get_n_grid(combo)

    bulk_eigs = bulk_population_eigs(combo)
    full_eigs = full_population_eigs(combo, bulk_eigs)

    info = edge_info[(int(combo.n), int(combo.p), str(combo.bulk_spectrum))]
    edge = float(info["edge"])
    bcrit = float(info["critical_b"])
    theta_threshold = float(info["theta_threshold"])
    tw_b_coeff = float(info["tw_b_coeff"])

    block_info = pd.DataFrame([{
        "combo_name": combo.name,
        "model_type": combo.model_type,
        "entry_dist": combo.entry_dist,
        "n": combo.n,
        "p": combo.p,
        "p_bulk": combo.p_bulk,
        "phi_edge": combo.phi_edge,
        "bulk_spectrum": combo.bulk_spectrum,
        "r": combo.r,
        "spike_1": combo.spike_1,
        "spike_2": combo.spike_2,
        "spike_label": combo.spike_label,
        "target_index": combo.target_index,
        "edge": edge,
        "critical_b": bcrit,
        "theta_threshold": theta_threshold,
        "tw_b_coeff": tw_b_coeff,
        "edge_uses": "p/n and p-coordinate reference spectrum for all model types",
        "p_edge_reference": int(info.get("p_edge_reference", combo.p)),
        "N_grid": ",".join(str(x) for x in N_grid),
        "n_reps": cfg.n_reps,
        "B": cfg.b_boot,
        "alpha_grid": ",".join(str(x) for x in ALPHA_GRID),
        "target_coverage_grid": ",".join(
            str(target_coverage_for_alpha(x)) for x in ALPHA_GRID
        ),
    }])
    block_info.to_csv(combo_dir / "block_info.csv", index=False)

    logger.info(
        "Starting block | %s | model=%s | spectrum=%s | spikes=%s | "
        "target=lambda_%d | N=%d..%d",
        combo.name,
        combo.model_type,
        combo.bulk_spectrum,
        combo.spike_label,
        combo.target_index,
        min(N_grid),
        max(N_grid),
    )

    combo_dict = combo.to_dict()
    cfg_dict = cfg.to_dict()

    all_detail_rows = []
    summary_rows = []
    ctx = mp.get_context(cfg.mp_start_method)

    for N in N_grid:
        tasks = [
            (
                combo_dict,
                int(N),
                int(rep),
                float(edge),
                full_eigs,
                cfg_dict,
            )
            for rep in range(1, cfg.n_reps + 1)
        ]

        rows_N = []
        with ctx.Pool(processes=cfg.n_jobs) as pool:
            for rep_rows in pool.imap_unordered(run_one_rep, tasks, chunksize=cfg.chunksize):
                rows_N.extend(rep_rows)

        df_N = pd.DataFrame(rows_N).sort_values(["alpha", "rep"]).reset_index(drop=True)
        all_detail_rows.extend(rows_N)

        summary_N = summarize_details(df_N, cfg)
        summary_rows.append(summary_N)

        summary_alpha = {
            round(float(s["alpha"]), 10): s
            for _, s in summary_N.sort_values("alpha").iterrows()
        }
        s05 = summary_alpha.get(0.05)
        s10 = summary_alpha.get(0.10)

        if s05 is not None and s10 is not None:
            logger.info(
                "[%s] N=%d | alpha=0.05: coverage=%.4f | "
                "alpha=0.10: coverage=%.4f | mean_boot_sd=%.6f | "
                "mean_center_corrected_minus_edge=%.6f | "
                "mean_target_minus_edge=%.6f",
                combo.name,
                N,
                float(s05["coverage"]),
                float(s10["coverage"]),
                float(s05["mean_boot_sd"]),
                float(s05["mean_center_corrected_minus_edge"]),
                float(s05["mean_target_minus_edge"]),
            )
        else:
            # Fallback for any future ALPHA_GRID with values other than 0.05/0.10.
            alpha_parts = []
            s_ref = None
            for _, s in summary_N.sort_values("alpha").iterrows():
                if s_ref is None:
                    s_ref = s
                alpha_parts.append(
                    "alpha={:.2f}: coverage={:.4f}".format(
                        float(s["alpha"]),
                        float(s["coverage"]),
                    )
                )
            logger.info(
                "[%s] N=%d | %s | mean_boot_sd=%.6f | "
                "mean_center_corrected_minus_edge=%.6f | "
                "mean_target_minus_edge=%.6f | q05_spike_bulk_gap=%.6f",
                combo.name,
                N,
                " | ".join(alpha_parts),
                float(s_ref["mean_boot_sd"]),
                float(s_ref["mean_center_corrected_minus_edge"]),
                float(s_ref["mean_target_minus_edge"]),
                float(s_ref["q05_spike_bulk_gap"]),
            )

        details_partial = pd.DataFrame(all_detail_rows)
        summary_partial = pd.concat(summary_rows, ignore_index=True).sort_values(["alpha", "N"])
        details_partial.to_csv(combo_dir / "details_partial.csv", index=False)
        summary_partial.to_csv(combo_dir / "summary_partial.csv", index=False)

    details = pd.DataFrame(all_detail_rows).sort_values(["alpha", "N", "rep"]).reset_index(drop=True)
    summary = pd.concat(summary_rows, ignore_index=True).sort_values(["alpha", "N"]).reset_index(drop=True)

    details.to_csv(combo_dir / "details.csv", index=False)
    summary.to_csv(combo_dir / "summary.csv", index=False)

    save_combo_plots(combo_dir, summary, cfg)

    logger.info("Finished block %s", combo.name)

    return details, summary


def config_for_distribution(base_cfg, entry_dist):
    return RunConfig(
        outdir=str(Path(base_cfg.outdir) / entry_dist),
        n_reps=base_cfg.n_reps,
        b_boot=base_cfg.b_boot,
        alpha=base_cfg.alpha,
        n_jobs=base_cfg.n_jobs,
        mp_start_method=base_cfg.mp_start_method,
        chunksize=base_cfg.chunksize,
        plot_dpi=base_cfg.plot_dpi,
        entry_dist=entry_dist,
    )


def run_distribution(entry_dist, base_cfg):
    cfg = config_for_distribution(base_cfg, entry_dist)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(outdir, entry_dist)
    combos = make_combos(entry_dist)

    logger.info("Distribution run started: %s", entry_dist)
    logger.info("Configuration: %s", cfg.to_dict())
    logger.info("ALPHA_GRID = %s", ALPHA_GRID)
    logger.info("No fixed random seed is used.")
    logger.info("ENTRY_DIST = %s", entry_dist)
    logger.info("NP_GRID = %s", NP_GRID)
    logger.info("BULK_SPECTRA = %s", BULK_SPECTRA)
    logger.info("MODEL_TYPES = %s", MODEL_TYPES)
    logger.info("Edge convention: use c=p/n for all model types.")
    logger.info(
        "Coverage method: updated held-out bootstrap center; "
        "first B-1 replicates estimate E0/variance and the last replicate "
        "gives the center. Multiple alpha values reuse the same "
        "bootstrap replicates; only z_{alpha/2} changes."
    )
    logger.info("Number of combos: %d", len(combos))

    edge_info = precompute_edge_info(logger, outdir)

    all_details = []
    all_summaries = []

    for combo in combos:
        details, summary = run_combo(
            combo,
            cfg,
            logger,
            outdir,
            edge_info,
        )
        all_details.append(details)
        all_summaries.append(summary)

        pd.concat(all_details, ignore_index=True).to_csv(
            outdir / "combined_details_partial.csv",
            index=False,
        )
        pd.concat(all_summaries, ignore_index=True).to_csv(
            outdir / "combined_summary_partial.csv",
            index=False,
        )

    combined_details = pd.concat(all_details, ignore_index=True)
    combined_summary = pd.concat(all_summaries, ignore_index=True)

    combined_details.to_csv(
        outdir / "combined_details.csv",
        index=False,
    )
    combined_summary.to_csv(
        outdir / "combined_summary.csv",
        index=False,
    )
    save_combined_plots(outdir, combined_summary, cfg)

    logger.info("Saved combined outputs for %s.", entry_dist)
    logger.info(
        "Distribution run finished successfully: %s",
        entry_dist,
    )

    return combined_summary


def main():
    base_cfg = CFG
    root_outdir = Path(base_cfg.outdir)
    root_outdir.mkdir(parents=True, exist_ok=True)

    distribution_summaries = []

    # Deliberately sequential: Gaussian finishes before t10 starts.
    for run_number, entry_dist in enumerate(ENTRY_DISTS, start=1):
        print(
            "Starting distribution {}/{}: {}".format(
                run_number,
                len(ENTRY_DISTS),
                entry_dist,
            ),
            flush=True,
        )
        combined_summary = run_distribution(
            entry_dist,
            base_cfg,
        ).copy()
        combined_summary["distribution_run_order"] = int(run_number)
        distribution_summaries.append(combined_summary)
        print(
            "Finished distribution {}/{}: {}".format(
                run_number,
                len(ENTRY_DISTS),
                entry_dist,
            ),
            flush=True,
        )

    pd.concat(
        distribution_summaries,
        ignore_index=True,
    ).to_csv(
        root_outdir / "combined_summary_all_distributions.csv",
        index=False,
    )

    print(
        "All distribution runs finished successfully in order: {}".format(
            " -> ".join(ENTRY_DISTS)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
