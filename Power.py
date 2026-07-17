#!/usr/bin/env python3
"""
Near-threshold power experiment to reproduce Fig 2 and Fig S1. 

Population model
----------------
For each bulk spectrum H, compute the right edge E and threshold

    theta_c = -1 / b,

where b solves f'(b)=0 for

    f(x) = -1/x + (p/n) mean_j 1/(x + sigma_j^{-1}).

Then simulate

    Sigma = diag(7, theta_c + epsilon, H),

with epsilon in {0.1,0.2,...,1.0,...,2.0}.  The statistic is the
second sample eigenvalue mu_2 = lambda_2.

For each bootstrap N, use the updated held-out multiplier-bootstrap
framework.  The first B-1 bootstrap replicates estimate the bootstrap
center E0 and variance, while the last bootstrap replicate is used as a
held-out randomized center.  The corrected center is

    mu_2_tilde = mu_2^*(B) - E0_hat + mu_2,

and the interval is

    mu_2_tilde +/- z_{0.995} * sd(mu_2^*(1),...,mu_2^*(B-1)).

The power event is that E is not contained in this corrected interval.

Saved rates
-----------
edge_containment_rate = P(E in CI)
power_reject_edge_rate = P(E not in CI) = 1 - edge_containment_rate

Defaults
--------
Gaussian entries first, then standardized t_10 entries.
The default test level is alpha = 0.01, so the two-sided normal
critical value is z_{0.995}.
(n,p) = (500,200), (500,750), (750,500).
Only the primary bootstrap N is run for each dimension:
    (500,200) : N = 4
    (500,750) : N = 15
    (750,500) : N = 9

No neighboring N grid is used in this version.

Run
---
mkdir -p logs
nohup python3 Fig2.py \
  >> logs/Fig2.log 2>&1 &

Monitor
-------
tail -f logs/Fig2.log
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.stats import norm


ENTRY_DISTS = ("gaussian", "t10")
T_DF = 10
FIRST_SPIKE = 7.0

PRIMARY_N = {
    (500, 200): 4,
    (500, 750): 15,
    (750, 500): 9,
}

# Only run the primary N for each dimension.  This replaces the old
NP_N_GRID = {dim: (int(N),) for dim, N in PRIMARY_N.items()}

BULK_SPECTRA = [
    # "identity",
    "identity_0p9",
    "uniform_0p75_1p25",
    "two_mass_0p75_1p25",
]

EPSILON_GRID = [round(x, 1) for x in np.arange(0.1, 2.0 + 1e-12, 0.1)]


class RunConfig(object):
    def __init__(
        self,
        outdir="Fig2_results",
        n_reps=500,
        b_boot=2000,
        alpha=0.01,
        n_jobs=50,
        mp_start_method="fork",
        chunksize=1,
        save_details=True,
    ):
        self.outdir = str(outdir)
        self.n_reps = int(n_reps)
        self.b_boot = int(b_boot)
        self.alpha = float(alpha)
        self.n_jobs = int(n_jobs)
        self.mp_start_method = str(mp_start_method)
        self.chunksize = int(chunksize)
        self.save_details = bool(save_details)

    @property
    def z_alpha(self):
        return float(norm.ppf(1.0 - self.alpha / 2.0))

    @property
    def nominal_level(self):
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
            "save_details": self.save_details,
        }


CFG = RunConfig()


def eps_label(epsilon):
    return "eps" + str(epsilon).replace(".", "p")


class PowerBlock(object):
    def __init__(self, n, p, bulk_spectrum, epsilon):
        self.n = int(n)
        self.p = int(p)
        self.bulk_spectrum = str(bulk_spectrum)
        self.epsilon = float(epsilon)

    @property
    def p_bulk(self):
        return self.p - 2

    @property
    def phi_edge(self):
        return float(self.p / self.n)

    @property
    def name(self):
        return "n{}_p{}_{}_{}".format(
            self.n, self.p, self.bulk_spectrum, eps_label(self.epsilon)
        )


class PowerCombo(object):
    def __init__(self, n, p, bulk_spectrum, epsilon, N):
        self.n = int(n)
        self.p = int(p)
        self.bulk_spectrum = str(bulk_spectrum)
        self.epsilon = float(epsilon)
        self.N = int(N)

    @property
    def p_bulk(self):
        return self.p - 2

    @property
    def phi_edge(self):
        return float(self.p / self.n)

    @property
    def target_index(self):
        return 2

    @property
    def k_needed(self):
        return 2

    @property
    def name(self):
        return "n{}_p{}_{}_{}_N{}".format(
            self.n, self.p, self.bulk_spectrum, eps_label(self.epsilon), self.N
        )

    def to_dict(self):
        return {
            "n": self.n,
            "p": self.p,
            "bulk_spectrum": self.bulk_spectrum,
            "epsilon": self.epsilon,
            "N": self.N,
        }


def make_blocks():
    blocks = []
    for n, p in NP_N_GRID.keys():
        for spectrum in BULK_SPECTRA:
            for epsilon in EPSILON_GRID:
                blocks.append(PowerBlock(n, p, spectrum, epsilon))
    return blocks


def setup_logger(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("near_threshold_two_spike_power")
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


def population_bulk_eigs_by_name(bulk_spectrum, p_bulk):
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

    raise ValueError("Unknown bulk_spectrum: {}".format(bulk_spectrum))


def f_edge(x, eigs, phi):
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
                    phi, sigma_max
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

    for n, p in NP_N_GRID.keys():
        phi = float(p / n)

        for spectrum in BULK_SPECTRA:
            # Edge and threshold use the p-coordinate reference spectrum.
            ref_eigs = population_bulk_eigs_by_name(spectrum, p)
            edge, bcrit, theta_threshold, tw_b_coeff = deformed_mp_upper_edge(
                ref_eigs, phi
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
                "tw_finite_n_scale": float(tw_b_coeff * (n ** (-2.0 / 3.0))),
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


def full_population_eigs(block, theta2):
    bulk_eigs = population_bulk_eigs_by_name(block.bulk_spectrum, block.p_bulk)
    return np.concatenate([
        np.array([FIRST_SPIKE, float(theta2)], dtype=float),
        bulk_eigs,
    ])


def sample_entries(n, p, rng, entry_dist):
    entry_dist = str(entry_dist)

    if entry_dist == "gaussian":
        return rng.standard_normal((n, p))

    if entry_dist in ("t10", "t_10"):
        # Standardize Student t_10 entries to variance one.
        # Var(t_nu) = nu / (nu - 2), so divide by sqrt(nu/(nu-2)).
        scale = np.sqrt(T_DF / (T_DF - 2.0))
        return rng.standard_t(df=T_DF, size=(n, p)) / scale

    raise ValueError("Unknown entry_dist: {}".format(entry_dist))


def generate_data(n, p, full_eigs, rng, entry_dist):
    Z = sample_entries(n, p, rng, entry_dist)
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


def run_one_rep(task):
    (
        combo_dict,
        rep,
        edge,
        critical_b,
        theta_threshold,
        theta2,
        full_eigs,
        cfg_dict,
        entry_dist,
    ) = task

    combo = PowerCombo(**combo_dict)
    cfg = RunConfig(**cfg_dict)

    rng = np.random.default_rng(None)

    X = generate_data(combo.n, combo.p, full_eigs, rng, entry_dist)
    vals = top_k_sample_cov_eigs(X, combo.k_needed)

    lambda1 = float(vals[0])
    mu2 = float(vals[1])
    gap12 = float(lambda1 - mu2)

    boot_mu2 = np.empty(cfg.b_boot, dtype=float)
    boot_gap12 = np.empty(cfg.b_boot, dtype=float)

    for b in range(cfg.b_boot):
        w = draw_chisq_weights(combo.n, combo.N, rng)
        boot_vals = top_k_weighted_cov_eigs(X, w, combo.k_needed)
        boot_mu2[b] = boot_vals[1]
        boot_gap12[b] = boot_vals[0] - boot_vals[1]

    if cfg.b_boot < 3:
        raise ValueError("The updated framework requires b_boot >= 3.")

    # ------------------------------------------------------------
    # Updated held-out bootstrap framework.
    # ------------------------------------------------------------
    # First B-1 bootstrap replicates estimate the conditional
    # bootstrap center E0 and the bootstrap standard error.  The
    # last replicate is a held-out randomized center.
    #
    # E_hat = mu_2(Q) estimates E under the null.  For this power
    # experiment, mu_2(Q) is intentionally a spiked eigenvalue under
    # the alternative, and the power event is E not in the interval.
    # ------------------------------------------------------------
    boot_train = boot_mu2[:-1]
    boot_holdout_mu2 = float(boot_mu2[-1])

    E_hat = float(mu2)
    E0_hat = float(np.mean(boot_train))
    boot_sd_train = float(np.std(boot_train, ddof=1))
    v0_hat = float(combo.n * boot_sd_train ** 2)

    bias_hat = float(E0_hat - E_hat)
    center_corrected = float(boot_holdout_mu2 - bias_hat)
    center_corrected_minus_edge = float(center_corrected - edge)

    ci_lower = float(center_corrected - cfg.z_alpha * boot_sd_train)
    ci_upper = float(center_corrected + cfg.z_alpha * boot_sd_train)
    ci_width = float(ci_upper - ci_lower)

    contain_edge = float(ci_lower <= edge <= ci_upper)
    reject_edge = float(1.0 - contain_edge)

    # Old interval, kept only for diagnostics.  The main columns
    # ci_lower, ci_upper, contain_edge, and reject_edge use the new
    # procedure above.
    old_boot_mean_mu2 = float(np.mean(boot_mu2))
    old_boot_sd_mu2 = float(np.std(boot_mu2, ddof=1))
    old_ci_lower = float(mu2 - cfg.z_alpha * old_boot_sd_mu2)
    old_ci_upper = float(mu2 + cfg.z_alpha * old_boot_sd_mu2)
    old_ci_width = float(old_ci_upper - old_ci_lower)
    old_contain_edge = float(old_ci_lower <= edge <= old_ci_upper)
    old_reject_edge = float(1.0 - old_contain_edge)

    return {
        "combo_name": combo.name,
        "entry_dist": str(entry_dist),
        "n": int(combo.n),
        "p": int(combo.p),
        "p_bulk": int(combo.p_bulk),
        "phi_edge": float(combo.phi_edge),
        "bulk_spectrum": combo.bulk_spectrum,
        "epsilon": float(combo.epsilon),
        "N": int(combo.N),
        "primary_N": int(PRIMARY_N[(combo.n, combo.p)]),
        "is_primary_N": bool(combo.N == PRIMARY_N[(combo.n, combo.p)]),
        "rep": int(rep),

        "first_spike": float(FIRST_SPIKE),
        "theta2": float(theta2),
        "theta_threshold": float(theta_threshold),
        "theta2_minus_threshold": float(theta2 - theta_threshold),
        "critical_b": float(critical_b),
        "edge": float(edge),

        "lambda1_hat": lambda1,
        "mu2_hat": mu2,
        "mu2_minus_edge": float(mu2 - edge),
        "gap12": gap12,

        "required_right_halfwidth": float(max(edge - center_corrected, 0.0)),
        "old_required_right_halfwidth": float(max(edge - mu2, 0.0)),

        "E_hat": E_hat,
        "E0_hat": E0_hat,
        "bias_hat": bias_hat,
        "boot_holdout_mu2": boot_holdout_mu2,
        "center_corrected": center_corrected,
        "center_corrected_minus_edge": center_corrected_minus_edge,
        "boot_sd_train": boot_sd_train,
        "v0_hat": v0_hat,

        # Keep these names as the main new-procedure outputs so that
        # existing summary code continues to work.
        "boot_mean_mu2": E0_hat,
        "boot_sd_mu2": boot_sd_train,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_width,

        "contain_edge": contain_edge,
        "reject_edge": reject_edge,

        # Old method diagnostics only.
        "old_boot_mean_mu2": old_boot_mean_mu2,
        "old_boot_sd_mu2": old_boot_sd_mu2,
        "old_ci_lower": old_ci_lower,
        "old_ci_upper": old_ci_upper,
        "old_ci_width": old_ci_width,
        "old_contain_edge": old_contain_edge,
        "old_reject_edge": old_reject_edge,

        "boot_gap12_mean": float(np.mean(boot_gap12)),
        "boot_gap12_q01": float(np.quantile(boot_gap12, 0.01)),
        "boot_gap12_q05": float(np.quantile(boot_gap12, 0.05)),
        "boot_gap12_q50": float(np.quantile(boot_gap12, 0.50)),
        "boot_gap12_min": float(np.min(boot_gap12)),

        "B": int(cfg.b_boot),
        "alpha": float(cfg.alpha),
        "nominal_level": float(cfg.nominal_level),
    }


def summarize_details(details):
    rows = []
    group_cols = ["n", "p", "bulk_spectrum", "epsilon", "N"]

    for _, sub in details.groupby(group_cols):
        contain_rate = float(sub["contain_edge"].mean())
        reject_rate = float(sub["reject_edge"].mean())

        rows.append({
            "combo_name": str(sub["combo_name"].iloc[0]),
            "entry_dist": str(sub["entry_dist"].iloc[0]),
            "n": int(sub["n"].iloc[0]),
            "p": int(sub["p"].iloc[0]),
            "p_bulk": int(sub["p_bulk"].iloc[0]),
            "phi_edge": float(sub["phi_edge"].iloc[0]),
            "bulk_spectrum": str(sub["bulk_spectrum"].iloc[0]),
            "epsilon": float(sub["epsilon"].iloc[0]),
            "N": int(sub["N"].iloc[0]),
            "primary_N": int(sub["primary_N"].iloc[0]),
            "is_primary_N": bool(sub["is_primary_N"].iloc[0]),

            "first_spike": float(sub["first_spike"].iloc[0]),
            "theta2": float(sub["theta2"].iloc[0]),
            "theta_threshold": float(sub["theta_threshold"].iloc[0]),
            "theta2_minus_threshold": float(
                sub["theta2_minus_threshold"].iloc[0]
            ),
            "critical_b": float(sub["critical_b"].iloc[0]),
            "edge": float(sub["edge"].iloc[0]),

            "edge_containment_rate": contain_rate,
            "power_reject_edge_rate": reject_rate,
            "mc_se_containment": float(
                np.sqrt(contain_rate * (1.0 - contain_rate) / len(sub))
            ),
            "mc_se_power": float(
                np.sqrt(reject_rate * (1.0 - reject_rate) / len(sub))
            ),

            "mean_mu2": float(sub["mu2_hat"].mean()),
            "sd_mu2": float(sub["mu2_hat"].std(ddof=1)),
            "mean_mu2_minus_edge": float(sub["mu2_minus_edge"].mean()),
            "sd_mu2_minus_edge": float(sub["mu2_minus_edge"].std(ddof=1)),
            "q01_mu2_minus_edge": float(sub["mu2_minus_edge"].quantile(0.01)),
            "q05_mu2_minus_edge": float(sub["mu2_minus_edge"].quantile(0.05)),
            "q10_mu2_minus_edge": float(sub["mu2_minus_edge"].quantile(0.10)),
            "q50_mu2_minus_edge": float(sub["mu2_minus_edge"].quantile(0.50)),
            "q90_mu2_minus_edge": float(sub["mu2_minus_edge"].quantile(0.90)),
            "q95_mu2_minus_edge": float(sub["mu2_minus_edge"].quantile(0.95)),
            "q99_mu2_minus_edge": float(sub["mu2_minus_edge"].quantile(0.99)),

            "mean_required_right_halfwidth": float(
                sub["required_right_halfwidth"].mean()
            ),
            "q90_required_right_halfwidth": float(
                sub["required_right_halfwidth"].quantile(0.90)
            ),
            "q95_required_right_halfwidth": float(
                sub["required_right_halfwidth"].quantile(0.95)
            ),

            "mean_E_hat": float(sub["E_hat"].mean()),
            "mean_E0_hat": float(sub["E0_hat"].mean()),
            "mean_bias_hat": float(sub["bias_hat"].mean()),
            "mean_boot_holdout_mu2": float(sub["boot_holdout_mu2"].mean()),
            "mean_center_corrected": float(sub["center_corrected"].mean()),
            "sd_center_corrected": float(sub["center_corrected"].std(ddof=1)),
            "mean_center_corrected_minus_edge": float(
                sub["center_corrected_minus_edge"].mean()
            ),
            "q05_center_corrected_minus_edge": float(
                sub["center_corrected_minus_edge"].quantile(0.05)
            ),
            "q50_center_corrected_minus_edge": float(
                sub["center_corrected_minus_edge"].quantile(0.50)
            ),
            "mean_boot_sd_mu2": float(sub["boot_sd_mu2"].mean()),
            "sd_boot_sd_mu2": float(sub["boot_sd_mu2"].std(ddof=1)),
            "mean_v0_hat": float(sub["v0_hat"].mean()),
            "mean_ci_width": float(sub["ci_width"].mean()),
            "sd_ci_width": float(sub["ci_width"].std(ddof=1)),

            "old_edge_containment_rate": float(sub["old_contain_edge"].mean()),
            "old_power_reject_edge_rate": float(sub["old_reject_edge"].mean()),
            "old_mean_boot_sd_mu2": float(sub["old_boot_sd_mu2"].mean()),
            "old_mean_ci_width": float(sub["old_ci_width"].mean()),

            "mean_gap12": float(sub["gap12"].mean()),
            "sd_gap12": float(sub["gap12"].std(ddof=1)),
            "q01_gap12": float(sub["gap12"].quantile(0.01)),
            "q05_gap12": float(sub["gap12"].quantile(0.05)),
            "q10_gap12": float(sub["gap12"].quantile(0.10)),
            "q50_gap12": float(sub["gap12"].quantile(0.50)),

            "mean_boot_gap12_q05": float(sub["boot_gap12_q05"].mean()),
            "mean_boot_gap12_min": float(sub["boot_gap12_min"].mean()),

            "n_reps": int(len(sub)),
            "B": int(sub["B"].iloc[0]),
            "alpha": float(sub["alpha"].iloc[0]),
            "nominal_level": float(sub["nominal_level"].iloc[0]),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["n", "p", "bulk_spectrum", "epsilon", "N"])
        .reset_index(drop=True)
    )


def add_smooth_columns(summary, window=5):
    out = []

    for _, sub in summary.groupby(["n", "p", "bulk_spectrum", "epsilon"]):
        ss = sub.sort_values("N").copy()
        ss["edge_containment_smooth_5"] = (
            ss["edge_containment_rate"]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
        ss["power_reject_edge_smooth_5"] = (
            ss["power_reject_edge_rate"]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
        ss["mean_ci_width_smooth_5"] = (
            ss["mean_ci_width"]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
        ss["mean_mu2_minus_edge_smooth_5"] = (
            ss["mean_mu2_minus_edge"]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
        ss["mean_center_corrected_minus_edge_smooth_5"] = (
            ss["mean_center_corrected_minus_edge"]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
        out.append(ss)

    return pd.concat(out, ignore_index=True)


def make_primary_summary(summary, outdir):
    primary = summary[summary["is_primary_N"]].copy()

    sort_cols = []
    if "entry_dist" in primary.columns:
        sort_cols.append("entry_dist")
    sort_cols.extend(["n", "p", "bulk_spectrum", "epsilon"])

    index_cols = []
    if "entry_dist" in primary.columns:
        index_cols.append("entry_dist")
    index_cols.extend(["n", "p", "bulk_spectrum"])

    primary = primary.sort_values(sort_cols)
    primary.to_csv(outdir / "primary_N_summary.csv", index=False)

    pivot_contain = primary.pivot_table(
        index=index_cols,
        columns="epsilon",
        values="edge_containment_rate",
    ).reset_index()
    pivot_contain.to_csv(outdir / "primary_N_edge_containment_pivot.csv", index=False)

    pivot_power = primary.pivot_table(
        index=index_cols,
        columns="epsilon",
        values="power_reject_edge_rate",
    ).reset_index()
    pivot_power.to_csv(outdir / "primary_N_power_reject_pivot.csv", index=False)

    return primary


def read_completed_Ns(block_dir):
    path = block_dir / "summary_partial.csv"
    if not path.exists():
        path = block_dir / "summary.csv"
    if not path.exists():
        return set()

    try:
        df = pd.read_csv(path)
    except Exception:
        return set()

    if "N" not in df.columns:
        return set()

    return set(int(x) for x in df["N"].dropna().unique())


def run_block(block, cfg, logger, outdir, edge_info, entry_dist):
    block_dir = outdir / block.name
    block_dir.mkdir(parents=True, exist_ok=True)

    info = edge_info[(block.n, block.p, block.bulk_spectrum)]
    edge = float(info["edge"])
    critical_b = float(info["critical_b"])
    theta_threshold = float(info["theta_threshold"])
    theta2 = float(theta_threshold + block.epsilon)

    full_eigs = full_population_eigs(block, theta2)
    n_grid = NP_N_GRID[(block.n, block.p)]

    block_info = pd.DataFrame([{
        "block_name": block.name,
        "entry_dist": str(entry_dist),
        "n": block.n,
        "p": block.p,
        "p_bulk": block.p_bulk,
        "phi_edge": block.phi_edge,
        "bulk_spectrum": block.bulk_spectrum,
        "epsilon": block.epsilon,
        "first_spike": FIRST_SPIKE,
        "theta2": theta2,
        "theta_threshold": theta_threshold,
        "theta2_minus_threshold": theta2 - theta_threshold,
        "critical_b": critical_b,
        "edge": edge,
        "tw_b_coeff": float(info["tw_b_coeff"]),
        "edge_uses": "p/n and p-coordinate reference spectrum",
        "p_edge_reference": int(info["p_edge_reference"]),
        "N_values_run": ",".join(str(x) for x in n_grid),
        "primary_N": PRIMARY_N[(block.n, block.p)],
        "n_reps": cfg.n_reps,
        "B": cfg.b_boot,
        "alpha": cfg.alpha,
        "nominal_level": cfg.nominal_level,
    }])
    block_info.to_csv(block_dir / "block_info.csv", index=False)

    completed = read_completed_Ns(block_dir)
    # Completed N values are silently skipped to keep one result log per setup.

    all_detail_frames = []
    all_summary_frames = []

    partial_details_path = block_dir / "details_partial.csv"
    partial_summary_path = block_dir / "summary_partial.csv"

    if partial_details_path.exists():
        try:
            all_detail_frames.append(pd.read_csv(partial_details_path))
        except Exception:
            pass

    if partial_summary_path.exists():
        try:
            all_summary_frames.append(pd.read_csv(partial_summary_path))
        except Exception:
            pass

    ctx = mp.get_context(cfg.mp_start_method)
    cfg_dict = cfg.to_dict()

    for N in n_grid:
        if int(N) in completed:
            continue

        combo = PowerCombo(
            n=block.n,
            p=block.p,
            bulk_spectrum=block.bulk_spectrum,
            epsilon=block.epsilon,
            N=int(N),
        )

        combo_dict = combo.to_dict()
        tasks = [
            (
                combo_dict,
                int(rep),
                float(edge),
                float(critical_b),
                float(theta_threshold),
                float(theta2),
                full_eigs,
                cfg_dict,
                str(entry_dist),
            )
            for rep in range(1, cfg.n_reps + 1)
        ]

        rows_N = []
        with ctx.Pool(processes=cfg.n_jobs) as pool:
            for row in pool.imap_unordered(run_one_rep, tasks, chunksize=cfg.chunksize):
                rows_N.append(row)

        df_N = pd.DataFrame(rows_N).sort_values("rep").reset_index(drop=True)
        summary_N = summarize_details(df_N)

        all_detail_frames.append(df_N)
        all_summary_frames.append(summary_N)

        details_partial = pd.concat(all_detail_frames, ignore_index=True)
        summary_partial = (
            pd.concat(all_summary_frames, ignore_index=True)
            .drop_duplicates(subset=["N"], keep="last")
            .sort_values("N")
            .reset_index(drop=True)
        )

        if cfg.save_details:
            details_partial.to_csv(block_dir / "details_partial.csv", index=False)
        summary_partial.to_csv(block_dir / "summary_partial.csv", index=False)

        s = summary_N.iloc[0]
        logger.info(
            "result | dist=%s | n=%d p=%d | spectrum=%s | eps=%.3f | "
            "N=%d | alpha=%.3f | contain_edge=%.4f | power_reject=%.4f",
            str(entry_dist),
            int(block.n),
            int(block.p),
            str(block.bulk_spectrum),
            float(block.epsilon),
            int(N),
            float(cfg.alpha),
            float(s["edge_containment_rate"]),
            float(s["power_reject_edge_rate"]),
        )

    if all_detail_frames:
        details = (
            pd.concat(all_detail_frames, ignore_index=True)
            .drop_duplicates(subset=["N", "rep"], keep="last")
            .sort_values(["N", "rep"])
            .reset_index(drop=True)
        )
    else:
        details = pd.DataFrame()

    if all_summary_frames:
        summary = (
            pd.concat(all_summary_frames, ignore_index=True)
            .drop_duplicates(subset=["N"], keep="last")
            .sort_values("N")
            .reset_index(drop=True)
        )
    else:
        summary = pd.DataFrame()

    if cfg.save_details and not details.empty:
        details.to_csv(block_dir / "details.csv", index=False)
    if not summary.empty:
        summary.to_csv(block_dir / "summary.csv", index=False)

    return details, summary


def run_for_entry_dist(entry_dist, cfg, logger, outdir):
    dist_outdir = outdir / str(entry_dist)
    dist_outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Entry distribution started: %s", str(entry_dist))
    logger.info("Output subdirectory: %s", str(dist_outdir))

    edge_info = precompute_edge_info(logger, dist_outdir)

    all_details = []
    all_summaries = []

    for block in make_blocks():
        details, summary = run_block(
            block,
            cfg,
            logger,
            dist_outdir,
            edge_info,
            entry_dist,
        )

        if not details.empty:
            all_details.append(details)
        if not summary.empty:
            all_summaries.append(summary)

        if all_summaries:
            combined_summary_partial = pd.concat(all_summaries, ignore_index=True)
            combined_summary_partial = add_smooth_columns(combined_summary_partial)
            combined_summary_partial.to_csv(
                dist_outdir / "combined_summary_partial.csv",
                index=False,
            )
            make_primary_summary(combined_summary_partial, dist_outdir)

        if cfg.save_details and all_details:
            pd.concat(all_details, ignore_index=True).to_csv(
                dist_outdir / "combined_details_partial.csv",
                index=False,
            )

    combined_details = pd.DataFrame()
    combined_summary = pd.DataFrame()

    if all_details:
        combined_details = pd.concat(all_details, ignore_index=True)
        combined_details.to_csv(dist_outdir / "combined_details.csv", index=False)

    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_summary = add_smooth_columns(combined_summary)
        combined_summary.to_csv(dist_outdir / "combined_summary.csv", index=False)
        make_primary_summary(combined_summary, dist_outdir)

    logger.info("Entry distribution finished: %s", str(entry_dist))
    return combined_details, combined_summary


def main():
    cfg = CFG
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(outdir)

    logger.info("Run started")
    logger.info("Configuration: %s", cfg.to_dict())
    logger.info("No fixed random seed is used.")
    logger.info("ENTRY_DISTS = %s", ENTRY_DISTS)
    logger.info("T_DF = %d", int(T_DF))
    logger.info("FIRST_SPIKE = %.4f", FIRST_SPIKE)
    logger.info("N_BY_DIM_PRIMARY_ONLY = %s", NP_N_GRID)
    logger.info("PRIMARY_N = %s", PRIMARY_N)
    logger.info("BULK_SPECTRA = %s", BULK_SPECTRA)
    logger.info("EPSILON_GRID = %s", EPSILON_GRID)
    logger.info("Number of blocks per entry distribution: %d", len(make_blocks()))
    logger.info("Method = updated held-out bias-corrected bootstrap.")
    logger.info("No matplotlib is used.")
    logger.info("Run order: finish gaussian first, then t_10.")

    combined_detail_frames = []
    combined_summary_frames = []

    for entry_dist in ENTRY_DISTS:
        details, summary = run_for_entry_dist(entry_dist, cfg, logger, outdir)

        if not details.empty:
            combined_detail_frames.append(details)
        if not summary.empty:
            combined_summary_frames.append(summary)

        if combined_summary_frames:
            combined_summary_partial = pd.concat(
                combined_summary_frames,
                ignore_index=True,
            )
            combined_summary_partial = add_smooth_columns(combined_summary_partial)
            combined_summary_partial.to_csv(
                outdir / "combined_summary_partial.csv",
                index=False,
            )
            make_primary_summary(combined_summary_partial, outdir)

        if cfg.save_details and combined_detail_frames:
            pd.concat(combined_detail_frames, ignore_index=True).to_csv(
                outdir / "combined_details_partial.csv",
                index=False,
            )

    if combined_detail_frames:
        combined_details = pd.concat(combined_detail_frames, ignore_index=True)
        combined_details.to_csv(outdir / "combined_details.csv", index=False)

    if combined_summary_frames:
        combined_summary = pd.concat(combined_summary_frames, ignore_index=True)
        combined_summary = add_smooth_columns(combined_summary)
        combined_summary.to_csv(outdir / "combined_summary.csv", index=False)
        make_primary_summary(combined_summary, outdir)

    logger.info("Saved combined gaussian+t_10 outputs.")
    logger.info("Run finished successfully.")


if __name__ == "__main__":
    main()
