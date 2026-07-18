#!/usr/bin/env python3
"""
Calibration of the chi-square multiplier parameter N.

The implementation follows Section 2.3 of the manuscript.

For each dimension pair (p, n):

1. Generate R_CAL independent Gaussian identity-reference samples.
2. Search the candidate grid

       G = {floor(n^(1/3)/5) + 1, ..., floor(5 n^(1/3))}.

3. For every candidate N and calibration sample, construct the
   bias-corrected multiplier-bootstrap confidence interval using r0 = 1:

       xi_i^2 ~ chi-square_N / N,

       delta = mu_1 - mean(lambda_1,1, ..., lambda_1,B),

       center = lambda_1,B+1 + delta,

       s^2 = (1/B) sum_{k=1}^B
             (lambda_1,k - mean(lambda_1,1, ..., lambda_1,B))^2.

4. At each alpha in ALPHA_GRID, estimate the empirical non-coverage
   probability.

5. Select the smallest N minimizing

       sum_alpha |noncoverage(N, alpha) - alpha|.

The reference edge is

       E_I = (1 + sqrt(p / n))^2.

Default dimension pairs are the three pairs used in the manuscript:

       (p, n) = (200, 500), (500, 750), (750, 500).

Environment-variable overrides
------------------------------
R_CAL=1000
B_BOOT=2000
N_JOBS=50
CHUNKSIZE=1
MP_START_METHOD=fork
BASE_SEED=20260718
OUTDIR=calibration_results
DIMENSIONS=200:500,500:750,750:500
ALPHA_GRID=0.05,0.10
SAVE_DETAILS=1
REUSE_RESULTS=1

Quick test
----------
R_CAL=10 B_BOOT=50 N_JOBS=4 \
python3 calibration.py

Full run
--------
mkdir -p logs
nohup python3 calibration.py \
  >> logs/calibration.log 2>&1 &

Monitor
-------
tail -f logs/calibration.log
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# Limit BLAS threading before importing NumPy/SciPy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import norm


Array = np.ndarray


# ============================================================
# Configuration
# ============================================================

# run this if only check the three dimensions used in Section 4
DEFAULT_DIMENSIONS: Tuple[Tuple[int, int], ...] = (
    (200, 500),
    (500, 750),
    (750, 500),
)

# run this if also check real data analysis in Section 5
# DEFAULT_DIMENSIONS: Tuple[Tuple[int, int], ...] = (
#     (200, 500),
#     (500, 750),
#     (750, 500),
#     (200, 400),
#     (1500, 400),
# )


def parse_dimensions(value: str) -> Tuple[Tuple[int, int], ...]:
    """Parse DIMENSIONS as comma-separated p:n pairs."""
    value = value.strip()
    if not value:
        return DEFAULT_DIMENSIONS

    pairs: List[Tuple[int, int]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(
                "DIMENSIONS must use comma-separated p:n pairs, "
                "for example 200:500,500:750."
            )
        p, n = int(parts[0]), int(parts[1])
        if p <= 0 or n <= 0:
            raise ValueError("All dimensions must be positive.")
        pairs.append((p, n))

    if not pairs:
        raise ValueError("DIMENSIONS produced an empty dimension list.")
    return tuple(pairs)


def parse_float_grid(value: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    if not values:
        raise ValueError("ALPHA_GRID cannot be empty.")
    for alpha in values:
        if not (0.0 < alpha < 1.0):
            raise ValueError("Every alpha must lie strictly between 0 and 1.")
    return values


@dataclass(frozen=True)
class RunConfig:
    dimensions: Tuple[Tuple[int, int], ...]
    alpha_grid: Tuple[float, ...]
    r_cal: int
    b_boot: int
    n_jobs: int
    chunksize: int
    mp_start_method: str
    base_seed: int
    outdir: str
    save_details: bool
    reuse_results: bool

    def validate(self) -> None:
        if self.r_cal < 1:
            raise ValueError("R_CAL must be at least 1.")
        if self.b_boot < 2:
            raise ValueError("B_BOOT must be at least 2.")
        if self.n_jobs < 1:
            raise ValueError("N_JOBS must be at least 1.")
        if self.chunksize < 1:
            raise ValueError("CHUNKSIZE must be at least 1.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": [list(x) for x in self.dimensions],
            "alpha_grid": list(self.alpha_grid),
            "r_cal": int(self.r_cal),
            "b_boot": int(self.b_boot),
            "n_jobs": int(self.n_jobs),
            "chunksize": int(self.chunksize),
            "mp_start_method": str(self.mp_start_method),
            "base_seed": int(self.base_seed),
            "outdir": str(self.outdir),
            "save_details": bool(self.save_details),
            "reuse_results": bool(self.reuse_results),
        }


CFG = RunConfig(
    dimensions=parse_dimensions(os.environ.get("DIMENSIONS", "")),
    alpha_grid=parse_float_grid(
        os.environ.get("ALPHA_GRID", "0.05,0.10")
    ),
    r_cal=int(os.environ.get("R_CAL", "1000")),
    b_boot=int(os.environ.get("B_BOOT", "2000")),
    n_jobs=int(os.environ.get("N_JOBS", "50")),
    chunksize=int(os.environ.get("CHUNKSIZE", "1")),
    mp_start_method=os.environ.get("MP_START_METHOD", "fork"),
    base_seed=int(os.environ.get("BASE_SEED", "20260718")),
    outdir=os.environ.get(
        "OUTDIR",
        "calibration_results",
    ),
    save_details=os.environ.get("SAVE_DETAILS", "1") != "0",
    reuse_results=os.environ.get("REUSE_RESULTS", "1") != "0",
)
CFG.validate()


# ============================================================
# General helpers
# ============================================================

def stable_seed(*parts: Any) -> int:
    text = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


def candidate_n_grid(n: int) -> Tuple[int, ...]:
    """Candidate grid in Section 2.3."""
    lower = int(np.floor((float(n) ** (1.0 / 3.0)) / 5.0)) + 1
    upper = int(np.floor(5.0 * (float(n) ** (1.0 / 3.0))))
    lower = max(lower, 1)
    if upper < lower:
        raise ValueError(
            f"Empty candidate grid for n={n}: lower={lower}, upper={upper}."
        )
    return tuple(range(lower, upper + 1))


def reference_edge(p: int, n: int) -> float:
    return float((1.0 + np.sqrt(float(p) / float(n))) ** 2)


def z_value(alpha: float) -> float:
    return float(norm.ppf(1.0 - float(alpha) / 2.0))


def setup_logger(outdir: Path) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("calibration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(outdir / "run.log", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ============================================================
# Eigenvalue routines
# ============================================================

def largest_eigenvalue_symmetric(matrix: Array) -> float:
    """Return the largest eigenvalue of a real symmetric matrix."""
    d = int(matrix.shape[0])
    value = eigh(
        matrix,
        subset_by_index=[d - 1, d - 1],
        eigvals_only=True,
        check_finite=False,
        overwrite_a=True,
    )
    return float(value[0])


def largest_sample_cov_eigenvalue(X: Array) -> float:
    """
    Largest eigenvalue of X.T @ X / n or X @ X.T / n.

    X is observations by variables and is already centered.
    """
    n, p = X.shape
    if p <= n:
        matrix = (X.T @ X) / float(n)
    else:
        matrix = (X @ X.T) / float(n)
    return largest_eigenvalue_symmetric(matrix)


def largest_weighted_cov_eigenvalue(X: Array, weights: Array) -> float:
    """
    Largest eigenvalue of

        (1/n) sum_i weights_i X_i X_i.T.
    """
    n, p = X.shape
    X_weighted = X * np.sqrt(weights)[:, None]

    if p <= n:
        matrix = (X_weighted.T @ X_weighted) / float(n)
    else:
        matrix = (X_weighted @ X_weighted.T) / float(n)

    return largest_eigenvalue_symmetric(matrix)


# ============================================================
# One calibration replication
# ============================================================

def run_one_replication(
    task: Tuple[int, int, int, int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run one Gaussian calibration replication for one fixed (p, n, N).

    Parameters
    ----------
    task
        (p, n, candidate_N, replication, config_dict)

    Returns
    -------
    list of dict
        One output row per alpha.
    """
    p, n, candidate_N, replication, config_dict = task

    alpha_grid = tuple(float(x) for x in config_dict["alpha_grid"])
    b_boot = int(config_dict["b_boot"])
    base_seed = int(config_dict["base_seed"])

    seed = stable_seed(
        base_seed,
        "calibration",
        int(p),
        int(n),
        int(candidate_N),
        int(replication),
    )
    rng = np.random.default_rng(seed)

    # Section 2.3 Gaussian identity reference model.
    X = rng.standard_normal(size=(int(n), int(p)))

    # Section 2.2 uses demeaned observations and divisor n.
    X = X - X.mean(axis=0, keepdims=True)

    mu_1 = largest_sample_cov_eigenvalue(X)

    # First B values estimate the center and scale; the final value is held out.
    bootstrap_values = np.empty(b_boot + 1, dtype=np.float64)
    for bootstrap_index in range(b_boot + 1):
        weights = (
            rng.chisquare(df=int(candidate_N), size=int(n))
            / float(candidate_N)
        )
        bootstrap_values[bootstrap_index] = (
            largest_weighted_cov_eigenvalue(X, weights)
        )

    training_values = bootstrap_values[:b_boot]
    held_out_value = float(bootstrap_values[b_boot])

    bootstrap_mean = float(np.mean(training_values))

    # The manuscript defines s^2 with divisor B.
    bootstrap_sd = float(
        np.sqrt(np.mean((training_values - bootstrap_mean) ** 2))
    )

    bias_correction = float(mu_1 - bootstrap_mean)
    interval_center = float(held_out_value + bias_correction)
    edge = reference_edge(p=int(p), n=int(n))

    rows: List[Dict[str, Any]] = []
    for alpha in alpha_grid:
        half_width = z_value(alpha) * bootstrap_sd
        lower = float(interval_center - half_width)
        upper = float(interval_center + half_width)
        covered = bool(lower <= edge <= upper)

        rows.append({
            "p": int(p),
            "n": int(n),
            "candidate_N": int(candidate_N),
            "replication": int(replication),
            "alpha": float(alpha),
            "target_noncoverage": float(alpha),
            "reference_edge": float(edge),
            "sample_lambda1": float(mu_1),
            "bootstrap_mean": float(bootstrap_mean),
            "bootstrap_sd": float(bootstrap_sd),
            "held_out_lambda1": float(held_out_value),
            "bias_correction": float(bias_correction),
            "interval_center": float(interval_center),
            "interval_lower": float(lower),
            "interval_upper": float(upper),
            "interval_length": float(upper - lower),
            "covered": int(covered),
            "noncovered": int(not covered),
            "seed": int(seed),
        })

    return rows


# ============================================================
# Aggregation and N selection
# ============================================================

def summarize_candidate_results(details: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        details.groupby(
            ["p", "n", "candidate_N", "alpha"],
            as_index=False,
        )
        .agg(
            r_cal=("replication", "nunique"),
            empirical_noncoverage=("noncovered", "mean"),
            empirical_coverage=("covered", "mean"),
            mean_interval_length=("interval_length", "mean"),
            mean_bootstrap_sd=("bootstrap_sd", "mean"),
            mean_interval_center=("interval_center", "mean"),
        )
    )

    grouped["target_noncoverage"] = grouped["alpha"]
    grouped["absolute_calibration_error"] = np.abs(
        grouped["empirical_noncoverage"]
        - grouped["target_noncoverage"]
    )

    return grouped.sort_values(
        ["p", "n", "candidate_N", "alpha"]
    ).reset_index(drop=True)


def select_calibrated_n(summary: pd.DataFrame) -> pd.DataFrame:
    objective = (
        summary.groupby(
            ["p", "n", "candidate_N"],
            as_index=False,
        )
        .agg(
            aggregate_calibration_error=(
                "absolute_calibration_error",
                "sum",
            ),
            maximum_calibration_error=(
                "absolute_calibration_error",
                "max",
            ),
            mean_calibration_error=(
                "absolute_calibration_error",
                "mean",
            ),
        )
        .sort_values(
            [
                "p",
                "n",
                "aggregate_calibration_error",
                "candidate_N",
            ]
        )
        .reset_index(drop=True)
    )

    selected_rows: List[pd.Series] = []
    for (_, _), block in objective.groupby(["p", "n"], sort=True):
        minimum = float(block["aggregate_calibration_error"].min())

        # Use a tiny numerical tolerance and then select the smallest N.
        tied = block[
            np.isclose(
                block["aggregate_calibration_error"],
                minimum,
                rtol=0.0,
                atol=1e-15,
            )
        ].sort_values("candidate_N")

        selected_rows.append(tied.iloc[0])

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected = selected.rename(columns={"candidate_N": "calibrated_N"})
    selected["selection_rule"] = (
        "minimum aggregate absolute noncoverage error; "
        "smallest N for ties"
    )

    return selected


# ============================================================
# Dimension-level execution
# ============================================================

def dimension_file_prefix(p: int, n: int) -> str:
    return f"calibration_p{int(p)}_n{int(n)}"


def run_dimension(
    p: int,
    n: int,
    cfg: RunConfig,
    logger: logging.Logger,
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_grid = candidate_n_grid(n)

    details_path = outdir / f"{dimension_file_prefix(p, n)}_details.csv"
    summary_path = outdir / f"{dimension_file_prefix(p, n)}_summary.csv"
    objective_path = outdir / f"{dimension_file_prefix(p, n)}_objective.csv"

    if (
        cfg.reuse_results
        and details_path.exists()
        and summary_path.exists()
        and objective_path.exists()
    ):
        logger.info(
            "Reusing completed calibration | p=%d | n=%d",
            p,
            n,
        )
        return (
            pd.read_csv(details_path),
            pd.read_csv(summary_path),
            pd.read_csv(objective_path),
        )

    logger.info(
        "Starting dimension | p=%d | n=%d | edge=%.10f | "
        "N grid=%d,...,%d (%d values) | R_CAL=%d | B_BOOT=%d",
        p,
        n,
        reference_edge(p, n),
        n_grid[0],
        n_grid[-1],
        len(n_grid),
        cfg.r_cal,
        cfg.b_boot,
    )

    config_dict = cfg.to_dict()
    tasks = [
        (
            int(p),
            int(n),
            int(candidate_N),
            int(replication),
            config_dict,
        )
        for candidate_N in n_grid
        for replication in range(cfg.r_cal)
    ]

    start_time = time.perf_counter()
    all_rows: List[Dict[str, Any]] = []

    if cfg.n_jobs == 1:
        iterator: Iterable[List[Dict[str, Any]]] = map(
            run_one_replication,
            tasks,
        )
        for task_index, rows in enumerate(iterator, start=1):
            all_rows.extend(rows)
            if task_index % max(1, cfg.r_cal) == 0:
                completed_n = task_index // cfg.r_cal
                logger.info(
                    "Progress | p=%d | n=%d | completed N values=%d/%d",
                    p,
                    n,
                    completed_n,
                    len(n_grid),
                )
    else:
        context = mp.get_context(cfg.mp_start_method)
        with context.Pool(processes=cfg.n_jobs) as pool:
            iterator = pool.imap_unordered(
                run_one_replication,
                tasks,
                chunksize=cfg.chunksize,
            )
            for task_index, rows in enumerate(iterator, start=1):
                all_rows.extend(rows)
                if task_index % max(1, cfg.r_cal) == 0:
                    logger.info(
                        "Progress | p=%d | n=%d | tasks=%d/%d",
                        p,
                        n,
                        task_index,
                        len(tasks),
                    )

    details = pd.DataFrame(all_rows)
    details = details.sort_values(
        ["candidate_N", "replication", "alpha"]
    ).reset_index(drop=True)

    summary = summarize_candidate_results(details)

    objective = (
        summary.groupby(
            ["p", "n", "candidate_N"],
            as_index=False,
        )
        .agg(
            aggregate_calibration_error=(
                "absolute_calibration_error",
                "sum",
            ),
            maximum_calibration_error=(
                "absolute_calibration_error",
                "max",
            ),
            mean_calibration_error=(
                "absolute_calibration_error",
                "mean",
            ),
        )
        .sort_values(
            ["aggregate_calibration_error", "candidate_N"]
        )
        .reset_index(drop=True)
    )

    if cfg.save_details:
        details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    objective.to_csv(objective_path, index=False)

    elapsed = time.perf_counter() - start_time
    selected = objective.iloc[0]

    logger.info(
        "Completed dimension | p=%d | n=%d | calibrated N=%d | "
        "objective=%.6f | elapsed=%.2f seconds",
        p,
        n,
        int(selected["candidate_N"]),
        float(selected["aggregate_calibration_error"]),
        elapsed,
    )

    return details, summary, objective


# ============================================================
# Main
# ============================================================

def main() -> None:
    outdir = Path(CFG.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir)

    logger.info("=" * 72)
    logger.info("Section 2.3 multiplier calibration")
    logger.info("Configuration: %s", CFG.to_dict())
    logger.info("=" * 72)

    config_frame = pd.DataFrame([{
        "r_cal": CFG.r_cal,
        "b_boot": CFG.b_boot,
        "n_jobs": CFG.n_jobs,
        "chunksize": CFG.chunksize,
        "mp_start_method": CFG.mp_start_method,
        "base_seed": CFG.base_seed,
        "alpha_grid": ",".join(str(x) for x in CFG.alpha_grid),
        "dimensions": ",".join(
            f"{p}:{n}" for p, n in CFG.dimensions
        ),
    }])
    config_frame.to_csv(outdir / "run_configuration.csv", index=False)

    all_summaries: List[pd.DataFrame] = []
    all_objectives: List[pd.DataFrame] = []

    for p, n in CFG.dimensions:
        _, summary, objective = run_dimension(
            p=int(p),
            n=int(n),
            cfg=CFG,
            logger=logger,
            outdir=outdir,
        )
        all_summaries.append(summary)
        all_objectives.append(objective)

    combined_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )
    combined_objective = pd.concat(
        all_objectives,
        ignore_index=True,
    )

    combined_summary.to_csv(
        outdir / "calibration_summary_all_dimensions.csv",
        index=False,
    )
    combined_objective.to_csv(
        outdir / "calibration_objective_all_dimensions.csv",
        index=False,
    )

    calibrated = select_calibrated_n(combined_summary)
    calibrated.to_csv(
        outdir / "calibrated_N_lookup.csv",
        index=False,
    )

    logger.info("Final calibrated lookup:")
    for row in calibrated.itertuples(index=False):
        logger.info(
            "p=%d | n=%d | calibrated N=%d | aggregate error=%.6f",
            int(row.p),
            int(row.n),
            int(row.calibrated_N),
            float(row.aggregate_calibration_error),
        )

    print("\nCalibrated N lookup")
    print(calibrated.to_string(index=False))
    print(f"\nResults saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
