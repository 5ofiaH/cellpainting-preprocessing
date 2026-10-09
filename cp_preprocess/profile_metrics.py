#!/usr/bin/env python3
"""
Standalone profile metrics for the Cell Painting pipeline (Level 4b).

This module is INDEPENDENT of the pipeline package: it imports nothing from the
workflow and modifies no existing file. It uses only numpy / pandas / scipy /
matplotlib so it can be dropped into any environment that can read the Level 4b
parquet. It is both importable (call the functions directly) and CLI-runnable
(``python profile_metrics.py ...``).

Conventions match the surrounding pipeline (see metrics.py / normalization.py):
  • Metadata columns are prefixed ``Metadata_``; every other numeric column is a
    morphological feature.
  • Compound label      → ``Metadata_compound``
  • Well id             → ``Metadata_Well`` (a replicate is identified by
                          ``Metadata_Barcode`` × ``Metadata_Well``)
  • Perturbation type   → ``Metadata_pert_type`` with negcon = "negcon" (DMSO)
                          and poscon = "poscon"
  • DMSO compound label → "DMSO"

Two metrics are provided, deliberately mirroring the pipeline's two questions:

  Metric 1 — Profile MATCHING (Biosimilarity / Pearson), a *shape/direction*
             measure, run on the per-COMPOUND consensus profiles.
  Metric 2 — Activity DETECTION (Euclidean magnitude vs the DMSO centroid), a
             *magnitude* measure, run on the WELL-LEVEL profiles and then
             aggregated per compound.

INPUT ASSUMPTIONS (confirmed against the pipeline, do not violate):
  • Level 4b values are ALREADY RobustMAD z-scores, (x − median_DMSO)/MAD_DMSO.
    Nothing here re-normalizes them.
  • The input is WELL-LEVEL: multiple replicate wells per compound, including many
    DMSO wells.
  • Features are already selected; we still defensively drop all-NaN / inf /
    zero-variance columns and warn which were dropped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import correlation as scipy_correlation_distance
from scipy.stats import percentileofscore
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("profile_metrics")

# Default z-score profile guides: solid reference at 0, dashed at ±3 (a common
# "clearly non-DMSO" band for MAD-normalized features).
Z_GUIDE = 3.0

# Channel keywords used only to GROUP/ORDER features along the x-axis of the
# profile plot. Order matters (first match wins). Detection is substring-based
# and case-insensitive, so it works whether the compartment is a prefix
# (Cells_/Nuclei_/Cytoplasm_) or a suffix (AreaShape_Area_cells).
CHANNEL_KEYWORDS: List[Tuple[str, Sequence[str]]] = [
    ("DNA", ("dna", "hoechst", "dapi")),
    ("ER", ("_er", "er_", "endoplasmic", "conc", "concanavalin")),
    ("RNA", ("rna", "syto")),
    ("AGP", ("agp", "actin", "golgi", "phalloidin", "pha", "wga")),
    ("Mito", ("mito",)),
    ("Morphology", ("areashape", "area_shape", "neighbors", "location")),
]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (column names / control labels), overridable from the CLI
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Columns:
    """Column-name and control-label bindings for a Level 4b table."""
    compound_col: str = "Metadata_compound"
    well_col: str = "Metadata_Well"
    plate_col: str = "Metadata_Barcode"
    pert_col: str = "Metadata_pert_type"
    negcon_value: str = "negcon"          # DMSO perturbation type
    poscon_value: str = "poscon"          # positive-control perturbation type
    dmso_compound: str = "DMSO"           # DMSO compound label
    metadata_prefix: str = "Metadata_"


# ─────────────────────────────────────────────────────────────────────────────
# Feature / metadata handling
# ─────────────────────────────────────────────────────────────────────────────
def feature_columns(df: pd.DataFrame, cols: Columns = Columns()) -> List[str]:
    """Feature columns = numeric columns that are not metadata (no Metadata_ prefix)."""
    return [
        c for c in df.columns
        if not c.startswith(cols.metadata_prefix) and pd.api.types.is_numeric_dtype(df[c])
    ]


def clean_features(
    df: pd.DataFrame,
    cols: Columns = Columns(),
    log: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Defensively drop degenerate feature columns and warn which were removed.

    Removes features that are entirely NaN, contain any ±inf, or have (near-)zero
    variance across the wells — they carry no information and would break the
    correlation / distance math. The values themselves are NOT re-normalized.

    Returns (cleaned_df, kept_feature_cols).
    """
    lg = log or logger
    feats = feature_columns(df, cols)
    if not feats:
        raise ValueError("No numeric feature columns found (all columns are metadata?).")

    X = df[feats]
    all_nan = X.columns[X.isna().all()].tolist()
    has_inf = X.columns[np.isinf(X.to_numpy(dtype=float, na_value=np.nan)).any(axis=0)].tolist()
    # Zero variance among finite values (constant feature).
    with np.errstate(invalid="ignore"):
        var = X.replace([np.inf, -np.inf], np.nan).var(axis=0, skipna=True)
    zero_var = var.index[(var.fillna(0.0) <= 0.0)].tolist()

    drop = sorted(set(all_nan) | set(has_inf) | set(zero_var))
    if drop:
        lg.warning(
            "Dropping %d degenerate feature(s): %d all-NaN, %d with inf, %d zero-variance",
            len(drop), len(all_nan), len(has_inf), len(zero_var),
        )
        lg.debug("  dropped features: %s", drop)

    kept = [c for c in feats if c not in drop]
    keep_cols = [c for c in df.columns if c not in drop]
    return df[keep_cols].copy(), kept


# ─────────────────────────────────────────────────────────────────────────────
# Data levels
# ─────────────────────────────────────────────────────────────────────────────
def consensus(df: pd.DataFrame, cols: Columns = Columns()) -> pd.DataFrame:
    """
    Per-COMPOUND consensus: feature-wise median across a compound's replicate wells.

    Profile MATCHING (Metric 1) runs on this compound-level table. The index is
    the compound label; columns are the feature medians. DMSO is included so it
    can be used as a reference row.
    """
    feats = feature_columns(df, cols)
    if cols.compound_col not in df.columns:
        raise KeyError(f"Compound column '{cols.compound_col}' not in dataframe.")
    # Normalize DMSO label casing to the canonical cols.dmso_compound before
    # grouping, otherwise case variants (e.g. "DMSO" vs "dmso") would form
    # separate consensus rows instead of one pooled DMSO reference.
    compound_key = df[cols.compound_col].astype(str)
    is_dmso = _dmso_well_mask(df, cols)
    compound_key = compound_key.mask(is_dmso, str(cols.dmso_compound))
    grouped = df[feats].groupby(compound_key, observed=True).median()
    grouped.index = grouped.index.astype(str)
    grouped.index.name = cols.compound_col
    return grouped


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Profile matching (Biosimilarity / Pearson)  [direction / shape]
# ─────────────────────────────────────────────────────────────────────────────
def biosimilarity(u: Sequence[float], v: Sequence[float]) -> float:
    """
    Biosimilarity between two profiles as a percentage in [0, 100].

    Defined via scipy's correlation distance:
        CD = scipy.spatial.distance.correlation(u, v)      # = 1 − Pearson r
        Biosimilarity = clip(1 − CD, 0, None) * 100        # = clip(r, 0, None)*100
    i.e. it is the Pearson correlation, floored at 0, expressed as a percent.
    """
    cd = scipy_correlation_distance(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    return float(np.clip(1.0 - cd, 0.0, None) * 100.0)


def _pearson_matrix(mat: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation matrix via mean-centering + cosine similarity.
    (No clipping — kept separate so the self-check can compare against scipy.)"""
    centered = mat - mat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for constant rows
    unit = centered / norms
    return unit @ unit.T


def biosimilarity_matrix(consensus_df: pd.DataFrame, cols: Columns = Columns()) -> pd.DataFrame:
    """
    Vectorized N×N biosimilarity (percent) between consensus profiles.

    Rows are mean-centered, cosine similarity is taken (= Pearson r), clipped to
    ≥0 and scaled to percent; the diagonal is forced to 100. Equivalent to calling
    :func:`biosimilarity` on every pair but far faster.
    """
    feats = feature_columns(consensus_df, cols) or list(consensus_df.columns)
    mat = consensus_df[feats].to_numpy(dtype=float)
    n_nan = int(np.isnan(mat).sum()) + int(np.isinf(mat).sum())
    if n_nan:
        logger.warning(
            "biosimilarity_matrix: imputing %d non-finite value(s) to 0 across %d compound(s) × %d feature(s).",
            n_nan, mat.shape[0], mat.shape[1],
        )
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    row_var = mat.var(axis=1)
    degenerate = row_var <= 0.0
    if degenerate.any():
        logger.warning(
            "biosimilarity_matrix: %d constant profile(s) (var=0) found — their off-diagonal "
            "similarity is meaningless (forced to NaN): %s",
            int(degenerate.sum()), consensus_df.index[degenerate].tolist(),
        )

    r = _pearson_matrix(mat)
    sim = np.clip(r, 0.0, None) * 100.0
    np.fill_diagonal(sim, 100.0)
    if degenerate.any():
        sim[degenerate, :] = np.nan
        sim[:, degenerate] = np.nan
        sim[degenerate, degenerate] = 100.0
    return pd.DataFrame(sim, index=consensus_df.index, columns=consensus_df.index)


def _ccc_matrix(mat: np.ndarray) -> np.ndarray:
    """Row-wise Lin's Concordance Correlation Coefficient (CCC) matrix.

    Unlike Pearson r (scale/offset-invariant — a profile scaled x0.3 still
    scores 100%), CCC penalizes both shape mismatch AND magnitude/location
    mismatch:  rho_c = 2*cov(x,y) / (var(x) + var(y) + (mean(x)-mean(y))**2).
    Two profiles that trend the same way but differ in amplitude score < 1.0.
    """
    n = mat.shape[0]
    means = mat.mean(axis=1)
    # population variance/covariance (ddof=0), consistent with the CCC definition
    centered = mat - means[:, None]
    var = (centered ** 2).mean(axis=1)
    cov = (centered @ centered.T) / mat.shape[1]
    mean_diff_sq = (means[:, None] - means[None, :]) ** 2
    denom = var[:, None] + var[None, :] + mean_diff_sq
    denom[denom == 0] = 1.0  # avoid divide-by-zero for two identical constant rows
    ccc = (2.0 * cov) / denom
    return ccc


def concordance_matrix(consensus_df: pd.DataFrame, cols: Columns = Columns()) -> pd.DataFrame:
    """
    Vectorized N×N concordance (percent) between consensus profiles — Lin's CCC,
    which requires matching magnitude/trend (not just shape) to score highly.
    Clipped to ≥0 and scaled to percent; the diagonal is forced to 100.
    """
    feats = feature_columns(consensus_df, cols) or list(consensus_df.columns)
    mat = consensus_df[feats].to_numpy(dtype=float)
    n_nan = int(np.isnan(mat).sum()) + int(np.isinf(mat).sum())
    if n_nan:
        logger.warning(
            "concordance_matrix: imputing %d non-finite value(s) to 0 across %d compound(s) × %d feature(s).",
            n_nan, mat.shape[0], mat.shape[1],
        )
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    row_var = mat.var(axis=1)
    degenerate = row_var <= 0.0
    if degenerate.any():
        logger.warning(
            "concordance_matrix: %d constant profile(s) (var=0) found — their off-diagonal "
            "concordance is meaningless (forced to NaN): %s",
            int(degenerate.sum()), consensus_df.index[degenerate].tolist(),
        )

    c = _ccc_matrix(mat)
    sim = np.clip(c, 0.0, None) * 100.0
    np.fill_diagonal(sim, 100.0)
    if degenerate.any():
        sim[degenerate, :] = np.nan
        sim[:, degenerate] = np.nan
        sim[degenerate, degenerate] = 100.0
    return pd.DataFrame(sim, index=consensus_df.index, columns=consensus_df.index)


def find_concordant(
    consensus_df: pd.DataFrame,
    query: str,
    threshold: float = 90.0,
    cols: Columns = Columns(),
) -> pd.DataFrame:
    """
    Compounds whose consensus profile is ≥ ``threshold`` concordant (same
    magnitude AND trend, via Lin's CCC) with ``query``.

    Returns a DataFrame (compound, concordance) sorted descending, with the
    query itself excluded. Use alongside :func:`find_similar` (Pearson,
    shape-only) — a compound can be biosimilar without being concordant if its
    response is much weaker/stronger in magnitude.
    """
    cmat = concordance_matrix(consensus_df, cols)
    if query not in cmat.index:
        raise KeyError(f"Query compound '{query}' not found among consensus profiles.")
    s = cmat.loc[query].drop(labels=[query])
    s = s[s >= threshold].sort_values(ascending=False)
    return s.rename("concordance").rename_axis(cols.compound_col).reset_index()


def _self_check_biosimilarity(seed: int = 0) -> None:
    """Assert the vectorized Pearson matrix matches scipy's pairwise correlation
    distance on a small random array (atol=1e-8). Raises AssertionError on drift."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((6, 20))
    got = _pearson_matrix(a)
    expected = np.empty((6, 6))
    for i in range(6):
        for j in range(6):
            expected[i, j] = 1.0 - scipy_correlation_distance(a[i], a[j])
    assert np.allclose(got, expected, atol=1e-8), "biosimilarity_matrix drifted from scipy"


def _self_check_ccc(seed: int = 1) -> None:
    """Assert the vectorized CCC matrix matches the textbook per-pair formula
    (population covariance/variance) on a small random array (atol=1e-8)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((6, 20))
    got = _ccc_matrix(a)
    expected = np.empty((6, 6))
    for i in range(6):
        for j in range(6):
            xi, yj = a[i], a[j]
            cov = np.mean((xi - xi.mean()) * (yj - yj.mean()))
            expected[i, j] = (2 * cov) / (xi.var() + yj.var() + (xi.mean() - yj.mean()) ** 2)
    assert np.allclose(got, expected, atol=1e-8), "concordance_matrix drifted from reference CCC"


def find_similar(
    consensus_df: pd.DataFrame,
    query: str,
    threshold: float = 90.0,
    cols: Columns = Columns(),
) -> pd.DataFrame:
    """
    Compounds whose consensus profile is ≥ ``threshold`` biosimilar to ``query``.

    Returns a DataFrame (compound, biosimilarity) sorted descending, with the
    query itself excluded.
    """
    bmat = biosimilarity_matrix(consensus_df, cols)
    if query not in bmat.index:
        raise KeyError(f"Query compound '{query}' not found among consensus profiles.")
    s = bmat.loc[query].drop(labels=[query])
    s = s[s >= threshold].sort_values(ascending=False)
    return s.rename("biosimilarity").rename_axis(cols.compound_col).reset_index()


def match_to_controls(
    consensus_df: pd.DataFrame,
    query: str,
    poscon_labels: Sequence[str],
    cols: Columns = Columns(),
) -> pd.DataFrame:
    """
    Biosimilarity of ``query`` to each positive-control compound.

    Interpretation: does the query's phenotype match a known-MoA positive control?
    Returns (poscon, biosimilarity) sorted descending.
    """
    bmat = biosimilarity_matrix(consensus_df, cols)
    if query not in bmat.index:
        raise KeyError(f"Query compound '{query}' not found among consensus profiles.")
    present = [p for p in poscon_labels if p in bmat.index]
    missing = [p for p in poscon_labels if p not in bmat.index]
    if missing:
        logger.warning("Positive controls absent from consensus: %s", missing)
    s = bmat.loc[query, present].sort_values(ascending=False)
    return s.rename("biosimilarity").rename_axis("poscon").reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — Activity detection (Euclidean magnitude vs DMSO centroid) [magnitude]
# ─────────────────────────────────────────────────────────────────────────────
def _dmso_well_mask(df: pd.DataFrame, cols: Columns) -> pd.Series:
    """Boolean mask of DMSO (negative-control) wells, via pert_type first,
    then the DMSO compound label as a fallback."""
    if cols.pert_col in df.columns:
        m = df[cols.pert_col].astype(str) == str(cols.negcon_value)
        if m.any():
            return m
    if cols.compound_col in df.columns:
        return df[cols.compound_col].astype(str).str.upper() == str(cols.dmso_compound).upper()
    return pd.Series(False, index=df.index)


def _compound_mask(df: pd.DataFrame, cols: Columns, cpd: str) -> pd.Series:
    """Boolean mask for wells belonging to compound ``cpd``.

    Routes DMSO through :func:`_dmso_well_mask` (case-insensitive, pert_type-aware)
    so callers never miss DMSO wells due to label-casing mismatches (e.g. "dmso" vs
    "DMSO"). All other compounds use a case-insensitive exact match against
    ``cols.compound_col`` for the same reason.
    """
    if str(cpd).upper() == str(cols.dmso_compound).upper():
        return _dmso_well_mask(df, cols)
    return df[cols.compound_col].astype(str).str.upper() == str(cpd).upper()


def _resolve_consensus_index(consensus_df: pd.DataFrame, cpd: str, cols: Columns) -> str:
    """Resolve ``cpd`` to its exact label in a consensus table's index,
    case-insensitively (DMSO routed through the canonical ``cols.dmso_compound``
    label, same as :func:`_compound_mask`), so a casing mismatch (e.g. "dmso" vs
    "DMSO") doesn't raise a spurious KeyError."""
    if cpd in consensus_df.index:
        return cpd
    target = str(cols.dmso_compound) if str(cpd).upper() == str(cols.dmso_compound).upper() else str(cpd)
    matches = [ix for ix in consensus_df.index if str(ix).upper() == target.upper()]
    if not matches:
        raise KeyError(f"'{cpd}' not found among consensus profiles.")
    return matches[0]


def dmso_centroid(df: pd.DataFrame, cols: Columns = Columns()) -> np.ndarray:
    """
    DMSO centroid = mean of the DMSO WELL profiles (feature-wise).

    For RobustMAD z-scores this is ~0 by construction, but it is computed
    explicitly (never assumed) so a shifted reference is caught.
    """
    feats = feature_columns(df, cols)
    mask = _dmso_well_mask(df, cols)
    if not mask.any():
        raise ValueError("No DMSO wells found — cannot compute the activity reference.")
    return df.loc[mask, feats].to_numpy(dtype=float).mean(axis=0)


def activity_scores(
    df: pd.DataFrame,
    cols: Columns = Columns(),
) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    """
    Per-WELL activity score = Euclidean distance from the well profile to the DMSO
    centroid.

    Returns (score_per_well, null_distances, centroid) where ``null_distances`` are
    the distances of each individual DMSO well to the centroid (the null model for
    "no phenotype").
    """
    feats = feature_columns(df, cols)
    centroid = dmso_centroid(df, cols)
    X = np.nan_to_num(df[feats].to_numpy(dtype=float), nan=0.0)
    scores = np.linalg.norm(X - centroid, axis=1)
    null = scores[_dmso_well_mask(df, cols).to_numpy()]
    return pd.Series(scores, index=df.index, name="activity_score"), null, centroid


def activity_table(
    df: pd.DataFrame,
    cols: Columns = Columns(),
    active_percentile: float = 95.0,
) -> pd.DataFrame:
    """
    Per-COMPOUND activity summary.

    Activity DETECTION runs on WELL-LEVEL profiles, then aggregates:
      • ``median_score``  — median well activity score for the compound
      • ``frac_active_wells`` — fraction of the compound's wells above the null
        percentile cutoff
      • ``active`` — True if the compound's median score exceeds the null cutoff
      • ``percentile`` — percentile of the compound's median score within the null
      • ``rank`` — 1 = most active (descending median score)

    The cutoff is the ``active_percentile`` (default 95th) of the DMSO-well null
    distances.
    """
    scores, null, _ = activity_scores(df, cols)
    cutoff = float(np.percentile(null, active_percentile)) if len(null) else np.inf

    work = pd.DataFrame({
        cols.compound_col: df[cols.compound_col].astype(str).values,
        "activity_score": scores.values,
    })
    work["well_active"] = work["activity_score"] > cutoff

    grp = work.groupby(cols.compound_col, observed=True)
    tbl = grp.agg(
        median_score=("activity_score", "median"),
        frac_active_wells=("well_active", "mean"),
        n_wells=("activity_score", "size"),
    ).reset_index()

    tbl["active"] = tbl["median_score"] > cutoff
    tbl["percentile"] = tbl["median_score"].apply(
        lambda s: float(percentileofscore(null, s, kind="mean")) if len(null) else np.nan
    )

    # Multiple-testing correction: a raw percentile cutoff (e.g. P95) applied
    # across many compounds yields ~5% expected false positives by chance alone
    # (e.g. ~50 false "active" calls out of 1000 compounds tested). Convert the
    # percentile to a pseudo-pvalue (1 - percentile/100) and BH-adjust it.
    pvals = (1.0 - tbl["percentile"] / 100.0).clip(lower=0.0, upper=1.0)
    if pvals.notna().all() and len(pvals):
        _, qvals, _, _ = multipletests(pvals.to_numpy(), method="fdr_bh")
        tbl["qval"] = qvals
    else:
        tbl["qval"] = np.nan
    expected_fp = float(len(tbl)) * (1.0 - active_percentile / 100.0)
    logger.info(
        "Activity: raw cutoff P%.0f over %d compounds → ~%.1f expected false positives "
        "by chance; see 'qval' column for FDR-corrected significance.",
        active_percentile, len(tbl), expected_fp,
    )

    tbl = tbl.sort_values("median_score", ascending=False).reset_index(drop=True)
    tbl["rank"] = np.arange(1, len(tbl) + 1)
    tbl.attrs["null_cutoff"] = cutoff
    tbl.attrs["active_percentile"] = active_percentile
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Z-score profile line plot (parallel coordinates)
# ─────────────────────────────────────────────────────────────────────────────
def _channel_of(feature: str) -> str:
    """Group label for a feature, used to order/annotate the x-axis."""
    fl = feature.lower()
    for name, keys in CHANNEL_KEYWORDS:
        if any(k in fl for k in keys):
            return name
    return "Other"


def _order_features(feats: Sequence[str]) -> List[str]:
    """Order features by channel/compartment group (stable within a group)."""
    order = {name: i for i, (name, _) in enumerate(CHANNEL_KEYWORDS)}
    order["Other"] = len(order)
    return sorted(feats, key=lambda f: (order[_channel_of(f)], f))


def _select_plot_features(
    df: pd.DataFrame,
    cols: Columns,
    features: Optional[Sequence[str]],
    top_n_var: int,
) -> List[str]:
    """Pick the features to plot: an explicit list if given, else the ``top_n_var``
    highest-variance features (keeps the parallel-coordinates plot readable)."""
    all_feats = feature_columns(df, cols)
    if features:
        chosen = [f for f in features if f in all_feats]
        missing = [f for f in features if f not in all_feats]
        if missing:
            logger.warning("Requested features not present (ignored): %s", missing)
    else:
        var = df[all_feats].var(axis=0, skipna=True).sort_values(ascending=False)
        chosen = var.head(int(top_n_var)).index.tolist()
    return _order_features(chosen)


def plot_zscore_profiles(
    df: pd.DataFrame,
    compounds: Sequence[str],
    color_map: Optional[Dict[str, str]] = None,
    cols: Columns = Columns(),
    features: Optional[Sequence[str]] = None,
    top_n_var: int = 40,
    include_dmso: bool = True,
    outpath: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Parallel-coordinates z-score profile plot.

    x = features (ordered, grouped by channel/compartment), y = profile value
    (z-score). One line per replicate WELL; wells are colored by their compound via
    ``color_map`` ({compound_label: color}). A solid guide line is drawn at y=0 and
    dashed guides at y=±``Z_GUIDE`` (default ±3). DMSO is overlaid by default so the
    query and any positive control can always be read against it.

    Feature-count guard: pass an explicit ``features`` list OR rely on ``top_n_var``
    (default 40) to plot only the highest-variance features.
    """
    plot_feats = _select_plot_features(df, cols, features, top_n_var)
    if not plot_feats:
        raise ValueError("No features selected for the profile plot.")

    to_plot = list(dict.fromkeys(compounds))  # de-dup, keep order
    if include_dmso and cols.dmso_compound not in to_plot:
        to_plot.append(cols.dmso_compound)

    # Default colors: DMSO grey, others from the matplotlib tab10 cycle.
    if color_map is None:
        color_map = {}
    palette = plt.get_cmap("tab10").colors
    ci = 0
    for cpd in to_plot:
        if cpd in color_map:
            continue
        if str(cpd).upper() == str(cols.dmso_compound).upper():
            color_map[cpd] = "#8A8F98"
        else:
            color_map[cpd] = palette[ci % len(palette)]
            ci += 1

    x = np.arange(len(plot_feats))
    fig, ax = plt.subplots(figsize=(max(9.0, 0.28 * len(plot_feats)), 5.5))

    drawn_labels = set()
    for cpd in to_plot:
        sub = df[_compound_mask(df, cols, cpd)]
        if sub.empty:
            logger.warning("Compound '%s' has no wells — skipped in plot", cpd)
            continue
        is_dmso = str(cpd).upper() == str(cols.dmso_compound).upper()
        for _, row in sub.iterrows():
            y = pd.to_numeric(row[plot_feats], errors="coerce").to_numpy(dtype=float)
            label = str(cpd) if str(cpd) not in drawn_labels else None
            drawn_labels.add(str(cpd))
            ax.plot(x, y, color=color_map[cpd], alpha=0.35 if is_dmso else 0.7,
                    lw=1.0 if is_dmso else 1.4, label=label, zorder=1 if is_dmso else 2)

    # y-guides
    ax.axhline(0.0, color="black", lw=1.0, zorder=0)
    ax.axhline(Z_GUIDE, color="black", lw=0.8, ls="--", alpha=0.6, zorder=0)
    ax.axhline(-Z_GUIDE, color="black", lw=0.8, ls="--", alpha=0.6, zorder=0)

    # x-axis grouped/annotated by channel/compartment
    _annotate_feature_groups(ax, plot_feats)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))  # headroom so channel labels don't crowd the title
    ax.set_ylabel("profile value (z-score)", fontweight="bold")
    ax.set_title(title or f"Z-score profiles ({len(plot_feats)} features)", fontweight="bold", pad=18)
    ax.legend(fontsize=8, ncol=min(4, len(to_plot)), framealpha=0.9)

    fig.tight_layout()
    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        logger.info("Saved profile plot → %s", outpath)
    return fig


def plot_paired_biosimilarity(
    consensus_df: pd.DataFrame,
    compound1: str,
    compound2: str,
    cols: Columns = Columns(),
    features: Optional[Sequence[str]] = None,
    top_n_var: int = 50,
    outpath: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Paired biosimilarity strip plot: two stacked 1-row heatmaps (compound1 on
    top, compound2 on bottom) over the same feature-ordered x-axis, diverging
    colormap centered at 0 (vmin = -vmax so 0 sits at the color midpoint).
    Compartment/channel block boundaries are drawn with tick labels, and the
    title reports the pairwise biosimilarity percent.
    """
    compound1 = _resolve_consensus_index(consensus_df, compound1, cols)
    compound2 = _resolve_consensus_index(consensus_df, compound2, cols)

    plot_feats = _select_plot_features(consensus_df, cols, features, top_n_var)
    if not plot_feats:
        raise ValueError("No features selected for the paired biosimilarity plot.")

    row1 = pd.to_numeric(consensus_df.loc[compound1, plot_feats], errors="coerce").to_numpy(dtype=float)
    row2 = pd.to_numeric(consensus_df.loc[compound2, plot_feats], errors="coerce").to_numpy(dtype=float)

    sim = biosimilarity(row1, row2)

    vmax = np.nanmax(np.abs(np.concatenate([row1, row2])))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0

    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 0.28 * len(plot_feats)), 3.2), sharex=True)
    fig.subplots_adjust(hspace=0.08, top=0.80)
    for ax, row, label in zip(axes, (row1, row2), (compound1, compound2)):
        im = ax.imshow(
            row[np.newaxis, :], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=9, fontweight="bold")

    groups = [_channel_of(f) for f in plot_feats]
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            if start != 0:
                for ax in axes:
                    ax.axvline(start - 0.5, color="black", lw=0.8, zorder=2)
            axes[0].text((start + i - 1) / 2.0, -0.62, groups[start], ha="center", va="bottom",
                         fontsize=8, fontweight="bold", color="#444444")
            start = i

    axes[1].set_xticks(np.arange(len(plot_feats)))
    axes[1].set_xticklabels(plot_feats, rotation=90, fontsize=5)
    axes[1].set_xlabel("features (grouped by channel / compartment)", fontweight="bold")

    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.02, label="z-score")
    fig.suptitle(title or f"{compound1} vs {compound2} — Biosimilarity {sim:.1f}%", fontweight="bold", y=0.99)

    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        logger.info("Saved paired biosimilarity plot → %s", outpath)
    return fig


def _annotate_feature_groups(ax, plot_feats: Sequence[str]) -> None:
    """Draw group separators and labels along the x-axis, tick every feature."""
    groups = [_channel_of(f) for f in plot_feats]
    ax.set_xlim(-0.5, len(plot_feats) - 0.5)
    ax.set_xticks(np.arange(len(plot_feats)))
    ax.set_xticklabels(plot_feats, rotation=90, fontsize=5)

    # vertical separators + centered group labels at the top
    start = 0
    ymax = ax.get_ylim()[1]
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            if start != 0:
                ax.axvline(start - 0.5, color="#CCCCCC", lw=0.8, zorder=0)
            ax.text((start + i - 1) / 2.0, ymax, groups[start], ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#444444")
            start = i
    ax.set_xlabel("features (grouped by channel / compartment)", fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# I/O + CLI
# ─────────────────────────────────────────────────────────────────────────────
def load_level4b(path: str) -> pd.DataFrame:
    """Load a WELL-LEVEL Level 4b table from parquet or CSV."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p, engine="python")


def _poscon_compounds(df: pd.DataFrame, cols: Columns) -> List[str]:
    """Compound labels whose pert_type is the positive control."""
    if cols.pert_col not in df.columns:
        return []
    m = df[cols.pert_col].astype(str) == str(cols.poscon_value)
    return sorted(df.loc[m, cols.compound_col].astype(str).unique().tolist())


def _build_columns(args: argparse.Namespace) -> Columns:
    return Columns(
        compound_col=args.compound_col,
        well_col=args.well_col,
        plate_col=args.plate_col,
        pert_col=args.pert_col,
        negcon_value=args.negcon,
        poscon_value=args.poscon,
        dmso_compound=args.dmso,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Standalone profile matching + activity detection on Level 4b profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Level 4b parquet/CSV (well-level).")
    ap.add_argument("--outdir", default=".", help="Directory for output tables/figures.")

    # Column / control bindings (defaults match the pipeline).
    ap.add_argument("--compound-col", default="Metadata_compound")
    ap.add_argument("--well-col", default="Metadata_Well")
    ap.add_argument("--plate-col", default="Metadata_Barcode")
    ap.add_argument("--pert-col", default="Metadata_pert_type")
    ap.add_argument("--negcon", default="negcon", help="DMSO/negative-control pert_type value.")
    ap.add_argument("--poscon", default="poscon", help="Positive-control pert_type value.")
    ap.add_argument("--dmso", default="DMSO", help="DMSO compound label.")

    # What to run.
    ap.add_argument("--activity", action="store_true", help="Write per-compound activity table.")
    ap.add_argument("--active-percentile", type=float, default=95.0,
                    help="Null percentile that defines 'active'.")
    ap.add_argument("--matching", action="store_true",
                    help="Write biosimilarity matrix (and find_similar/match_to_controls if --query).")
    ap.add_argument("--query", help="Query compound for find_similar / match_to_controls / plot.")
    ap.add_argument("--threshold", type=float, default=90.0, help="find_similar biosimilarity threshold.")
    ap.add_argument("--plot-profile", action="store_true",
                    help="Z-score profile line plot for --query (+ positive controls + DMSO).")
    ap.add_argument("--features", nargs="+", help="Explicit feature list for the profile plot.")
    ap.add_argument("--top-n-var", type=int, default=40,
                    help="If --features not given, plot the N highest-variance features.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
    )

    # Self-check the vectorized biosimilarity against scipy before doing anything.
    _self_check_biosimilarity()

    cols = _build_columns(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_level4b(args.input)
    df, kept = clean_features(df, cols)
    logger.info("Loaded %d wells × %d features from %s", len(df), len(kept), args.input)

    did_something = False

    if args.activity:
        tbl = activity_table(df, cols, active_percentile=args.active_percentile)
        path = outdir / "profile_activity_table.csv"
        tbl.to_csv(path, index=False)
        n_active = int(tbl["active"].sum())
        logger.info("Activity: %d/%d compounds active (cutoff=%.3f) → %s",
                    n_active, len(tbl), tbl.attrs.get("null_cutoff", float("nan")), path)
        did_something = True

    if args.matching:
        cons = consensus(df, cols)
        bmat = biosimilarity_matrix(cons, cols)
        bmat.to_csv(outdir / "profile_biosimilarity_matrix.csv")
        logger.info("Biosimilarity matrix: %d×%d → %s",
                    bmat.shape[0], bmat.shape[1], outdir / "profile_biosimilarity_matrix.csv")
        if args.query:
            sim = find_similar(cons, args.query, threshold=args.threshold, cols=cols)
            sim.to_csv(outdir / f"profile_similar_to_{_slug(args.query)}.csv", index=False)
            logger.info("find_similar('%s', ≥%.0f): %d hits", args.query, args.threshold, len(sim))
            poscons = _poscon_compounds(df, cols)
            if poscons:
                mc = match_to_controls(cons, args.query, poscons, cols=cols)
                mc.to_csv(outdir / f"profile_match_controls_{_slug(args.query)}.csv", index=False)
                logger.info("match_to_controls('%s'): %d positive controls compared",
                            args.query, len(mc))
            if len(sim):
                top_hit = sim.iloc[0][cols.compound_col]
                fig = plot_paired_biosimilarity(
                    cons, args.query, top_hit, cols=cols, top_n_var=args.top_n_var,
                    outpath=str(outdir / f"profile_paired_{_slug(args.query)}_{_slug(top_hit)}.png"),
                )
                plt.close(fig)
        did_something = True

    if args.plot_profile:
        if not args.query:
            logger.error("--plot-profile requires --query")
            return 2
        overlay = [args.query] + _poscon_compounds(df, cols)
        fig = plot_zscore_profiles(
            df, overlay, cols=cols, features=args.features, top_n_var=args.top_n_var,
            outpath=str(outdir / f"profile_zscore_{_slug(args.query)}.png"),
            title=f"Z-score profile — {args.query} vs positive controls + DMSO",
        )
        plt.close(fig)
        did_something = True

    if not did_something:
        logger.warning("Nothing to do — pass one of --activity / --matching / --plot-profile.")
        ap.print_help()
        return 1
    return 0


def _slug(s: str) -> str:
    """Filesystem-safe token from a compound label."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))[:60]


if __name__ == "__main__":
    sys.exit(main())
