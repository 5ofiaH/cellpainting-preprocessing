"""
Site-level outlier detection (opt-in, runs on Level 2.5b — normalized,
feature-selected, site-level data; see config.site_level.run).

Two simple, MAD-robust methods, computed within each compound x dose (or
custom) replicate group:

  1. distance_to_median — Euclidean distance of each site to its group's
     median profile; flagged if its robust Z-score (median/MAD) exceeds
     ``distance_z_thresh``.
  2. median_replicate_corr — each site's median pairwise Pearson correlation
     to the other sites in its group; flagged if its robust Z-score falls
     below ``corr_z_thresh`` (i.e. unusually poorly correlated with its
     replicates).

Writes one row per site with both scores/flags and a combined ``outlier_flag``
plus an empty ``drop`` column for manual review — this is a CSV meant to be
opened, inspected, and edited by hand before drop_outlier_sites.py uses it.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if not mad or np.isnan(mad):
        mad = 1e-9
    return (s - med) / mad


def compute_site_outliers(
    df: pd.DataFrame,
    config,
    group_cols: Optional[List[str]] = None,
    min_group_size: int = 3,
    distance_z_thresh: float = 3.0,
    corr_z_thresh: float = -3.0,
) -> pd.DataFrame:
    """
    Compute per-site outlier flags on Level 2.5b (site-level, normalized +
    feature-selected). Returns one row per site.

    Groups with fewer than ``min_group_size`` sites are marked
    ``evaluable=False`` (not enough replicates to judge an outlier).
    """
    if group_cols is None:
        group_cols = [c for c in ("Metadata_compound", "Metadata_conc_uM") if c in df.columns]
    if not group_cols:
        raise ValueError("compute_site_outliers: no compound/dose columns found to group by")

    m = config.metadata
    key_cols = [c for c in (m.barcode_col, m.well_col, m.site_col) if c in df.columns]
    feat_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    rows = []
    for _gkey, idx in df.groupby(group_cols, dropna=False, observed=True).groups.items():
        idx = list(idx)
        n = len(idx)
        base = {c: df.loc[idx[0], c] for c in group_cols}
        sub = X.loc[idx]

        if n < min_group_size:
            for i in idx:
                rows.append({
                    "_orig_idx": i,
                    **{c: df.loc[i, c] for c in key_cols}, **base,
                    "group_n": n, "evaluable": False,
                    "distance_to_median": np.nan, "distance_z": np.nan, "distance_flag": False,
                    "median_replicate_corr": np.nan, "corr_z": np.nan, "corr_flag": False,
                })
            continue

        # Method 1: Euclidean distance to the group's median profile.
        median = sub.median(axis=0)
        dist = np.sqrt(((sub - median) ** 2).sum(axis=1))
        dist_z = _robust_z(dist)
        dist_flag = dist_z > distance_z_thresh

        # Method 2: median pairwise correlation to the other sites in the group.
        corr = sub.T.corr()
        np.fill_diagonal(corr.values, np.nan)
        med_corr = corr.median(axis=1)
        corr_z = _robust_z(med_corr)
        corr_flag = corr_z < corr_z_thresh

        for i in idx:
            rows.append({
                "_orig_idx": i,
                **{c: df.loc[i, c] for c in key_cols}, **base,
                "group_n": n, "evaluable": True,
                "distance_to_median": float(dist.loc[i]), "distance_z": float(dist_z.loc[i]),
                "distance_flag": bool(dist_flag.loc[i]),
                "median_replicate_corr": float(med_corr.loc[i]), "corr_z": float(corr_z.loc[i]),
                "corr_flag": bool(corr_flag.loc[i]),
            })

    out = pd.DataFrame(rows)
    # Restore the input dataframe's row order (groupby scrambles it) so callers
    # can zip this 1:1 with the Level 2.5b dataframe by position (e.g. for plots).
    out = out.sort_values("_orig_idx").drop(columns="_orig_idx").reset_index(drop=True)
    out["outlier_flag"] = out["distance_flag"].fillna(False) | out["corr_flag"].fillna(False)
    out["drop"] = False  # for manual review: set True on rows to actually drop

    n_flagged = int(out["outlier_flag"].sum())
    logger.info(f"  [site-outliers] {n_flagged}/{len(out)} sites flagged "
                f"(distance_z>{distance_z_thresh} or corr_z<{corr_z_thresh})")
    return out
