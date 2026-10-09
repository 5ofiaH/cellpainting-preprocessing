"""
Viability computation: cell count-based cytotoxicity assessment.

Viability is a COUNT RATIO, not a morphological feature, so it never goes through
the feature ladder (MAD-normalize → feature-select → ComBat → consensus). It is
computed at the level it is measured — the imaging SITE — and rolled up to wells
(and later to compound×dose) by nested medians of the ratio.

Pipeline of a single measurement:

    site_viability_pct = site_count / dmso_site_median_clean × 100

where ``dmso_site_median_clean`` is the per-plate median count of DMSO (negcon)
sites AFTER dropping low-count DMSO outliers (a dead/dying control site would
otherwise drag the reference down and make every treatment look healthier than
it is). Outliers are detected with a per-plate robust Z-score on the low side
only (``viability_dmso_z_threshold``).

Rolling up the *ratio* (median of site %) rather than the pooled counts makes the
well estimate unbiased to how many sites a well retained — fewer sites give a
noisier estimate, not a biased one. ``n_sites`` is carried so low-coverage wells
can be flagged low-confidence instead of silently trusted.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _robust_zscore(series: pd.Series) -> pd.Series:
    """Median/MAD-based robust Z-score. Median and MAD are used (not mean/std)
    because the outliers we want to catch would themselves inflate the std and
    shift the mean, masking themselves."""
    x = pd.to_numeric(series, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        # Degenerate spread: fall back to std so a genuine outlier is still caught.
        std = x.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return pd.Series(np.zeros(len(x)), index=series.index)
        return (x - x.mean()) / std
    return (x - med) / (1.4826 * mad)


def compute_viability(
    df_sites: pd.DataFrame,
    config: PipelineConfig,
    log: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """
    Site-level viability with a cleaned per-plate DMSO reference, rolled up to wells.

    Parameters
    ----------
    df_sites : pd.DataFrame
        SITE-level data (one row per Barcode × Well × Site) carrying a measured
        per-site cell-count column and the perturbation-type / well metadata.
    config : PipelineConfig
    log : logging.Logger, optional

    Returns
    -------
    dict
        {
          "sites": df_sites + Metadata_site_viability_pct,
                   Metadata_site_viability_flag, Metadata_dmso_ref_outlier
          "wells": well-level DataFrame keyed by (barcode, well) with
                   Metadata_Viability_pct, Metadata_Viability_flag,
                   Metadata_Viability_n_sites, Metadata_frac_sites_cytotoxic,
                   Metadata_Viability_low_confidence
          "stats": dict of run-summary counts
        }
    """
    lg = log or logger
    m = config.metrics
    bc = config.metadata.barcode_col
    well = config.metadata.well_col
    site = config.metadata.site_col
    pert_col = m.pert_type_col
    negcon = m.negcon_value

    sites = df_sites.copy().reset_index(drop=True)
    n = len(sites)

    count_col = _find_cell_count_col(sites, preferred=m.viability_site_count_col)
    threshold_pct = m.viability_threshold_pct * 100.0
    z_thr = abs(m.viability_dmso_z_threshold)
    min_dmso = config.batch_correction.min_dmso_per_plate

    # Default outputs (safe fallbacks)
    site_via = np.full(n, np.nan)
    site_flag = np.zeros(n, dtype=bool)
    dmso_outlier = np.zeros(n, dtype=bool)

    if count_col is None or pert_col not in sites.columns or bc not in sites.columns:
        lg.warning(
            "  Viability: missing count/pert_type/barcode column — defaulting to 100%%"
        )
        site_via = np.full(n, 100.0)
    else:
        counts = pd.to_numeric(sites[count_col], errors="coerce")
        is_dmso = sites[pert_col].astype(str) == str(negcon)

        for plate_id in sites[bc].dropna().unique():
            mask_plate = (sites[bc] == plate_id).values
            mask_dmso = mask_plate & is_dmso.values

            dmso_counts = counts[mask_dmso]
            n_dmso = int(dmso_counts.notna().sum())
            if n_dmso == 0:
                lg.debug("  Viability: plate %s has no DMSO sites — 100%%", plate_id)
                site_via[mask_plate] = 100.0
                continue

            # Flag low-side DMSO outliers on this plate and drop them from the reference.
            plate_outlier = np.zeros(mask_plate.sum(), dtype=bool)
            if n_dmso >= 3:
                z = _robust_zscore(dmso_counts)
                is_low = (z < -z_thr).reindex(dmso_counts.index).fillna(False).values
                # write outlier flags back into the plate-aligned vector
                dmso_pos = np.where(mask_dmso)[0]
                dmso_outlier[dmso_pos[is_low]] = True
                clean_dmso_counts = dmso_counts[~is_low]
            else:
                clean_dmso_counts = dmso_counts

            if int(clean_dmso_counts.notna().sum()) < max(1, min_dmso):
                # Not enough clean DMSO sites to trust the reference — use all DMSO.
                clean_dmso_counts = dmso_counts

            dmso_median = clean_dmso_counts.median()
            if not np.isfinite(dmso_median) or dmso_median <= 0:
                site_via[mask_plate] = 100.0
                continue

            plate_pct = (counts[mask_plate] / dmso_median) * 100.0
            site_via[mask_plate] = plate_pct.values

            below = (plate_pct < threshold_pct).values
            non_dmso_plate = (~is_dmso.values[mask_plate])
            site_flag[mask_plate] = below & non_dmso_plate

    sites["Metadata_site_viability_pct"] = site_via
    sites["Metadata_site_viability_flag"] = site_flag.astype(int)
    sites["Metadata_dmso_ref_outlier"] = dmso_outlier.astype(int)

    n_dmso_out = int(dmso_outlier.sum())
    n_sites_flag = int(site_flag.sum())
    if n_dmso_out:
        lg.info("  Viability: %d DMSO reference sites dropped (robust Z < -%.1f)", n_dmso_out, z_thr)
    if n_sites_flag:
        lg.info("  Viability: %d treatment sites below %.0f%% of DMSO", n_sites_flag, threshold_pct)

    wells = _aggregate_to_wells(sites, config, threshold_pct)

    stats = {
        "sites_total": n,
        "dmso_ref_sites_dropped": n_dmso_out,
        "sites_below_threshold": n_sites_flag,
        "wells_total": len(wells),
        "wells_cytotoxic_flagged": int(wells["Metadata_Viability_flag"].sum()) if len(wells) else 0,
        "wells_low_confidence": int(wells["Metadata_Viability_low_confidence"].sum()) if len(wells) else 0,
        "dmso_z_threshold": z_thr,
        "threshold_pct_of_DMSO": threshold_pct,
        "count_col": count_col,
    }

    # Sanity-check the computation and fold the results into stats.
    stats.update(_validate_viability(sites, wells, config, lg))

    return {"sites": sites, "wells": wells, "stats": stats}


def _validate_viability(
    sites: pd.DataFrame,
    wells: pd.DataFrame,
    config: PipelineConfig,
    lg: logging.Logger,
) -> Dict[str, object]:
    """
    Post-hoc checks that the site-level viability behaves as it must.

    These are diagnostics, not hard failures — anything suspicious is logged as a
    warning and reported in the run summary so it is visible without crashing a
    long pipeline. Checks:

      1. DMSO self-consistency — because each site is divided by its plate's DMSO
         median, the DMSO wells themselves must centre on ~100%. A large deviation
         means the reference is being computed on the wrong rows.
      2. Coverage — fraction of sites/wells that ended up with a usable (finite)
         viability value; a high NaN fraction points at a missing/!numeric count
         column.
      3. Range — negative percentages are impossible (counts are non-negative);
         flag if any slip through.
    """
    checks: Dict[str, object] = {}
    m = config.metrics
    pert_col = m.pert_type_col
    negcon = m.negcon_value

    site_pct = pd.to_numeric(sites.get("Metadata_site_viability_pct"), errors="coerce")
    checks["sites_with_viability_pct"] = int(site_pct.notna().sum())
    checks["sites_nan_pct_fraction"] = (
        round(float(site_pct.isna().mean()), 4) if len(site_pct) else 0.0
    )
    if len(site_pct) and site_pct.isna().mean() > 0.20:
        lg.warning("  Viability CHECK: %.0f%% of sites have no viability value "
                   "(missing/non-numeric count column?)", 100 * site_pct.isna().mean())

    n_negative = int((site_pct < 0).sum())
    checks["sites_negative_pct"] = n_negative
    if n_negative:
        lg.warning("  Viability CHECK: %d sites have negative viability %% — impossible, "
                   "inspect the count column", n_negative)

    # DMSO self-consistency (well level): median of DMSO wells should sit near 100%.
    if pert_col in wells.columns and "Metadata_Viability_pct" in wells.columns and len(wells):
        dmso_wells = pd.to_numeric(
            wells.loc[wells[pert_col].astype(str) == str(negcon), "Metadata_Viability_pct"],
            errors="coerce",
        ).dropna()
        if len(dmso_wells):
            dmso_med = float(dmso_wells.median())
            checks["dmso_well_median_pct"] = round(dmso_med, 2)
            # DMSO median can drift below 100 when low-count DMSO outliers were
            # dropped from the reference (the surviving reference is higher). A
            # drift above 100 or far below is what we want to surface.
            if not (70.0 <= dmso_med <= 130.0):
                lg.warning("  Viability CHECK: DMSO wells median = %.1f%% (expected ≈100%%) — "
                           "the per-plate DMSO reference may be off", dmso_med)
            else:
                lg.info("  Viability CHECK: DMSO wells median = %.1f%% (≈100%% as expected)", dmso_med)

    return checks


def _aggregate_to_wells(sites: pd.DataFrame, config: PipelineConfig, threshold_pct: float) -> pd.DataFrame:
    """Roll site-level viability up to wells by median-of-ratios."""
    m = config.metrics
    bc = config.metadata.barcode_col
    well = config.metadata.well_col
    pert_col = m.pert_type_col
    negcon = m.negcon_value
    min_sites = config.metrics.viability_min_sites

    key = [c for c in [bc, well] if c in sites.columns]
    if not key:
        return pd.DataFrame()

    # Carry per-well annotation columns (first value within the well).
    carry = [c for c in [pert_col, config.metadata.compound_col, config.metadata.conc_col]
             if c in sites.columns]

    grp = sites.groupby(key, observed=True, sort=False)
    wells = grp.agg(
        Metadata_Viability_pct=("Metadata_site_viability_pct", "median"),
        Metadata_frac_sites_cytotoxic=("Metadata_site_viability_flag", "mean"),
        Metadata_Viability_n_sites=("Metadata_site_viability_pct", "count"),
        Metadata_dmso_ref_outlier_sites=("Metadata_dmso_ref_outlier", "sum"),
    ).reset_index()

    if carry:
        ann = grp[carry].first().reset_index()
        wells = wells.merge(ann, on=key, how="left")

    is_neg = (wells[pert_col].astype(str) == str(negcon)) if pert_col in wells.columns \
        else pd.Series(False, index=wells.index)
    wells["Metadata_Viability_flag"] = (
        (wells["Metadata_Viability_pct"] < threshold_pct) & (~is_neg)
    ).astype(int)
    wells["Metadata_Viability_low_confidence"] = (
        wells["Metadata_Viability_n_sites"] < min_sites
    ).astype(int)
    return wells


def _find_cell_count_col(metadata: pd.DataFrame, preferred: str = None) -> str:
    """
    Find the MEASURED per-site cell-count column.

    Prefers the configured column, then the measured nuclei count, then falls back
    to other count columns. Never returns Metadata_cell_line (it merely contains the
    substring "cell") and avoids the seeded density column when a measured count
    is available.
    """
    priority = [
        preferred,
        "Metadata_Count_nuclei",           # measured per-site nuclei count
        "Metadata_Cells_per_well_total",   # summed site nuclei per well (well-level fallback)
        "Metadata_Count_nuclei_sum",
        "Metadata_cells_per_well",         # last resort (may be the seeded density)
    ]
    for c in priority:
        if c and c in metadata.columns and pd.api.types.is_numeric_dtype(metadata[c]):
            return c

    # Generic fallback: a numeric column that looks like a count, excluding
    # cell_line / perturbation / concentration columns.
    for c in metadata.columns:
        cl = c.lower()
        if ("count" in cl or "nuclei" in cl) and "cell_line" not in cl \
                and "pert" not in cl and "conc" not in cl \
                and pd.api.types.is_numeric_dtype(metadata[c]):
            return c

    return None
