"""
Normalization pipeline: Level 3, 4a, 4b.
Aggregates sites to wells, applies MAD normalization, feature selection.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from pycytominer import normalize, aggregate
from pycytominer import feature_select as pycytominer_feature_select
import logging

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def aggregate_sites_to_wells(
    df: pd.DataFrame,
    config: PipelineConfig,
    cell_line: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Level 3: Aggregate sites to wells (median or mean).

    Parameters
    ----------
    df : pd.DataFrame
        Level 2.5 site-level data
    config : PipelineConfig
    cell_line : str

    Returns
    -------
    df_agg : pd.DataFrame
        Aggregated (well-level) features
    metadata_agg : pd.DataFrame
        Metadata (one row per well)
    """
    # Separate metadata from features
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]

    metadata = df[meta_cols].copy()
    features = df[feature_cols].copy()

    # Apply cell count filters at site level
    min_cells_site = config.aggregation.min_dmso_cells_per_site
    min_cells_trt = config.aggregation.min_cells_per_site

    # Filter sites by cell count. Prefer the real per-site nuclei count
    # (Metadata_Count_nuclei from image QC); fall back to legacy count columns.
    cell_count_col = None
    for candidate in ("Metadata_Count_nuclei", "Metadata_Count_nuclei_merged"):
        if candidate in metadata.columns:
            cell_count_col = candidate
            break
    if cell_count_col is None:
        for col in metadata.columns:
            if "Count_Cells" in col or "count_cells" in col.lower():
                cell_count_col = col
                break

    n_dropped_lowcount = 0
    if cell_count_col:
        counts = pd.to_numeric(metadata[cell_count_col], errors="coerce")
        is_dmso = metadata[config.metrics.pert_type_col] == config.metrics.negcon_value
        is_low_dmso = is_dmso & (counts < min_cells_site)
        is_low_trt = (~is_dmso) & (counts < min_cells_trt)

        n_dropped_lowcount = int((is_low_dmso | is_low_trt).sum())
        if n_dropped_lowcount > 0:
            logger.info(
                f"  Dropping {n_dropped_lowcount} sites due to low cell count "
                f"(<{min_cells_site} DMSO / <{min_cells_trt} trt, col={cell_count_col})"
            )

        keep_mask = ~(is_low_dmso | is_low_trt)
        metadata = metadata[keep_mask].reset_index(drop=True)
        features = features[keep_mask].reset_index(drop=True)
    else:
        logger.warning("  No cell count column found — site cell-count filter skipped")

    # Preserve a real per-well total cell count (sum of surviving site counts)
    well_count_series = None
    if cell_count_col:
        _wk = [c for c in (config.metadata.barcode_col, config.metadata.well_col) if c in metadata.columns]
        if _wk:
            well_count_series = (
                pd.to_numeric(metadata[cell_count_col], errors="coerce")
                .groupby([metadata[k] for k in _wk]).transform("sum")
            )

    # Reattach (possibly cell-count-filtered) metadata to features for aggregation
    df_full = pd.concat(
        [metadata.reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )

    # Well identity = plate + well
    well_keys = [config.metadata.barcode_col, config.metadata.well_col]
    well_keys = [c for c in well_keys if c in df_full.columns]

    # Strata = metadata columns that are constant within a well. This keeps the
    # treatment annotations (compound, concentration, pert_type, cell line, ...)
    # and drops per-site columns (Site, AcqID, z_*) that vary within a well.
    other_meta = [c for c in meta_cols if c not in well_keys]
    nunique = df_full.groupby(well_keys, dropna=False, observed=True)[other_meta].nunique(dropna=False)
    constant_meta = [c for c in other_meta if bool((nunique[c] <= 1).all())]
    strata = well_keys + constant_meta

    agg_func = "median" if config.aggregation.operation == "median" else "mean"

    # Aggregate numeric features (median/mean) grouped by well.
    numeric_feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df_full[c])
    ]
    df_agg = aggregate(
        population_df=df_full,
        strata=strata,
        features=numeric_feature_cols,
        operation=agg_func,
    )

    # Attach real per-well total cell count (sum across surviving sites)
    if cell_count_col:
        _wk = [c for c in (config.metadata.barcode_col, config.metadata.well_col) if c in metadata.columns]
        if _wk:
            well_totals = (
                pd.to_numeric(metadata[cell_count_col], errors="coerce")
                .groupby([metadata[k] for k in _wk]).sum()
                .rename("Metadata_Cells_per_well_total").reset_index()
            )
            df_agg = df_agg.merge(well_totals, on=_wk, how="left")

    metadata_agg = df_agg[[c for c in df_agg.columns if c.startswith("Metadata_")]].copy()

    logger.info(
        f"  {len(df)} sites → {len(df_agg)} wells "
        f"({len(numeric_feature_cols)} features)"
    )

    # Expose low-count drop count for the run summary
    aggregate_sites_to_wells.last_n_dropped_lowcount = n_dropped_lowcount

    return df_agg, metadata_agg


def _filter_low_count_sites(df: pd.DataFrame, config: PipelineConfig) -> Tuple[pd.DataFrame, int]:
    """Same per-site cell-count cutoffs used inside aggregate_sites_to_wells,
    factored out so the site-level (Level 2.5a/2.5b) path can apply the
    identical filter without aggregating to wells."""
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    min_cells_site = config.aggregation.min_dmso_cells_per_site
    min_cells_trt = config.aggregation.min_cells_per_site

    cell_count_col = None
    for candidate in ("Metadata_Count_nuclei", "Metadata_Count_nuclei_merged"):
        if candidate in df.columns:
            cell_count_col = candidate
            break
    if cell_count_col is None:
        for col in meta_cols:
            if "Count_Cells" in col or "count_cells" in col.lower():
                cell_count_col = col
                break

    if cell_count_col is None:
        logger.warning("  [site-level] No cell count column found — site cell-count filter skipped")
        return df.copy(), 0

    counts = pd.to_numeric(df[cell_count_col], errors="coerce")
    is_dmso = df[config.metrics.pert_type_col] == config.metrics.negcon_value
    is_low_dmso = is_dmso & (counts < min_cells_site)
    is_low_trt = (~is_dmso) & (counts < min_cells_trt)
    n_dropped = int((is_low_dmso | is_low_trt).sum())
    keep_mask = ~(is_low_dmso | is_low_trt)
    return df[keep_mask].reset_index(drop=True), n_dropped


def normalize_and_select_sites(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Level 2.5a / 2.5b: per-plate MAD normalization + feature selection applied
    at the imaging SITE level (no site→well aggregation). This is the
    site-level analogue of Level 4a (normalize_per_plate) / 4b (feature_select) —
    same DMSO reference, same clipping, same feature_select operations — just
    one row per site instead of one row per well, so outlier SITES within a
    compound×dose replicate group can be inspected/flagged before Level 3
    smooths them into the well median/mean.

    Parameters
    ----------
    df : pd.DataFrame
        Level 2.5 site-level data (already harmonized/filtered as pipeline.run
        does before Level 3 aggregation).
    config : PipelineConfig

    Returns
    -------
    df_25a, metadata_25a : normalized site-level table (Level 2.5a)
    df_25b, metadata_25b : + feature-selected site-level table (Level 2.5b)
    n_dropped_lowcount   : sites dropped by the site cell-count filter
    """
    # ImageID is a string site key without the Metadata_ prefix; left as-is it
    # is treated as a feature and pycytominer normalize fails to cast it.
    if "ImageID" in df.columns and "Metadata_ImageID" not in df.columns:
        df = df.rename(columns={"ImageID": "Metadata_ImageID"})

    df_filt, n_dropped = _filter_low_count_sites(df, config)
    if n_dropped:
        logger.info(f"  [site-level] Dropping {n_dropped} sites due to low cell count")

    meta_cols = [c for c in df_filt.columns if c.startswith("Metadata_")]
    df_25a, metadata_25a = normalize_per_plate(df_filt, df_filt[meta_cols], config)
    df_25b, metadata_25b = feature_select(df_25a, metadata_25a, config)

    logger.info(f"  [site-level] Level 2.5a/2.5b: {len(df_25b)} sites x "
                f"{len([c for c in df_25b.columns if not c.startswith('Metadata_')])} features")

    return df_25a, metadata_25a, df_25b, metadata_25b, n_dropped


def normalize_per_plate(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Level 4a: Per-plate MAD normalization with pycytominer.

    Mirrors CP_MS/BA/03_cellpainting_pipeline.py step_normalize():
      1. Filter globally-invariant features on RAW DMSO (MAD ≤ threshold)
      2. Drop rows whose features are all-NaN
      3. Normalize each plate against its own DMSO with pycytominer.normalize()
      4. Replace inf/NaN (from zero-MAD features) with 0
      5. Clip extreme values to [clip_min, clip_max]

    Blocklist removal is deferred to Level 4b (pycytominer feature_select
    "blocklist" operation), exactly as in the reference pipeline.
    """
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]

    pert_col = config.metadata.pert_type_col
    negcon = config.metrics.negcon_value
    mad_threshold = config.normalization.mad_threshold
    samples_query = config.normalization.samples_query
    clip_min = float(config.normalization.clip_min)
    clip_max = float(config.normalization.clip_max)
    plate_col = config.metadata.barcode_col

    # ── Step 1: filter globally-invariant features on RAW DMSO ───────────────
    ref_query = config.normalization.invariant_filter_reference_query
    try:
        df_ref = df.query(ref_query) if ref_query else df[df[pert_col] == negcon]
    except Exception:
        df_ref = df[df[pert_col] == negcon] if pert_col in df.columns else df
    if df_ref.empty:
        df_ref = df
    mad_raw = (df_ref[feature_cols] - df_ref[feature_cols].median()).abs().median()
    feat_list = mad_raw[mad_raw > mad_threshold].index.tolist()
    n_invariant = len(feature_cols) - len(feat_list)
    if n_invariant:
        logger.info(f"  Dropping {n_invariant} invariant features (MAD ≤ {mad_threshold} on DMSO)")

    df = df[meta_cols + feat_list].copy()

    # ── Step 2: drop rows where all features are NaN ─────────────────────────
    n_before = len(df)
    df = df.dropna(subset=feat_list, how="all").copy()
    if n_before - len(df):
        logger.info(f"  Dropped {n_before - len(df)} rows with all-NaN features")

    # ── Step 3: normalize per plate against that plate's own DMSO ────────────
    if plate_col in df.columns:
        parts = []
        for plate, grp in df.groupby(plate_col, sort=False):
            try:
                parts.append(normalize(
                    profiles=grp,
                    features=feat_list,
                    meta_features=meta_cols,
                    samples=samples_query if samples_query else "all",
                    method=config.normalization.method,
                    mad_robustize_epsilon=mad_threshold,
                ))
            except Exception as e:
                logger.warning(f"  Normalization failed for plate {plate}: {e} — using raw values")
                parts.append(grp)
        df_norm = pd.concat(parts, ignore_index=True)
    else:
        logger.warning(f"  Plate column '{plate_col}' not found — normalizing globally")
        df_norm = normalize(
            df, features=feat_list, meta_features=meta_cols,
            samples=samples_query if samples_query else "all",
            method=config.normalization.method, mad_robustize_epsilon=mad_threshold,
        )

    # ── Step 4: replace inf/NaN from zero-MAD features with 0 ────────────────
    df_norm[feat_list] = df_norm[feat_list].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ── Step 5: clip extreme values ─────────────────────────────────────────
    df_norm[feat_list] = np.clip(df_norm[feat_list].to_numpy(dtype=float), clip_min, clip_max)

    logger.info(f"  {len(feat_list)} features after normalization (clipped to [{clip_min}, {clip_max}])")

    metadata_df = df_norm[[c for c in df_norm.columns if c.startswith("Metadata_")]].copy()

    # ── Step 6: validate DMSO is centered near 0 after normalization ─────────
    normalize_per_plate.last_dmso_center_stats = _validate_dmso_centering(
        df_norm, feat_list, config,
    )

    return df_norm, metadata_df


def _validate_dmso_centering(
    df_norm: pd.DataFrame,
    feat_list: list,
    config: PipelineConfig,
) -> dict:
    """
    Post-normalization sanity check: DMSO (negcon) wells should have ~0 mean
    feature value, per plate, once mad_robustize has been fit against that
    plate's own DMSO reference. A plate whose DMSO mean drifts far from 0
    indicates a normalization failure for that plate (e.g. too few DMSO wells,
    or the fallback-to-raw-values path in step 3 above).

    Returns a stats dict (per-plate + overall mean-of-means, and the count of
    plates outside tolerance) that the caller can fold into the run summary.
    Never raises — this is diagnostic only.
    """
    pert_col = config.metadata.pert_type_col
    negcon = config.metrics.negcon_value
    plate_col = config.metadata.barcode_col
    tolerance = getattr(config.normalization, "dmso_center_tolerance", 0.5)

    if pert_col not in df_norm.columns or not feat_list:
        return {}

    dmso = df_norm[df_norm[pert_col] == negcon]
    if dmso.empty:
        logger.warning("  DMSO-centering check skipped — no negcon wells found")
        return {}

    if plate_col in dmso.columns:
        per_plate_mean = dmso.groupby(plate_col, observed=True)[feat_list].mean().mean(axis=1)
    else:
        per_plate_mean = pd.Series({"all": dmso[feat_list].mean().mean()})

    overall_mean = float(per_plate_mean.mean())
    off_center = per_plate_mean[per_plate_mean.abs() > tolerance]
    if len(off_center):
        logger.warning(
            "  DMSO-centering check: %d/%d plate(s) have |mean DMSO feature value| > %.2f "
            "after normalization: %s",
            len(off_center), len(per_plate_mean), tolerance,
            {k: round(v, 3) for k, v in off_center.items()},
        )
    else:
        logger.info(
            "  DMSO-centering check: OK — all %d plate(s) within ±%.2f of 0 "
            "(overall mean %.4f)", len(per_plate_mean), tolerance, overall_mean,
        )

    return {
        "dmso_center_overall_mean": overall_mean,
        "dmso_center_tolerance": tolerance,
        "dmso_center_n_plates": int(len(per_plate_mean)),
        "dmso_center_n_off_tolerance": int(len(off_center)),
        "dmso_center_per_plate_mean": {str(k): float(v) for k, v in per_plate_mean.items()},
    }


def feature_select(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Level 4b: Feature selection (variance, correlation, outliers).

    Parameters
    ----------
    df : pd.DataFrame
        Level 4a normalized data
    metadata : pd.DataFrame
    config : PipelineConfig

    Returns
    -------
    df_selected : pd.DataFrame
        Features after selection
    metadata : pd.DataFrame
    """
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]

    # Run pycytominer feature selection
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    df_selected = pycytominer_feature_select(
        df,
        features=feature_cols,
        samples="all",
        operation=config.feature_selection.operations,
    )

    n_remaining = len(df_selected.columns) - len(meta_cols)
    logger.info(f"  {n_remaining} features after selection")

    return df_selected, metadata


