"""
Level 2 Data Loading & Preparation
===================================
Fetches CellProfiler features from database, applies QC,
merges with experiment metadata, outputs Level 2.5 site-level profiles.

Implementation uses the exact database schema and compartment merging from there.
"""

import glob
import logging
import pandas as pd
import numpy as np
import sqlalchemy
import time
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

HAS_SQLALCHEMY = True

import re


def canonical_well(well) -> str:
    """
    Normalize a well identifier to a canonical zero-padded form: <ROW><2-digit COL>.

    Feature tables use padded wells (e.g. "B02") while hand-made metadata files
    often mix padded and unpadded wells ("B2", "B20"). Without normalization a
    left join silently drops every well whose padding differs between sources.

    Examples: "B2" -> "B02", "b2" -> "B02", "B02" -> "B02", "B20" -> "B20".
    """
    m = re.match(r"^([A-Za-z]+)(\d+)$", str(well).strip())
    if not m:
        return str(well).strip().upper()
    return f"{m.group(1).upper()}{int(m.group(2)):02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Image-level QC helpers (blur/focus outliers + low cell-count sites)
# Adapted from CP_MS/BA/01_data_prep.py
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_site_id(series: pd.Series) -> pd.Series:
    """Normalize site identifiers to a clean string form for joins."""
    def _norm(v):
        if pd.isna(v):
            return v
        try:
            # 3.0 -> "3"
            return str(int(float(v)))
        except (ValueError, TypeError):
            return str(v).strip()
    return series.map(_norm)


def _robust_zscore(series: pd.Series) -> pd.Series:
    """Robust Z-score using median and MAD (1.4826 scaling)."""
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or pd.isna(mad):
        mad = s.std() or 1e-9
    return (s - med) / (1.4826 * mad)


def _find_qc_parquet(results_path: str) -> Optional[str]:
    """Find the ImageQuality parquet in a cp-qc results directory."""
    candidates = ["image.parquet", "qc_image.parquet", "Image.parquet",
                  "imageqc.parquet", "featICF_image.parquet"]
    for c in candidates:
        p = os.path.join(results_path, c)
        if os.path.exists(p):
            return p
    # Fallback: any parquet containing ImageQuality columns
    for p in glob.glob(os.path.join(results_path, "*.parquet")):
        try:
            cols = pd.read_parquet(p).columns.tolist()
            if any("ImageQuality" in c for c in cols):
                return p
        except Exception:
            continue
    return None


def _compute_image_qc_flags(df_qc: pd.DataFrame, config) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute per-site image QC outlier flags from ImageQuality_*_{channel} columns
    and the per-site nuclei count.

    Returns (df with OutlierScaled_* + Total columns, list of flag columns).
    """
    threshold = config.image_qc.z_threshold
    t_str = f"-{threshold:.0f}_{threshold:.0f}" if threshold == int(threshold) else f"-{threshold}_{threshold}"
    metrics = config.image_qc.qc_metrics

    df = df_qc.copy()
    flag_cols: List[str] = []

    for metric in metrics:
        metric_cols = [c for c in df.columns if f"ImageQuality_{metric}_" in c]
        if not metric_cols:
            continue
        # Focus-like metrics: worst channel (min); intensity metrics: mean
        if metric in ("FocusScore", "LocalFocusScore"):
            df[f"_agg_{metric}"] = df[metric_cols].min(axis=1)
        else:
            df[f"_agg_{metric}"] = df[metric_cols].mean(axis=1)

        z = _robust_zscore(df[f"_agg_{metric}"])
        flag_col = f"OutlierScaled_{metric}_{t_str}"
        df[flag_col] = (z.abs() > threshold).astype(int)
        flag_cols.append(flag_col)

    # Flag low cell-count sites
    count_col = config.image_qc.count_col
    if count_col in df.columns:
        min_n = config.image_qc.min_nuclei_per_site
        df["OutlierScaled_LowCellCount"] = (
            pd.to_numeric(df[count_col], errors="coerce") < min_n
        ).astype(int)
        flag_cols.append("OutlierScaled_LowCellCount")

    df["Total"] = df[flag_cols].sum(axis=1) if flag_cols else 0
    df.drop(columns=[c for c in df.columns if c.startswith("_agg_")], inplace=True)
    return df, flag_cols


def run_image_qc(df_qc_analyses: pd.DataFrame, config, out_base: str) -> Optional[pd.DataFrame]:
    """
    For each plate, load the cp-qc ImageQuality parquet, compute per-site flags,
    save per-plate + combined flag CSVs, and return the combined QC flags.
    """
    logger.info("Running image-level QC (blur/focus + low cell count)...")
    all_qc = []

    for _, row in df_qc_analyses.iterrows():
        barcode = row["plate_barcode"]
        acq_id = row["plate_acq_id"]
        results = row["results"]

        qc_path = _find_qc_parquet(results)
        if qc_path is None:
            logger.warning(f"  No ImageQuality parquet in {results} — {barcode} lacks image QC")
            continue

        df_qc = pd.read_parquet(qc_path)

        # Standardize identifying columns for the join
        rename_map = {}
        if "Metadata_Barcode" not in df_qc.columns:
            for cand in ("Image_Metadata_Barcode", "Barcode", "plate_barcode"):
                if cand in df_qc.columns:
                    rename_map[cand] = "Metadata_Barcode"
                    break
        if "Metadata_Well" not in df_qc.columns:
            for cand in ("Image_Metadata_Well", "Well"):
                if cand in df_qc.columns:
                    rename_map[cand] = "Metadata_Well"
                    break
        if "Metadata_Site" not in df_qc.columns:
            for cand in ("Image_Metadata_Site", "Site"):
                if cand in df_qc.columns:
                    rename_map[cand] = "Metadata_Site"
                    break
        if rename_map:
            df_qc = df_qc.rename(columns=rename_map)

        # Force barcode/acq for consistency
        df_qc["Metadata_Barcode"] = barcode
        if "Metadata_Site" in df_qc.columns:
            df_qc["Metadata_Site"] = _normalize_site_id(df_qc["Metadata_Site"])

        df_qc, flag_cols = _compute_image_qc_flags(df_qc, config)
        n_flagged = int((df_qc["Total"] > 0).sum())
        logger.info(f"  {barcode} ({acq_id}): {n_flagged}/{len(df_qc)} sites flagged "
                    f"({100*n_flagged/max(len(df_qc),1):.1f}%)")

        qc_out_dir = os.path.join(out_base, "_image_qc")
        os.makedirs(qc_out_dir, exist_ok=True)
        keep = ["Metadata_Barcode", "Metadata_Well", "Metadata_Site", config.image_qc.count_col,
                "Total"] + flag_cols
        keep = [c for c in keep if c in df_qc.columns]
        df_qc[keep].to_csv(os.path.join(qc_out_dir, f"qc_flags_{acq_id}.csv"), index=False)

        all_qc.append(df_qc[keep])

    if not all_qc:
        logger.warning("  No image QC data loaded — filter/count step will be skipped.")
        return None

    df_qc_all = pd.concat(all_qc, ignore_index=True)
    qc_out_dir = os.path.join(out_base, "_image_qc")
    os.makedirs(qc_out_dir, exist_ok=True)
    df_qc_all[df_qc_all["Total"] > 0].to_csv(
        os.path.join(qc_out_dir, "flagged_sites.csv"), index=False
    )
    logger.info(f"  Total QC rows: {len(df_qc_all)} | flagged: {int((df_qc_all['Total']>0).sum())}")
    return df_qc_all


def apply_image_qc(df_features: pd.DataFrame, df_qc_flags: Optional[pd.DataFrame],
                   config) -> pd.DataFrame:
    """
    Merge QC flags + real cell count (Count_nuclei) onto features.
    Drops flagged sites when config.image_qc.apply_filter is True; otherwise keeps
    them and stores the flag in Metadata_image_qc_flag.
    """
    if df_qc_flags is None:
        return df_features

    df_features = df_features.copy()
    df_features["Metadata_Site"] = _normalize_site_id(df_features["Metadata_Site"])
    df_qc_flags = df_qc_flags.copy()
    df_qc_flags["Metadata_Site"] = _normalize_site_id(df_qc_flags["Metadata_Site"])

    count_col = config.image_qc.count_col
    join_cols = ["Metadata_Barcode", "Metadata_Well", "Metadata_Site", "Total"]
    if count_col in df_qc_flags.columns:
        join_cols.append(count_col)
    flag_prefix = [c for c in df_qc_flags.columns if c.startswith("OutlierScaled")]
    qc_join = df_qc_flags[join_cols + flag_prefix].drop_duplicates(
        subset=["Metadata_Barcode", "Metadata_Well", "Metadata_Site"]
    )

    n_before = len(df_features)
    df_merged = df_features.merge(
        qc_join, on=["Metadata_Barcode", "Metadata_Well", "Metadata_Site"], how="left"
    )

    # Use the authoritative QC count as Metadata_Count_nuclei
    if count_col in df_merged.columns:
        df_merged["Metadata_Count_nuclei"] = pd.to_numeric(df_merged[count_col], errors="coerce")
        if count_col != "Metadata_Count_nuclei":
            df_merged.drop(columns=[count_col], inplace=True)
    elif "Metadata_Count_nuclei_merged" in df_merged.columns:
        df_merged["Metadata_Count_nuclei"] = df_merged["Metadata_Count_nuclei_merged"]

    # Record the flag
    df_merged["Metadata_image_qc_flag"] = (df_merged["Total"].fillna(0) > 0).astype(int)

    if config.image_qc.apply_filter:
        df_out = df_merged[(df_merged["Total"].isna()) | (df_merged["Total"] == 0)].copy()
        n_dropped = n_before - len(df_out)
        logger.info(f"  Image QC filter: dropped {n_dropped}/{n_before} sites "
                    f"({100*n_dropped/max(n_before,1):.1f}%)")
    else:
        df_out = df_merged
        logger.info(f"  Image QC: {int(df_merged['Metadata_image_qc_flag'].sum())} sites flagged (not dropped)")

    df_out.drop(columns=[c for c in (["Total"] + flag_prefix) if c in df_out.columns], inplace=True)
    return df_out


def apply_dmso_site_qc(df: pd.DataFrame, config, out_base: str) -> Tuple[pd.DataFrame, Dict]:
    """
    QC the DMSO (negcon) sites specifically, since they define the normalization
    reference. Flags DMSO sites with abnormally low cell count (absolute threshold
    OR robust-Z outlier on the low side) and drops them.

    Writes a per-site DMSO stats CSV and returns (filtered_df, stats).
    """
    stats: Dict = {"dmso_sites_total": 0, "dmso_sites_dropped": 0}
    pert_col = config.metadata.pert_type_col
    negcon = config.metrics.negcon_value
    count_col = "Metadata_Count_nuclei"

    if pert_col not in df.columns or count_col not in df.columns:
        logger.warning("  DMSO site QC skipped (missing pert_type or cell count)")
        return df, stats

    counts = pd.to_numeric(df[count_col], errors="coerce")
    is_dmso = df[pert_col] == negcon
    dmso_idx = df.index[is_dmso]
    stats["dmso_sites_total"] = int(is_dmso.sum())

    if is_dmso.sum() == 0:
        logger.warning("  DMSO site QC: no negcon sites found")
        return df, stats

    dmso_counts = counts.loc[dmso_idx]
    # Robust Z on DMSO cell counts computed PER PLATE. Within a plate all wells
    # are seeded at the same density, so a healthy plate has a tight DMSO count
    # distribution. Median/MAD (robust Z) is used rather than mean/std because
    # the outliers we want to drop would themselves inflate the std and shift the
    # mean, masking themselves; median and MAD are unaffected. Computing it per
    # plate avoids penalising a whole plate that simply ran denser or sparser.
    barcode_col = config.metadata.barcode_col
    if barcode_col in df.columns:
        plate_of_dmso = df.loc[dmso_idx, barcode_col]
        z = dmso_counts.groupby(plate_of_dmso).transform(_robust_zscore)
    else:
        z = _robust_zscore(dmso_counts)
    min_abs = config.image_qc.dmso_min_cells_per_site
    z_thr = config.image_qc.dmso_count_z_threshold

    # Drop a DMSO site if it is a low-side robust-Z outlier on its own plate,
    # or below a hard absolute floor (safety net for a whole-plate failure where
    # even the median is low and robust Z alone would not catch it).
    low_abs = dmso_counts < min_abs
    low_z = z < -abs(z_thr)
    dmso_bad = (low_abs | low_z)

    # Record per-DMSO-site stats
    site_col = config.metadata.site_col
    keep_cols = [c for c in ["Metadata_Barcode", "Metadata_Well", site_col] if c in df.columns]
    dmso_stats = df.loc[dmso_idx, keep_cols].copy()
    dmso_stats["cell_count"] = dmso_counts.values
    dmso_stats["robust_z"] = z.values
    dmso_stats["flag_low_abs"] = low_abs.values
    dmso_stats["flag_low_z"] = low_z.values
    dmso_stats["dropped"] = dmso_bad.values

    stats["dmso_sites_dropped"] = int(dmso_bad.sum())
    stats["dmso_count_min"] = float(dmso_counts.min())
    stats["dmso_count_max"] = float(dmso_counts.max())
    stats["dmso_count_mean"] = float(dmso_counts.mean())
    stats["dmso_count_median"] = float(dmso_counts.median())

    logger.info(
        f"  DMSO site QC: {stats['dmso_sites_dropped']}/{stats['dmso_sites_total']} DMSO sites flagged "
        f"(count min={stats['dmso_count_min']:.0f} / mean={stats['dmso_count_mean']:.0f} / "
        f"max={stats['dmso_count_max']:.0f}; drop if <{min_abs} or Z<-{z_thr})"
    )

    # Save DMSO site stats CSV
    qc_out_dir = os.path.join(out_base, "_image_qc")
    os.makedirs(qc_out_dir, exist_ok=True)
    dmso_stats.sort_values("cell_count").to_csv(
        os.path.join(qc_out_dir, "dmso_site_stats.csv"), index=False
    )

    if config.image_qc.drop_dmso_outliers and dmso_bad.any():
        drop_index = dmso_idx[dmso_bad.values]
        df = df.drop(index=drop_index).reset_index(drop=True)

    return df, stats


def write_qc_drop_summary(out_base: str, cell_line: str, summary: Dict) -> None:
    """Write a compact QC drop-accounting summary CSV for verification."""
    qc_out_dir = os.path.join(out_base, "_image_qc")
    os.makedirs(qc_out_dir, exist_ok=True)
    row = {"cell_line": cell_line, **summary}
    path = os.path.join(qc_out_dir, f"qc_drop_summary_{cell_line}.csv")
    pd.DataFrame([row]).to_csv(path, index=False)
    logger.info(f"  QC drop summary → {path}")


def query_db_analyses(db_uri: str, project: str, analysis_type: str, barcodes=None) -> pd.DataFrame:
    """
    Query image_analyses_per_plate for a project and analysis type.

    Returns DataFrame with columns:
      project, plate_barcode, plate_acq_name, plate_acq_id,
      analysis_id, analysis_date, results (path), pipeline_name

    Based on: EnvTox/01_data_prep.py query_db_analyses()
    """
    if not HAS_SQLALCHEMY:
        raise RuntimeError("sqlalchemy required for database queries.")

    project_like = f"{project}%"
    query = f"""
        SELECT project, plate_barcode, plate_acq_name, plate_acq_id,
               analysis_id, analysis_date, analysis_error, results, pipeline_name
        FROM   image_analyses_per_plate
        WHERE  project LIKE '{project_like}'
          AND  meta->>'type' = '{analysis_type}'
          AND  analysis_date IS NOT NULL
          AND  analysis_error IS NULL
        ORDER  BY plate_acq_id, analysis_id
    """

    logger.info(f"Querying database for project LIKE '{project_like}', type='{analysis_type}'")
    t0 = time.time()

    if not db_uri:
        raise ValueError(
            "No database URI configured. Set the IMAGEDB_URI environment variable "
            "(see README → 'Database access') or database.uri in the config."
        )
    try:
        engine = sqlalchemy.create_engine(db_uri)
        with engine.connect() as conn:
            df = pd.read_sql_query(sqlalchemy.text(query), conn)
    finally:
        engine.dispose()

    logger.info(f"Retrieved {len(df)} analysis rows in {time.time() - t0:.1f}s")

    if barcodes is not None:
        before = len(df)
        df = df[df["plate_barcode"].isin(barcodes)].copy()
        logger.info(f"Filtered to metadata barcodes: {len(df)}/{before} rows kept")

    return df


def _rename_compartment_cols(df: pd.DataFrame, compartment: str) -> pd.DataFrame:
    """
    Rename feature columns from suffix to CellProfiler prefix format.
    e.g. AreaShape_Area_nuclei → Nuclei_AreaShape_Area

    Based on: EnvTox/01_data_prep.py _rename_compartment_cols()
    """
    suffix = f"_{compartment.lower()}"
    prefix = compartment.capitalize()
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_") and c not in ("ImageID",)]

    matching = [c for c in feature_cols if c.endswith(suffix)]
    if not matching:
        return df

    new_cols = []
    for c in df.columns:
        if c in matching:
            base = c[: -len(suffix)]
            new_cols.append(f"{prefix}_{base}")
        else:
            new_cols.append(c)
    df.columns = new_cols

    return df


def _merge_compartments(results_path: str, plate_meta: dict, config) -> pd.DataFrame:
    """
    Merge nuclei, cells, cytoplasm parquets at cell level,
    aggregate to site level (mean/median).

    Returns one row per (Barcode, Well, Site).

    Based on: EnvTox/01_data_prep.py _merge_compartments()
    """
    comp_files = config.data_loading.compartment_files
    agg_fn = "mean" if config.aggregation.operation == "mean" else "median"
    plate_name = plate_meta.get("plate_acq_name", "unknown")
    acq_id = plate_meta.get("plate_acq_id", "na")

    # ── Cache: skip the expensive cell-level merge if a site-level cache exists ──
    cache_path = None
    if getattr(config.data_loading, "use_cache", False):
        cache_dir = os.path.join(
            config.output.base_dir or ".", config.data_loading.cache_subdir
        )
        os.makedirs(cache_dir, exist_ok=True)
        # Key on aggregation op so mean/median caches don't collide
        cache_path = os.path.join(cache_dir, f"sitelevel_{acq_id}_{agg_fn}.parquet")
        if os.path.exists(cache_path) and not config.data_loading.refresh_cache:
            try:
                cached = pd.read_parquet(cache_path)
                logger.info(f"    ✓ cache hit: {os.path.basename(cache_path)} "
                            f"({len(cached)} sites) — skipping DB merge")
                return cached
            except Exception as e:
                logger.warning(f"    cache read failed ({e}) — recomputing")

    logger.debug(f"Loading compartments from {results_path}")

    frames = {}
    for comp, fname in comp_files.items():
        fpath = os.path.join(results_path, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing {comp} parquet: {fpath}")

        df_comp = pd.read_parquet(fpath)
        # Add compartment suffix
        df_comp.columns = [f"{c}_{comp}" for c in df_comp.columns]
        frames[comp] = df_comp
        logger.debug(f"Loaded {comp}: {df_comp.shape[0]} cells x {df_comp.shape[1]} cols")

    # Merge: nuclei <- cells (via Parent_cells_nuclei = ObjectNumber_cells)
    df = frames["nuclei"].merge(
        frames["cells"],
        left_on=["Metadata_Barcode_nuclei", "Metadata_Site_nuclei", "Metadata_Well_nuclei", "Parent_cells_nuclei"],
        right_on=["Metadata_Barcode_cells", "Metadata_Site_cells", "Metadata_Well_cells", "ObjectNumber_cells"],
        how="left",
    )

    # Merge: + cytoplasm
    df = df.merge(
        frames["cytoplasm"],
        left_on=["Metadata_Barcode_nuclei", "Metadata_Site_nuclei", "Metadata_Well_nuclei", "Parent_cells_nuclei"],
        right_on=["Metadata_Barcode_cytoplasm", "Metadata_Site_cytoplasm", "Metadata_Well_cytoplasm", "ObjectNumber_cytoplasm"],
        how="left",
    )

    df.reset_index(drop=True, inplace=True)
    logger.debug(f"Merged compartments: {df.shape[0]} cells x {df.shape[1]} cols")

    # Patch metadata
    df["Metadata_Barcode_nuclei"] = plate_meta["plate_barcode"]
    df["Metadata_AcqID_nuclei"] = plate_meta["plate_acq_id"]

    # Standardize column names
    df.rename(
        columns={
            "Metadata_Barcode_nuclei": "Metadata_Barcode",
            "Metadata_Well_nuclei": "Metadata_Well",
            "Metadata_Site_nuclei": "Metadata_Site",
            "Metadata_AcqID_nuclei": "Metadata_AcqID",
        },
        inplace=True,
    )

    # ImageID
    df["ImageID"] = (
        df["Metadata_AcqID"].astype(str)
        + "_"
        + df["Metadata_Barcode"]
        + "_"
        + df["Metadata_Well"]
        + "_"
        + df["Metadata_Site"].astype(str)
    )

    # Aggregate cell-level → site-level
    group_keys = ["ImageID", "Metadata_Barcode", "Metadata_Well", "Metadata_Site", "Metadata_AcqID"]
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    agg_dict = {c: agg_fn for c in numeric_cols}
    grouped = df.groupby(group_keys, as_index=False).agg(agg_dict)

    # Capture the number of cells (nuclei) per site as a fallback cell count.
    # The authoritative count comes from the cp-qc image parquet (Count_nuclei),
    # but this preserves a real count even when image QC is unavailable.
    cell_counts = (
        df.groupby(group_keys, as_index=False)
          .size()
          .rename(columns={"size": "Metadata_Count_nuclei_merged"})
    )
    grouped = grouped.merge(cell_counts, on=group_keys, how="left")

    logger.debug(f"Aggregated to site level: {grouped.shape[0]} sites x {grouped.shape[1]} cols")

    # Rename cell count if the CellProfiler count column survived aggregation
    if "Count_nuclei_nuclei" in grouped.columns:
        grouped.rename(columns={"Count_nuclei_nuclei": "Metadata_Count_nuclei"}, inplace=True)

    # Rename compartment suffix → prefix format
    for compartment in ["nuclei", "cells", "cytoplasm"]:
        grouped = _rename_compartment_cols(grouped, compartment)

    # Write the site-level cache so future runs skip the cell-level merge
    if cache_path is not None:
        try:
            grouped.to_parquet(cache_path, index=False)
            logger.info(f"    ✓ cached site-level table → {os.path.basename(cache_path)}")
        except Exception as e:
            logger.warning(f"    cache write failed ({e})")

    return grouped


class DataLoader:
    """Load and prepare Level 2 data from database."""

    def __init__(self, config):
        self.config = config
        self.logger = logger

    def _select_analyses_from_metadata(self, df_analyses, metadata_df, id_col, kind):
        """Restrict queried DB analyses to the exact analysis_ids named in the
        metadata `id_col` (e.g. Metadata_pipeline_id_feat_texture / _qc).

        The metadata is authoritative about which acquisition/analysis to use per
        plate, so this avoids loading duplicate acquisitions that share a barcode.
        No-op (returns df unchanged) when `id_col` is not configured or absent from
        the metadata, so projects that don't provide ID columns keep the old
        load-everything behaviour.
        """
        if not id_col or metadata_df is None or id_col not in metadata_df.columns:
            return df_analyses
        wanted = set(
            pd.to_numeric(metadata_df[id_col], errors="coerce").dropna().astype(int).tolist()
        )
        if not wanted:
            self.logger.warning(f"  {kind}: metadata '{id_col}' has no usable ids — not filtering")
            return df_analyses
        aid = pd.to_numeric(df_analyses["analysis_id"], errors="coerce")
        before = len(df_analyses)
        filtered = df_analyses[aid.isin(wanted)].copy()
        self.logger.info(
            f"  {kind}: selected {len(filtered)}/{before} analyses via metadata "
            f"'{id_col}' ({len(wanted)} ids)"
        )
        found = set(pd.to_numeric(filtered["analysis_id"], errors="coerce").dropna().astype(int).tolist())
        missing = wanted - found
        if missing:
            self.logger.warning(
                f"  {kind}: {len(missing)} metadata id(s) not found in DB result: {sorted(missing)}"
            )
        return filtered

    def load_cell_line_data(self, metadata_df: pd.DataFrame, cell_line: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load Level 2 data for a cell line from database.

        Parameters
        ----------
        metadata_df : pd.DataFrame
            Metadata with barcode, well, cell_line, etc.
        cell_line : str

        Returns
        -------
        df_l2 : pd.DataFrame
            Level 2.5 (site-level) data with metadata
        stats : dict
            Loading statistics
        """
        self.logger.info(f"[{cell_line}] Loading Level 2 from database...")

        stats = {"cell_line": cell_line, "plates": 0, "sites": 0}

        # Get barcodes for this cell line
        meta_cond = metadata_df[self.config.metadata.cell_line_col] == cell_line
        if not meta_cond.any():
            self.logger.error(f"No metadata for {cell_line}")
            return None, stats

        barcodes = metadata_df.loc[meta_cond, self.config.metadata.barcode_col].unique().tolist()
        self.logger.info(f"  Found {len(barcodes)} plates for {cell_line}")

        # Query database
        try:
            df_analyses = query_db_analyses(
                self.config.database.uri,
                self.config.project_name,  # Use project name from config
                self.config.database.feature_analysis_type,
                barcodes=barcodes,
            )
        except Exception as e:
            self.logger.error(f"Database query failed: {e}")
            return None, stats

        # Metadata is authoritative: keep only the exact feature analysis named
        # per plate (feat_id_col), avoiding duplicate acquisitions on one barcode.
        df_analyses = self._select_analyses_from_metadata(
            df_analyses, metadata_df, self.config.metadata.feat_id_col, "cp-features"
        )

        if len(df_analyses) == 0:
            self.logger.warning("No analyses found in database")
            return None, stats

        # Extract and merge compartments for each plate
        all_sites = []
        for idx, (_, row) in enumerate(df_analyses.iterrows()):
            try:
                plate_name = row["plate_acq_name"]
                self.logger.info(f"  [{idx+1}/{len(df_analyses)}] {plate_name}")

                df_plate = _merge_compartments(row["results"], row.to_dict(), self.config)
                all_sites.append(df_plate)

            except Exception as e:
                self.logger.error(f"    Failed: {e}")

        if not all_sites:
            self.logger.error("No plates processed successfully")
            return None, stats

        # Combine all plates
        df_l2 = pd.concat(all_sites, ignore_index=True)
        out_base = self.config.output.base_dir or "."

        # QC drop accounting (for verification of how many sites were dropped)
        drop_summary = {"sites_loaded": int(len(df_l2))}

        # ── Image-level QC (blur/focus outliers + real per-site cell count) ──
        if getattr(self.config, "image_qc", None) and self.config.image_qc.run:
            try:
                df_qc_analyses = query_db_analyses(
                    self.config.database.uri,
                    self.config.project_name,
                    self.config.database.qc_analysis_type,
                    barcodes=barcodes,
                )
                df_qc_analyses = self._select_analyses_from_metadata(
                    df_qc_analyses, metadata_df, self.config.metadata.qc_id_col, "cp-qc"
                )
                if len(df_qc_analyses) > 0:
                    qc_flags = run_image_qc(df_qc_analyses, self.config, out_base)
                    n_before_imgqc = len(df_l2)
                    df_l2 = apply_image_qc(df_l2, qc_flags, self.config)
                    drop_summary["sites_dropped_image_qc"] = int(n_before_imgqc - len(df_l2))
                    stats["qc_flagged_sites"] = int(
                        df_l2["Metadata_image_qc_flag"].sum()
                    ) if "Metadata_image_qc_flag" in df_l2.columns else 0
                else:
                    logger.warning(f"  No cp-qc analyses found for {cell_line} — skipping image QC")
                    drop_summary["sites_dropped_image_qc"] = 0
            except Exception as e:
                logger.warning(f"  Image QC failed ({type(e).__name__}: {e}) — continuing without it")
                drop_summary["sites_dropped_image_qc"] = 0

        # Ensure a real cell count column exists (fallback to merge-based count)
        if "Metadata_Count_nuclei" not in df_l2.columns and "Metadata_Count_nuclei_merged" in df_l2.columns:
            df_l2["Metadata_Count_nuclei"] = df_l2["Metadata_Count_nuclei_merged"]

        # Merge with metadata on a canonical zero-padded well key so wells like
        # "B2" (metadata) and "B02" (features) join correctly. A raw join drops
        # every well whose padding differs between the two sources.
        right_bc = self.config.metadata.barcode_col
        right_well = self.config.metadata.well_col

        meta_join = metadata_df.copy()
        if right_bc != "Metadata_Barcode":
            meta_join = meta_join.rename(columns={right_bc: "Metadata_Barcode"})
        meta_join["_well_key"] = meta_join[right_well].map(canonical_well)
        # Drop the metadata well column so the features' padded well is kept.
        meta_join = meta_join.drop(columns=[right_well], errors="ignore")

        df_l2["_well_key"] = df_l2["Metadata_Well"].map(canonical_well)
        n_before = len(df_l2)
        df_l2 = df_l2.merge(
            meta_join,
            on=["Metadata_Barcode", "_well_key"],
            how="left",
        ).drop(columns=["_well_key"])

        # Report annotation coverage so unmatched wells are visible, not silent.
        if right_well in metadata_df.columns and self.config.metadata.pert_type_col in df_l2.columns:
            n_unmatched = int(df_l2[self.config.metadata.pert_type_col].isna().sum())
            if n_unmatched:
                logger.warning(
                    f"  {n_unmatched}/{n_before} sites have no metadata match "
                    f"(wells not present in {Path(self.config.metadata.file_path).name})"
                )

        # ── DMSO (negcon) site QC — protect the normalization reference ──
        if getattr(self.config, "image_qc", None) and self.config.image_qc.dmso_site_qc:
            try:
                n_before_dmso = len(df_l2)
                df_l2, dmso_stats = apply_dmso_site_qc(df_l2, self.config, out_base)
                drop_summary["sites_dropped_dmso_qc"] = int(n_before_dmso - len(df_l2))
                drop_summary.update(dmso_stats)
            except Exception as e:
                logger.warning(f"  DMSO site QC failed ({type(e).__name__}: {e}) — continuing")
                drop_summary["sites_dropped_dmso_qc"] = 0

        drop_summary["sites_remaining"] = int(len(df_l2))
        write_qc_drop_summary(out_base, cell_line, drop_summary)

        # Statistics
        stats["plates"] = df_l2["Metadata_Barcode"].nunique()
        stats["sites"] = len(df_l2)
        stats["wells"] = df_l2["Metadata_Well"].nunique()
        stats["drop_summary"] = drop_summary

        self.logger.info(
            f"[{cell_line}] Loaded {stats['sites']} sites, "
            f"{stats['wells']} wells, {stats['plates']} plates"
        )
        self.logger.info(
            f"[{cell_line}] QC accounting: loaded={drop_summary.get('sites_loaded')} | "
            f"image-QC dropped={drop_summary.get('sites_dropped_image_qc', 0)} | "
            f"DMSO-QC dropped={drop_summary.get('sites_dropped_dmso_qc', 0)} | "
            f"remaining={drop_summary.get('sites_remaining')}"
        )

        return df_l2, stats


def load_and_prepare_all_cell_lines(config) -> Dict[str, pd.DataFrame]:
    """
    Load and prepare Level 2.5 data for all configured cell lines.

    Orchestrates complete data loading workflow:
    1. Load metadata
    2. Query database for features
    3. Merge compartments (nuclei, cells, cytoplasm)
    4. Aggregate to site level
    5. Merge with metadata
    6. Return {cell_line: dataframe} dict

    Parameters
    ----------
    config : PipelineConfig
        Configuration with database URI, metadata file, compartment info

    Returns
    -------
    dict
        {cell_line_name: level_2_5_dataframe, ...}
        Returns empty dict if no data loaded
    """
    loader = DataLoader(config)

    # Load metadata
    try:
        metadata_path = config.metadata.file_path
        logger.info(f"Loading metadata from {metadata_path}...")
        metadata_df = pd.read_csv(metadata_path, sep=config.metadata.sep, dtype=str)
        logger.info(f"Loaded {len(metadata_df)} rows from metadata")
        logger.info(f"Metadata columns: {metadata_df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}", exc_info=True)
        return {}

    # Determine cell lines to process
    cell_lines = config.metadata.cell_lines
    if not cell_lines:
        try:
            cell_lines = sorted(metadata_df[config.metadata.cell_line_col].unique().tolist())
            logger.info(f"Auto-discovered cell lines: {cell_lines}")
        except Exception as e:
            logger.error(f"Failed to discover cell lines: {e}", exc_info=True)
            return {}

    # Load Level 2.5 for each cell line
    data_by_cellline = {}
    for cell_line in cell_lines:
        df_l2, stats = loader.load_cell_line_data(metadata_df, cell_line)
        if df_l2 is not None and len(df_l2) > 0:
            data_by_cellline[cell_line] = df_l2
            logger.info(f"✓ {cell_line}: {stats}")
        else:
            logger.warning(f"✗ {cell_line}: No data loaded")

    logger.info(f"\nLoaded {len(data_by_cellline)} cell line(s)")
    return data_by_cellline
