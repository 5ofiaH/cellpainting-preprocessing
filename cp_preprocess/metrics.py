"""
Proper Metrics Computation


Computes real metrics using external libraries:
- Phenotypic Activity (copairs): Mean Average Precision of replicates vs controls
- Phenotypic Consistency (copairs): Mean Average Precision of compounds grouped by MOA/target
- mAP (copairs): Mean Average Precision
- GRIT (cytominer_eval): Gene Rank Index Threshold
- Reproducibility: Replicate correlation

All metrics are computed on Level 4b (feature-selected, PRE batch correction)
well-level data, so they reflect the raw replicate structure rather than the
residuals of a batch-correction model.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Fix NumPy 2.x compatibility issue in cytominer_eval
import numpy
if not hasattr(numpy, 'float'):
    # numpy.float / numpy.int are aliases removed in NumPy 1.24+, but
    # cytominer_eval still imports them. Re-add only these removed aliases.
    # NOTE: do NOT reassign numpy.float64 — it is a real, in-use dtype and
    # overwriting it with the Python float builtin corrupts pandas' C-parser
    # dtype table (KeyError: <class 'float'> on any pd.read_csv).
    numpy.float = float
    numpy.int = int

# Now safe to import
try:
    from cytominer_eval import evaluate as cytominer_evaluate
    HAS_CYTOMINER_EVAL = True
except ImportError as e:
    logger.warning(f"cytominer_eval import failed: {e}")
    HAS_CYTOMINER_EVAL = False

try:
    from copairs import map as copairs_map
    from copairs.matching import assign_reference_index
    HAS_COPAIRS = True
except ImportError as e:
    logger.warning(f"copairs import failed: {e}")
    HAS_COPAIRS = False


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return columns that are CellProfiler features (not Metadata_*)."""
    return [c for c in df.columns if not c.startswith("Metadata_")]


def get_meta_cols(df: pd.DataFrame) -> list:
    """Return Metadata_* columns."""
    return [c for c in df.columns if c.startswith("Metadata_")]


def compute_phenotypic_activity(
    df: pd.DataFrame,
    config,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Compute phenotypic activity using mAP (copairs).

    Phenotypic activity measures how distinguishable each perturbation is from
    negative controls. Positive pairs are replicate wells of the SAME COMPOUND
    at the SAME DOSE. Negative pairs are any wells from different compounds.

    Returns both replicate-level AP scores and compound-level mAP scores.

    Reference: Kalinin et al., Nature Communications (2025)
    https://github.com/cytomining/copairs

    Parameters
    ----------
    df : pd.DataFrame
        Well-level profiles (Level 4b) with metadata and features
    config : PipelineConfig

    Returns
    -------
    tuple
        (replicate_ap_df, compound_map_df) or (None, None) if failed
    """
    if not HAS_COPAIRS:
        logger.warning("copairs not installed — skipping phenotypic activity")
        return None, None

    print("  Computing phenotypic activity (mAP via copairs)...")

    pert_col = config.metrics.pert_type_col
    negcon = config.metrics.negcon_value
    ref_col = "Metadata_reference_index"

    feat_cols = get_feature_cols(df)

    # Need compound and concentration columns
    if "Metadata_compound" not in df.columns or "Metadata_conc_uM" not in df.columns:
        logger.warning("Metadata_compound or Metadata_conc_uM missing — skipping phenotypic activity")
        return None, None

    try:
        # Assign reference index to DMSO controls
        df_act = assign_reference_index(
            df,
            condition=f"{pert_col} == '{negcon}'",
            reference_col=ref_col,
            default_value=-1,
        )

        # Define pair rules
        pos_sameby = ["Metadata_compound", "Metadata_conc_uM", ref_col]
        pos_diffby = []
        neg_sameby = []
        neg_diffby = ["Metadata_compound", "Metadata_conc_uM", ref_col]

        # Clean up rows with missing metadata
        required_meta = [c for c in [pert_col, "Metadata_compound", "Metadata_conc_uM", ref_col]
                         if c in df_act.columns]
        before_meta = len(df_act)
        df_act = df_act.dropna(subset=required_meta).copy()
        dropped_meta = before_meta - len(df_act)
        if dropped_meta:
            logger.info(f"    Dropped {dropped_meta} rows with missing metadata")

        metadata = df_act[required_meta].copy()
        profiles = df_act[feat_cols].fillna(0).values

        # Compute replicate-level average precision
        ap_df = copairs_map.average_precision(
            metadata,
            profiles,
            pos_sameby,
            pos_diffby,
            neg_sameby,
            neg_diffby,
            distance="cosine",
            batch_size=5000,
        )

        # Add -log10(p-value) for replicates
        if "p_value" in ap_df.columns:
            ap_df["-log10(p-value)"] = -np.log10(ap_df["p_value"].clip(lower=1e-10))

        logger.info(f"    Replicate AP computed for {len(ap_df)} replicates")

        # Filter out DMSO rows for compound-level aggregation
        ap_df_trt = ap_df[ap_df[pert_col] != negcon].copy()

        # Compute compound-level mean average precision
        map_df = copairs_map.mean_average_precision(
            ap_df_trt,
            pos_sameby,
            null_size=getattr(config.metrics, "map_null_size", 500),
            threshold=getattr(config.metrics, "map_fdr_threshold", 0.05),
            seed=42,
        )

        # Add -log10(p-value)
        if "corrected_p_value" in map_df.columns:
            map_df["-log10(p-value)"] = -np.log10(map_df["corrected_p_value"].clip(lower=1e-10))
        elif "p_value" in map_df.columns:
            map_df["-log10(p-value)"] = -np.log10(map_df["p_value"].clip(lower=1e-10))

        # Add compound_dose column
        if {"Metadata_compound", "Metadata_conc_uM"}.issubset(map_df.columns):
            map_df["Metadata_compound_dose"] = (
                map_df["Metadata_compound"].astype(str) + "_" +
                map_df["Metadata_conc_uM"].astype(str)
            )

        logger.info(f"    Phenotypic activity (mAP) computed for {len(map_df)} compound×dose groups")
        return ap_df, map_df

    except Exception as e:
        logger.error(f"    Phenotypic activity computation failed: {e}")
        return None, None


def compute_phenotypic_consistency(
    df: pd.DataFrame,
    map_df: pd.DataFrame,
    config,
) -> Optional[pd.DataFrame]:
    """
    Compute phenotypic consistency using mAP for grouped compounds.

    Phenotypic consistency assesses whether compounds targeting the same gene/MOA
    have similar phenotypic profiles. Groups are defined by a configurable metadata
    column (e.g., Metadata_moa, Metadata_target).

    Only includes compounds that passed phenotypic activity threshold (mAP p-value < 0.05).

    Reference: Kalinin et al., Nature Communications (2025)
    https://github.com/cytomining/copairs

    Parameters
    ----------
    df : pd.DataFrame
        Original well-level profiles (Level 4b)
    map_df : pd.DataFrame
        Compound-level mAP scores from phenotypic activity
    config : PipelineConfig
        Must have config.metrics.consistency_col set (e.g., "Metadata_moa")

    Returns
    -------
    consistency_df : pd.DataFrame or None
        Consistency results with columns: grouping_col, mean_average_precision, p_value, ...
    """
    if not HAS_COPAIRS:
        logger.warning("copairs not installed — skipping phenotypic consistency")
        return None

    if not hasattr(config.metrics, "consistency_col") or not config.metrics.consistency_col:
        logger.warning("consistency_col not configured — skipping phenotypic consistency")
        return None

    print(f"  Computing phenotypic consistency (grouped by {config.metrics.consistency_col})...")

    feat_cols = get_feature_cols(df)
    consistency_col = config.metrics.consistency_col

    # Check that consistency column exists
    if consistency_col not in df.columns:
        logger.warning(f"Column {consistency_col} not in dataframe — skipping phenotypic consistency")
        return None

    try:
        # Filter to phenotypically active compounds (mAP corrected p-value < 0.05)
        if "corrected_p_value" not in map_df.columns:
            logger.warning("No corrected_p_value in mAP results — using p_value threshold")
            active_threshold_col = "p_value"
        else:
            active_threshold_col = "corrected_p_value"

        active_compounds = map_df.query(f"{active_threshold_col} < 0.05")["Metadata_compound"].unique()
        logger.info(f"    Filtering to {len(active_compounds)} phenotypically active compounds")

        if len(active_compounds) == 0:
            logger.warning("No phenotypically active compounds found")
            return None

        # Filter to active compounds only
        df_active = df[df["Metadata_compound"].isin(active_compounds)].copy()

        # Aggregate replicates by taking median of features
        # Group by compound and consistency column
        group_cols = ["Metadata_compound", consistency_col]
        df_active_agg = df_active.groupby(group_cols, as_index=False)[feat_cols].median()

        # Check if consistency column contains multi-label data (pipe-separated)
        if df_active_agg[consistency_col].dtype == object:
            has_multilabel = (
                df_active_agg[consistency_col].astype(str).str.contains("|", regex=False).any()
            )
        else:
            has_multilabel = False

        # Positive pairs need >=2 active compounds sharing a label; drop
        # singleton labels (copairs raises UnpairedException otherwise).
        if has_multilabel:
            df_active_agg[consistency_col] = (
                df_active_agg[consistency_col].astype(str).str.split("|", regex=False)
                .apply(lambda labels: [l.strip() for l in labels if l.strip()])
            )
            label_counts = df_active_agg[consistency_col].explode().value_counts()
            shared = set(label_counts[label_counts >= 2].index)
            df_active_agg = df_active_agg[
                df_active_agg[consistency_col].apply(lambda labels: any(l in shared for l in labels))
            ].reset_index(drop=True)
        else:
            label_counts = df_active_agg[consistency_col].value_counts()
            shared = set(label_counts[label_counts >= 2].index)
            df_active_agg = df_active_agg[
                df_active_agg[consistency_col].isin(shared)
            ].reset_index(drop=True)

        if len(shared) == 0 or len(df_active_agg) < 2:
            logger.warning(
                f"No {consistency_col} label is shared by >=2 phenotypically active "
                f"compounds ({len(active_compounds)} active) — skipping phenotypic consistency"
            )
            return None
        logger.info(
            f"    {len(shared)} {consistency_col} groups with >=2 active compounds "
            f"({len(df_active_agg)} compounds)"
        )

        if has_multilabel:
            logger.info(f"    Multi-label data detected in {consistency_col}, using multilabel AP")
            ap_fn = copairs_map.multilabel.average_precision
            multilabel_col = consistency_col
        else:
            ap_fn = copairs_map.average_precision
            multilabel_col = None

        # Define pair rules for consistency
        pos_sameby = [consistency_col]
        pos_diffby = []
        neg_sameby = []
        neg_diffby = [consistency_col]

        metadata = df_active_agg[["Metadata_compound", consistency_col]].copy()
        profiles = df_active_agg[feat_cols].fillna(0).values

        # Compute average precision for each compound
        if multilabel_col:
            consistency_ap = ap_fn(
                metadata,
                profiles,
                pos_sameby=pos_sameby,
                pos_diffby=pos_diffby,
                neg_sameby=neg_sameby,
                neg_diffby=neg_diffby,
                multilabel_col=multilabel_col,
                distance="cosine",
            )
        else:
            consistency_ap = ap_fn(
                metadata,
                profiles,
                pos_sameby,
                pos_diffby,
                neg_sameby,
                neg_diffby,
                distance="cosine",
            )

        # Compute mean average precision per group
        consistency_map = copairs_map.mean_average_precision(
            consistency_ap,
            pos_sameby,
            null_size=getattr(config.metrics, "consistency_null_size", 500),
            threshold=getattr(config.metrics, "consistency_fdr_threshold", 0.05),
            seed=42,
        )

        # Add -log10(p-value)
        if "corrected_p_value" in consistency_map.columns:
            consistency_map["-log10(p-value)"] = -np.log10(
                consistency_map["corrected_p_value"].clip(lower=1e-10)
            )
        elif "p_value" in consistency_map.columns:
            consistency_map["-log10(p-value)"] = -np.log10(
                consistency_map["p_value"].clip(lower=1e-10)
            )

        logger.info(f"    Phenotypic consistency computed for {len(consistency_map)} groups")
        return consistency_map

    except Exception as e:
        logger.error(f"    Phenotypic consistency computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_map(
    df: pd.DataFrame,
    config,
) -> Optional[pd.DataFrame]:
    """
    Legacy wrapper for compute_phenotypic_activity.
    Returns compound-level mAP scores only.
    """
    _, map_df = compute_phenotypic_activity(df, config)
    return map_df


def compute_grit(
    df: pd.DataFrame,
    config,
) -> Optional[pd.DataFrame]:
    """
    Compute GRIT (Gene Rank Index Threshold) using cytominer_eval.

    GRIT measures the robustness of feature expression and is computed per well
    using similarity metrics between technical/biological replicates.

    Reference: cytominer_eval library
    https://github.com/cytomining/cytominer_eval

    Parameters
    ----------
    df : pd.DataFrame
        Well-level profiles (Level 4b) with metadata and features
    config : PipelineConfig

    Returns
    -------
    grit_df : pd.DataFrame or None
        GRIT results with columns: Metadata_compound, Metadata_conc_uM, grit, grit_std, ...
    """
    if not HAS_CYTOMINER_EVAL:
        logger.warning("cytominer_eval not installed — skipping GRIT")
        return None

    print("  Computing GRIT (cytominer_eval / pearson)...")

    pert_col = config.metrics.pert_type_col
    negcon = config.metrics.negcon_value
    feat_cols = get_feature_cols(df)

    if "Metadata_compound" not in df.columns:
        logger.warning("Metadata_compound missing — skipping GRIT")
        return None

    try:
        df = df.copy()

        # Ensure required columns exist
        if "Metadata_conc_uM" not in df.columns:
            df["Metadata_conc_uM"] = "NA"

        if "Metadata_ImageID" not in df.columns:
            # Fallback: create from Barcode_Well[_Site]. At well level (Level 4b)
            # there is no Site column, so Barcode_Well already identifies each
            # well uniquely; only append Site when the column is present (df.get
            # otherwise returns the scalar default, which has no .astype).
            image_id = (
                df["Metadata_Barcode"].astype(str) + "_" +
                df["Metadata_Well"].astype(str)
            )
            if "Metadata_Site" in df.columns:
                image_id = image_id + "_" + df["Metadata_Site"].astype(str)
            df["Metadata_ImageID"] = image_id

        # Create replicate identifiers (based on your script)
        # Metadata_Replicate: unique per image (compound_dose_ImageID)
        df["Metadata_Replicate"] = (
            df["Metadata_compound"].astype(str) + "_" +
            df["Metadata_conc_uM"].astype(str) + "_" +
            df["Metadata_ImageID"].astype(str)
        )

        # Metadata_ReplicateGroup: groups by compound-dose
        df["Metadata_ReplicateGroup"] = (
            df["Metadata_compound"].astype(str) + "_" +
            df["Metadata_conc_uM"].astype(str)
        )

        # Recompute meta columns AFTER adding the replicate identifiers so that
        # Metadata_Replicate / Metadata_ReplicateGroup are passed as meta_features
        # to cytominer_eval — otherwise it raises "replicate_group not found in
        # melted dataframe columns".
        meta_cols = get_meta_cols(df)

        # Get all DMSO replicate IDs globally
        control_perts = df[df[pert_col] == negcon]["Metadata_Replicate"].unique().tolist()
        if not control_perts:
            logger.warning("No control wells found — skipping GRIT")
            return None

        # Run GRIT globally, not per plate. Compound-dose replicate wells are
        # spread across multiple plates, so restricting cytominer_eval to one
        # plate at a time starves most replicate groups of enough replicates
        # (matches last_dnt/03_cellpainting_pipeline.py's step_grit).
        try:
            grit_combined = cytominer_evaluate(
                profiles=df,
                features=feat_cols,
                # Only the two ID columns GRIT uses. cytominer_eval copies every
                # meta column onto each of the n^2 well pairs, so passing all
                # Metadata_* columns OOM-kills at ~15k wells (same GRIT values).
                meta_features=["Metadata_Replicate", "Metadata_ReplicateGroup"],
                replicate_groups={
                    "profile_col": "Metadata_Replicate",
                    "replicate_group_col": "Metadata_ReplicateGroup",
                },
                operation="grit",
                similarity_metric="pearson",
                grit_control_perts=control_perts,
                grit_replicate_summary_method="mean",
            )
        except Exception as e:
            logger.warning(f"    GRIT calculation failed: {e}")
            return None

        if grit_combined is None or grit_combined.empty:
            logger.warning("No GRIT results computed")
            return None

        # cytominer_eval's grit output uses the columns ['perturbation', 'group',
        # 'grit'] (one row per well), where 'group' is the compound_dose replicate
        # group. Older versions emitted 'Metadata_ReplicateGroup'; normalise both.
        if "Metadata_ReplicateGroup" not in grit_combined.columns:
            group_col = "group" if "group" in grit_combined.columns else None
            if group_col is None:
                logger.warning(
                    f"GRIT output missing group column (got {list(grit_combined.columns)}) — skipping GRIT"
                )
                return None
            grit_combined["Metadata_ReplicateGroup"] = grit_combined[group_col]

        # Extract compound and dose from replicate group (using rsplit to handle _ in names)
        split = grit_combined["Metadata_ReplicateGroup"].str.rsplit("_", n=1, expand=True)
        grit_combined["Metadata_compound"] = split[0]
        grit_combined["Metadata_conc_uM"] = pd.to_numeric(split[1], errors="coerce")
        grit_combined["Metadata_compound_dose"] = grit_combined["Metadata_ReplicateGroup"]

        # Calculate standard deviation of GRIT per compound-dose
        grit_std = grit_combined.groupby("Metadata_ReplicateGroup")["grit"].std().reset_index()
        grit_std.columns = ["Metadata_ReplicateGroup", "grit_std"]
        grit_combined = grit_combined.merge(grit_std, on="Metadata_ReplicateGroup", how="left")

        # Select output columns (matching your script)
        grit_df = grit_combined[[
            "Metadata_compound", "Metadata_conc_uM", "Metadata_compound_dose", "grit", "grit_std",
        ]].sort_values("grit", ascending=False).reset_index(drop=True)

        logger.info(f"    GRIT computed for {len(grit_df)} compound×dose groups")
        return grit_df

    except Exception as e:
        logger.error(f"    GRIT computation failed: {e}")
        return None


def add_grit_flag(grit_df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Add GRIT flag column to GRIT results.

    Wells with GRIT > threshold are flagged as low quality.

    Parameters
    ----------
    grit_df : pd.DataFrame
        GRIT results with 'grit' column
    config : PipelineConfig

    Returns
    -------
    grit_df : pd.DataFrame
        Input dataframe with added 'grit_flag' column
    """
    if grit_df is None or grit_df.empty or "grit" not in grit_df.columns:
        return grit_df

    threshold = getattr(config.metrics, "grit_flag_threshold", 1.0)
    grit_df = grit_df.copy()
    grit_df["grit_flag"] = grit_df["grit"] > threshold

    n_flagged = grit_df["grit_flag"].sum()
    if n_flagged > 0:
        logger.info(f"    {n_flagged} wells flagged for low quality (GRIT > {threshold})")

    return grit_df


def compute_all_metrics(
    df: pd.DataFrame,
    config,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Compute all metrics (phenotypic activity, phenotypic consistency, GRIT, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Level 4b data (feature-selected, well level, pre batch correction)
    config : PipelineConfig

    Returns
    -------
    dict
        {metric_name: dataframe or None}
    """
    metrics_data = {}

    # Phenotypic activity is always computed if copairs is available
    ap_df, map_df = compute_phenotypic_activity(df, config)
    metrics_data["activity_ap_df"] = ap_df
    metrics_data["activity_map_df"] = map_df
    metrics_data["map_df"] = map_df  # Legacy key for backward compatibility

    # Phenotypic consistency requires activity results
    if config.metrics.compute_consistency and map_df is not None:
        consistency_df = compute_phenotypic_consistency(df, map_df, config)
        metrics_data["consistency_df"] = consistency_df
    else:
        metrics_data["consistency_df"] = None

    if config.metrics.compute_grit:
        grit_df = compute_grit(df, config)
        grit_df = add_grit_flag(grit_df, config)
        metrics_data["grit_df"] = grit_df
    else:
        metrics_data["grit_df"] = None

    return metrics_data
