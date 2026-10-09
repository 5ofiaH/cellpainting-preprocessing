"""
Level 5: Compute consensus profiles.

Uses pycytominer.consensus() for profile-level aggregation.
Default operation: median (pycytominer default, recommended for most use cases)
Optional: modz (see pycytominer docs for details)

Reference:
  pycytominer.consensus() forms level 5 consensus profiles by aggregating
  replicates (e.g., same compound + concentration) across replicates/wells/plates.
"""

import pandas as pd
import logging
from typing import Tuple

from pycytominer import consensus as pycytominer_consensus

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def compute_consensus(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Level 5: Compute consensus profiles using pycytominer.consensus().

    This function aggregates replicate profiles (typically within the same
    compound + concentration) into single consensus profiles.

    Parameters
    ----------
    df : pd.DataFrame
        Level 4c data (batch-corrected features + metadata)
    metadata : pd.DataFrame
    config : PipelineConfig
        config.consensus.operation should be "median" or "modz"
        config.consensus.replicate_cols defines grouping (typically compound+conc)

    Returns
    -------
    df_consensus : pd.DataFrame
        Consensus profiles (one row per unique replicate group)
    metadata_consensus : pd.DataFrame
        Metadata for consensus profiles
    """
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]

    # Determine replicate columns (what to group by)
    replicate_cols = _get_replicate_columns(df, config)

    logger.info(
        f"  Computing {config.consensus.operation} consensus "
        f"(grouping by: {replicate_cols})"
    )

    try:
        # Use pycytominer.consensus() for consensus computation
        df_consensus = pycytominer_consensus(
            profiles=df,
            replicate_columns=replicate_cols,
            operation=config.consensus.operation,  # "median", "mean" or "modz"
            # Pass the feature columns explicitly rather than features="infer".
            # pycytominer's inference keys on the canonical CellProfiler naming
            # (Cells_/Nuclei_/Cytoplasm_ prefixes); datasets that put the
            # compartment as a SUFFIX (e.g. SPECS: AreaShape_Area_cells) infer to
            # zero features and raise. The explicit list is exactly what inference
            # would find for prefix-named data, so this is behaviour-preserving
            # there and simply also correct for suffix-named data.
            features=feature_cols,
            output_file=None,  # Return DataFrame, don't write
        )

        # pycytominer.consensus keeps only the replicate columns + features.
        # Re-attach every metadata column that is constant within a treatment
        # group (pert_type, cell line, batch, project, ids, ...) so the Level 5
        # profile is self-describing. Well/site/plate columns are intentionally
        # dropped because a compound×dose treatment spans multiple wells/plates.
        meta_only = df[[c for c in df.columns if c.startswith("Metadata_")]]
        nunique = meta_only.groupby(replicate_cols, dropna=False, observed=True).nunique(dropna=False)
        constant_meta = [
            c for c in meta_only.columns
            if c not in replicate_cols and bool((nunique[c] <= 1).all())
        ]
        if constant_meta:
            group_meta = (
                meta_only.groupby(replicate_cols, dropna=False, observed=True)[constant_meta]
                .first()
                .reset_index()
            )
            df_consensus = df_consensus.merge(group_meta, on=replicate_cols, how="left")

        metadata_consensus = df_consensus[
            [c for c in df_consensus.columns if c.startswith("Metadata_")]
        ].copy()

        logger.info(
            f"  {len(df)} wells → {len(df_consensus)} consensus profiles "
            f"(grouped by {replicate_cols})"
        )

        return df_consensus, metadata_consensus

    except Exception as e:
        logger.error(f"  Consensus computation failed: {e}")
        raise


def _get_replicate_columns(df: pd.DataFrame, config: PipelineConfig) -> list:
    """
    Determine which metadata columns define replicates for the Level 5 consensus.

    A replicate group is one treatment: the same compound at the same
    concentration. Wells sharing that (compound, concentration) are aggregated
    into a single consensus profile. Grouping is taken from the configured
    compound and concentration columns.
    """
    compound_col = config.metadata.compound_col
    conc_col = config.metadata.conc_col

    # Explicit override from config wins (e.g. group by compound only). Checked
    # FIRST so a configured replicate_cols is actually honored; only columns that
    # exist in the data are used.
    if config.consensus.replicate_cols:
        cols = [c for c in config.consensus.replicate_cols if c in df.columns]
        if cols:
            return cols

    # Default when no override: compound + concentration = treatment identity
    if compound_col in df.columns:
        cols = [c for c in (compound_col, conc_col) if c in df.columns]
        return cols

    raise ValueError(
        f"Cannot determine Level 5 replicate columns: compound column "
        f"'{compound_col}' not found in data. Set metadata.compound_col / "
        f"metadata.conc_col in the config."
    )
