"""
I/O utilities: save parquet files with metadata, viability, metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_level_parquet(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    level_name: str,
) -> Path:
    """
    Save a level (3-5) as parquet with metadata + metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Features + metadata dataframe
    metadata : pd.DataFrame
        Metadata dataframe
    output_dir : Path
        Output directory
    level_name : str
        "level_3", "level_4a", etc.

    Returns
    -------
    Path
        Path to saved parquet file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure metadata columns are first (for readability)
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    metric_cols = [c for c in df.columns if c in ["mAP", "GRIT", "Reproducibility", "Viability_pct", "Viability_flag"]]
    feature_cols = [c for c in df.columns if c not in meta_cols + metric_cols]

    # Reorder
    col_order = meta_cols + metric_cols + feature_cols
    df_ordered = df[[c for c in col_order if c in df.columns]]

    # Save
    output_path = output_dir / f"{level_name}.parquet"
    df_ordered.to_parquet(output_path, index=False)

    n_rows, n_cols = df_ordered.shape
    n_features = len(feature_cols)
    n_metadata = len(meta_cols)
    n_metrics = len(metric_cols)

    logger.info(
        f"  Saved {output_path.name}: {n_rows} samples, "
        f"{n_metadata} metadata + {n_metrics} metrics + {n_features} features"
    )

    return output_path


def save_summary_stats(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    level_name: str,
) -> None:
    """Save summary statistics for a level."""
    output_dir = Path(output_dir)
    summary_path = output_dir / f"{level_name}_summary.csv"

    stats = pd.DataFrame({
        "metric": ["n_samples", "n_features"],
        "value": [
            len(df),
            len([c for c in df.columns if not c.startswith("Metadata_")]),
        ],
    })

    stats.to_csv(summary_path, index=False)
    logger.debug(f"  Saved {summary_path.name}")
