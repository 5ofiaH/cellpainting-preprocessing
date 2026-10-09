#!/usr/bin/env python3
"""
Convert any pipeline level parquet (Level 3 → 5) into an AnnData object
compatible with scanpy.

Standalone module (no pipeline imports) — only pandas/numpy/anndata, so it can
be dropped anywhere that can read the parquet.

Convention (matches the rest of the pipeline, see io_utils.py / CLAUDE.md):
  • Columns prefixed ``Metadata_`` -> obs (one row per well/compound, whatever
    the level's grain is).
  • Every other column -> var / X (a morphological feature). Non-numeric
    "feature" columns are defensively dropped into obs instead of raising,
    since some levels attach compound-like annotation columns without the
    prefix (rare, but silent data loss would be worse than a warning).
  • obs_names default to the row's positional RangeIndex as a string; pass
    --obs-names to use a metadata column instead (e.g. Metadata_Well), when it
    is unique.

Usage
-----
    python -m cp_preprocess.parquet_to_anndata --input level_4b.parquet --output level_4b.h5ad
    python -m cp_preprocess.parquet_to_anndata --input level_5_batch_corrected.parquet \\
        --output level_5.h5ad --obs-names Metadata_compound_dose

Importable:
    from cp_preprocess.parquet_to_anndata import parquet_to_anndata
    adata = parquet_to_anndata("level_4b.parquet")
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger("parquet_to_anndata")


def split_metadata_features(
    df: pd.DataFrame,
    metadata_prefix: str = "Metadata_",
    log: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a level dataframe into (obs, feature) frames.

    Non-numeric columns without the metadata prefix are moved into obs rather
    than dropped, so no column silently disappears; a warning is logged for
    those. Returns (obs_df, feature_df) with the original row order/index.
    """
    lg = log or logger

    meta_cols = [c for c in df.columns if c.startswith(metadata_prefix)]
    other_cols = [c for c in df.columns if c not in meta_cols]

    numeric_mask = {c: pd.api.types.is_numeric_dtype(df[c]) for c in other_cols}
    feature_cols = [c for c in other_cols if numeric_mask[c]]
    stray_cols = [c for c in other_cols if not numeric_mask[c]]

    if stray_cols:
        lg.warning(
            "%d non-metadata, non-numeric column(s) found — moved into obs "
            "instead of treated as features: %s",
            len(stray_cols), stray_cols,
        )

    obs_cols = meta_cols + stray_cols
    obs = df[obs_cols].copy() if obs_cols else pd.DataFrame(index=df.index)
    features = df[feature_cols].copy()

    return obs, features


def parquet_to_anndata(
    path: str,
    metadata_prefix: str = "Metadata_",
    obs_names_col: Optional[str] = None,
    layer_name: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> AnnData:
    """
    Load a level parquet/CSV and build an AnnData: X = feature matrix
    (float32), obs = Metadata_* columns, var = feature names.

    Parameters
    ----------
    path : str
        Path to a level parquet (or CSV, e.g. level_2_5.csv).
    metadata_prefix : str
        Prefix identifying metadata/obs columns.
    obs_names_col : str, optional
        A metadata column to use as ``obs_names`` (must be unique). Defaults
        to a stringified RangeIndex.
    layer_name : str, optional
        If given, also store the (unmodified) feature matrix under
        ``adata.layers[layer_name]`` in addition to X — useful when you plan
        to let scanpy overwrite X (e.g. after scaling) but want the original
        values preserved.
    """
    lg = log or logger
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, engine="python")

    obs, features = split_metadata_features(df, metadata_prefix=metadata_prefix, log=lg)

    if features.empty:
        raise ValueError(f"No numeric feature columns found in {path} (all metadata?).")

    X = features.to_numpy(dtype=np.float32)

    var = pd.DataFrame(index=pd.Index(features.columns, name="feature"))

    obs = obs.reset_index(drop=True)
    obs.index = obs.index.astype(str)
    if obs_names_col:
        if obs_names_col not in obs.columns:
            raise KeyError(f"obs_names_col '{obs_names_col}' not found among metadata columns.")
        names = obs[obs_names_col].astype(str)
        if names.duplicated().any():
            n_dup = int(names.duplicated().sum())
            lg.warning(
                "obs_names_col '%s' is not unique (%d duplicate value(s)) — "
                "falling back to the default RangeIndex obs_names.",
                obs_names_col, n_dup,
            )
        else:
            obs.index = names

    adata = AnnData(X=X, obs=obs, var=var)

    if layer_name:
        adata.layers[layer_name] = X.copy()

    lg.info(
        "Built AnnData: %d obs x %d vars (from %s, %d obs column(s))",
        adata.n_obs, adata.n_vars, path, obs.shape[1],
    )
    return adata


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert a pipeline level parquet/CSV into a scanpy-compatible AnnData (.h5ad).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Level parquet/CSV (e.g. level_4b.parquet).")
    ap.add_argument("--output", required=True, help="Output .h5ad path.")
    ap.add_argument("--metadata-prefix", default="Metadata_", help="Prefix identifying obs columns.")
    ap.add_argument("--obs-names", default=None,
                    help="Metadata column to use as obs_names (must be unique), e.g. Metadata_Well.")
    ap.add_argument("--layer-name", default=None,
                    help="Also store the raw feature matrix under adata.layers[NAME].")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
    )

    adata = parquet_to_anndata(
        args.input,
        metadata_prefix=args.metadata_prefix,
        obs_names_col=args.obs_names,
        layer_name=args.layer_name,
    )

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    adata.write(outpath)
    logger.info("Wrote %s (%d obs x %d vars)", outpath, adata.n_obs, adata.n_vars)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
