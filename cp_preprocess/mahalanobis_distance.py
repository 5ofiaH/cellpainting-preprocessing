#!/usr/bin/env python3
"""
Mahalanobis Distance (MD) — OASIS-style, project-agnostic
===========================================================

Ported from an OASIS-style BMC/PAC dose-response pipeline. This script extracts ONLY the MD
computation (the input to that pipeline's BMC/PAC curve fitting), wired to
this repo's YAML config system so it can run against any project's pipeline
output (Level 4b feature-selected, or Level 4c batch-corrected) instead of
being hardcoded to one project's cell lines/paths.

Method (unchanged from the BA script)
--------------------------------------
Per feature category (global + compartment x channel), fit ONE PCA
(retaining `cover_var` explained variance) + treatment-group residual
covariance model GLOBALLY across all plates (OASIS method: regress out
per-treatment-group PC-score means via a dummy design matrix, estimate
covariance from the residuals — this is more stable than a per-plate
covariance estimate when a plate has few DMSO wells).

Then, per PLATE, per category: project that plate's wells into the same PCA
space, take the plate's own DMSO centroid as the reference point, and compute
the Mahalanobis distance of every well from that centroid using the GLOBAL
inverse covariance. This keeps the plate-specific control baseline (handles
plate-to-plate drift) while sharing statistical power for the covariance
estimate across the whole run.

Feature categories: "global" (all features) + compartment x channel, e.g.
Nuclei_DNA, Cells_Mito, Cytoplasm_AreaShape, ... (a category is skipped if it
has fewer than 3 features). Compartments and channel-marker substrings are
configurable (defaults match this repo's Cell Painting feature-naming
convention, e.g. "Nuclei_Correlation_Correlation_illumCONC_illumMITO").

Usage
-----
    python mahalanobis_distance.py --config configs/gromics/6_plates_normal.yaml
    python mahalanobis_distance.py --config configs/gromics/6_plates_normal.yaml --cell-line MCF7
    python mahalanobis_distance.py --config configs/gromics/6_plates_normal.yaml --level 4b --keep-poscon

Output
------
<output.base_dir>/<cell_line>/<features_subdir>/mahalanobis_distances.csv, one
row per (well, feature category):
    Metadata_compound, Metadata_conc_uM, Metadata_pert_type, Metadata_plate,
    Metadata_conc_log10_uM, category, md
"""

import argparse
import itertools
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import load_config_from_yaml, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mahalanobis_distance")


# ─────────────────────────────────────────────────────────────────────────────
# Default channel markers (this repo's feature-naming convention — see
# plotting_analysis.CHANNELS for the same tags used in the QC channel-analysis
# plot). Override via --channel-map for a project with different marker names.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CHANNEL_MARKERS = {
    "DNA": ["HOECHST"],
    "ER": ["CONC"],
    "RNA": ["SYTO"],
    "AGP": ["PHAandWGA"],
    "Mito": ["MITO"],
    "AreaShape": [],  # matched by compartment prefix + "_AreaShape_" instead of a marker
}
DEFAULT_COMPARTMENTS = ["Nuclei", "Cells", "Cytoplasm"]


# ─────────────────────────────────────────────────────────────────────────────
# OASIS PCA + treatment-group residual covariance (verbatim from
# 05_pod_pac_analysis.py, generalized: negcon label is passed in rather than
# hardcoded, no other behavior change)
# ─────────────────────────────────────────────────────────────────────────────

def build_treatment_labels(df: pd.DataFrame, compound_col: str, conc_col: str,
                            pert_col: str, negcon_value: str) -> np.ndarray:
    """Group wells by compound-dose for the residual-covariance design matrix;
    all negcon wells share one label ('DMSO') regardless of their nominal
    vehicle-match concentration."""
    labels = []
    for _, row in df[[compound_col, conc_col, pert_col]].iterrows():
        if str(row.get(pert_col, "")) == negcon_value:
            labels.append("DMSO")
        else:
            cmpd = str(row.get(compound_col, "NA"))
            conc = row.get(conc_col, np.nan)
            if pd.isna(conc):
                labels.append(cmpd)
            else:
                labels.append(f"{cmpd}__{float(conc):.12g}")
    return np.asarray(labels, dtype=object)


def prep_oasis_distance_model(X: np.ndarray, treatment_labels: np.ndarray,
                               cover_var: float = 0.95, seed: int = 42):
    """1. center/scale features, 2. retain PCs covering `cover_var`,
    3. estimate covariance from residuals of PC scores after removing
    treatment-group means. Returns None if there isn't enough data."""
    if X.shape[0] < 3 or X.shape[1] < 1:
        return None

    mask_col = ~np.isnan(X).all(axis=0)
    if mask_col.sum() < 1:
        return None

    X_sub = X[:, mask_col].astype(float, copy=True)
    col_med = np.nanmedian(X_sub, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    X_sub = np.where(np.isnan(X_sub), col_med, X_sub)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_sub)
    scaler.scale_[scaler.scale_ == 0] = 1.0

    pca = PCA(svd_solver="full", random_state=seed)
    X_pca = pca.fit_transform(X_scaled)
    if X_pca.ndim != 2 or X_pca.shape[1] < 1:
        return None

    cum_prop = np.cumsum(pca.explained_variance_ratio_)
    n_pc = int(np.searchsorted(cum_prop, cover_var) + 1)
    n_pc = max(1, min(n_pc, X_pca.shape[1], X.shape[0]))
    X_pc = X_pca[:, :n_pc]

    design = pd.get_dummies(pd.Series(treatment_labels, dtype="object"), drop_first=False)
    D = design.to_numpy(dtype=float)
    if D.shape[1] == 0:
        return None

    coef, _, _, _ = np.linalg.lstsq(D, X_pc, rcond=None)
    resid = X_pc - D @ coef
    dof = max(1, X_pc.shape[0] - np.linalg.matrix_rank(D))
    cov = (resid.T @ resid) / dof
    cov = cov + np.eye(cov.shape[0]) * 1e-9
    inv_cov = np.linalg.pinv(cov)

    return {
        "mask_col": mask_col,
        "col_med": col_med,
        "scaler": scaler,
        "pca": pca,
        "n_pc": n_pc,
        "inv_cov": inv_cov,
    }


def compute_oasis_mahalanobis(X_treat: np.ndarray, X_dmso: np.ndarray, model: dict) -> np.ndarray:
    """Mahalanobis distance in the retained PCA space, from THIS PLATE's DMSO
    centroid, using the GLOBAL inverse covariance from `model`."""
    if model is None or X_treat.shape[0] == 0 or X_dmso.shape[0] == 0:
        return np.full(X_treat.shape[0], np.nan)

    mask_col = model["mask_col"]
    col_med = model["col_med"]

    X_t = X_treat[:, mask_col].astype(float, copy=True)
    X_d = X_dmso[:, mask_col].astype(float, copy=True)
    X_t = np.where(np.isnan(X_t), col_med, X_t)
    X_d = np.where(np.isnan(X_d), col_med, X_d)

    X_t_scaled = model["scaler"].transform(X_t)
    X_d_scaled = model["scaler"].transform(X_d)
    X_t_pc = model["pca"].transform(X_t_scaled)[:, :model["n_pc"]]
    X_d_pc = model["pca"].transform(X_d_scaled)[:, :model["n_pc"]]

    ctrl_mean = X_d_pc.mean(axis=0)
    delta = X_t_pc - ctrl_mean
    md = np.sqrt(np.einsum("ij,jk,ik->i", delta, model["inv_cov"], delta))
    return np.round(md, 3)


def get_feature_categories(feat_cols: list, compartments: list, channel_markers: dict) -> dict:
    """global + compartment x channel categories (e.g. Nuclei_DNA,
    Cytoplasm_AreaShape, ...). A category is dropped if it has < 3 features."""
    cats = {"global": feat_cols}
    for comp, channel_name in itertools.product(compartments, channel_markers):
        if channel_name == "AreaShape":
            cols = [c for c in feat_cols if c.startswith(f"{comp}_AreaShape_")]
        else:
            markers = channel_markers[channel_name]
            cols = [
                c for c in feat_cols
                if c.startswith(f"{comp}_") and any(marker in c for marker in markers)
            ]
        if len(cols) >= 3:
            cats[f"{comp}_{channel_name}"] = cols
    return cats


# ─────────────────────────────────────────────────────────────────────────────
# Main per-cell-line computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_mahalanobis_for_df(
    df: pd.DataFrame,
    plate_col: str,
    compound_col: str,
    conc_col: str,
    pert_col: str,
    negcon_value: str,
    poscon_value: str,
    compartments: list,
    channel_markers: dict,
    cover_var: float = 0.95,
    min_dmso_wells: int = 4,
    drop_poscon: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    if not feat_cols:
        raise ValueError("No feature columns found (everything starts with 'Metadata_')")

    cats = get_feature_categories(feat_cols, compartments, channel_markers)
    logger.info("Feature categories (%d): %s", len(cats), list(cats.keys()))
    feat_idx_map = {feat: idx for idx, feat in enumerate(feat_cols)}
    cat_idx_map = {
        cat_name: [feat_idx_map[c] for c in cat_cols if c in feat_idx_map]
        for cat_name, cat_cols in cats.items()
    }
    cat_idx_map = {k: v for k, v in cat_idx_map.items() if len(v) >= 3}

    X_all = df[feat_cols].to_numpy(dtype=float, copy=True)
    treatment_labels = build_treatment_labels(df, compound_col, conc_col, pert_col, negcon_value)

    oasis_models = {}
    for cat_name, idxs in cat_idx_map.items():
        model = prep_oasis_distance_model(X_all[:, idxs], treatment_labels, cover_var=cover_var, seed=seed)
        if model is not None:
            oasis_models[cat_name] = model
    logger.info("OASIS PCA/covariance models prepared for %d/%d categories", len(oasis_models), len(cat_idx_map))

    plates = df[plate_col].unique()
    md_records = []
    for plate in plates:
        df_plate = df[df[plate_col] == plate].reset_index(drop=True)
        dmso_mask = (df_plate[pert_col] == negcon_value).values
        if dmso_mask.sum() < min_dmso_wells:
            logger.warning("[%s] only %d DMSO wells (< min_dmso_wells=%d) — skipping plate",
                           plate, int(dmso_mask.sum()), min_dmso_wells)
            continue

        X_plate = df_plate[feat_cols].to_numpy(dtype=float, copy=True)
        X_dmso_full = X_plate[dmso_mask]

        for cat_name, idxs in cat_idx_map.items():
            model = oasis_models.get(cat_name)
            if model is None:
                continue
            md = compute_oasis_mahalanobis(X_plate[:, idxs], X_dmso_full[:, idxs], model)
            md_df = pd.DataFrame({
                "Metadata_compound": df_plate[compound_col].values,
                "Metadata_conc_uM": df_plate[conc_col].values,
                "Metadata_pert_type": df_plate[pert_col].values,
                "Metadata_plate": plate,
                "category": cat_name,
                "md": md,
            })
            conc = pd.to_numeric(md_df["Metadata_conc_uM"], errors="coerce")
            log10_uM = pd.Series(np.nan, index=conc.index)
            pos = conc > 0
            log10_uM[pos] = np.log10(conc[pos])
            md_df["Metadata_conc_log10_uM"] = log10_uM.values

            if drop_poscon:
                md_df = md_df[md_df["Metadata_pert_type"] != poscon_value]

            if not md_df.empty:
                md_records.append(md_df)

    if not md_records:
        raise RuntimeError("No MD records computed — check min_dmso_wells / plate_col / input data")

    return pd.concat(md_records, ignore_index=True)


def resolve_input_path(base_dir: str, cell_line: str, features_subdir: str, level: str) -> str:
    candidates = [
        os.path.join(base_dir, cell_line, features_subdir, f"level_{level}.parquet"),
    ]
    # Fall back through the pipeline's other level names if the requested one is missing
    fallback_order = ["4c", "4b"] if level == "4c" else ["4b"]
    for lv in fallback_order:
        candidates.append(os.path.join(base_dir, cell_line, features_subdir, f"level_{lv}.parquet"))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No level_{level}(.parquet) input found for {cell_line} under {base_dir} "
        f"(tried: {candidates})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="OASIS-style Mahalanobis distance (MD) computation, per project config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Pipeline YAML config (e.g. configs/gromics/6_plates_normal.yaml)")
    parser.add_argument("--cell-line", help="Process a single cell line only (default: all in config.metadata.cell_lines)")
    parser.add_argument("--level", choices=["4b", "4c"], default="4c",
                        help="Input level: 4c (batch-corrected, default) or 4b (feature-selected, pre-batch-correction)")
    parser.add_argument("--output-dir", help="Override output directory (default: alongside the input parquet)")
    parser.add_argument("--cover-var", type=float, default=0.95, help="PCA explained-variance to retain (default 0.95)")
    parser.add_argument("--min-dmso-wells", type=int, default=4, help="Minimum DMSO wells per plate to compute MD (default 4)")
    parser.add_argument("--keep-poscon", action="store_true", help="Keep poscon rows in the output (default: dropped, matching the BA script)")
    parser.add_argument("--compartments", nargs="+", default=DEFAULT_COMPARTMENTS,
                        help=f"Compartment prefixes for category names (default: {DEFAULT_COMPARTMENTS})")
    parser.add_argument("--compound-col", default="Metadata_compound",
                        help="Compound column name in the Level 4b/4c parquet (default: 'Metadata_compound', "
                             "the canonical name pipeline.py harmonizes to at Level 3+ — NOT config.metadata.compound_col, "
                             "which names the RAW pre-pipeline metadata column and is usually different, e.g. 'Metadata_cmpdFullName')")
    parser.add_argument("--conc-col", default="Metadata_conc_uM", help="Concentration column (default: 'Metadata_conc_uM')")
    parser.add_argument("--pert-col", default="Metadata_pert_type", help="Pert-type column (default: 'Metadata_pert_type')")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config: PipelineConfig = load_config_from_yaml(args.config)
    if not config.output.base_dir:
        config.output.base_dir = str(Path(args.config).parent)

    cell_lines = [args.cell_line] if args.cell_line else (config.metadata.cell_lines or [])
    if not cell_lines:
        logger.error("No cell lines to process (pass --cell-line or set metadata.cell_lines in the config)")
        sys.exit(1)

    negcon_value = config.metrics.negcon_value
    poscon_value = config.metrics.poscon_value
    plate_col = config.batch_correction.plate_col
    compound_col = args.compound_col
    conc_col = args.conc_col
    pert_col = args.pert_col

    failed = []
    for cell_line in cell_lines:
        logger.info("=" * 70)
        logger.info("Processing: %s", cell_line)
        try:
            input_path = resolve_input_path(config.output.base_dir, cell_line, config.output.features_subdir, args.level)
            logger.info("Input: %s", input_path)
            df = pd.read_parquet(input_path)

            for col in (plate_col, compound_col, conc_col, pert_col):
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in {input_path}")

            df_md = compute_mahalanobis_for_df(
                df,
                plate_col=plate_col,
                compound_col=compound_col,
                conc_col=conc_col,
                pert_col=pert_col,
                negcon_value=negcon_value,
                poscon_value=poscon_value,
                compartments=args.compartments,
                channel_markers=DEFAULT_CHANNEL_MARKERS,
                cover_var=args.cover_var,
                min_dmso_wells=args.min_dmso_wells,
                drop_poscon=not args.keep_poscon,
                seed=args.seed,
            )

            out_dir = args.output_dir or os.path.dirname(input_path)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "mahalanobis_distances.csv")
            df_md.to_csv(out_path, index=False)
            logger.info("Wrote %s (%d rows)", out_path, len(df_md))
        except Exception as e:
            logger.error("✗ %s failed: %s", cell_line, e)
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed.append(cell_line)

    logger.info("=" * 70)
    logger.info("Processed: %d/%d cell lines", len(cell_lines) - len(failed), len(cell_lines))
    if failed:
        logger.error("Failed: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
