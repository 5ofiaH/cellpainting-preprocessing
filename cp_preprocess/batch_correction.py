"""
Level 4c: batch correction for plate effects.

Three interchangeable methods are provided, selected via
``config.batch_correction.method``. All operate in feature space and return a
table with the same metadata + feature columns as the input, so downstream
levels (metrics, consensus) are agnostic to which was used:

  "combat"   — scanpy ComBat, mirroring CP_MS/BA/03_cellpainting_pipeline.py
               step_combat(): per-plate location/scale correction. Using scanpy's
               ComBat (rather than the standalone combat.pycombat package) matches
               the reference pipeline and is far more robust.

Well ROW / COLUMN positional effects are handled by two METHOD-INDEPENDENT steps
that the orchestrator applies AFTER whichever correction above ran, so they work
with ComBat, Harmony or Spherize alike:
  • regress_out_covariates      — least-squares removal of row/column (per
                                  config.batch_correction.batch_effect_fix)
  • regress_out_column_effect   — control-based within-plate column de-trend
                                  (per config.batch_correction.regress_out_column)
  "harmony"  — harmonypy soft-kmeans integration with the plate as the batch
               variable, run directly in the (already normalized) feature space
               so the returned profiles stay interpretable.
  "spherize" — pycytominer ZCA whitening fit on the DMSO (negcon) reference,
               decorrelating the technical covariance structure shared by every
               well while leaving treatment-specific signal intact.

The public entry point is ``apply_batch_correction`` which dispatches on the
configured method; the per-method functions can also be called directly.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

import scanpy as sc
import anndata as ad
from sklearn.linear_model import LinearRegression

sc.settings.verbosity = 0

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _covariate_columns(obs: pd.DataFrame, config) -> pd.DataFrame:
    """
    Derive positional covariates from the well id according to batch_effect_fix:
      "barcode"            -> none
      "barcode_row"        -> row (A–P)
      "barcode_column"     -> column (1–24)
      "barcode_row_column" -> both
    """
    mode = getattr(config.batch_correction, "batch_effect_fix", "barcode")
    well_col = config.metadata.well_col
    cov = pd.DataFrame(index=obs.index)
    if well_col in obs.columns:
        wells = obs[well_col].astype(str)
        if "row" in mode:
            cov["row"] = wells.str.extract(r"^([A-Za-z]+)")[0].str.upper()
        if "column" in mode:
            cov["column"] = wells.str.extract(r"(\d+)$")[0]
    return cov


def regress_out_column_effect(df: pd.DataFrame, config, logger=None) -> pd.DataFrame:
    """
    Remove the within-plate COLUMN positional gradient from feature columns.

    The offset for each plate column is estimated from the negcon (DMSO) wells —
    the same biological condition across all 24 columns, so any column-dependent
    variation there is pure position — as (median negcon profile in that column −
    global negcon median), then subtracted from EVERY well in that column. This is
    a control-based de-trend, pooled across plates (per-plate estimates are too
    sparse: ~20 negcon wells over 24 columns). Metadata columns are untouched.

    Returns a copy of df with adjusted features (or df unchanged if it cannot run).
    """
    lg = logger or logging.getLogger(__name__)
    well_col = config.metadata.well_col
    pert_col = config.metadata.pert_type_col
    negcon = config.metrics.negcon_value
    min_neg = getattr(config.batch_correction, "column_min_dmso", 4)

    feat = [c for c in df.columns if not c.startswith("Metadata_")]
    if well_col not in df.columns or pert_col not in df.columns or not feat:
        lg.warning("  Column de-trend skipped (missing well/pert_type column or no features)")
        return df

    col = pd.to_numeric(df[well_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce").to_numpy()
    is_neg = (df[pert_col].astype(str) == str(negcon)).to_numpy()
    if is_neg.sum() == 0:
        lg.warning("  Column de-trend skipped (no negcon wells to estimate offsets)")
        return df

    F = df[feat].to_numpy(float)
    glob = np.nanmedian(F[is_neg], axis=0)
    out = F.copy()
    n_adj = 0
    for c in np.unique(col[~np.isnan(col)]):
        m_all = col == c
        m_neg = m_all & is_neg
        if m_neg.sum() < min_neg:
            continue
        out[m_all] = F[m_all] - (np.nanmedian(F[m_neg], axis=0) - glob)
        n_adj += 1

    df = df.copy()
    df[feat] = out
    lg.info("  Column de-trend: adjusted %d columns using %d negcon wells (min %d/col)",
            n_adj, int(is_neg.sum()), min_neg)
    return df


def regress_out_covariates(df: pd.DataFrame, config, logger=None) -> pd.DataFrame:
    """
    Regress out well ROW / COLUMN positional effects from the feature block.

    The covariates are derived from the well id according to
    ``config.batch_correction.batch_effect_fix`` ("barcode_row" / "barcode_column"
    / "barcode_row_column"); "barcode" (plate only) yields no covariates and this
    is a no-op. Each requested covariate is one-hot encoded and its linear
    contribution is subtracted from every feature via least squares.

    This step is METHOD-INDEPENDENT: it used to be inlined inside ComBat, but is
    now applied by the orchestrator AFTER whichever batch correction ran, so
    row/column effects can be removed with ComBat, Harmony or Spherize alike.
    Metadata columns are untouched.
    """
    lg = logger or logging.getLogger(__name__)
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    if not feature_cols:
        return df

    cov = _covariate_columns(df, config)
    if cov.shape[1] == 0:
        return df  # batch_effect_fix == "barcode" (or no well column) → nothing to do

    df = df.copy()
    X = df[feature_cols].to_numpy(dtype=np.float64)
    for col in cov.columns:
        n_levels = cov[col].nunique()
        if n_levels < 2:
            lg.info("  Covariate regression (%s): only %d level — skipping", col, n_levels)
            continue
        dummies = pd.get_dummies(cov[col], drop_first=True).to_numpy(dtype=np.float64)
        reg = LinearRegression(fit_intercept=True).fit(dummies, X)
        X_before = X.copy()
        X = X - reg.predict(dummies)
        shift = float(np.abs(X - X_before).mean())
        lg.info("  Covariate regression (%s, %d levels): mean absolute shift = %.4f",
                col, n_levels, shift)
    df[feature_cols] = X
    return df


def apply_combat(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply ComBat batch correction for plate effects (+ optional covariate regression).

    Parameters
    ----------
    df : pd.DataFrame
        Level 4b data (features + metadata)
    metadata : pd.DataFrame
        Unused (kept for signature compatibility); metadata is taken from df.
    config : PipelineConfig

    Returns
    -------
    df_corrected : pd.DataFrame
        Batch-corrected data (metadata + features)
    metadata : pd.DataFrame
        Metadata of the corrected data
    """
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    plate_col = config.batch_correction.plate_col

    # ComBat needs ≥2 batches (plates) to estimate/adjust batch effects.
    if plate_col not in df.columns or df[plate_col].nunique() < 2:
        logger.info("  Batch correction skipped — need ≥2 plates")
        return df.copy(), df[meta_cols].copy()

    # Warn about plates with too few DMSO wells (weak batch reference).
    pert_col = config.metadata.pert_type_col
    negcon = config.metrics.negcon_value
    min_dmso = config.batch_correction.min_dmso_per_plate
    if pert_col in df.columns:
        dmso_per_plate = (
            df.loc[df[pert_col] == negcon, plate_col].value_counts().to_dict()
        )
        low = {k: v for k, v in dmso_per_plate.items() if v < min_dmso}
        if low:
            logger.warning(f"  Plates with <{min_dmso} DMSO wells: {low}")

    # Build AnnData (ComBat requires float64).
    obs = df[meta_cols].reset_index(drop=True)
    obs.index = obs.index.astype(str)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    adata = ad.AnnData(X=X, obs=obs)
    adata.obs[plate_col] = adata.obs[plate_col].astype("category")

    # ── Step 1: ComBat plate correction ──────────────────────────────────────
    try:
        X_before = adata.X.copy()
        sc.pp.combat(adata, key=plate_col)
        shift = float(np.abs(adata.X - X_before).mean())
        logger.info(f"  ComBat (plate): mean absolute shift = {shift:.4f}")
        if shift < 1e-6:
            logger.warning("  ComBat produced ~no change — check that plate labels vary")
    except Exception as e:
        logger.error(f"  ✗ ComBat FAILED ({type(e).__name__}: {e}) — Level 4c will equal Level 4b")
        return df.copy(), df[meta_cols].copy()

    # NOTE: well row / column covariate regression is NO LONGER done here. It has
    # moved to the shared, method-independent step `regress_out_covariates`, which
    # the orchestrator applies after batch correction (so Harmony/Spherize get it
    # too). ComBat behaviour is preserved: the same regression now runs one step
    # later on this ComBat output.

    # AnnData stores string obs columns as `category`. Cast them back to object
    # so downstream groupby() (e.g. Level 5 consensus) does not treat them as
    # categoricals and emit every unobserved compound×dose combination.
    obs_out = adata.obs.reset_index(drop=True)
    for c in obs_out.columns:
        if isinstance(obs_out[c].dtype, pd.CategoricalDtype):
            obs_out[c] = obs_out[c].astype(object)

    df_corrected = pd.concat(
        [obs_out, pd.DataFrame(adata.X, columns=feature_cols)],
        axis=1,
    )
    return df_corrected, df_corrected[meta_cols].copy()


def apply_batch_correction(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dispatch to the configured Level 4c batch-correction method.

    ``config.batch_correction.method`` selects "combat" (default), "harmony" or
    "spherize". Unknown methods fall back to ComBat with a warning. Each method
    returns (df_corrected, metadata) with identical column layout to the input.
    """
    method = str(getattr(config.batch_correction, "method", "combat")).lower()
    dispatch = {
        "combat": apply_combat,
        "harmony": apply_harmony,
        "spherize": apply_spherize,
    }
    if method not in dispatch:
        logger.warning("  Unknown batch_correction.method '%s' — falling back to ComBat", method)
        method = "combat"
    logger.info("  Batch correction method: %s", method)
    return dispatch[method](df, metadata, config)


def apply_harmony(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Harmony batch integration (harmonypy) with the plate as the batch variable.

    Harmony is run directly on the (already MAD-normalized, feature-selected)
    feature block rather than on PCs, so the corrected output stays in the
    original feature space and remains interpretable for consensus/metrics.
    harmonypy returns the corrected embedding as ``Z_corr`` with shape
    (n_features, n_wells); its transpose is the corrected well × feature matrix.

    Falls back to the uncorrected data (with a warning) if harmonypy is missing,
    there are <2 plates, or the run fails.
    """
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    plate_col = config.batch_correction.plate_col

    if plate_col not in df.columns or df[plate_col].nunique() < 2:
        logger.info("  Batch correction skipped — need ≥2 plates")
        return df.copy(), df[meta_cols].copy()

    try:
        import harmonypy
    except ImportError:
        logger.error("  ✗ harmonypy not installed — Level 4c will equal Level 4b")
        return df.copy(), df[meta_cols].copy()

    try:
        X = df[feature_cols].to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        obs = df[[plate_col]].reset_index(drop=True).astype(str)

        # Floor nclust: harmonypy's auto value min(N/30, 100) collapses to 1 on
        # small inputs, which crashes this fork ("object of type 'float' has no
        # len()"). Keep ≥2 clusters. See docs/HARMONY_NOTES.md.
        nclust = max(2, min(X.shape[0] // 30, 100))
        ho = harmonypy.run_harmony(X, obs, vars_use=[plate_col], nclust=nclust)
        # Z_corr orientation varies by harmonypy fork: classic returns
        # (n_features, n_wells), this PyTorch fork returns (n_wells, n_features).
        # Pick whichever matches the input (wells × features).
        Z = np.asarray(ho.Z_corr)
        if Z.shape == X.shape:
            X_corr = Z
        elif Z.T.shape == X.shape:
            X_corr = Z.T
        else:
            logger.error("  ✗ Harmony output shape %s incompatible with input %s — using Level 4b",
                         Z.shape, X.shape)
            return df.copy(), df[meta_cols].copy()

        shift = float(np.abs(X_corr - X).mean())
        logger.info("  Harmony (plate): mean absolute shift = %.4f", shift)

        obs_out = df[meta_cols].reset_index(drop=True)
        df_corrected = pd.concat(
            [obs_out, pd.DataFrame(X_corr, columns=feature_cols)], axis=1
        )
        return df_corrected, df_corrected[meta_cols].copy()
    except Exception as e:
        logger.error("  ✗ Harmony FAILED (%s: %s) — Level 4c will equal Level 4b",
                     type(e).__name__, e)
        return df.copy(), df[meta_cols].copy()


def apply_spherize(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Spherize (ZCA whitening) batch correction via pycytominer.

    The whitening transform is FIT on the negative controls (DMSO wells, the
    same biology on every plate) and APPLIED to all wells. This decorrelates and
    unit-scales the technical covariance structure shared across plates while
    preserving treatment-specific deviations. ZCA keeps the feature space aligned
    with the input (one output column per input feature).

    Falls back to the uncorrected data (with a warning) if the transform fails.
    """
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]

    # Fit on the negcon reference (same query used for Level 4a normalization).
    samples_query = config.normalization.samples_query or "all"
    pert_col = config.metadata.pert_type_col
    negcon = config.metrics.negcon_value
    if pert_col in df.columns:
        n_ref = int((df[pert_col].astype(str) == str(negcon)).sum())
        if n_ref < 2:
            logger.warning("  Spherize: <2 negcon wells — fitting on all wells instead")
            samples_query = "all"

    try:
        from pycytominer import normalize
        df_corrected = normalize(
            profiles=df,
            features=feature_cols,
            meta_features=meta_cols,
            samples=samples_query,
            method="spherize",
            spherize_method="ZCA",
            spherize_center=True,
        )
        # Preserve column ordering (metadata first, then features) for parity with
        # the other methods; normalize() may reorder.
        ordered = meta_cols + [c for c in df_corrected.columns if c not in meta_cols]
        df_corrected = df_corrected[[c for c in ordered if c in df_corrected.columns]]
        shift = float(
            np.abs(df_corrected[feature_cols].to_numpy(float)
                   - df[feature_cols].to_numpy(float)).mean()
        )
        logger.info("  Spherize (ZCA, fit on %s): mean absolute shift = %.4f",
                    "negcon" if samples_query != "all" else "all wells", shift)
        return df_corrected, df_corrected[meta_cols].copy()
    except Exception as e:
        logger.error("  ✗ Spherize FAILED (%s: %s) — Level 4c will equal Level 4b",
                     type(e).__name__, e)
        return df.copy(), df[meta_cols].copy()
