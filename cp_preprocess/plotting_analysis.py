"""
Staged QC & analysis plots for the cell painting preprocessing workflow.

Each plot is a check on a specific stage of the experiment, mirroring the
CP_MS/BA/HOG/plots/qc reference style:

  1. plot_image_qc_flags        — sites dropped by image QC (blur/focus/low-count), by plate & reason
  2. plot_cell_count_per_plate  — well-level cell count distribution + plate CV
  3. plot_cell_count_per_site   — negcon cell count per site + plate×site heatmap
  4. plot_dmso_site_qc          — DMSO site cell-count QC: min/max/mean + flagged outliers (+CSV)
  5. plot_feature_preprocessing — feature counts across L3→L4a→L4b→L4c + variance
  6. plot_dmso_normalization_check — per-plate DMSO mean feature value after L4a normalization (should be ≈0)
  7. plot_batch_effects         — prebatch (L4b) vs afterbatch (L4c) PCA by plate & row
  8. plot_dmso_correlation      — DMSO control correlation by plate (Caicedo et al. 2017 batch-effect diagnostic)

All functions are defensive: if a required column is missing they log and skip
rather than producing an empty figure.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent palette
C_MAIN = "#9B59B6"
C_ACCENT = "#E07B39"
C_OK = "#5BAD6F"
C_WARN = "#E0A030"
C_FAIL = "#CC3333"
C_BLUE = "#3A7FC1"

COUNT_CANDIDATES = ["Metadata_Count_nuclei", "Metadata_Count_nuclei_merged",
                    "Metadata_Cells_per_well_total"]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _feat_cols(df: pd.DataFrame):
    return [c for c in df.columns if not c.startswith("Metadata_")]


def _find_col(df, *substrs, exact=None):
    if exact:
        for e in exact:
            if e in df.columns:
                return e
    low = {c.lower(): c for c in df.columns}
    for c in df.columns:
        cl = c.lower()
        if all(s.lower() in cl for s in substrs):
            return c
    return None


def _count_col(df):
    for c in COUNT_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _well_row_col(well):
    s = str(well).strip()
    if len(s) < 2:
        return None, None
    row = s[0].upper()
    try:
        col = int(s[1:])
    except ValueError:
        return row, None
    return row, col


def _add_footnote(fig, text):
    """Write ``text`` in small grey type just below everything else on ``fig``
    (once per figure)."""
    if getattr(fig, "_source_note", None) == text:
        return
    try:
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        y = bbox.y0 / fig.get_figheight() - 0.01
    except Exception:
        y = -0.02
    fig.text(0.01, y, text, ha="left", va="top", fontsize=7, color="0.4", wrap=False)
    fig._source_note = text


@contextmanager
def source_note(text):
    """Inside this block every saved figure (Figure.savefig, plt.savefig,
    seaborn grid .savefig) gets ``text`` as a footnote — used by
    run_analysis_plots to label each plot with its data level and run folder."""
    from matplotlib.figure import Figure
    orig = Figure.savefig

    def savefig(fig, *args, **kwargs):
        _add_footnote(fig, text)
        return orig(fig, *args, **kwargs)

    Figure.savefig = savefig
    try:
        yield
    finally:
        Figure.savefig = orig


def _save(fig, plot_dir, name):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. Image QC flagged sites
# ─────────────────────────────────────────────────────────────────────────────

def plot_image_qc_flags(image_qc_dir: Path, plots_dir: Path, cell_line: str) -> None:
    """Stacked bar of sites flagged by image QC per plate, broken down by reason."""
    image_qc_dir = Path(image_qc_dir)
    flagged = image_qc_dir / "flagged_sites.csv"
    # Prefer combining per-plate qc_flags for full totals; fall back to flagged_sites
    flag_files = sorted(image_qc_dir.glob("qc_flags_*.csv"))
    if not flag_files and not flagged.exists():
        logger.info("  No image QC files found — skipping image QC plot")
        return

    try:
        if flag_files:
            df = pd.concat([pd.read_csv(f) for f in flag_files], ignore_index=True)
        else:
            df = pd.read_csv(flagged)

        flag_cols = [c for c in df.columns if c.startswith("OutlierScaled")]
        plate_col = _find_col(df, exact=["Metadata_Barcode"])
        if plate_col is None or not flag_cols:
            logger.info("  Image QC file lacks flag columns — skipping")
            return

        plates = sorted(df[plate_col].astype(str).unique())
        counts = df.assign(**{plate_col: df[plate_col].astype(str)}) \
                   .groupby(plate_col)[flag_cols].sum().reindex(plates).fillna(0)
        total_sites = df.assign(**{plate_col: df[plate_col].astype(str)}) \
                        .groupby(plate_col).size().reindex(plates).fillna(0)
        flagged_any = df[df.get("Total", 0) > 0] if "Total" in df.columns else None

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Left: stacked bar by reason
        ax = axes[0]
        bottom = np.zeros(len(plates))
        palette = sns.color_palette("Set2", n_colors=len(flag_cols))
        for i, col in enumerate(flag_cols):
            vals = counts[col].to_numpy()
            ax.bar(plates, vals, bottom=bottom, color=palette[i],
                   label=col.replace("OutlierScaled_", "").split("_-")[0],
                   edgecolor="white", linewidth=0.5)
            bottom += vals
        ax.set_ylabel("Flagged sites (by reason)", fontweight="bold")
        ax.set_xlabel("Plate", fontweight="bold")
        ax.set_title(f"{cell_line} — Image QC flags by plate", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8, ncol=2)
        sns.despine(ax=ax)

        # Right: % of sites dropped per plate
        ax = axes[1]
        n_flag = (flagged_any.assign(**{plate_col: flagged_any[plate_col].astype(str)})
                  .groupby(plate_col).size().reindex(plates).fillna(0)) if flagged_any is not None else bottom
        pct = 100 * np.asarray(n_flag) / np.maximum(total_sites.to_numpy(), 1)
        bars = ax.bar(plates, pct, color=C_ACCENT, edgecolor="white", alpha=0.85)
        for b, p, n in zip(bars, pct, np.asarray(n_flag)):
            ax.text(b.get_x() + b.get_width() / 2, p, f"{int(n)}\n({p:.1f}%)",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("% sites flagged (dropped)", fontweight="bold")
        ax.set_xlabel("Plate", fontweight="bold")
        ax.set_title(f"{cell_line} — Sites dropped by image QC", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        sns.despine(ax=ax)

        _save(fig, plots_dir, f"{cell_line}_qc_image_flags.png")
    except Exception as e:
        logger.warning(f"Image QC flags plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cell count per plate (well level)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cell_count_per_plate(df_l3: pd.DataFrame, plots_dir: Path, cell_line: str,
                               min_cells_per_well: int = 50) -> None:
    """Violin of per-well cell count by plate + plate-level CV (like HOG qc_cell_counts)."""
    plate_col = _find_col(df_l3, exact=["Metadata_Barcode"])
    count_col = _count_col(df_l3)
    if plate_col is None or count_col is None:
        logger.info("  No cell count / plate column — skipping cell-count-per-plate plot")
        return

    try:
        d = df_l3[[plate_col, count_col]].copy()
        d[count_col] = pd.to_numeric(d[count_col], errors="coerce")
        d = d.dropna()
        plates = sorted(d[plate_col].astype(str).unique())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: violin per plate
        ax = axes[0]
        data = [d[d[plate_col].astype(str) == p][count_col].values for p in plates]
        parts = ax.violinplot(data, showmeans=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(C_MAIN)
            pc.set_alpha(0.6)
        ax.axhline(min_cells_per_well, color=C_FAIL, linestyle="--",
                   label=f"Min threshold ({min_cells_per_well})")
        ax.set_xticks(range(1, len(plates) + 1))
        ax.set_xticklabels(plates, rotation=45)
        ax.set_ylabel("Cell Count per Well", fontweight="bold")
        ax.set_title(f"{cell_line} — Cell Count per Plate", fontweight="bold")
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

        # Right: plate CV
        ax = axes[1]
        stats = d.groupby(d[plate_col].astype(str))[count_col].agg(["mean", "std"])
        stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
        stats = stats.reindex(plates)
        ax.bar(plates, stats["cv"].to_numpy(), color=C_MAIN, alpha=0.85, edgecolor="white")
        ax.axhline(0.30, color=C_WARN, linestyle="--", label="Warn (30%)")
        ax.axhline(0.50, color=C_FAIL, linestyle="--", label="Fail (50%)")
        ax.set_ylabel("Cell Count CV", fontweight="bold")
        ax.set_xlabel("Plate", fontweight="bold")
        ax.set_title(f"{cell_line} — Plate-Level Cell Count CV", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

        _save(fig, plots_dir, f"{cell_line}_qc_cell_count_per_plate.png")
    except Exception as e:
        logger.warning(f"Cell count per plate plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cell count per site (site level, negcon)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cell_count_per_site(df_sites: pd.DataFrame, plots_dir: Path, cell_line: str,
                             min_cells_per_site: int = 20,
                             negcon_value: str = "negcon") -> None:
    """Negcon cell count per site number (boxplot) + median count per (plate, site) heatmap."""
    plate_col = _find_col(df_sites, exact=["Metadata_Barcode"])
    site_col = _find_col(df_sites, exact=["Metadata_Site"])
    pert_col = _find_col(df_sites, exact=["Metadata_pert_type"])
    count_col = _count_col(df_sites)
    if not all([plate_col, site_col, count_col]):
        logger.info("  Missing plate/site/count columns — skipping cell-count-per-site plot")
        return

    try:
        d = df_sites.copy()
        d[count_col] = pd.to_numeric(d[count_col], errors="coerce")
        d[site_col] = pd.to_numeric(d[site_col], errors="coerce").astype("Int64")
        d = d.dropna(subset=[count_col, site_col])
        if pert_col is not None:
            dctrl = d[d[pert_col] == negcon_value]
            if len(dctrl) == 0:
                dctrl = d
                subtitle = "all sites"
            else:
                subtitle = "negcon"
        else:
            dctrl = d
            subtitle = "all sites"

        sites = sorted(dctrl[site_col].dropna().unique())
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Left: boxplot per site number
        ax = axes[0]
        data = [dctrl[dctrl[site_col] == s][count_col].values for s in sites]
        bp = ax.boxplot(data, labels=[str(int(s)) for s in sites], patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor(C_BLUE)
            box.set_alpha(0.75)
        ax.axhline(min_cells_per_site, color=C_FAIL, linestyle="--",
                   label=f"threshold={min_cells_per_site}")
        ax.set_xlabel("Site", fontweight="bold")
        ax.set_ylabel("Cell count", fontweight="bold")
        ax.set_title(f"{cell_line} — {subtitle} cell count per site", fontweight="bold")
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

        # Right: median count per (plate, site) heatmap
        ax = axes[1]
        pivot = (dctrl.assign(**{plate_col: dctrl[plate_col].astype(str)})
                 .groupby([plate_col, site_col])[count_col].median().unstack())
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Site", fontweight="bold")
        ax.set_ylabel("Plate", fontweight="bold")
        ax.set_title(f"{cell_line} — Median cell count per (plate, site)", fontweight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v < np.nanmedian(pivot.values) else "black")
        plt.colorbar(im, ax=ax)

        _save(fig, plots_dir, f"{cell_line}_qc_cell_count_per_site.png")
    except Exception as e:
        logger.warning(f"Cell count per site plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DMSO site QC — min/max/mean + outlier flagging (+ CSV)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dmso_site_qc(df_sites: pd.DataFrame, plots_dir: Path, cell_line: str,
                      dmso_min_cells: int = 30, z_threshold: float = 3.5,
                      negcon_value: str = "negcon") -> None:
    """
    DMSO (negcon) site cell-count QC. Spots problematic DMSO sites (low count)
    using an absolute threshold + robust-Z, and writes a CSV of flagged sites.
    This protects the normalization reference.
    """
    plate_col = _find_col(df_sites, exact=["Metadata_Barcode"])
    well_col = _find_col(df_sites, exact=["Metadata_Well"])
    site_col = _find_col(df_sites, exact=["Metadata_Site"])
    pert_col = _find_col(df_sites, exact=["Metadata_pert_type"])
    count_col = _count_col(df_sites)
    if not all([plate_col, count_col, pert_col]):
        logger.info("  Missing columns for DMSO site QC — skipping")
        return

    try:
        d = df_sites[df_sites[pert_col] == negcon_value].copy()
        if len(d) == 0:
            logger.info("  No negcon sites — skipping DMSO site QC")
            return
        d[count_col] = pd.to_numeric(d[count_col], errors="coerce")
        d = d.dropna(subset=[count_col]).reset_index(drop=True)

        counts = d[count_col]
        med = counts.median()
        mad = (counts - med).abs().median() or (counts.std() or 1e-9)
        z = (counts - med) / (1.4826 * mad)
        low_abs = counts < dmso_min_cells
        low_z = z < -abs(z_threshold)
        bad = low_abs | low_z

        cmin, cmax, cmean = counts.min(), counts.max(), counts.mean()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Left: per-site scatter with thresholds
        ax = axes[0]
        colors = [C_FAIL if b else C_OK for b in bad]
        ax.scatter(range(len(counts)), counts, c=colors, s=45, alpha=0.8,
                   edgecolors="black", linewidth=0.4)
        ax.axhline(cmean, color="black", linestyle="-", linewidth=1, label=f"mean={cmean:.0f}")
        ax.axhline(cmin, color=C_BLUE, linestyle=":", linewidth=1, label=f"min={cmin:.0f}")
        ax.axhline(cmax, color=C_BLUE, linestyle=":", linewidth=1, label=f"max={cmax:.0f}")
        ax.axhline(dmso_min_cells, color=C_FAIL, linestyle="--", linewidth=1.5,
                   label=f"abs cutoff={dmso_min_cells}")
        z_cut = med - abs(z_threshold) * 1.4826 * mad
        ax.axhline(z_cut, color=C_WARN, linestyle="--", linewidth=1.5,
                   label=f"robust-Z cutoff={z_cut:.0f}")
        ax.set_xlabel("DMSO site index", fontweight="bold")
        ax.set_ylabel("Cell count", fontweight="bold")
        ax.set_title(f"{cell_line} — DMSO site cell count (n={len(counts)}, "
                     f"{int(bad.sum())} flagged)", fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        sns.despine(ax=ax)

        # Right: histogram
        ax = axes[1]
        ax.hist(counts[~bad], bins=25, color=C_OK, alpha=0.8, edgecolor="black", label="keep")
        if bad.any():
            ax.hist(counts[bad], bins=15, color=C_FAIL, alpha=0.8, edgecolor="black",
                    label=f"drop (n={int(bad.sum())})")
        ax.axvline(med, color="black", linestyle="--", label=f"median={med:.0f}")
        ax.set_xlabel("Cell count", fontweight="bold")
        ax.set_ylabel("DMSO sites", fontweight="bold")
        ax.set_title(f"{cell_line} — DMSO count distribution", fontweight="bold")
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

        _save(fig, plots_dir, f"{cell_line}_qc_dmso_site.png")

        # CSV of flagged DMSO sites
        keep_cols = [c for c in [plate_col, well_col, site_col] if c in d.columns]
        out = d[keep_cols].copy()
        out["cell_count"] = counts.values
        out["robust_z"] = z.values
        out["flag_low_abs"] = low_abs.values
        out["flag_low_z"] = low_z.values
        out["drop"] = bad.values
        csv_path = Path(plots_dir) / f"{cell_line}_dmso_flagged_sites.csv"
        out[out["drop"]].sort_values("cell_count").to_csv(csv_path, index=False)
        logger.info(f"  Saved: {csv_path.name} ({int(bad.sum())} flagged DMSO sites)")
    except Exception as e:
        logger.warning(f"DMSO site QC plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature preprocessing progression
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_preprocessing(df_l3, df_l4a, df_l4b, df_l4c, plots_dir, cell_line) -> None:
    """Feature counts through the preprocessing steps + value-distribution change."""
    try:
        levels = [("L3\nraw", df_l3), ("L4a\nnormalized", df_l4a),
                  ("L4b\nselected", df_l4b), ("L4c\nbatch-corr", df_l4c)]
        names = [n for n, _ in levels]
        n_feats = [len(_feat_cols(df)) for _, df in levels]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: feature count funnel
        ax = axes[0]
        colors = [C_ACCENT, C_OK, C_BLUE, C_MAIN]
        bars = ax.bar(names, n_feats, color=colors, alpha=0.85, edgecolor="black")
        for b, n in zip(bars, n_feats):
            ax.text(b.get_x() + b.get_width() / 2, n, f"{n}", ha="center", va="bottom",
                    fontweight="bold")
        ax.set_ylabel("Number of features", fontweight="bold")
        ax.set_title(f"{cell_line} — Feature count through preprocessing", fontweight="bold")
        sns.despine(ax=ax)

        # Right: value distribution before/after normalization
        ax = axes[1]
        raw_vals = df_l3[_feat_cols(df_l3)].to_numpy().flatten()
        norm_vals = df_l4a[_feat_cols(df_l4a)].to_numpy().flatten()
        raw_vals = raw_vals[np.isfinite(raw_vals)]
        norm_vals = norm_vals[np.isfinite(norm_vals)]
        # clip for display
        ax.hist(np.clip(norm_vals, -10, 10), bins=80, color=C_OK, alpha=0.7,
                density=True, label="L4a normalized")
        ax.set_xlabel("Feature value (normalized, clipped ±10)", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.set_title(f"{cell_line} — Normalized feature distribution", fontweight="bold")
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

        _save(fig, plots_dir, f"{cell_line}_qc_feature_preprocessing.png")
    except Exception as e:
        logger.warning(f"Feature preprocessing plot failed: {e}")


def plot_dmso_normalization_check(df_l4a, plots_dir, cell_line, negcon_value="negcon",
                                   tolerance=0.5) -> None:
    """
    DMSO-centering QC for Level 4a (post mad_robustize normalization).

    Each DMSO/negcon well is normalized against its own plate's DMSO reference,
    so the mean feature value across DMSO wells on a given plate should sit
    close to 0. Plots one point per plate (mean-of-feature-means across that
    plate's DMSO wells) against a 0-centered tolerance band — a plate outside
    the band signals a normalization problem for that plate (too few DMSO
    wells, or a fallback-to-raw-values on that plate).
    """
    plate_col = _find_col(df_l4a, exact=["Metadata_Barcode"])
    pert_col = _find_col(df_l4a, exact=["Metadata_pert_type"])
    if plate_col is None or pert_col is None:
        logger.info("  No plate/pert_type column — skipping DMSO-centering check plot")
        return
    try:
        feats = _feat_cols(df_l4a)
        dmso = df_l4a[df_l4a[pert_col].astype(str) == negcon_value]
        if dmso.empty or not feats:
            logger.info("  No DMSO wells / features — skipping DMSO-centering check plot")
            return

        per_plate_mean = dmso.groupby(plate_col, observed=True)[feats].mean().mean(axis=1)
        per_plate_mean = per_plate_mean.sort_index()
        overall = float(per_plate_mean.mean())
        off = per_plate_mean[per_plate_mean.abs() > tolerance]

        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(per_plate_mean)), 4.5))
        colors = [C_FAIL if abs(v) > tolerance else C_OK for v in per_plate_mean.values]
        ax.bar(per_plate_mean.index.astype(str), per_plate_mean.values, color=colors,
               edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", lw=1)
        ax.axhspan(-tolerance, tolerance, color=C_OK, alpha=0.1,
                   label=f"±{tolerance} tolerance")
        ax.set_ylabel("Mean DMSO feature value (post-normalization)", fontweight="bold")
        ax.set_xlabel("Plate", fontweight="bold")
        ax.set_title(
            f"{cell_line} — DMSO-centering check (L4a, {len(off)}/{len(per_plate_mean)} "
            f"plate(s) outside ±{tolerance})",
            fontweight="bold",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        _save(fig, plots_dir, f"{cell_line}_qc_dmso_normalization_check.png")
    except Exception as e:
        logger.warning(f"DMSO-centering check plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6b. Batch effects: prebatch (L4b) vs afterbatch (L4c) PCA by plate & row
# ─────────────────────────────────────────────────────────────────────────────

def _pca_2d(df):
    from sklearn.decomposition import PCA
    X = df[_feat_cols(df)].fillna(0).to_numpy(dtype=float)
    X = X[:, np.isfinite(X).all(axis=0)]
    n = min(2, X.shape[1])
    if n < 2:
        return None
    pcs = PCA(n_components=2, random_state=42).fit_transform(X)
    return pcs


def _scatter_categorical(ax, emb, labels, title):
    """Discrete colors + legend — for low-cardinality labels like plate."""
    cats = pd.Series(labels).astype(str)
    uniq = sorted(cats.unique())
    palette = sns.color_palette("tab10", n_colors=max(len(uniq), 1))
    for i, u in enumerate(uniq):
        m = (cats == u).to_numpy()
        ax.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.6, color=palette[i % len(palette)],
                   label=u, edgecolors="none")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7, markerscale=1.5, title=None)
    sns.despine(ax=ax)


def _scatter_ordinal(fig, ax, emb, values, title, cbar_label):
    """Ordered colormap + colorbar — for positional labels like well row/column."""
    vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
    finite = np.isfinite(vals)
    sc = ax.scatter(emb[finite, 0], emb[finite, 1], s=12, alpha=0.65,
                    c=vals[finite], cmap="viridis", edgecolors="none")
    ax.set_title(title, fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label, fontsize=8)
    sns.despine(ax=ax)


def _umap_2d(df):
    """2-D UMAP embedding of the feature block (PCA-initialised)."""
    try:
        import scanpy as sc
    except ImportError:
        return None
    feats = _feat_cols(df)
    X = df[feats].fillna(0).to_numpy(dtype=np.float32)
    X = X[:, np.isfinite(X).all(axis=0)]
    if X.shape[1] < 2:
        return None
    adata = sc.AnnData(X=X)
    sc.pp.pca(adata, n_comps=min(50, X.shape[1] - 1))
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    sc.tl.umap(adata, min_dist=0.1)
    return adata.obsm["X_umap"]


def _row_to_num(well):
    """Well row letter → number (A=1 … P=16)."""
    r = _well_row_col(well)[0]
    return (ord(r) - ord("A") + 1) if r else np.nan


def _col_num(well):
    """Well column number (1–24)."""
    return _well_row_col(well)[1]


def _batch_effect_figure(emb_before, emb_after, df_l4b, df_l4c, plate_col, well_col,
                         cell_line, method, plots_dir, fname, method_label="Batch Correction"):
    axis = "UMAP" if method == "UMAP" else "PC"
    have_well = well_col is not None and well_col in df_l4b.columns
    nrows = 3 if have_well else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 5.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, 2)

    # Row 1: by plate (categorical)
    _scatter_categorical(axes[0, 0], emb_before, df_l4b[plate_col], f"{cell_line} — Before {method_label}: by plate")
    _scatter_categorical(axes[0, 1], emb_after, df_l4c[plate_col], f"{cell_line} — After {method_label}: by plate")

    if have_well:
        # Row 2: by well row (ordinal A–P)
        rb = df_l4b[well_col].map(_row_to_num)
        ra = df_l4c[well_col].map(_row_to_num)
        _scatter_ordinal(fig, axes[1, 0], emb_before, rb, f"Before {method_label}: by well row", "row (A=1…P=16)")
        _scatter_ordinal(fig, axes[1, 1], emb_after, ra, f"After {method_label}: by well row", "row (A=1…P=16)")
        # Row 3: by well column (ordinal 1–24)
        cb = df_l4b[well_col].map(_col_num)
        ca = df_l4c[well_col].map(_col_num)
        _scatter_ordinal(fig, axes[2, 0], emb_before, cb, f"Before {method_label}: by well column", "column (1–24)")
        _scatter_ordinal(fig, axes[2, 1], emb_after, ca, f"After {method_label}: by well column", "column (1–24)")

    for ax in axes.flat:
        if ax.has_data():
            ax.set_xlabel(f"{axis} 1"); ax.set_ylabel(f"{axis} 2")
    fig.suptitle(f"{cell_line} — Batch effect ({method}: plate, row & column) before vs after {method_label}",
                 fontsize=14, fontweight="bold")
    _save(fig, plots_dir, fname)


def plot_spatial_effects(df_l4b, plots_dir, cell_line) -> None:
    """
    Spatial/positional effect diagnostic: mean PC1 score by well row and by well
    column, one line per plate (like HOG qc_spatial_effects). Reveals edge/gradient
    artefacts that survive normalization.
    """
    plate_col = _find_col(df_l4b, exact=["Metadata_Barcode"])
    well_col = _find_col(df_l4b, exact=["Metadata_Well"])
    if plate_col is None or well_col is None:
        logger.info("  No plate/well column — skipping spatial effects plot")
        return
    try:
        pcs = _pca_2d(df_l4b)
        if pcs is None:
            logger.info("  Too few features for PCA — skipping spatial effects plot")
            return
        d = pd.DataFrame({
            "plate": df_l4b[plate_col].astype(str).values,
            "row": df_l4b[well_col].map(_row_to_num).values,
            "col": df_l4b[well_col].map(_col_num).values,
            "pc1": pcs[:, 0],
        })
        plates = sorted(d["plate"].unique())
        palette = sns.color_palette("tab10", n_colors=len(plates))

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        # by row
        for i, p in enumerate(plates):
            g = d[d["plate"] == p].groupby("row")["pc1"].mean()
            axes[0].plot(g.index, g.values, "o-", color=palette[i], label=p, alpha=0.8, ms=4)
        row_ticks = sorted(d["row"].dropna().unique())
        axes[0].set_xticks(row_ticks)
        axes[0].set_xticklabels([chr(int(r) - 1 + ord("A")) for r in row_ticks])
        axes[0].set_xlabel("Plate Row", fontweight="bold")
        axes[0].set_ylabel("Mean PC1 Score", fontweight="bold")
        axes[0].set_title(f"{cell_line} — Spatial Effect: Plate Row", fontweight="bold")
        axes[0].legend(title="Plate", fontsize=8)
        axes[0].grid(alpha=0.2)
        sns.despine(ax=axes[0])
        # by column
        for i, p in enumerate(plates):
            g = d[d["plate"] == p].groupby("col")["pc1"].mean()
            axes[1].plot(g.index, g.values, "o-", color=palette[i], label=p, alpha=0.8, ms=4)
        axes[1].set_xlabel("Plate Column", fontweight="bold")
        axes[1].set_ylabel("Mean PC1 Score", fontweight="bold")
        axes[1].set_title(f"{cell_line} — Spatial Effect: Plate Column", fontweight="bold")
        axes[1].legend(title="Plate", fontsize=8)
        axes[1].grid(alpha=0.2)
        sns.despine(ax=axes[1])

        _save(fig, plots_dir, f"{cell_line}_qc_spatial_effects.png")
    except Exception as e:
        logger.warning(f"Spatial effects plot failed: {e}")


def plot_well_count_heatmap(df_l3, plots_dir, cell_line) -> None:
    """
    Per-plate 384-well heatmap of total cell count per well (like HOG
    'Cell counts per well'). Needs the real per-well count column.
    """
    import matplotlib.gridspec as gridspec
    plate_col = _find_col(df_l3, exact=["Metadata_Barcode"])
    well_col = _find_col(df_l3, exact=["Metadata_Well"])
    count_col = _count_col(df_l3)
    if not all([plate_col, well_col, count_col]):
        logger.info("  No plate/well/count column — skipping well-count heatmap")
        return
    try:
        d = df_l3[[plate_col, well_col, count_col]].copy()
        d[count_col] = pd.to_numeric(d[count_col], errors="coerce")
        d["row"] = d[well_col].map(_row_to_num)
        d["col"] = d[well_col].map(_col_num)
        d = d.dropna(subset=["row", "col", count_col])
        plates = sorted(d[plate_col].astype(str).unique())
        n = len(plates)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig = plt.figure(figsize=(7 * ncol, 4 * nrow))
        gs = gridspec.GridSpec(nrow, ncol, figure=fig)

        all_rows = list(range(1, 17))   # A–P
        all_cols = list(range(1, 25))   # 1–24
        for i, p in enumerate(plates):
            ax = fig.add_subplot(gs[i // ncol, i % ncol])
            sub = d[d[plate_col].astype(str) == p]
            grid = np.full((16, 24), np.nan)
            for _, r in sub.iterrows():
                rr, cc = int(r["row"]), int(r["col"])
                if 1 <= rr <= 16 and 1 <= cc <= 24:
                    grid[rr - 1, cc - 1] = r[count_col]
            im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
            ax.set_title(str(p), fontsize=10, fontweight="bold")
            ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels(range(1, 25, 2), fontsize=6)
            ax.set_yticks(range(16)); ax.set_yticklabels([chr(ord("A") + k) for k in range(16)], fontsize=6)
            plt.colorbar(im, ax=ax, label="Total nuclei/well")
        fig.suptitle(f"{cell_line} — Cell counts per well (384-well)", fontsize=13, fontweight="bold")
        _save(fig, plots_dir, f"{cell_line}_qc_well_count_heatmap.png")
    except Exception as e:
        logger.warning(f"Well count heatmap failed: {e}")


def plot_batch_effects(df_l4b, df_l4c, plots_dir, cell_line, method_label="Batch Correction") -> None:
    """PCA colored by plate and by well-row, before (L4b) and after (L4c) batch correction."""
    plate_col = _find_col(df_l4b, exact=["Metadata_Barcode"])
    well_col = _find_col(df_l4b, exact=["Metadata_Well"])
    if plate_col is None:
        logger.info("  No plate column — skipping batch effect plot")
        return
    try:
        pcs_before = _pca_2d(df_l4b)
        pcs_after = _pca_2d(df_l4c)
        if pcs_before is None or pcs_after is None:
            logger.info("  Not enough features for PCA — skipping batch effect plot")
            return
        _batch_effect_figure(pcs_before, pcs_after, df_l4b, df_l4c, plate_col, well_col,
                             cell_line, "PCA", plots_dir, f"{cell_line}_qc_batch_effects.png",
                             method_label=method_label)
    except Exception as e:
        logger.warning(f"Batch effects (PCA) plot failed: {e}")


def plot_batch_effects_umap(df_l4b, df_l4c, plots_dir, cell_line, method_label="Batch Correction") -> None:
    """UMAP colored by plate and by well-row, before (L4b) and after (L4c) batch correction."""
    plate_col = _find_col(df_l4b, exact=["Metadata_Barcode"])
    well_col = _find_col(df_l4b, exact=["Metadata_Well"])
    if plate_col is None:
        logger.info("  No plate column — skipping batch effect UMAP plot")
        return
    try:
        emb_before = _umap_2d(df_l4b)
        emb_after = _umap_2d(df_l4c)
        if emb_before is None or emb_after is None:
            logger.info("  scanpy unavailable / too few features — skipping batch effect UMAP")
            return
        _batch_effect_figure(emb_before, emb_after, df_l4b, df_l4c, plate_col, well_col,
                             cell_line, "UMAP", plots_dir, f"{cell_line}_qc_batch_effects_umap.png",
                             method_label=method_label)
    except Exception as e:
        logger.warning(f"Batch effects (UMAP) plot failed: {e}")


def plot_dmso_correlation(df, plots_dir, cell_line, label="L4b",
                          negcon_value="negcon", n_plates=5, seed=42) -> None:
    """
    Batch-effect diagnostic following Caicedo et al. (2017, Nat. Methods,
    https://doi.org/10.1038/nmeth.4397).

    Plots the pairwise Pearson correlation between all DMSO (negcon) control
    profiles, with each control sample color-annotated by its plate. Because the
    controls are biologically identical, any structure is technical: a high
    *intra-plate* correlation relative to the *cross-plate* correlation shows up
    as blocks along the diagonal and indicates that plate (batch) effects are
    present.

    The full control × control matrix is too large to render, so we draw
    ``n_plates`` random plates without replacement to demonstrate the plot
    (``RandomState(seed)`` keeps the selection reproducible). Run on Level 4b
    (before ComBat) the blocks reveal batch effects; on Level 4c (after ComBat)
    they should be attenuated.
    """
    plate_col = _find_col(df, exact=["Metadata_Barcode"])
    pert_col = _find_col(df, exact=["Metadata_pert_type"])
    if plate_col is None or pert_col is None:
        logger.info("  No plate/pert_type column — skipping DMSO correlation plot")
        return
    try:
        feats = _feat_cols(df)
        dmso = df.loc[df[pert_col].astype(str) == negcon_value, [plate_col] + feats].copy()
        if len(dmso) < 3:
            logger.info("  <3 DMSO wells — skipping DMSO correlation plot")
            return

        plates = pd.unique(dmso[plate_col])
        if len(plates) < 2:
            logger.info("  <2 plates with DMSO — skipping DMSO correlation plot")
            return

        # Draw up to n_plates plates without replacement (reproducible selection).
        rng = np.random.RandomState(seed)
        k = min(n_plates, len(plates))
        chosen = sorted(rng.choice(plates, size=k, replace=False).tolist(), key=str)

        # Index each control sample by (plate, running id) so we can color by plate,
        # then transpose so the correlation is computed sample × sample.
        sub = dmso[dmso[plate_col].isin(chosen)].reset_index(drop=True)
        sub = sub.set_index([plate_col, sub.index])
        corr = sub[feats].transpose().corr()

        # One color per plate, in the given (sorted) plate order.
        palette = sns.cubehelix_palette(len(chosen), light=.6, dark=.1,
                                         reverse=True, start=1, rot=-2)
        plate_to_color = dict(zip(map(str, chosen), palette))
        side_colors = pd.Series({c: plate_to_color[str(c[0])] for c in corr.columns})

        # Center the diverging scale halfway between the lowest correlation and 1
        # so within-plate blocks stand out against the cross-plate background.
        vmin = float(np.nanmin(corr.values))
        center = 1 - ((1 - vmin) / 2)
        g = sns.clustermap(
            corr, cmap="coolwarm", center=center, vmin=vmin, vmax=1,
            row_cluster=False, col_cluster=False,
            row_colors=side_colors, col_colors=side_colors,
            linewidths=0, xticklabels=False, yticklabels=False)
        g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")

        # Legend mapping plate colors → plate IDs.
        for pid in chosen:
            g.ax_col_dendrogram.bar(0, 0, color=plate_to_color[str(pid)],
                                    label=str(pid), linewidth=0)
        g.ax_col_dendrogram.legend(loc="center", ncol=min(5, len(chosen)),
                                   title="Plate ID")
        g.cax.set_position([1, .2, .03, .45])
        g.fig.suptitle(
            f"{cell_line} — DMSO control correlation by plate ({label}, "
            f"{k} random plates)\nstronger within-plate than cross-plate "
            f"correlation ⇒ batch effect (Caicedo et al. 2017)",
            fontsize=11, fontweight="bold")

        plots_dir = Path(plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"{cell_line}_qc_dmso_correlation_{label}.png"
        g.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(g.fig)
        logger.info(f"  Saved: {out.name}")
    except Exception as e:
        logger.warning(f"DMSO correlation plot failed: {e}")


def _compound_col(df):
    for c in ("Metadata_compound", "Metadata_cmpd_name"):
        if c in df.columns:
            return c
    for c in df.columns:
        if c.startswith("Metadata_") and ("compound" in c.lower() or "cmpd" in c.lower()):
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. Controls PCA — separation of poscon / negcon / treatments
# ─────────────────────────────────────────────────────────────────────────────

def plot_controls_pca(df_wells, plots_dir, cell_line,
                      negcon_value="negcon", poscon_value="poscon") -> None:
    """
    PCA of well profiles colored by control type, with each poscon compound
    highlighted — shows whether positive controls separate from DMSO and each other.
    """
    pert_col = _find_col(df_wells, exact=["Metadata_pert_type"])
    cpd_col = _compound_col(df_wells)
    if pert_col is None:
        logger.info("  No pert_type column — skipping controls PCA")
        return
    try:
        pcs = _pca_2d(df_wells)
        if pcs is None:
            logger.info("  Too few features for controls PCA")
            return
        pert = df_wells[pert_col].astype(str).values
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Panel 1: by control type
        ax = axes[0]
        order = [("treatment/other", "#C7C7C7", 12, 0.35),
                 (negcon_value, "black", 45, 0.9),
                 (poscon_value, C_FAIL, 45, 0.9)]
        is_ctrl = np.isin(pert, [negcon_value, poscon_value])
        ax.scatter(pcs[~is_ctrl, 0], pcs[~is_ctrl, 1], s=12, c="#C7C7C7", alpha=0.35,
                   label="treatment", edgecolors="none")
        for val, color, s, a in [(negcon_value, "black", 40, 0.9), (poscon_value, C_FAIL, 40, 0.9)]:
            m = pert == val
            if m.any():
                ax.scatter(pcs[m, 0], pcs[m, 1], s=s, c=color, alpha=a, label=val,
                           edgecolors="white", linewidth=0.4)
        ax.set_xlabel("PC1", fontweight="bold"); ax.set_ylabel("PC2", fontweight="bold")
        ax.set_title(f"{cell_line} — Controls in PCA space", fontweight="bold")
        ax.legend(fontsize=9); sns.despine(ax=ax)

        # Panel 2: each poscon compound distinctly
        ax = axes[1]
        ax.scatter(pcs[:, 0], pcs[:, 1], s=10, c="#E5E5E5", alpha=0.5, edgecolors="none")
        if cpd_col is not None:
            posc = pert == poscon_value
            cpds = sorted(pd.Series(df_wells.loc[posc, cpd_col]).astype(str).unique())
            palette = sns.color_palette("tab10", n_colors=max(len(cpds), 1))
            for i, cpd in enumerate(cpds):
                m = posc & (df_wells[cpd_col].astype(str).values == cpd)
                ax.scatter(pcs[m, 0], pcs[m, 1], s=55, color=palette[i], label=cpd,
                           edgecolors="black", linewidth=0.5)
            # DMSO reference
            mneg = pert == negcon_value
            if mneg.any():
                ax.scatter(pcs[mneg, 0], pcs[mneg, 1], s=45, marker="x", c="black",
                           linewidths=1.1, label="DMSO")
            ax.legend(fontsize=8, title="poscon")
        ax.set_xlabel("PC1", fontweight="bold"); ax.set_ylabel("PC2", fontweight="bold")
        ax.set_title(f"{cell_line} — Positive controls by compound", fontweight="bold")
        sns.despine(ax=ax)

        _save(fig, plots_dir, f"{cell_line}_controls_pca.png")
    except Exception as e:
        logger.warning(f"Controls PCA plot failed: {e}")


def plot_controls_clustering(df_wells, plots_dir, cell_line,
                             negcon_value="negcon", poscon_value="poscon",
                             out_name=None, level_label="") -> None:
    """
    PCA and UMAP of the CONTROL rows only (poscon + negcon), each colored by
    compound and by plate. Answers: do controls cluster by compound (reproducible
    across plates) or split by plate (residual batch effect)?

    Works at any granularity (wells or, e.g., Level 2.5b sites) — pass
    ``out_name``/``level_label`` to distinguish the output file and title from
    the default well-level call.
    """
    pert_col = _find_col(df_wells, exact=["Metadata_pert_type"])
    plate_col = _find_col(df_wells, exact=["Metadata_Barcode"])
    cpd_col = _compound_col(df_wells)
    conc_col = _find_col(df_wells, exact=["Metadata_conc_uM"])
    if pert_col is None or cpd_col is None:
        logger.info("  No pert_type/compound column — skipping controls clustering")
        return
    try:
        ctrl_mask = df_wells[pert_col].isin([negcon_value, poscon_value]).to_numpy()
        if ctrl_mask.sum() < 5:
            logger.info("  Too few control wells — skipping controls clustering")
            return
        dctrl = df_wells.loc[ctrl_mask].reset_index(drop=True)

        # label = compound (DMSO for negcon); poscon rows get compound @ dose so
        # different doses of the same poscon don't collapse into one cluster color
        cpd = dctrl[cpd_col].astype(str)
        pert = dctrl[pert_col].astype(str)
        if conc_col is not None:
            conc = dctrl[conc_col]
            dose_str = conc.map(lambda v: f" ({v:g} µM)" if pd.notna(v) else "").values
            cpd_dose = cpd.values + dose_str
        else:
            cpd_dose = cpd.values
        label = np.where(pert.values == negcon_value, "DMSO", cpd_dose)
        plate = dctrl[plate_col].astype(str).values if plate_col else np.array(["?"] * len(dctrl))

        pcs = _pca_2d(dctrl)
        emb = _umap_2d(dctrl)
        if pcs is None:
            logger.info("  Too few features for controls clustering")
            return
        have_umap = emb is not None

        nrows = 2 if have_umap else 1
        fig, axes = plt.subplots(nrows, 2, figsize=(15, 6.5 * nrows), squeeze=False)

        def _by_label(ax, emb2, title, axis):
            uniq = sorted(pd.unique(label))
            palette = sns.color_palette("tab10", n_colors=max(len(uniq), 1))
            for i, u in enumerate(uniq):
                m = label == u
                mk = "X" if u == "DMSO" else "o"
                ax.scatter(emb2[m, 0], emb2[m, 1], s=70, color=palette[i % len(palette)],
                           marker=mk, label=u, edgecolors="black", linewidth=0.4, alpha=0.85)
            ax.set_xlabel(f"{axis}1", fontweight="bold"); ax.set_ylabel(f"{axis}2", fontweight="bold")
            ax.set_title(title, fontweight="bold"); ax.legend(fontsize=7, title="control")
            sns.despine(ax=ax)

        def _by_plate(ax, emb2, title, axis):
            uniq = sorted(pd.unique(plate))
            palette = sns.color_palette("Set2", n_colors=max(len(uniq), 1))
            for i, u in enumerate(uniq):
                m = plate == u
                ax.scatter(emb2[m, 0], emb2[m, 1], s=55, color=palette[i % len(palette)],
                           label=u, edgecolors="black", linewidth=0.3, alpha=0.8)
            ax.set_xlabel(f"{axis}1", fontweight="bold"); ax.set_ylabel(f"{axis}2", fontweight="bold")
            ax.set_title(title, fontweight="bold"); ax.legend(fontsize=7, title="plate")
            sns.despine(ax=ax)

        _by_label(axes[0][0], pcs, f"{cell_line} — Controls PCA: by compound", "PC")
        _by_plate(axes[0][1], pcs, f"{cell_line} — Controls PCA: by plate", "PC")
        if have_umap:
            _by_label(axes[1][0], emb, "Controls UMAP: by compound", "UMAP")
            _by_plate(axes[1][1], emb, "Controls UMAP: by plate", "UMAP")

        title_suffix = f" ({level_label})" if level_label else ""
        fig.suptitle(f"{cell_line} — Control clustering{title_suffix} (compound vs plate) — "
                     f"cluster by compound = reproducible; by plate = batch effect",
                     fontsize=13, fontweight="bold")
        _save(fig, plots_dir, out_name or f"{cell_line}_controls_clustering.png")
    except Exception as e:
        logger.warning(f"Controls clustering plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Site-level outlier plots (Level 2.5b sites + site_outliers.csv)
# ─────────────────────────────────────────────────────────────────────────────

def _worst_outlier_compounds(outliers_df, top_n, compound_col):
    """Compounds with flagged sites, ranked by flagged-site count (desc).
    Returns (top_compound_list, full_counts_series). If a compound has fewer
    than ``top_n`` flagged sites, fewer than top_n compounds are returned —
    only compounds that actually have flags are ever included."""
    counts = (
        outliers_df.loc[outliers_df["outlier_flag"], compound_col]
        .astype(str).value_counts()
    )
    return counts.head(top_n).index.tolist(), counts


def plot_site_outliers_pca(df_25b, outliers_df, plots_dir, cell_line, top_n_compounds=5) -> None:
    """
    PCA of Level 2.5b sites. Flagged outlier sites are colored by compound —
    up to the ``top_n_compounds`` with the most flagged sites (fewer if fewer
    compounds have any flags at all). If flagged sites from the same compound
    land near each other in PCA space, that's a real, reproducible per-compound
    effect (or plate artifact); if they're scattered, it's more likely noise.
    """
    compound_col = "Metadata_compound" if "Metadata_compound" in outliers_df.columns else None
    try:
        pcs = _pca_2d(df_25b)
        if pcs is None:
            logger.info("  Too few features for site-outlier PCA")
            return
        if len(outliers_df) != len(df_25b):
            logger.warning("  site_outliers row count doesn't match Level 2.5b — skipping PCA plot")
            return
        flag = outliers_df["outlier_flag"].to_numpy(dtype=bool)

        fig, ax = plt.subplots(figsize=(8.5, 7))
        ax.scatter(pcs[~flag, 0], pcs[~flag, 1], s=12, c="#D8DEE4", alpha=0.4,
                   edgecolors="none", label=f"OK (n={int((~flag).sum())})", zorder=1)

        n_flagged_compounds = 0
        if compound_col is None or not flag.any():
            if flag.any():
                ax.scatter(pcs[flag, 0], pcs[flag, 1], s=40, c=C_FAIL, alpha=0.9,
                           edgecolors="black", linewidth=0.4,
                           label=f"Flagged outlier (n={int(flag.sum())})", zorder=2)
        else:
            cpd = outliers_df[compound_col].astype(str).to_numpy()
            top_compounds, _counts = _worst_outlier_compounds(outliers_df, top_n_compounds, compound_col)
            n_flagged_compounds = int(pd.unique(cpd[flag]).size)

            other_flagged = flag & ~np.isin(cpd, top_compounds)
            if other_flagged.any():
                ax.scatter(pcs[other_flagged, 0], pcs[other_flagged, 1], s=26, c="#9AA5B1",
                           marker="x", linewidths=1.0, alpha=0.7,
                           label=f"Other flagged compounds (n={int(other_flagged.sum())})", zorder=2)

            palette = sns.color_palette("tab10", n_colors=max(len(top_compounds), 1))
            for i, c in enumerate(top_compounds):
                m = flag & (cpd == c)
                if m.any():
                    ax.scatter(pcs[m, 0], pcs[m, 1], s=55, color=palette[i], marker="o",
                               edgecolors="black", linewidth=0.5, alpha=0.95,
                               label=f"{c} (n={int(m.sum())})", zorder=3)

        note = f" — {n_flagged_compounds} compound(s) with flagged sites" if n_flagged_compounds else ""
        ax.set_xlabel("PC1", fontweight="bold"); ax.set_ylabel("PC2", fontweight="bold")
        ax.set_title(f"{cell_line} — Site-level outliers by compound (Level 2.5b, PCA){note}",
                     fontweight="bold", fontsize=11)
        ax.legend(fontsize=8, markerscale=1.0, framealpha=0.9)
        sns.despine(ax=ax)
        _save(fig, plots_dir, f"{cell_line}_site_outliers_pca.png")
    except Exception as e:
        logger.warning(f"Site-outlier PCA plot failed: {e}")


def plot_site_outliers_heatmap(df_25b, outliers_df, plots_dir, cell_line, top_n_compounds=5) -> None:
    """
    Clustered heatmap of flagged outlier sites belonging to the ``top_n_compounds``
    compounds with the most flags, row-colored by compound. If same-compound
    outlier sites cluster together (adjacent rows / matching row color blocks
    after clustering), that supports a real per-compound effect rather than
    scattered per-site noise.
    """
    compound_col = "Metadata_compound" if "Metadata_compound" in outliers_df.columns else None
    if compound_col is None:
        logger.info("  No Metadata_compound in site_outliers.csv — skipping site-outlier heatmap")
        return
    try:
        if len(outliers_df) != len(df_25b):
            logger.warning("  site_outliers row count doesn't match Level 2.5b — skipping heatmap")
            return
        flag = outliers_df["outlier_flag"].to_numpy(dtype=bool)
        if not flag.any():
            logger.info("  No flagged sites — skipping site-outlier heatmap")
            return

        cpd = outliers_df[compound_col].astype(str).to_numpy()
        top_compounds, _counts = _worst_outlier_compounds(outliers_df, top_n_compounds, compound_col)
        sel = flag & np.isin(cpd, top_compounds)
        positions = np.where(sel)[0]
        if len(positions) < 2:
            logger.info("  Too few flagged sites among top compounds — skipping heatmap")
            return

        # outliers_df is row-order-aligned (by position) with df_25b — see
        # compute_site_outliers()'s _orig_idx restore step.
        rows = df_25b.iloc[positions]
        row_compounds = cpd[positions]

        feats = _feat_cols(rows)
        mat = rows[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        keep = mat.std(axis=0) > 0
        mat = mat.loc[:, keep]
        if mat.shape[0] < 2 or mat.shape[1] < 2:
            logger.info("  Not enough flagged sites/features for heatmap — skipping")
            return

        palette = sns.color_palette("tab10", n_colors=max(len(top_compounds), 1))
        color_map = {c: palette[i] for i, c in enumerate(top_compounds)}
        row_colors = pd.Series(row_compounds, index=mat.index).map(color_map)

        vmax = float(np.nanpercentile(np.abs(mat.values), 99)) or 1.0
        g = sns.clustermap(mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                           row_colors=row_colors,
                           figsize=(12, max(4, 0.22 * mat.shape[0] + 2)),
                           xticklabels=False, yticklabels=True, dendrogram_ratio=0.12)
        g.ax_heatmap.set_ylabel("Flagged site", fontsize=9)
        g.ax_heatmap.set_xlabel(f"{mat.shape[1]} features", fontsize=9)

        from matplotlib.patches import Patch
        handles = [Patch(color=color_map[c], label=f"{c} (n={int((row_compounds == c).sum())})")
                   for c in top_compounds]
        g.ax_heatmap.legend(handles=handles, title="compound", bbox_to_anchor=(1.25, 1),
                            loc="upper left", fontsize=8, frameon=False)

        g.fig.suptitle(f"{cell_line} — Flagged outlier sites, top {len(top_compounds)} "
                       f"compound(s) by outlier count, clustered", y=1.02, fontweight="bold")
        out = Path(plots_dir); out.mkdir(parents=True, exist_ok=True)
        outpath = out / f"{cell_line}_site_outliers_heatmap.png"
        g.fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(g.fig)
        logger.info(f"  Saved: {outpath.name}")
    except Exception as e:
        logger.warning(f"Site-outlier heatmap failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Control morphology boxplots — DMSO vs positive controls (Count_nuclei, areas)
# ─────────────────────────────────────────────────────────────────────────────

def plot_control_morphology_boxplots(df_wells, plots_dir, cell_line,
                                      negcon_value="negcon", poscon_value="poscon") -> None:
    """
    Boxplots (+ jittered points) of nuclei count, nuclei area and cytoplasm area
    for DMSO (navy) vs each positive-control compound (orange), one box per
    compound. A quick check that poscons produce the expected morphological
    phenotype and that DMSO replicates are tight, before any normalization.
    """
    pert_col = _find_col(df_wells, exact=["Metadata_pert_type"])
    cpd_col = _compound_col(df_wells)
    if pert_col is None or cpd_col is None:
        logger.info("  No pert_type/compound column — skipping control morphology boxplots")
        return

    count_col = _count_col(df_wells)
    nuc_area_col = _find_col(df_wells, "nuclei", "area") or _find_col(df_wells, "nucleus", "area")
    cyto_area_col = _find_col(df_wells, "cytoplasm", "area")

    metrics = [
        ("Count_nuclei", count_col),
        ("Nuclei area", nuc_area_col),
        ("Cytoplasm area", cyto_area_col),
    ]
    metrics = [(label, col) for label, col in metrics if col is not None]
    if not metrics:
        logger.info("  No Count_nuclei/area columns found — skipping control morphology boxplots")
        return

    try:
        pert = df_wells[pert_col].astype(str)
        ctrl_mask = pert.isin([negcon_value, poscon_value]).to_numpy()
        if ctrl_mask.sum() < 3:
            logger.info("  Too few control rows — skipping control morphology boxplots")
            return
        dctrl = df_wells.loc[ctrl_mask].copy()
        dctrl["_label"] = np.where(pert.loc[ctrl_mask].values == negcon_value,
                                    "DMSO", dctrl[cpd_col].astype(str).values)

        poscon_labels = sorted(dctrl.loc[dctrl[pert_col].astype(str) == poscon_value, "_label"].unique())
        order_all = ["DMSO"] + poscon_labels
        palette = {lbl: C_ACCENT for lbl in poscon_labels}
        palette["DMSO"] = "navy"

        fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 5.5), squeeze=False)
        axes = axes[0]
        for ax, (label, col) in zip(axes, metrics):
            vals = pd.to_numeric(dctrl[col], errors="coerce")
            plot_df = pd.DataFrame({"label": dctrl["_label"].values, "value": vals.values}).dropna()
            order = [o for o in order_all if o in plot_df["label"].unique()]
            if not order:
                ax.set_visible(False)
                continue
            sns.boxplot(data=plot_df, x="label", y="value", order=order,
                        palette=palette, ax=ax, showfliers=False, linewidth=1.0)
            sns.stripplot(data=plot_df, x="label", y="value", order=order,
                         color="black", size=3, alpha=0.4, ax=ax)
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel(""); ax.set_ylabel(col, fontsize=8)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            sns.despine(ax=ax)

        fig.suptitle(f"{cell_line} — Control morphology: DMSO (navy) vs positive controls (orange)",
                     fontsize=13, fontweight="bold")
        _save(fig, plots_dir, f"{cell_line}_control_morphology_boxplots.png")
    except Exception as e:
        logger.warning(f"Control morphology boxplot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Poscon analysis — top phenotype-driving features per positive control
# ─────────────────────────────────────────────────────────────────────────────

def plot_poscon_analysis(df_l5, plots_dir, cell_line, top_n=15,
                         poscon_value="poscon") -> None:
    """
    For each positive control (consensus L5 profile, already normalized to DMSO),
    show the top features driving its phenotype (largest |value|). Reveals the
    distinct mechanism signature of each poscon.
    """
    pert_col = _find_col(df_l5, exact=["Metadata_pert_type"])
    cpd_col = _compound_col(df_l5)
    if pert_col is None or cpd_col is None:
        logger.info("  No pert_type/compound column — skipping poscon analysis")
        return
    try:
        posc = df_l5[df_l5[pert_col] == poscon_value]
        if len(posc) == 0:
            logger.info("  No poscon profiles — skipping poscon analysis")
            return
        feats = _feat_cols(df_l5)
        cpds = sorted(posc[cpd_col].astype(str).unique())
        n = len(cpds)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 5 * nrow), squeeze=False)

        for i, cpd in enumerate(cpds):
            ax = axes[i // ncol][i % ncol]
            row = posc[posc[cpd_col].astype(str) == cpd][feats].iloc[0]
            top = row.reindex(row.abs().sort_values(ascending=False).index).head(top_n)[::-1]
            colors = [C_BLUE if v < 0 else C_FAIL for v in top.values]
            labels = [f.replace("_illum", " ").replace("Metadata_", "")[:40] for f in top.index]
            ax.barh(range(len(top)), top.values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
            ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=6)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Consensus value (vs DMSO)", fontsize=9)
            ax.set_title(_short(_disp(_name_map(df_l5), cpd)), fontsize=11, fontweight="bold")
            ax.grid(axis="x", alpha=0.2)
            sns.despine(ax=ax)
        # hide any unused axes
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")

        fig.suptitle(f"{cell_line} — Positive control phenotypes: top {top_n} driving features",
                     fontsize=14, fontweight="bold")
        _save(fig, plots_dir, f"{cell_line}_poscon_analysis.png")
    except Exception as e:
        logger.warning(f"Poscon analysis plot failed: {e}")


def plot_poscon_radar_combined(df_l5, plots_dir, cell_line,
                               poscon_value="poscon", negcon_value="negcon",
                               value_range=(-5, 5)) -> None:
    """
    All positive controls on ONE radar: category-colored rim (stain × feature type
    + AreaShape compartments) with one colored polygon per poscon compound, so their
    morphological signatures can be compared directly. DMSO drawn as a grey reference.
    """
    from math import pi
    import matplotlib.patches as mpatches
    from .radarplots import MAIN_GROUPS, categorize_features

    pert_col = _find_col(df_l5, exact=["Metadata_pert_type"])
    cpd_col = _compound_col(df_l5)
    if pert_col is None or cpd_col is None or MAIN_GROUPS is None:
        logger.info("  Missing columns / radarplots — skipping combined radar")
        return
    try:
        feats = _feat_cols(df_l5)
        categorized = categorize_features(feats, MAIN_GROUPS)
        group_names = list(MAIN_GROUPS.keys())
        all_labels = [sg for mg in MAIN_GROUPS for sg in MAIN_GROUPS[mg]]
        angles = np.linspace(0, 2 * pi, len(all_labels), endpoint=False).tolist()
        angles += angles[:1]

        def _stats_for(row):
            vals = []
            for mg in MAIN_GROUPS:
                for sg in MAIN_GROUPS[mg]:
                    fl = categorized[mg][sg]
                    vals.append(float(row[fl].mean()) if fl else 0.0)
            return vals + vals[:1]

        # rim segment colors (match radarplots.py)
        custom = {"HOECHST": "#1f77b4", "MITO": "#d62728", "SYTO": "#ffff00",
                  "PHAandWGA": "#ff7f0e", "CONC": "#2ca02c"}
        areashape = {"AreaShape_Nuclei": "#4a0080", "AreaShape_Cytoplasm": "#ff69b4",
                     "AreaShape_Cells": "#da70d6"}
        seg_colors = [areashape.get(mg, custom.get(mg, "#7f7f7f")) for mg in group_names]

        max_value = max(abs(v) for v in value_range)
        outer = max_value * 1.25
        inner = max_value * 1.50
        radial_ticks = np.linspace(value_range[0], value_range[1], 5)

        fig = plt.figure(figsize=(11, 11))
        ax = plt.subplot(111, polar=True)

        # rim background per stain/compartment
        idx = 0
        awidth = angles[1] - angles[0]
        for mg, col in zip(group_names, seg_colors):
            n = len(MAIN_GROUPS[mg])
            if n == 0:
                continue
            s, e = angles[idx], angles[idx + n - 1]
            seg = np.linspace(s - awidth / 2, e + awidth / 2, 100)
            ax.fill_between(seg, inner, outer, color=col, alpha=0.5, zorder=0)
            idx += n
        for ang, lab in zip(angles[:-1], all_labels):
            ax.text(ang, outer * 1.08, lab, ha="center", va="center", fontsize=8, zorder=1)
        for ang in angles[:-1]:
            ax.plot([ang, ang], [value_range[0], outer], color="gray", lw=0.5, alpha=0.5, zorder=1)
        ax.set_rgrids(radial_ticks, labels=[f"{r:.1f}" for r in radial_ticks], angle=0, fontsize=8)
        ax.set_xticks([]); ax.xaxis.grid(False)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="gray", alpha=0.5)
        ax.spines["polar"].set_visible(False)

        # one polygon per poscon compound + DMSO reference
        posc = df_l5[df_l5[pert_col] == poscon_value]
        cpds = sorted(posc[cpd_col].astype(str).unique())
        palette = sns.color_palette("tab10", n_colors=max(len(cpds), 1))
        cpd_handles = []
        neg = df_l5[df_l5[pert_col] == negcon_value]
        if len(neg):
            s = _stats_for(neg[feats].apply(pd.to_numeric, errors="coerce").iloc[0])
            ax.plot(angles, s, color="black", lw=1.3, ls="--", alpha=0.7, zorder=2)
            cpd_handles.append(mpatches.Patch(color="black", label="DMSO (ref)"))
        for i, cpd in enumerate(cpds):
            row = posc[posc[cpd_col].astype(str) == cpd][feats].apply(pd.to_numeric, errors="coerce").iloc[0]
            s = _stats_for(row)
            ax.plot(angles, s, color=palette[i], lw=2, zorder=3, label=_disp(_name_map(df_l5), cpd))
            ax.fill(angles, s, color=palette[i], alpha=0.12, zorder=2)
            cpd_handles.append(mpatches.Patch(color=palette[i], label=_disp(_name_map(df_l5), cpd)))

        leg1 = ax.legend(handles=cpd_handles, loc="center left", bbox_to_anchor=(1.08, 0.72),
                         fontsize=9, frameon=False, title="Compound")
        ax.add_artist(leg1)
        stain_handles = [mpatches.Patch(color=c, alpha=0.5, label=g) for c, g in zip(seg_colors, group_names)]
        ax.legend(handles=stain_handles, loc="center left", bbox_to_anchor=(1.08, 0.30),
                  fontsize=8, frameon=False, title="Channel / compartment")
        fig.text(1.08, 0.05, "I=Intensity  G=Granularity  T=Texture\nR=RadialDist  S=Shape",
                 fontsize=8, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_title(f"{cell_line} — Positive controls (combined radar, consensus vs DMSO)",
                     fontsize=13, fontweight="bold", pad=24)
        _save(fig, plots_dir, f"{cell_line}_radar_poscon_combined.png")
    except Exception as e:
        logger.warning(f"Combined poscon radar failed: {e}")


def plot_poscon_clustermap(df_l5, plots_dir, cell_line, top_n=60,
                           poscon_value="poscon", negcon_value="negcon",
                           include_dmso=True) -> None:
    """
    Clustered heatmap of positive-control consensus profiles (compounds × features).
    Rows = poscon compounds (+ DMSO reference), columns = the top-N most variable
    features across those controls. Hierarchical clustering on both axes reveals
    which poscons share mechanisms and which features define them.
    """
    pert_col = _find_col(df_l5, exact=["Metadata_pert_type"])
    cpd_col = _compound_col(df_l5)
    if pert_col is None or cpd_col is None:
        logger.info("  No pert_type/compound column — skipping poscon clustermap")
        return
    try:
        sel = df_l5[pert_col] == poscon_value
        if include_dmso:
            sel = sel | (df_l5[pert_col] == negcon_value)
        d = df_l5[sel].copy()
        if len(d) < 2:
            logger.info("  <2 control profiles — skipping poscon clustermap")
            return

        feats = _feat_cols(d)
        # Row label = compound (DMSO for negcon), aggregate to one row per compound
        _nm = _name_map(d)
        labels = np.where(d[pert_col].values == negcon_value, "DMSO",
                          d[cpd_col].astype(str).map(lambda x: _disp(_nm, x)).values)
        mat = d[feats].apply(pd.to_numeric, errors="coerce")
        mat.index = labels
        mat = mat.groupby(level=0).mean()

        # Keep the most variable features across controls (readable clustermap)
        variances = mat.var(axis=0).sort_values(ascending=False)
        keep = variances.head(min(top_n, len(feats))).index.tolist()
        sub = mat[keep]
        # short, readable feature labels
        sub.columns = [c.replace("_illum", " ").replace("Metadata_", "")[:38] for c in sub.columns]

        # symmetric color scale around 0 (profiles are normalized vs DMSO)
        vmax = float(np.nanpercentile(np.abs(sub.values), 98)) or 1.0

        n_rows, n_cols = sub.shape
        g = sns.clustermap(
            sub, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
            figsize=(max(12, n_cols * 0.22), max(5, n_rows * 0.6)),
            linewidths=0.3, linecolor="white",
            dendrogram_ratio=(0.12, 0.28),
            cbar_pos=(0.02, 0.83, 0.03, 0.15),
            xticklabels=True, yticklabels=True,
        )
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=6, rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=10, rotation=0)
        g.ax_heatmap.set_xlabel(f"Top {len(keep)} most variable features", fontweight="bold")
        g.figure.suptitle(f"{cell_line} — Positive control clustermap (consensus vs DMSO)",
                          fontsize=13, fontweight="bold")
        out = Path(plots_dir) / f"{cell_line}_poscon_clustermap.png"
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        g.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(g.figure)
        logger.info(f"  Saved: {out.name}")
    except Exception as e:
        logger.warning(f"Poscon clustermap failed: {e}")


CHANNELS = {
    "HOECHST (DNA)": "illumhoechst",
    "SYTO (RNA/nucleoli)": "illumsyto",
    "MITO (mitochondria)": "illummito",
    "PHAandWGA (actin/membrane)": "illumphaandwga",
    "CONC": "illumconc",
}
FEATURE_TYPES = ["Intensity", "Granularity", "Texture", "RadialDistribution",
                 "AreaShape", "Correlation", "Neighbors"]


def _channel_of(feat_lower):
    """Return the set of imaging channels a feature mentions (AreaShape → morphology)."""
    if "areashape" in feat_lower:
        return ["AreaShape (morphology)"]
    hits = [name for name, tag in CHANNELS.items() if tag in feat_lower]
    return hits if hits else []


def plot_channel_analysis(df_l5, plots_dir, cell_line,
                          poscon_value="poscon", negcon_value="negcon") -> None:
    """
    Where are the changes? Aggregate each positive control's consensus profile
    (vs DMSO) by imaging channel and by feature type, so you can see which
    organelle/stain and which measurement class carries the phenotype.

    Produces a 2-panel heatmap: compounds × channel and compounds × feature-type,
    colored by mean |change| (magnitude vs DMSO).
    """
    pert_col = _find_col(df_l5, exact=["Metadata_pert_type"])
    cpd_col = _compound_col(df_l5)
    if pert_col is None or cpd_col is None:
        logger.info("  No pert_type/compound column — skipping channel analysis")
        return
    try:
        d = df_l5[df_l5[pert_col] == poscon_value].copy()
        if len(d) == 0:
            logger.info("  No poscon profiles — skipping channel analysis")
            return
        feats = _feat_cols(d)
        feat_lower = {f: f.lower() for f in feats}

        # Precompute channel/type membership
        chan_map = {name: [f for f in feats if name in _channel_of(feat_lower[f])]
                    for name in list(CHANNELS.keys()) + ["AreaShape (morphology)"]}
        type_map = {t: [f for f in feats if t.lower() in feat_lower[f]] for t in FEATURE_TYPES}

        cpds = sorted(d[cpd_col].astype(str).unique())
        chan_names = [c for c in chan_map if chan_map[c]]
        type_names = [t for t in type_map if type_map[t]]

        chan_mat = pd.DataFrame(index=cpds, columns=chan_names, dtype=float)
        type_mat = pd.DataFrame(index=cpds, columns=type_names, dtype=float)
        for cpd in cpds:
            row = d[d[cpd_col].astype(str) == cpd][feats].apply(pd.to_numeric, errors="coerce").iloc[0]
            for c in chan_names:
                chan_mat.loc[cpd, c] = row[chan_map[c]].abs().mean()
            for t in type_names:
                type_mat.loc[cpd, t] = row[type_map[t]].abs().mean()

        # Relabel rows with real compound names.
        _nm = _name_map(d)
        chan_mat.index = [_disp(_nm, c) for c in chan_mat.index]
        type_mat.index = [_disp(_nm, c) for c in type_mat.index]

        fig_w = 7 + 0.5 * len(chan_names) + 0.5 * len(type_names)
        fig_h = max(4.5, 1.2 + 0.7 * len(cpds))
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                 gridspec_kw={"width_ratios": [len(chan_names), len(type_names)]})

        for ax, mat, title in [
            (axes[0], chan_mat, "by imaging channel"),
            (axes[1], type_mat, "by feature type"),
        ]:
            im = ax.imshow(mat.values.astype(float), cmap="magma_r", aspect="auto")
            ax.set_xticks(range(mat.shape[1]))
            ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(mat.shape[0]))
            ax.set_yticklabels(mat.index, fontsize=9)
            ax.set_title(f"{cell_line} — mean |change| {title}", fontsize=11, fontweight="bold")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                                color="white" if v > np.nanmax(mat.values) * 0.5 else "black")
            plt.colorbar(im, ax=ax, label="mean |consensus vs DMSO|", fraction=0.046, pad=0.04)

        fig.suptitle(f"{cell_line} — Positive control changes by channel & feature type",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, plots_dir, f"{cell_line}_poscon_channel_analysis.png")
    except Exception as e:
        logger.warning(f"Channel analysis plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. UMAP (compound structure) — kept from analysis set
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap_embedding(df_l4b, plots_dir, cell_line, negcon_value="negcon", level_label="4b") -> None:
    """UMAP of Level {level_label} profiles, controls highlighted."""
    try:
        import scanpy as sc
    except ImportError:
        logger.info("  scanpy unavailable — skipping UMAP")
        return
    pert_col = _find_col(df_l4b, exact=["Metadata_pert_type"])
    try:
        feats = _feat_cols(df_l4b)
        if not feats:
            return
        adata = sc.AnnData(X=df_l4b[feats].fillna(0).values.astype(np.float32))
        sc.pp.pca(adata, n_comps=min(50, len(feats) - 1))
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
        sc.tl.umap(adata, min_dist=0.1)
        emb = adata.obsm["X_umap"]

        fig, ax = plt.subplots(figsize=(9, 7))
        _scatter_by_perttype(ax, emb, df_l4b, pert_col, negcon_value)
        ax.set_xlabel("UMAP 1", fontweight="bold"); ax.set_ylabel("UMAP 2", fontweight="bold")
        ax.set_title(f"{cell_line} — UMAP (Level {level_label})", fontweight="bold")
        sns.despine(ax=ax)
        _save(fig, plots_dir, f"{cell_line}_umap.png")
    except Exception as e:
        logger.warning(f"UMAP plot failed: {e}")


def _scatter_by_perttype(ax, emb, df, pert_col, negcon_value="negcon", poscon_value="poscon"):
    """Draw a 2-D embedding colored by perturbation type: treatment (orange),
    positive control (red ▲) and negative/DMSO control (blue ✕). Treatments are
    drawn first so the sparse, important controls sit on top."""
    if pert_col and pert_col in df.columns:
        pt = df[pert_col].astype(str).values
    else:
        pt = np.array(["trt"] * len(df))

    layers = [
        (pt == "trt",                  C_ACCENT, 16, 0.40, "o", "none",  "Treatment"),
        (pt == str(poscon_value),      C_FAIL,   42, 0.90, "^", "white", "Positive control"),
        (pt == str(negcon_value),      C_BLUE,   42, 0.90, "X", "white", "Negative control (negcon)"),
    ]
    known = np.zeros(len(df), dtype=bool)
    for mask, *_ in layers:
        known |= mask
    if (~known).any():  # any other pert_type value
        ax.scatter(emb[~known, 0], emb[~known, 1], c="#B9C4CF", s=14, alpha=0.4,
                   edgecolors="none", label="Other")
    for mask, color, s, a, marker, edge, label in layers:
        if mask.any():
            ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=s, alpha=a, marker=marker,
                       edgecolors=edge, linewidths=0.4, label=f"{label} (n={int(mask.sum())})")
    ax.legend(fontsize=8, markerscale=1.1, framealpha=0.9)


def _embedding_scatter(emb, df, pert_col, negcon_value, xlab, ylab, title,
                       plots_dir, cell_line, fname):
    """Shared 2-D embedding scatter colored by perturbation type."""
    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter_by_perttype(ax, emb, df, pert_col, negcon_value)
    ax.set_xlabel(xlab, fontweight="bold"); ax.set_ylabel(ylab, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    sns.despine(ax=ax)
    _save(fig, plots_dir, fname)


def plot_pca_embedding(df_l4b, plots_dir, cell_line, negcon_value="negcon", level_label="4b") -> None:
    """PCA of Level {level_label} profiles, controls highlighted; axes show % variance explained."""
    from sklearn.decomposition import PCA
    pert_col = _find_col(df_l4b, exact=["Metadata_pert_type"])
    try:
        feats = _feat_cols(df_l4b)
        X = df_l4b[feats].fillna(0).to_numpy(dtype=float)
        X = X[:, np.isfinite(X).all(axis=0)]
        if X.shape[1] < 2:
            logger.info("  Too few features for PCA overview — skipping")
            return
        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_ * 100
        _embedding_scatter(emb, df_l4b, pert_col, negcon_value,
                           f"PC1 ({ev[0]:.1f}% var)", f"PC2 ({ev[1]:.1f}% var)",
                           f"{cell_line} — PCA (Level {level_label})", plots_dir, cell_line,
                           f"{cell_line}_pca.png")
    except Exception as e:
        logger.warning(f"PCA overview plot failed: {e}")


def plot_tsne_embedding(df_l4b, plots_dir, cell_line, negcon_value="negcon",
                        perplexity=30, level_label="4b") -> None:
    """t-SNE of Level {level_label} profiles, controls highlighted. PCA-reduced first
    (standard practice) for speed and denoising; perplexity auto-capped for n."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    pert_col = _find_col(df_l4b, exact=["Metadata_pert_type"])
    try:
        feats = _feat_cols(df_l4b)
        X = df_l4b[feats].fillna(0).to_numpy(dtype=float)
        X = X[:, np.isfinite(X).all(axis=0)]
        if X.shape[1] < 2 or X.shape[0] < 5:
            logger.info("  Too few features/samples for t-SNE — skipping")
            return
        n_pca = min(50, X.shape[1] - 1, X.shape[0] - 1)
        Xr = PCA(n_components=n_pca, random_state=42).fit_transform(X)
        # perplexity must stay well below the sample count
        perp = int(min(perplexity, max(5, (X.shape[0] - 1) // 3)))
        emb = TSNE(n_components=2, perplexity=perp, init="pca",
                   random_state=42).fit_transform(Xr)
        _embedding_scatter(emb, df_l4b, pert_col, negcon_value,
                           "t-SNE 1", "t-SNE 2",
                           f"{cell_line} — t-SNE (Level {level_label}, perplexity={perp})",
                           plots_dir, cell_line, f"{cell_line}_tsne.png")
    except Exception as e:
        logger.warning(f"t-SNE overview plot failed: {e}")


def plot_feature_heatmap(df_l4b, plots_dir, cell_line, negcon_value="negcon", level_label="4b") -> None:
    """
    Hierarchically-clustered heatmap of per-compound median morphological profiles
    (Level {level_label}, MAD units). Rows = compound (DMSO included as the reference),
    columns = features (clustered, unlabeled). Blocks of compounds with similar
    signatures cluster together, and the feature columns driving them line up.
    """
    pert_col = _find_col(df_l4b, exact=["Metadata_pert_type"])
    cpd_col = _compound_col(df_l4b)
    if cpd_col is None:
        logger.info("  No compound column — skipping feature heatmap")
        return
    try:
        feats = _feat_cols(df_l4b)
        prof = df_l4b.groupby(cpd_col)[feats].median()
        # drop features that are constant/NaN across compounds (uninformative)
        keep = prof.std(axis=0).fillna(0) > 0
        prof = prof.loc[:, keep].fillna(0)
        if prof.shape[0] < 2 or prof.shape[1] < 2:
            logger.info("  Not enough compounds/features for heatmap — skipping")
            return
        # symmetric color scale, clipped at the 99th percentile of |value|
        vmax = float(np.nanpercentile(np.abs(prof.values), 99)) or 1.0
        row_colors = None
        if pert_col:
            ptype = df_l4b.groupby(cpd_col)[pert_col].first()
            cmap_ptype = {"negcon": "black", "poscon": C_FAIL, "trt": "#B9C4CF"}
            row_colors = ptype.reindex(prof.index).map(
                lambda v: cmap_ptype.get(str(v), "#B9C4CF"))

        g = sns.clustermap(
            prof, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax,
            row_colors=row_colors, xticklabels=False, yticklabels=True,
            figsize=(12, max(6, 0.22 * prof.shape[0])), linewidths=0)
        g.ax_heatmap.set_xlabel(f"{prof.shape[1]} features (clustered)", fontweight="bold")
        g.ax_heatmap.set_ylabel("compound (median profile)", fontweight="bold")
        g.ax_heatmap.tick_params(axis="y", labelsize=7)
        g.fig.suptitle(f"{cell_line} — compound morphological profiles "
                       f"(Level {level_label}, MAD units; {prof.shape[0]} compounds)",
                       fontsize=12, fontweight="bold")
        plots_dir = Path(plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"{cell_line}_feature_heatmap.png"
        g.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(g.fig)
        logger.info(f"  Saved: {out.name}")
    except Exception as e:
        logger.warning(f"Feature heatmap failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Metric plots: phenotypic activity (mAP), GRIT, viability
# ─────────────────────────────────────────────────────────────────────────────

def plot_map_distribution(map_df, plots_dir, cell_line, fdr=0.05):
    """
    Phenotypic activity: distribution of mean average precision (mAP) across
    compound×dose, plus mAP vs significance. Reads metrics_map.parquet.
    """
    if map_df is None or getattr(map_df, "empty", True) or "mean_average_precision" not in map_df.columns:
        logger.info("  mAP plot skipped — no mean_average_precision column")
        return
    m = map_df["mean_average_precision"].dropna()
    p_col = "-log10(p-value)" if "-log10(p-value)" in map_df.columns else None

    fig, axes = plt.subplots(1, 2 if p_col else 1, figsize=(11 if p_col else 6, 4.2), squeeze=False)
    ax = axes[0][0]
    ax.hist(m, bins=30, color=C_MAIN, alpha=0.85, edgecolor="white")
    ax.axvline(float(m.median()), color=C_FAIL, ls="--", lw=1.2, label=f"median={m.median():.2f}")
    ax.set_xlabel("mean average precision (mAP)"); ax.set_ylabel("compound×dose count")
    ax.set_title(f"{cell_line} — phenotypic activity (mAP)"); ax.legend(fontsize=9); sns.despine(ax=ax)

    if p_col:
        ax2 = axes[0][1]
        sig = None
        if "below_corrected_p" in map_df.columns:
            sig = map_df["below_corrected_p"].astype(bool)
        colors = np.where(sig, C_OK, C_WARN) if sig is not None else C_BLUE
        ax2.scatter(map_df["mean_average_precision"], map_df[p_col], s=14, c=colors, alpha=0.8, edgecolors="none")
        ax2.axhline(-np.log10(fdr), color=C_FAIL, ls=":", lw=1, label=f"FDR={fdr}")
        ax2.set_xlabel("mAP"); ax2.set_ylabel("-log10(p-value)")
        ax2.set_title(f"{cell_line} — mAP significance"); ax2.legend(fontsize=9); sns.despine(ax=ax2)
    _save(fig, plots_dir, f"{cell_line}_map_distribution.png")


def plot_grit_distribution(grit_df, plots_dir, cell_line, grit_threshold=1.0):
    """Distribution of GRIT across compound×dose. Reads metrics_grit.parquet."""
    if grit_df is None or getattr(grit_df, "empty", True) or "grit" not in grit_df.columns:
        logger.info("  GRIT plot skipped — no grit column")
        return
    g = pd.to_numeric(grit_df["grit"], errors="coerce").dropna()
    if g.empty:
        logger.info("  GRIT plot skipped — no finite grit values")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(g, bins=30, color=C_ACCENT, alpha=0.85, edgecolor="white")
    ax.axvline(grit_threshold, color=C_FAIL, ls="--", lw=1.2, label=f"threshold={grit_threshold}")
    ax.axvline(float(g.median()), color=C_BLUE, ls=":", lw=1.2, label=f"median={g.median():.2f}")
    n_pass = int((g >= grit_threshold).sum())
    pct = 100 * n_pass / len(g) if len(g) else float("nan")
    ax.set_xlabel("GRIT"); ax.set_ylabel("compound×dose count")
    ax.set_title(f"{cell_line} — GRIT ({n_pass}/{len(g)} compound×dose ≥ {grit_threshold}, {pct:.1f}%)")

    # Compound-level stats too — a compound with multiple doses can pass at one
    # dose and not another, so this is a different (usually higher) percentage
    # than the compound×dose count above.
    if "Metadata_compound" in grit_df.columns:
        gg = grit_df.copy()
        gg["grit"] = pd.to_numeric(gg["grit"], errors="coerce")
        n_cpd_total = gg["Metadata_compound"].nunique()
        n_cpd_pass = gg.loc[gg["grit"] >= grit_threshold, "Metadata_compound"].nunique()
        cpd_pct = 100 * n_cpd_pass / n_cpd_total if n_cpd_total else float("nan")
        fig.text(0.5, 0.965,
                  f"{n_cpd_pass}/{n_cpd_total} treatment compounds GRIT ≥ {grit_threshold:g} ({cpd_pct:.1f}%)",
                  ha="center", va="bottom", fontsize=9.5, color="#333333")

    ax.legend(fontsize=9); sns.despine(ax=ax)
    _save(fig, plots_dir, f"{cell_line}_grit_distribution.png")


def plot_viability(df_wells, plots_dir, cell_line, threshold_pct=80.0,
                   plate_col="Metadata_Barcode", negcon_value="negcon"):
    """
    Viability distribution (% of plate DMSO median cell count) across wells,
    with the cytotoxicity threshold. Reads a well-level table (Level 4c).

    Right panel overlays the DMSO (negcon) wells per plate so the control
    variability — the spread of the reference the viability % is measured
    against — is visible alongside the full well population.
    """
    if df_wells is None or "Metadata_Viability_pct" not in df_wells.columns:
        logger.info("  Viability plot skipped — no Metadata_Viability_pct column")
        return
    v = pd.to_numeric(df_wells["Metadata_Viability_pct"], errors="coerce").dropna()
    if v.empty:
        logger.info("  Viability plot skipped — no finite viability values")
        return
    has_plate = plate_col in df_wells.columns
    fig, axes = plt.subplots(1, 2 if has_plate else 1, figsize=(11 if has_plate else 6, 4.2), squeeze=False)
    ax = axes[0][0]
    ax.hist(v, bins=30, color=C_OK, alpha=0.85, edgecolor="white")
    ax.axvline(threshold_pct, color=C_FAIL, ls="--", lw=1.2, label=f"cytotox < {threshold_pct:.0f}%")
    n_below = int((v < threshold_pct).sum())
    ax.set_xlabel("viability (% of plate DMSO median)"); ax.set_ylabel("well count")
    ax.set_title(f"{cell_line} — viability ({n_below}/{len(v)} below threshold)")
    ax.legend(fontsize=9); sns.despine(ax=ax)
    if has_plate:
        ax2 = axes[0][1]
        order = sorted(df_wells[plate_col].dropna().astype(str).unique())
        plate_str = df_wells[plate_col].astype(str)
        data = [pd.to_numeric(df_wells.loc[plate_str == p, "Metadata_Viability_pct"],
                              errors="coerce").dropna().values for p in order]
        ax2.boxplot(data, labels=order, showfliers=False)
        ax2.axhline(threshold_pct, color=C_FAIL, ls="--", lw=1.2, label=f"cytotox < {threshold_pct:.0f}%")

        # Overlay DMSO (negcon) wells per plate to show control variability.
        pert_col = _find_col(df_wells, exact=["Metadata_pert_type"])
        if pert_col is not None:
            rng = np.random.RandomState(42)
            labeled = False
            for i, p in enumerate(order):
                dv = pd.to_numeric(
                    df_wells.loc[(plate_str == p) & (df_wells[pert_col].astype(str) == negcon_value),
                                 "Metadata_Viability_pct"],
                    errors="coerce").dropna().values
                if len(dv) == 0:
                    continue
                x = (i + 1) + rng.uniform(-0.16, 0.16, size=len(dv))  # jitter around box
                ax2.scatter(x, dv, s=16, color="black", alpha=0.55, edgecolors="none",
                            zorder=3, label=("DMSO wells" if not labeled else None))
                # DMSO robust spread as a vertical tick: median ± MAD
                med = float(np.median(dv))
                mad = float(np.median(np.abs(dv - med)))
                ax2.errorbar(i + 1, med, yerr=1.4826 * mad, fmt="_", color=C_BLUE,
                             elinewidth=1.6, capsize=4, capthick=1.6, zorder=4,
                             label=("DMSO median ± MAD" if not labeled else None))
                labeled = True
        ax2.set_xlabel("plate"); ax2.set_ylabel("viability (%)")
        ax2.set_title(f"{cell_line} — viability per plate (+ DMSO controls)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend(fontsize=7, loc="lower left")
        sns.despine(ax=ax2)
    _save(fig, plots_dir, f"{cell_line}_viability.png")


def plot_viability_site_level(df_sites, plots_dir, cell_line, threshold_pct=50.0,
                              dmso_z=3.0, plate_col="Metadata_Barcode",
                              negcon_value="negcon", poscon_value="poscon"):
    """
    Site-level viability diagnostics (reads features/viability_sites.parquet):

      A) Per-SITE viability distribution split by perturbation type — the raw
         count ratio before any roll-up, with the cytotoxicity threshold.
      B) Site→well roll-up: well median viability vs number of retained sites,
         showing that low-site-count wells give noisier (not biased) estimates;
         low-confidence wells (< min sites) highlighted.
      C) DMSO reference QC: per-plate DMSO site counts with the low-side robust-Z
         cutoff that defines a dropped reference outlier.
    """
    if df_sites is None or "Metadata_site_viability_pct" not in df_sites.columns:
        logger.info("  Site viability plot skipped — no viability_sites table")
        return
    pert_col = _find_col(df_sites, exact=["Metadata_pert_type"])
    count_col = _find_col(df_sites, exact=["Metadata_Count_nuclei"])
    v = pd.to_numeric(df_sites["Metadata_site_viability_pct"], errors="coerce")
    pt = df_sites[pert_col].astype(str) if pert_col else pd.Series(["trt"] * len(df_sites))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))

    # ── A) per-site viability by pert_type ───────────────────────────────────
    axA = axes[0]
    bins = np.linspace(0, min(220, np.nanpercentile(v, 99.5) if v.notna().any() else 200), 45)
    groups = [("trt", C_ACCENT, "Treatment"), (poscon_value, C_FAIL, "Positive control"),
              (negcon_value, C_BLUE, "DMSO (negcon)")]
    for val, color, label in groups:
        gv = v[pt == str(val)].dropna()
        if gv.empty:
            continue
        axA.hist(np.clip(gv, 0, bins[-1]), bins=bins, color=color, alpha=0.55,
                 density=True, label=f"{label} (n={len(gv)})", edgecolor="none")
    axA.axvline(threshold_pct, color=C_FAIL, ls="--", lw=1.3,
                label=f"cytotox < {threshold_pct:.0f}%")
    n_below = int((v[pt != str(negcon_value)] < threshold_pct).sum())
    axA.set_xlabel("site viability (% of plate DMSO median)")
    axA.set_ylabel("density")
    axA.set_title(f"{cell_line} — per-site viability ({n_below} sites below threshold)")
    axA.legend(fontsize=7.5); sns.despine(ax=axA)

    # ── B) well median vs n_sites (confidence) ───────────────────────────────
    axB = axes[1]
    key = [c for c in [plate_col, "Metadata_Well"] if c in df_sites.columns]
    if key:
        w = (df_sites.groupby(key, observed=True)
             .agg(vmed=("Metadata_site_viability_pct", "median"),
                  nsite=("Metadata_site_viability_pct", "count")).reset_index())
        rng = np.random.RandomState(42)
        jx = w["nsite"] + rng.uniform(-0.18, 0.18, size=len(w))
        low_conf = w["nsite"] < 2
        axB.scatter(jx[~low_conf], w["vmed"][~low_conf], s=10, alpha=0.35,
                    color=C_BLUE, edgecolors="none", label="well")
        if low_conf.any():
            axB.scatter(jx[low_conf], w["vmed"][low_conf], s=28, alpha=0.9,
                        color=C_FAIL, marker="D", edgecolors="white", linewidths=0.4,
                        label=f"low-confidence (<2 sites, n={int(low_conf.sum())})")
        axB.axhline(threshold_pct, color=C_FAIL, ls="--", lw=1.1)
        axB.set_xlabel("retained sites per well")
        axB.set_ylabel("well viability (median of site %)")
        axB.set_title(f"{cell_line} — roll-up confidence")
        axB.legend(fontsize=7.5); sns.despine(ax=axB)

    # ── C) DMSO reference per-plate counts + robust-Z cutoff ─────────────────
    axC = axes[2]
    if count_col and pert_col:
        d = df_sites[pt == str(negcon_value)].copy()
        d[count_col] = pd.to_numeric(d[count_col], errors="coerce")
        order = sorted(d[plate_col].dropna().astype(str).unique()) if plate_col in d.columns else []
        drawn_cut = False
        for i, p in enumerate(order):
            cc = d.loc[d[plate_col].astype(str) == p, count_col].dropna()
            if cc.empty:
                continue
            med = float(cc.median()); mad = float((cc - med).abs().median())
            x = (i + 1) + np.random.RandomState(1).uniform(-0.15, 0.15, size=len(cc))
            axC.scatter(x, cc, s=12, color="black", alpha=0.45, edgecolors="none", zorder=3)
            if mad > 0:
                cut = med - dmso_z * 1.4826 * mad
                axC.plot([i + 0.7, i + 1.3], [cut, cut], color=C_FAIL, lw=1.6, zorder=4,
                         label=(f"drop if count < median − {dmso_z:.0f}·MAD" if not drawn_cut else None))
                drawn_cut = True
        axC.set_xticks(range(1, len(order) + 1))
        axC.set_xticklabels(order, rotation=45, ha="right", fontsize=6)
        axC.set_xlabel("plate"); axC.set_ylabel("DMSO site cell count")
        axC.set_title(f"{cell_line} — DMSO reference (low-side outliers dropped)")
        if drawn_cut:
            axC.legend(fontsize=7.5)
        sns.despine(ax=axC)

    fig.tight_layout()
    _save(fig, plots_dir, f"{cell_line}_viability_sites.png")


# ─────────────────────────────────────────────────────────────────────────────
# Top-compound rankings: which compounds are most active / most toxic
# ─────────────────────────────────────────────────────────────────────────────

def _short(name, n=32):
    """Truncate a long compound name for axis labels."""
    s = str(name)
    return s if len(s) <= n else s[: n - 1] + "…"


def _name_map(df):
    """Map Metadata_compound (id) -> real display name (Metadata_compound_name),
    when the dataframe carries it. Used so plots show real compound names, not ids."""
    try:
        if "Metadata_compound_name" in df.columns and "Metadata_compound" in df.columns:
            disp = df["Metadata_compound_name"].astype("string").fillna(
                df["Metadata_compound"].astype("string"))
            return dict(zip(df["Metadata_compound"].astype(str), disp.astype(str)))
    except Exception:
        pass
    return {}


def _disp(name_map, cpd):
    """Resolve a compound id to its display name via name_map (fallback: the id)."""
    return name_map.get(str(cpd), str(cpd))


def _barh(ax, labels, values, colors, value_fmt="{:.2f}"):
    """Horizontal bar chart with value annotations, highest bar at the top."""
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    span = (max(values) - min(0, min(values))) or 1.0
    for yi, v in zip(y, values):
        ax.text(v + 0.01 * span, yi, value_fmt.format(v), va="center",
                fontsize=7, color="#333")
    sns.despine(ax=ax)


def plot_top_map_compounds(map_df, plots_dir, cell_line, top_n=20, fdr=0.05):
    """
    Compounds with the highest phenotypic activity (mean average precision).

    mAP is computed per compound×dose; each compound is represented by its most
    active dose. Bars are colored by whether that dose passes the FDR-corrected
    significance threshold. Reads metrics_map.parquet.
    """
    if (map_df is None or getattr(map_df, "empty", True)
            or "mean_average_precision" not in map_df.columns
            or "Metadata_compound" not in map_df.columns):
        logger.info("  Top-mAP compounds plot skipped — missing mAP/compound columns")
        return
    d = map_df.copy()
    d["mean_average_precision"] = pd.to_numeric(d["mean_average_precision"], errors="coerce")
    d = d.dropna(subset=["mean_average_precision"])
    nm = _name_map(d)
    # drop DMSO / negcon if present
    if "Metadata_compound" in d.columns:
        d = d[~d["Metadata_compound"].astype(str).str.upper().isin(["DMSO", "NEGCON"])]
    if d.empty:
        logger.info("  Top-mAP compounds plot skipped — no finite mAP values")
        return
    try:
        # each compound's strongest dose
        best = d.loc[d.groupby("Metadata_compound")["mean_average_precision"].idxmax()]
        best = best.sort_values("mean_average_precision", ascending=False).head(top_n).iloc[::-1]

        sig = (best["below_corrected_p"].astype(bool).values
               if "below_corrected_p" in best.columns else None)
        colors = np.where(sig, C_OK, "#B9C4CF") if sig is not None else C_MAIN

        labels = []
        for _, r in best.iterrows():
            dose = f" ({r['Metadata_conc_uM']:g} µM)" if "Metadata_conc_uM" in best.columns and pd.notna(r["Metadata_conc_uM"]) else ""
            labels.append(_short(_disp(nm, r["Metadata_compound"])) + dose)

        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.36 * len(best))))
        _barh(ax, labels, best["mean_average_precision"].values, colors)
        ax.set_xlabel("mean average precision (mAP, strongest dose)", fontweight="bold")
        n_sig = int(sig.sum()) if sig is not None else 0
        ax.set_title(f"{cell_line} — top {len(best)} compounds by phenotypic activity (mAP)\n"
                     f"green = significant at FDR {fdr:g} ({n_sig} of {len(best)} shown)",
                     fontsize=11, fontweight="bold")
        _save(fig, plots_dir, f"{cell_line}_top_map_compounds.png")
    except Exception as e:
        logger.warning(f"Top-mAP compounds plot failed: {e}")


def plot_top_grit_compounds(grit_df, plots_dir, cell_line, top_n=20, grit_threshold=1.0):
    """
    Compounds with the highest GRIT — the strongest, most reproducible phenotypes
    relative to DMSO. GRIT is computed per well; we aggregate to the mean per
    compound (across doses & replicates). Reads metrics_grit.parquet.
    """
    if (grit_df is None or getattr(grit_df, "empty", True)
            or "grit" not in grit_df.columns
            or "Metadata_compound" not in grit_df.columns):
        logger.info("  Top-GRIT compounds plot skipped — missing grit/compound columns")
        return
    d = grit_df.copy()
    d["grit"] = pd.to_numeric(d["grit"], errors="coerce")
    d = d.dropna(subset=["grit"])
    d = d[~d["Metadata_compound"].astype(str).str.upper().isin(["DMSO", "NEGCON"])]
    nm = _name_map(d)
    if d.empty:
        logger.info("  Top-GRIT compounds plot skipped — no finite grit values")
        return
    try:
        agg = d.groupby("Metadata_compound")["grit"].mean().sort_values(ascending=False)
        agg = agg.head(top_n).iloc[::-1]
        colors = np.where(agg.values >= grit_threshold, C_ACCENT, "#B9C4CF")
        labels = [_short(_disp(nm, c)) for c in agg.index]

        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.36 * len(agg))))
        _barh(ax, labels, agg.values, colors)
        ax.axvline(grit_threshold, color=C_FAIL, ls="--", lw=1.1, label=f"threshold={grit_threshold:g}")
        ax.set_xlabel("mean GRIT (across doses & replicates)", fontweight="bold")
        n_pass = int((agg.values >= grit_threshold).sum())
        ax.set_title(f"{cell_line} — top {len(agg)} compounds by GRIT\n"
                     f"orange = passes threshold ({n_pass} of {len(agg)} shown)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        _save(fig, plots_dir, f"{cell_line}_top_grit_compounds.png")
    except Exception as e:
        logger.warning(f"Top-GRIT compounds plot failed: {e}")


def plot_top_toxic_compounds(df_wells, plots_dir, cell_line, top_n=20, threshold_pct=80.0):
    """
    Most cytotoxic compounds — the lowest viability (cell count as % of the plate
    DMSO median). Viability is per well; we take, for each compound, its most
    toxic dose (lowest dose-mean viability). Excludes DMSO/negcon controls.
    Reads a well-level table with Metadata_Viability_pct (Level 4b/4c).
    """
    if (df_wells is None or "Metadata_Viability_pct" not in df_wells.columns
            or "Metadata_compound" not in df_wells.columns):
        logger.info("  Top-toxic compounds plot skipped — missing viability/compound columns")
        return
    d = df_wells.copy()
    d["Metadata_Viability_pct"] = pd.to_numeric(d["Metadata_Viability_pct"], errors="coerce")
    d = d.dropna(subset=["Metadata_Viability_pct"])
    if "Metadata_pert_type" in d.columns:
        d = d[d["Metadata_pert_type"].astype(str) != "negcon"]
    d = d[~d["Metadata_compound"].astype(str).str.upper().isin(["DMSO", "NEGCON"])]
    nm = _name_map(d)
    if d.empty:
        logger.info("  Top-toxic compounds plot skipped — no finite viability values")
        return
    try:
        # mean viability per compound×dose, then each compound's most toxic dose
        has_dose = "Metadata_conc_uM" in d.columns
        group_cols = ["Metadata_compound"] + (["Metadata_conc_uM"] if has_dose else [])
        dose_mean = d.groupby(group_cols)["Metadata_Viability_pct"].mean().reset_index()
        idx = dose_mean.groupby("Metadata_compound")["Metadata_Viability_pct"].idxmin()
        worst = dose_mean.loc[idx].sort_values("Metadata_Viability_pct").head(top_n)
        worst = worst.iloc[::-1]  # most toxic at top after barh

        colors = np.where(worst["Metadata_Viability_pct"].values < threshold_pct, C_FAIL, "#B9C4CF")
        labels = []
        for _, r in worst.iterrows():
            dose = f" ({r['Metadata_conc_uM']:g} µM)" if has_dose and pd.notna(r["Metadata_conc_uM"]) else ""
            labels.append(_short(_disp(nm, r["Metadata_compound"])) + dose)

        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.36 * len(worst))))
        _barh(ax, labels, worst["Metadata_Viability_pct"].values, colors, value_fmt="{:.0f}%")
        ax.axvline(threshold_pct, color=C_FAIL, ls="--", lw=1.1, label=f"cytotox < {threshold_pct:.0f}%")
        ax.set_xlabel("viability at most-toxic dose (% of plate DMSO median)", fontweight="bold")
        n_tox = int((worst["Metadata_Viability_pct"].values < threshold_pct).sum())
        ax.set_title(f"{cell_line} — top {len(worst)} most cytotoxic compounds\n"
                     f"red = below viability threshold ({n_tox} of {len(worst)} shown)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        _save(fig, plots_dir, f"{cell_line}_top_toxic_compounds.png")
    except Exception as e:
        logger.warning(f"Top-toxic compounds plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# L4b metric-driven embeddings: GRIT-filtered and phenotypically-active maps
#
# Both operate on Level 4b (well-level, feature-selected, pre batch correction) —
# the same level GRIT and mAP are computed on — so the embedding and the metric
# annotation describe exactly the same data. Each figure has two panels:
#   (A) HIGHLIGHTED — all wells embedded, the selected subset drawn on top
#   (B) FILTERED    — only the selected subset embedded coordinates, colored by
#                     perturbation type
# ─────────────────────────────────────────────────────────────────────────────

def _l4b_embedding(df):
    """2-D embedding of the L4b feature block: UMAP if scanpy is available,
    otherwise PCA. Returns (emb, label) or (None, None) if it cannot be built."""
    emb = _umap_2d(df)
    if emb is not None:
        return emb, "UMAP"
    emb = _pca_2d(df)
    if emb is not None:
        return emb, "PCA"
    return None, None


def _map_metric_to_wells(df_l4b, metric_df, value_col):
    """Left-join a per-compound×dose metric column onto the L4b wells (order and
    length preserved). Returns the aligned metric Series, or None if the join keys
    are unavailable."""
    key = [c for c in ["Metadata_compound", "Metadata_conc_uM"]
           if c in df_l4b.columns and c in metric_df.columns]
    if not key or value_col not in metric_df.columns:
        return None
    take = metric_df[key + [value_col]].drop_duplicates(subset=key)
    merged = df_l4b[key].merge(take, on=key, how="left")
    return merged[value_col]


def _highlight_and_filtered_figure(df_l4b, emb, emb_label, mask, subset_label,
                                   highlight_color, plots_dir, cell_line, fname,
                                   title_prefix, negcon_value="negcon",
                                   compound_stats_text=None):
    """Shared two-panel figure: (A) subset highlighted over all wells,
    (B) subset only, colored by perturbation type.

    ``compound_stats_text``, if given, is printed as a figure-level caption
    below the titles — e.g. "1234/1500 treatment compounds active (82.3%)" —
    since the highlight counts (n wells) above are per-well, not per-compound.
    """
    mask = np.asarray(pd.Series(mask).fillna(False), dtype=bool)
    pert_col = _find_col(df_l4b, exact=["Metadata_pert_type"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel A — highlight the subset on top of a grey "all wells" backdrop.
    ax = axes[0]
    ax.scatter(emb[~mask, 0], emb[~mask, 1], s=12, c="#D0D5DB", alpha=0.45,
               edgecolors="none", label=f"other (n={int((~mask).sum())})")
    ax.scatter(emb[mask, 0], emb[mask, 1], s=28, c=highlight_color, alpha=0.85,
               edgecolors="white", linewidths=0.3, label=f"{subset_label} (n={int(mask.sum())})")
    ax.set_title(f"{title_prefix} — highlighted", fontweight="bold")
    ax.set_xlabel(f"{emb_label} 1", fontweight="bold")
    ax.set_ylabel(f"{emb_label} 2", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    sns.despine(ax=ax)

    # Panel B — only the selected subset, colored by perturbation type.
    ax2 = axes[1]
    if mask.any():
        idx = np.where(mask)[0]
        sub_df = df_l4b.iloc[idx].reset_index(drop=True)
        _scatter_by_perttype(ax2, emb[idx], sub_df, pert_col, negcon_value)
        ax2.set_title(f"{title_prefix} — subset only", fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "no wells in subset", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11, color=C_WARN)
        ax2.set_title(f"{title_prefix} — subset only", fontweight="bold")
    ax2.set_xlabel(f"{emb_label} 1", fontweight="bold")
    ax2.set_ylabel(f"{emb_label} 2", fontweight="bold")
    sns.despine(ax=ax2)

    if compound_stats_text:
        fig.text(0.5, 0.965, compound_stats_text, ha="center", va="bottom",
                  fontsize=10, color="#333333")

    _save(fig, plots_dir, fname)


def plot_l4b_grit_filtered(df_l4b, grit_df, plots_dir, cell_line,
                           grit_threshold=1.0, negcon_value="negcon"):
    """
    L4b GRIT-filtered map: wells whose compound×dose GRIT exceeds the threshold
    (default > 1) highlighted on the L4b embedding, then shown in isolation.

    High GRIT marks a strongly reproducible, control-distinct phenotype, so this
    view answers "where in morphology space do the high-GRIT treatments sit, and
    do they separate from the negcon cloud?".
    """
    if grit_df is None or getattr(grit_df, "empty", True) or "grit" not in grit_df.columns:
        logger.info("  L4b GRIT-filtered map skipped — no GRIT results")
        return
    if not _feat_cols(df_l4b):
        logger.info("  L4b GRIT-filtered map skipped — no features")
        return
    try:
        grit_vals = _map_metric_to_wells(df_l4b, grit_df, "grit")
        if grit_vals is None:
            logger.info("  L4b GRIT-filtered map skipped — cannot join GRIT to wells")
            return
        mask = pd.to_numeric(grit_vals, errors="coerce") > grit_threshold
        n_hi = int(mask.fillna(False).sum())
        emb, emb_label = _l4b_embedding(df_l4b)
        if emb is None:
            logger.info("  L4b GRIT-filtered map skipped — embedding unavailable")
            return

        # Compound-level stats (GRIT is computed per compound×dose — this is a
        # different denominator than the per-well "n_hi" count in the title).
        stats_text = None
        if "Metadata_compound" in grit_df.columns:
            g = grit_df.copy()
            g["grit"] = pd.to_numeric(g["grit"], errors="coerce")
            n_compounds_total = g["Metadata_compound"].nunique()
            n_compounds_pass = g.loc[g["grit"] > grit_threshold, "Metadata_compound"].nunique()
            pct = 100 * n_compounds_pass / n_compounds_total if n_compounds_total else float("nan")
            stats_text = (
                f"{n_compounds_pass}/{n_compounds_total} treatment compounds "
                f"GRIT > {grit_threshold:g} ({pct:.1f}%)"
            )

        _highlight_and_filtered_figure(
            df_l4b, emb, emb_label, mask,
            subset_label=f"GRIT > {grit_threshold:g}", highlight_color=C_ACCENT,
            plots_dir=plots_dir, cell_line=cell_line,
            fname=f"{cell_line}_l4b_grit_filtered.png",
            title_prefix=f"{cell_line} — L4b GRIT > {grit_threshold:g} ({n_hi} wells)",
            negcon_value=negcon_value,
            compound_stats_text=stats_text,
        )
    except Exception as e:
        logger.warning(f"L4b GRIT-filtered map failed: {e}")


def plot_l4b_map_active(df_l4b, map_df, plots_dir, cell_line,
                        fdr=0.05, negcon_value="negcon"):
    """
    L4b phenotypically-active map: wells whose compound×dose is phenotypically
    active by mAP (significant after FDR correction) highlighted on the L4b
    embedding, then shown in isolation.

    Active = below the corrected-p threshold (falls back to raw p_value, then to
    the `below_corrected_p` boolean if present). This shows which treatments carry
    a real, replicate-consistent phenotype and how they lie relative to controls.
    """
    if map_df is None or getattr(map_df, "empty", True):
        logger.info("  L4b phenotypically-active map skipped — no mAP results")
        return
    if not _feat_cols(df_l4b):
        logger.info("  L4b phenotypically-active map skipped — no features")
        return
    try:
        # Build a per-compound×dose boolean "active" column from whatever the mAP
        # table provides, most-trustworthy first.
        md = map_df.copy()
        if "below_corrected_p" in md.columns:
            md["_active"] = md["below_corrected_p"].astype(bool)
        elif "corrected_p_value" in md.columns:
            md["_active"] = pd.to_numeric(md["corrected_p_value"], errors="coerce") < fdr
        elif "p_value" in md.columns:
            md["_active"] = pd.to_numeric(md["p_value"], errors="coerce") < fdr
        else:
            logger.info("  L4b phenotypically-active map skipped — no p-value/flag column")
            return

        active_vals = _map_metric_to_wells(df_l4b, md, "_active")
        if active_vals is None:
            logger.info("  L4b phenotypically-active map skipped — cannot join mAP to wells")
            return
        mask = active_vals.fillna(False).astype(bool)
        n_active = int(mask.sum())
        emb, emb_label = _l4b_embedding(df_l4b)
        if emb is None:
            logger.info("  L4b phenotypically-active map skipped — embedding unavailable")
            return

        # Compound-level stats (mAP is computed per compound×dose — this is a
        # different denominator than the per-well "n_active" count in the title).
        stats_text = None
        if "Metadata_compound" in md.columns:
            n_compounds_total = md["Metadata_compound"].nunique()
            n_compounds_active = md.loc[md["_active"], "Metadata_compound"].nunique()
            pct = 100 * n_compounds_active / n_compounds_total if n_compounds_total else float("nan")
            stats_text = (
                f"{n_compounds_active}/{n_compounds_total} treatment compounds "
                f"phenotypically active (mAP FDR<{fdr:g}) ({pct:.1f}%)"
            )

        _highlight_and_filtered_figure(
            df_l4b, emb, emb_label, mask,
            subset_label=f"active (mAP FDR<{fdr:g})", highlight_color=C_OK,
            plots_dir=plots_dir, cell_line=cell_line,
            fname=f"{cell_line}_l4b_phenotypically_active.png",
            title_prefix=f"{cell_line} — L4b phenotypically active ({n_active} wells)",
            negcon_value=negcon_value,
            compound_stats_text=stats_text,
        )
    except Exception as e:
        logger.warning(f"L4b phenotypically-active map failed: {e}")
