#!/usr/bin/env python3
"""
Generate staged QC & analysis plots from completed pipeline outputs.

Each plot checks one stage of the cell painting experiment (image QC, cell number,
DMSO reference quality, feature preprocessing, batch effects). Reads parquets from
{output_dir}/{cell_line}/features/ and the image QC CSVs from {output_dir}/_image_qc/.

Usage
-----
    python run_analysis_plots.py --output-dir /path/to/testwithba [--cell-line U2OS]
    python run_analysis_plots.py --output-dir ... --config cfg.yaml --plots qc,metrics
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import plotting_analysis as pa
from .config import load_config_from_yaml
from . import radar_wrapper


def _radar(df_l5, meta5, plots_dir, cell_line, dose_level_df=None):
    radar_wrapper.generate_radar_plots(df_l5, meta5, plots_dir, cell_line,
                                       dose_level_df=dose_level_df)


def setup_logging(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _read(path):
    return pd.read_parquet(path) if Path(path).exists() else None


def _level_label(name: str) -> str:
    """'level_4b' → 'Level 4b', 'level_2_5b' → 'Level 2.5b'; anything else unchanged."""
    stem = name.split(".")[0].split(" ")[0]
    if not stem.startswith("level_"):
        return name
    lvl = stem[len("level_"):].replace("2_5", "2.5")
    return f"Level {lvl.split('_')[0]}" + ("" if "_" not in lvl else f" ({lvl.split('_', 1)[1].replace('_', ' ')})")


def _source_text(files, feat: Path) -> str:
    """Footnote for a plot: which data level(s)/files it uses, and from which run."""
    parts = []
    for f in files or []:
        if "/" in f or "(" in f:  # full path or annotated name: keep as is
            parts.append(f)
            continue
        p = feat / f
        if not p.suffix:
            p = next((feat / (f + ext) for ext in (".parquet", ".csv") if (feat / (f + ext)).exists()),
                     feat / (f + ".parquet"))
        label = _level_label(p.name)
        txt = p.name if label == p.name else f"{label} ({p.name})"
        if p.is_symlink():
            txt += f" → {p.resolve()}"
        parts.append(txt)
    data = "; ".join(parts) if parts else "see plot title"
    return f"Data: {data}\nRun features folder: {feat}"


# Plots are organized into these subfolders under <cell_line>/plots/
PLOT_SUBDIRS = ["01_qc", "02_batch", "03_controls", "04_overview", "05_metrics", "06_site_level"]


def parse_plot_groups(groups):
    """Normalize a plot-group selection to PLOT_SUBDIRS names.

    Accepts a list or a comma-separated string of full ("01_qc") or short
    ("qc", "batch", "controls", "overview", "metrics", "site_level") names.
    None / "all" → None (every group)."""
    if groups is None:
        return None
    if isinstance(groups, str):
        groups = groups.split(",")
    groups = [g.strip() for g in groups if g and g.strip()]
    if not groups or "all" in groups:
        return None
    by_short = {d.split("_", 1)[1]: d for d in PLOT_SUBDIRS}
    out = []
    for g in groups:
        name = g if g in PLOT_SUBDIRS else by_short.get(g)
        if name is None:
            raise ValueError(f"Unknown plot group '{g}'. Choose from: "
                             f"{', '.join(by_short)} (or {', '.join(PLOT_SUBDIRS)})")
        out.append(name)
    return out


def generate_plots(output_dir, cell_line, config=None, verbose=False,
                   features_subdir=None, plots_subdir=None, groups=None):
    """Single entry point that generates ALL plots, organized into subfolders.

    Used both standalone (run_analysis_plots.py) and by the pipeline at the end,
    so every plot always comes from the same code path.

    ``features_subdir``/``plots_subdir`` let a caller point at a nested feature
    folder (e.g. the SPECS "features/6k" pooled reference) or a dated plots
    folder. When not given explicitly, they fall back to
    ``config.output.features_subdir`` / ``config.output.plots_subdir`` (so a
    dated run, e.g. ``features_subdir: "features_2026-09-15"`` in the YAML,
    reads/writes the matching dated folders), and finally to the plain
    "features" / "plots" defaults when there's no config either.

    ``groups`` restricts output to some plot subfolders, e.g. ["qc"] or
    "qc,metrics" (see parse_plot_groups); default is every group.
    """
    log = logging.getLogger("qc_plots")
    groups = parse_plot_groups(groups)
    base = Path(output_dir)
    if features_subdir is None:
        features_subdir = getattr(config.output, "features_subdir", "features") if config else "features"
    if plots_subdir is None:
        plots_subdir = getattr(config.output, "plots_subdir", "plots") if config else "plots"
    feat = base / cell_line / features_subdir
    plots = base / cell_line / plots_subdir
    image_qc_dir = base / "_image_qc"

    # Category subfolders
    qc = plots / "01_qc"
    batch = plots / "02_batch"
    controls = plots / "03_controls"
    overview = plots / "04_overview"
    metrics_dir = plots / "05_metrics"
    site_level_dir = plots / "06_site_level"
    for d in (qc, batch, controls, overview, metrics_dir, site_level_dir):
        d.mkdir(parents=True, exist_ok=True)

    # thresholds from config (fallback to defaults)
    min_site = config.aggregation.min_cells_per_site if config else 20
    min_well = config.aggregation.min_cells_per_well if config else 50
    dmso_min = config.image_qc.dmso_min_cells_per_site if config else 30
    dmso_z = config.image_qc.dmso_count_z_threshold if config else 3.5
    negcon = config.metrics.negcon_value if config else "negcon"
    poscon = config.metrics.poscon_value if config else "poscon"
    grit_thr = config.metrics.grit_flag_threshold if config else 1.0
    map_fdr = config.metrics.map_fdr_threshold if config else 0.05
    viab_thr = (config.metrics.viability_threshold_pct * 100) if config else 80.0
    dmso_center_tol = config.normalization.dmso_center_tolerance if config else 0.5
    method_label = (config.batch_correction.method if config else "combat").title()

    log.info(f"Generating plots for {cell_line} → {plots}/ (01_qc, 02_batch, 03_controls, 04_overview)")

    # Site-level source (has per-site cell count + pert_type): level_2_5.parquet/.csv
    l25 = next((p for p in (feat / "level_2_5.parquet", feat / "level_2_5.csv") if p.exists()), None)
    # (large file — only read when the 01_qc plots that use it are requested)
    df_sites = None
    if l25 is not None and (groups is None or "01_qc" in groups):
        df_sites = pd.read_parquet(l25) if l25.suffix == ".parquet" \
            else pd.read_csv(l25, engine="python")
    df_l3 = _read(feat / "level_3.parquet")
    df_l4a = _read(feat / "level_4a.parquet")
    df_l4b = _read(feat / "level_4b.parquet")
    df_l4c = _read(feat / "level_4c.parquet")

    # (subdir, name, fn)
    steps = [
        (qc, "Image QC flags", lambda: pa.plot_image_qc_flags(image_qc_dir, qc, cell_line)),
    ]
    if df_sites is not None:
        steps += [
            (qc, "Cell count per site", lambda: pa.plot_cell_count_per_site(
                df_sites, qc, cell_line, min_site, negcon)),
            (qc, "DMSO site QC", lambda: pa.plot_dmso_site_qc(
                df_sites, qc, cell_line, dmso_min, dmso_z, negcon)),
        ]
    if df_l3 is not None:
        steps.append((qc, "Cell count per plate", lambda: pa.plot_cell_count_per_plate(
            df_l3, qc, cell_line, min_well)))
        steps.append((qc, "Well count heatmap", lambda: pa.plot_well_count_heatmap(
            df_l3, qc, cell_line)))
        steps.append((controls, "Control morphology boxplots (DMSO vs poscon)",
                      lambda: pa.plot_control_morphology_boxplots(
                          df_l3, controls, cell_line, negcon, poscon)))
    if df_l4b is not None:
        steps.append((qc, "Spatial effects", lambda: pa.plot_spatial_effects(
            df_l4b, qc, cell_line)))
    if all(x is not None for x in (df_l3, df_l4a, df_l4b, df_l4c)):
        steps.append((qc, "Feature preprocessing", lambda: pa.plot_feature_preprocessing(
            df_l3, df_l4a, df_l4b, df_l4c, qc, cell_line)))
    if df_l4a is not None:
        steps.append((qc, "DMSO normalization check", lambda: pa.plot_dmso_normalization_check(
            df_l4a, qc, cell_line, negcon, dmso_center_tol)))
    if df_l4b is not None and df_l4c is not None:
        steps.append((batch, "Batch effects (PCA)", lambda: pa.plot_batch_effects(
            df_l4b, df_l4c, batch, cell_line, method_label=method_label)))
        steps.append((batch, "Batch effects (UMAP)", lambda: pa.plot_batch_effects_umap(
            df_l4b, df_l4c, batch, cell_line, method_label=method_label)))
        steps.append((batch, "DMSO correlation (before, L4b)", lambda: pa.plot_dmso_correlation(
            df_l4b, batch, cell_line, label="L4b", negcon_value=negcon)))
        steps.append((batch, "DMSO correlation (after, L4c)", lambda: pa.plot_dmso_correlation(
            df_l4c, batch, cell_line, label="L4c", negcon_value=negcon)))
    if df_l4b is not None:
        # 04_overview embeddings run on batch-corrected Level 4c (fall back to 4b if absent)
        df_ov = df_l4c if df_l4c is not None else df_l4b
        ov_lvl = "4c" if df_l4c is not None else "4b"
        steps.append((overview, "UMAP", lambda: pa.plot_umap_embedding(df_ov, overview, cell_line, negcon, level_label=ov_lvl)))
        steps.append((overview, "PCA", lambda: pa.plot_pca_embedding(df_ov, overview, cell_line, negcon, level_label=ov_lvl)))
        steps.append((overview, "t-SNE", lambda: pa.plot_tsne_embedding(df_ov, overview, cell_line, negcon, level_label=ov_lvl)))
        steps.append((overview, "Feature heatmap", lambda: pa.plot_feature_heatmap(df_ov, overview, cell_line, negcon, level_label=ov_lvl)))
        steps.append((controls, "Controls PCA", lambda: pa.plot_controls_pca(
            df_l4b, controls, cell_line, negcon)))
        steps.append((controls, "Controls clustering (PCA+UMAP)", lambda: pa.plot_controls_clustering(
            df_l4b, controls, cell_line, negcon)))

    # Site-level Level 2.5b (opt-in via config.site_level.run): PCA + UMAP of
    # DMSO/poscon SITES (pre-aggregation), for spotting site-level outliers.
    df_l25b = _read(feat / "level_2_5b.parquet")
    if df_l25b is not None:
        steps.append((site_level_dir, "Site-level (L2.5b) controls PCA+UMAP",
                      lambda: pa.plot_controls_clustering(
                          df_l25b, site_level_dir, cell_line, negcon, poscon,
                          out_name=f"{cell_line}_controls_clustering_sitelevel.png",
                          level_label="Level 2.5b, site level")))
        df_site_outliers = None
        outliers_csv = feat / "site_outliers.csv"
        if outliers_csv.exists():
            df_site_outliers = pd.read_csv(outliers_csv)
        if df_site_outliers is not None:
            steps.append((site_level_dir, "Site outliers PCA", lambda: pa.plot_site_outliers_pca(
                df_l25b, df_site_outliers, site_level_dir, cell_line)))
            steps.append((site_level_dir, "Site outliers heatmap", lambda: pa.plot_site_outliers_heatmap(
                df_l25b, df_site_outliers, site_level_dir, cell_line)))

    # Level 5 (consensus) plots: radar + poscon analysis.
    # Prefer the batch-corrected consensus; fall back to the legacy single-file
    # name (level_5.parquet) so outputs from older pipeline runs still plot.
    l5 = feat / "level_5_batch_corrected.parquet"
    if not l5.exists():
        l5 = feat / "level_5.parquet"
    if l5.exists():
        df_l5 = pd.read_parquet(l5)
        meta5 = df_l5[[c for c in df_l5.columns if c.startswith("Metadata_")]].copy()
        steps.append((controls, "Radar (poscon/negcon, dose-shaded)",
                      lambda: _radar(df_l5, meta5, controls, cell_line, dose_level_df=df_l4c)))
        steps.append((controls, "Radar combined (all poscon)", lambda: pa.plot_poscon_radar_combined(df_l5, controls, cell_line)))
        steps.append((controls, "Poscon analysis", lambda: pa.plot_poscon_analysis(df_l5, controls, cell_line)))
        steps.append((controls, "Poscon clustermap", lambda: pa.plot_poscon_clustermap(df_l5, controls, cell_line)))
        steps.append((controls, "Channel analysis", lambda: pa.plot_channel_analysis(df_l5, controls, cell_line)))

    # Metric plots: phenotypic activity (mAP), GRIT, viability
    map_df = _read(feat / "metrics_map.parquet")
    grit_df = _read(feat / "metrics_grit.parquet")
    if map_df is not None:
        steps.append((metrics_dir, "mAP distribution", lambda: pa.plot_map_distribution(
            map_df, metrics_dir, cell_line, map_fdr)))
        steps.append((metrics_dir, "Top mAP compounds", lambda: pa.plot_top_map_compounds(
            map_df, metrics_dir, cell_line, fdr=map_fdr)))
    if grit_df is not None:
        steps.append((metrics_dir, "GRIT distribution", lambda: pa.plot_grit_distribution(
            grit_df, metrics_dir, cell_line, grit_thr)))
        steps.append((metrics_dir, "Top GRIT compounds", lambda: pa.plot_top_grit_compounds(
            grit_df, metrics_dir, cell_line, grit_threshold=grit_thr)))
    # L4b metric-driven maps (GRIT and mAP are computed on L4b, so embed L4b).
    if df_l4b is not None and grit_df is not None:
        steps.append((metrics_dir, "L4b GRIT-filtered map (GRIT > 1)",
                      lambda: pa.plot_l4b_grit_filtered(
                          df_l4b, grit_df, metrics_dir, cell_line,
                          grit_threshold=grit_thr, negcon_value=negcon)))
    if df_l4b is not None and map_df is not None:
        steps.append((metrics_dir, "L4b phenotypically-active map",
                      lambda: pa.plot_l4b_map_active(
                          df_l4b, map_df, metrics_dir, cell_line,
                          fdr=map_fdr, negcon_value=negcon)))
    # Viability / toxicity: prefer L4c, fall back to L4b
    df_viab = df_l4c if (df_l4c is not None and "Metadata_Viability_pct" in getattr(df_l4c, "columns", [])) else \
              (df_l4b if (df_l4b is not None and "Metadata_Viability_pct" in getattr(df_l4b, "columns", [])) else None)
    if df_viab is not None:
        steps.append((metrics_dir, "Viability", lambda: pa.plot_viability(
            df_viab, metrics_dir, cell_line, viab_thr)))
        steps.append((metrics_dir, "Top toxic compounds", lambda: pa.plot_top_toxic_compounds(
            df_viab, metrics_dir, cell_line, threshold_pct=viab_thr)))
    # Site-level viability diagnostics (per-site distribution, roll-up confidence, DMSO reference)
    df_via_sites = _read(feat / "viability_sites.parquet")
    if df_via_sites is not None:
        viab_dmso_z = config.metrics.viability_dmso_z_threshold if config else 3.0
        steps.append((metrics_dir, "Viability (site level)", lambda: pa.plot_viability_site_level(
            df_via_sites, metrics_dir, cell_line, threshold_pct=viab_thr,
            dmso_z=viab_dmso_z, negcon_value=negcon)))

    if groups is not None:
        steps = [s for s in steps if s[0].name in groups]
        log.info(f"  Plot groups: {', '.join(groups)} ({len(steps)} plots)")

    # Data each plot is drawn from — written as a footnote on every figure,
    # together with the features folder of this run.
    l5_name = l5.name if l5.exists() else "level_5_batch_corrected.parquet"
    viab_lvl = ("level_4c" if df_viab is df_l4c else "level_4b") if df_viab is not None else "level_4b"
    ov_lvl_file = "level_4c" if df_l4c is not None else "level_4b"
    sources = {
        "Image QC flags": [str(image_qc_dir / "qc_flags_*.csv")],
        "Cell count per site": [l25.name if l25 else "level_2_5.csv"],
        "DMSO site QC": [l25.name if l25 else "level_2_5.csv"],
        "Cell count per plate": ["level_3"],
        "Well count heatmap": ["level_3"],
        "Control morphology boxplots (DMSO vs poscon)": ["level_3"],
        "Spatial effects": ["level_4b"],
        "Feature preprocessing": ["level_3", "level_4a", "level_4b", "level_4c"],
        "DMSO normalization check": ["level_4a"],
        "Batch effects (PCA)": ["level_4b", "level_4c"],
        "Batch effects (UMAP)": ["level_4b", "level_4c"],
        "DMSO correlation (before, L4b)": ["level_4b"],
        "DMSO correlation (after, L4c)": ["level_4c"],
        "UMAP": [ov_lvl_file], "PCA": [ov_lvl_file], "t-SNE": [ov_lvl_file],
        "Feature heatmap": [ov_lvl_file],
        "Controls PCA": ["level_4b"],
        "Controls clustering (PCA+UMAP)": ["level_4b"],
        "Site-level (L2.5b) controls PCA+UMAP": ["level_2_5b"],
        "Site outliers PCA": ["level_2_5b", "site_outliers.csv"],
        "Site outliers heatmap": ["level_2_5b", "site_outliers.csv"],
        "Radar (poscon/negcon, dose-shaded)": [l5_name, "level_4c"],
        "Radar combined (all poscon)": [l5_name],
        "Poscon analysis": [l5_name],
        "Poscon clustermap": [l5_name],
        "Channel analysis": [l5_name],
        "mAP distribution": ["metrics_map.parquet (mAP computed on Level 4b)"],
        "Top mAP compounds": ["metrics_map.parquet (mAP computed on Level 4b)"],
        "GRIT distribution": ["metrics_grit.parquet (GRIT computed on Level 4b)"],
        "Top GRIT compounds": ["metrics_grit.parquet (GRIT computed on Level 4b)"],
        "L4b GRIT-filtered map (GRIT > 1)": ["level_4b", "metrics_grit"],
        "L4b phenotypically-active map": ["level_4b", "metrics_map"],
        "Viability": [viab_lvl],
        "Top toxic compounds": [viab_lvl],
        "Viability (site level)": ["viability_sites"],
    }

    ok = True
    for _sub, name, fn in steps:
        try:
            log.info(f"  ▶ {name}")
            with pa.source_note(_source_text(sources.get(name), feat)):
                fn()
        except Exception as e:
            log.warning(f"  ✗ {name} failed: {e}")
            ok = False
    log.info("✓ Plot generation complete")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cell-line")
    ap.add_argument("--config", help="Optional YAML config for thresholds")
    ap.add_argument("--plots", default=None,
                    help="Comma-separated plot groups to generate: qc, batch, controls, "
                         "overview, metrics, site_level (default: all)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    base = Path(args.output_dir)
    if not base.exists():
        print(f"Error: {args.output_dir} not found")
        sys.exit(1)

    config = load_config_from_yaml(args.config) if args.config else None

    if args.cell_line:
        cell_lines = [args.cell_line]
    else:
        cell_lines = sorted(d.name for d in base.iterdir()
                            if d.is_dir() and not d.name.startswith("_"))
    if not cell_lines:
        print("No cell lines found")
        sys.exit(1)

    failed = [cl for cl in cell_lines if not generate_plots(args.output_dir, cl, config, args.verbose,
                                                                   groups=args.plots)]
    if failed:
        print(f"\n⚠ Some plots failed for: {', '.join(failed)}")
    else:
        print(f"\n✓ All plots generated for: {', '.join(cell_lines)}")


if __name__ == "__main__":
    main()
