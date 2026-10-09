"""
Wrapper to call radar plot functions from the pipeline.
Generates radar plots for poscon, negcon, and specific compounds.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from .radarplots import (create_radar_plot, categorize_features,
                         calculate_appropriate_range, MAIN_GROUPS)


# Distinct base hues cycled across compounds; each radar deepens this hue by dose.
_BASE_HUES = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]


def generate_radar_plots(
    df_level5: pd.DataFrame,
    metadata_level5: pd.DataFrame,
    output_dir: Path,
    cell_line: str,
    compounds_to_plot=None,
    dose_level_df: pd.DataFrame = None,
    conc_col: str = "Metadata_conc_uM",
) -> None:
    """
    Generate one categorized radar plot per poscon/negcon compound.

    Parameters
    ----------
    df_level5, metadata_level5 : Level 5 consensus profiles + metadata.
    output_dir, cell_line : where / what to name the plots.
    compounds_to_plot : optional treatment compounds to also render.
    dose_level_df : optional WELL-LEVEL frame (e.g. Level 4c) carrying
        `conc_col`. When given, each compound's radar overlays ALL its
        concentrations as polygons in progressively darker shades of one hue
        (light = lowest dose, dark = highest), instead of a single aggregated
        polygon. Compounds with a single dose still render as one polygon.
    """

    if create_radar_plot is None or MAIN_GROUPS is None:
        logger.warning("Radarplots module not available")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find relevant columns
    pert_col = None
    compound_col = None
    for col in metadata_level5.columns:
        if "pert_type" in col.lower():
            pert_col = col
        if col in ("Metadata_compound", "Metadata_cmpd_name") and compound_col is None:
            compound_col = col
    if compound_col is None:
        for col in metadata_level5.columns:
            if "compound" in col.lower() or "cmpd" in col.lower():
                compound_col = col
                break

    if pert_col is None or compound_col is None:
        logger.warning("Cannot find pert_type / compound column for radar plots")
        return

    feature_cols = [c for c in df_level5.columns if not c.startswith("Metadata_")]
    if not feature_cols:
        logger.warning("No features found for radar plots")
        return

    # Build a combined frame (metadata + features) that create_radar_plot filters
    combined = pd.concat([metadata_level5.reset_index(drop=True),
                          df_level5[feature_cols].reset_index(drop=True)], axis=1)
    # Categorize features once into compartment/stain radar axes
    categorized = categorize_features(feature_cols, MAIN_GROUPS)

    # Well-level dose source (optional): must carry the compound + conc columns.
    have_doses = (dose_level_df is not None and conc_col in dose_level_df.columns
                  and compound_col in dose_level_df.columns)
    if have_doses:
        logger.info(f"Generating dose-resolved radar plots for {cell_line} "
                    f"(shaded by {conc_col})...")
    else:
        logger.info(f"Generating categorized radar plots for {cell_line}...")

    # Map compound id -> real display name (Metadata_compound_name), if present.
    name_map = {}
    if "Metadata_compound_name" in metadata_level5.columns:
        disp = (metadata_level5["Metadata_compound_name"].astype("string")
                .fillna(metadata_level5[compound_col].astype("string")))
        name_map = dict(zip(metadata_level5[compound_col].astype(str), disp.astype(str)))

    def _safe(s):
        return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))

    def _radar(compound, label, base_color=None):
        try:
            # Build ordered per-dose data when a well-level dose source is given.
            doses_data, plot_df = None, combined
            if have_doses:
                sub = dose_level_df[dose_level_df[compound_col].astype(str) == str(compound)]
                concs = sorted(pd.to_numeric(sub[conc_col], errors="coerce").dropna().unique())
                if len(concs) >= 1:
                    plot_df = sub
                    if len(concs) > 1:
                        doses_data = {}
                        cnum = pd.to_numeric(sub[conc_col], errors="coerce")
                        for c in concs:
                            doses_data[c] = sub.loc[cnum == c, feature_cols]
            fig = create_radar_plot(
                df=plot_df,
                compound_name=compound,
                display_name=name_map.get(str(compound), str(compound)),
                main_groups=MAIN_GROUPS,
                categorized_features=categorized,
                id_col=compound_col,
                feature_cols=feature_cols,
                value_range=(-5, 5),
                doses_data=doses_data,
                base_color=base_color,
                save_path=str(output_dir / f"{cell_line}_radar_{label}.png"),
            )
            plt.close(fig)
            logger.info(f"  Saved: {cell_line}_radar_{label}.png")
        except Exception as e:
            logger.warning(f"  Radar plot for {label} failed: {e}")

    # One categorized radar per poscon compound (filename by real name), each a
    # distinct hue that darkens with dose.
    poscon = metadata_level5[metadata_level5[pert_col] == "poscon"][compound_col].unique()
    for i, cpd in enumerate(poscon):
        _radar(cpd, f"poscon_{_safe(name_map.get(str(cpd), cpd))}",
               base_color=_BASE_HUES[i % len(_BASE_HUES)])

    # Negcon (DMSO) baseline
    negcon = metadata_level5[metadata_level5[pert_col] == "negcon"][compound_col].unique()
    for cpd in negcon:
        _radar(cpd, "negcon", base_color="#555555")

    # Requested treatment compounds
    if compounds_to_plot:
        for cpd in compounds_to_plot:
            if cpd in combined[compound_col].values:
                _radar(cpd, f"trt_{str(cpd).replace('/', '_')}")
            else:
                logger.warning(f"Compound {cpd} not found for radar")
