#!/usr/bin/env python
"""
Radar plot generator for compound feature analysis.
Creates radar plots from h5ad data showing feature categories organized by
stain type, feature type, and cellular compartment.

Usage:
    python radarplots.py --h5ad HOG_adata_L5.h5ad --compound "7-Ketocholesterol"
    python radarplots.py --h5ad HOG_adata_L5.h5ad --compound "3-Epideoxycholic acid" --output ./plots
    python radarplots.py --csv features.csv --compound "7-Ketocholesterol"
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from math import pi
from typing import Dict, List, Optional

# Feature group definitions for HOG staining
# Groups by stain/marker (biological target) with subgroups by measurement type
# Abbreviations: I=Intensity, G=Granularity, T=Texture, R=RadialDistribution, S=Shape
MAIN_GROUPS = {
    "HOECHST": {
        "I": ["Intensity", "illumHOECHST"],
        "G": ["Granularity", "illumHOECHST"],
        "T": ["Texture", "illumHOECHST"],
        "R": ["RadialDistribution", "illumHOECHST"]
    },
    "CONC": {
        "I": ["Intensity", "illumCONC"],
        "G": ["Granularity", "illumCONC"],
        "T": ["Texture", "illumCONC"],
        "R": ["RadialDistribution", "illumCONC"]
    },
    "SYTO": {
        "I": ["Intensity", "illumSYTO"],
        "G": ["Granularity", "illumSYTO"],
        "T": ["Texture", "illumSYTO"],
        "R": ["RadialDistribution", "illumSYTO"]
    },
    "PHAandWGA": {
        "I": ["Intensity", "illumPHAandWGA"],
        "G": ["Granularity", "illumPHAandWGA"],
        "T": ["Texture", "illumPHAandWGA"],
        "R": ["RadialDistribution", "illumPHAandWGA"]
    },
    "MITO": {
        "I": ["Intensity", "illumMITO"],
        "G": ["Granularity", "illumMITO"],
        "T": ["Texture", "illumMITO"],
        "R": ["RadialDistribution", "illumMITO"]
    },
    "AreaShape_Nuclei": {
        "S": ["AreaShape", "Nuclei"]
    },
    "AreaShape_Cytoplasm": {
        "S": ["AreaShape", "Cytoplasm"]
    },
    "AreaShape_Cells": {
        "S": ["AreaShape", "Cells"]
    }
}

# Legend for abbreviations
ABBREVIATION_LEGEND = {
    "I": "Intensity",
    "G": "Granularity",
    "T": "Texture",
    "R": "Radial Distribution",
    "S": "Shape"
}


def load_h5ad_as_dataframe(h5ad_path: str) -> pd.DataFrame:
    """Load h5ad file and convert to DataFrame."""
    try:
        import anndata
    except ImportError:
        raise ImportError("anndata is required. Install with: pip install anndata")

    adata = anndata.read_h5ad(h5ad_path)
    df = adata.to_df()

    # If metadata is in obs, add it
    if hasattr(adata, 'obs') and len(adata.obs) > 0:
        df = pd.concat([adata.obs.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    return df


def load_data(csv_path: Optional[str] = None, h5ad_path: Optional[str] = None, sep: str = ";") -> pd.DataFrame:
    """Load data from CSV or h5ad file."""
    if h5ad_path:
        print(f"Loading h5ad from: {h5ad_path}")
        return load_h5ad_as_dataframe(h5ad_path)
    elif csv_path:
        print(f"Loading CSV from: {csv_path}")
        return pd.read_csv(csv_path, sep=sep)
    else:
        raise ValueError("Either csv_path or h5ad_path must be provided")


def load_grit_metrics(metrics_path: str, cell_line: str, min_grit: float) -> Optional[pd.DataFrame]:
    """Load grit metrics from parquet file for filtering by grit value.

    ``metrics_path`` may contain ``{cell_line}``.
    """
    try:
        metrics_path = Path(metrics_path.format(cell_line=cell_line))
        if metrics_path.exists():
            print(f"Loading grit metrics from: {metrics_path}")
            metrics_df = pd.read_parquet(metrics_path)
            # Filter to only doses with grit > min_grit
            filtered = metrics_df[metrics_df['grit'] > min_grit]
            print(f"Found {len(filtered)} doses with grit > {min_grit}")
            return filtered
        else:
            print(f"Warning: Metrics file not found at {metrics_path}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Warning: Could not load grit metrics: {e}", file=sys.stderr)
        return None


def categorize_features(
    feature_names: List[str],
    main_groups: Dict[str, Dict[str, List[str]]],
    verbose: bool = False
) -> Dict[str, Dict[str, List[str]]]:
    """Categorize features based on keyword matching."""
    categorized = {mg: {sg: [] for sg in subs} for mg, subs in main_groups.items()}

    for feature in feature_names:
        feature_lower = feature.lower()
        for mg, subgroups in main_groups.items():
            for sg, keywords in subgroups.items():
                if all(kw.lower() in feature_lower for kw in keywords):
                    categorized[mg][sg].append(feature)
                    break

    if verbose:
        for mg, subs in categorized.items():
            for sg, feats in subs.items():
                if feats:
                    print(f"[MATCH] {mg}/{sg}: {len(feats)} features")

    return categorized


def calculate_appropriate_range(df: pd.DataFrame, feature_cols: List[str], padding: float = 1.5) -> tuple:
    """Calculate appropriate value range based on data min/max with padding."""
    # Get only numeric feature values
    numeric_cols = []
    for col in feature_cols:
        try:
            pd.to_numeric(df[col])
            numeric_cols.append(col)
        except (ValueError, TypeError):
            pass

    if not numeric_cols:
        return (-10, 10)

    # Get all values from numeric columns
    all_values = []
    for col in numeric_cols:
        col_data = pd.to_numeric(df[col], errors='coerce').dropna().values
        all_values.extend(col_data)

    all_values = np.array(all_values)

    if len(all_values) == 0:
        return (-10, 10)

    data_min = float(np.min(all_values))
    data_max = float(np.max(all_values))

    # Add padding (percentage of range)
    data_range = data_max - data_min
    if data_range == 0:
        data_range = 1

    pad = data_range * (padding - 1) / 2
    range_min = data_min - pad
    range_max = data_max + pad

    # Round to nearest 5 for nice numbers
    range_min = int(np.floor(range_min / 5) * 5)
    range_max = int(np.ceil(range_max / 5) * 5)

    # Ensure symmetric range around 0 for better visualization
    max_abs = max(abs(range_min), abs(range_max))
    return (-max_abs, max_abs)


def create_radar_plot(
    df: pd.DataFrame,
    compound_name: str,
    main_groups: Dict[str, Dict[str, List[str]]],
    categorized_features: Dict[str, Dict[str, List[str]]],
    id_col: str = "Metadata_compound",
    feature_cols: Optional[List[str]] = None,
    palette: str = "Paired",
    value_range: Optional[tuple] = (-5, 5),
    seg_alpha: float = 0.5,
    poly_alpha: float = 0.30,
    save_path: Optional[str] = None,
    batch_id: Optional[str] = None,
    dose: Optional[float] = None,
    doses_data: Optional[Dict] = None,
    display_name: Optional[str] = None,
    base_color: Optional[str] = None,
) -> plt.Figure:
    """Create a radar plot for a single compound, optionally with multiple doses.

    display_name, if given, is shown in the title instead of compound_name (which
    is still used to select the row), so plots can show a real name while filtering
    by a stable id.

    When doses_data (an ORDERED {dose_value: df} mapping, ascending by dose) is
    given, one polygon is drawn per dose in progressively darker SHADES of a single
    hue (base_color) — light = lowest dose, dark = highest — so a compound's
    dose-response reads as one colour deepening with concentration."""

    # Get feature columns if not specified
    if feature_cols is None:
        metadata_cols = [col for col in df.columns if col.startswith("Metadata_")]
        feature_cols = [col for col in df.columns if col not in metadata_cols]

    # Filter data for the compound
    data = df.loc[df[id_col] == compound_name, feature_cols]

    if data.empty:
        raise ValueError(f"No data found for compound: {compound_name}")

    # Prepare stats for plotting
    all_labels = [sg for mg in main_groups for sg in main_groups[mg]]

    # Setup angles
    angles = np.linspace(0, 2 * pi, len(all_labels), endpoint=False).tolist()
    angles += angles[:1]

    # Colors for segments - AreaShape compartments get distinct colors
    group_names = list(main_groups.keys())
    custom_colors = {
        "HOECHST": "#1f77b4",  # blue
        "MITO": "#d62728",     # red
        "SYTO": "#ffff00",     # yellow
        "PHAandWGA": "#ff7f0e",  # orange
        "CONC": "#2ca02c",     # green
    }

    # AreaShape compartment colors (shades of pink and purple)
    areashape_colors = {
        "AreaShape_Nuclei": "#4a0080",     # Dark purple
        "AreaShape_Cytoplasm": "#ff69b4",  # Hot pink
        "AreaShape_Cells": "#da70d6"       # Orchid
    }

    # Create color list with custom colors where specified, fallback to palette for others
    seg_colors = []
    for mg in group_names:
        if mg in areashape_colors:
            seg_colors.append(areashape_colors[mg])
        elif mg in custom_colors:
            seg_colors.append(custom_colors[mg])
        else:
            seg_colors.append("#7f7f7f")  # gray

    # Background layout calculations
    total_subgroups = sum(len(subs) for subs in main_groups.values())
    max_value = max(abs(v) for v in value_range)
    outer_edge_value = max_value * 1.25
    inner_ring_value = max_value * 1.50
    radial_ticks = np.linspace(value_range[0], value_range[1], num=5)

    # Create figure
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    # Calculate angle range for each main group based on expected order
    group_angle_ranges = {}
    label_idx = 0
    for mg in group_names:
        num_subgroups = len(main_groups[mg])
        if num_subgroups > 0:
            group_angle_ranges[mg] = (label_idx, label_idx + num_subgroups - 1)
            label_idx += num_subgroups

    # Draw colored background segments for each main group
    for mg_idx, (mg, col) in enumerate(zip(group_names, seg_colors)):
        if mg in group_angle_ranges:
            start_idx, end_idx = group_angle_ranges[mg]

            # Get the angle range for this main group
            if start_idx < len(angles) and end_idx < len(angles):
                start_ang = angles[start_idx]
                end_ang = angles[end_idx]

                # Calculate the angle width for proper spacing
                angle_width = angles[1] - angles[0] if len(angles) > 1 else 2 * pi / len(all_labels)

                # Draw the segment without overlap
                seg_start = start_ang - angle_width / 2
                seg_end = end_ang + angle_width / 2

                seg_angles = np.linspace(seg_start, seg_end, 120)
                ax.fill_between(seg_angles, inner_ring_value, outer_edge_value,
                               color=col, alpha=seg_alpha, zorder=0)

    # Place subgroup labels on rim
    for ang, lab in zip(angles[:-1], all_labels):
        ax.text(ang, outer_edge_value * 1.08, lab, ha="center", va="center",
               fontsize=8, color="black", zorder=1)

    # Radial ticks and grid with value labels
    ax.set_rgrids(radii=radial_ticks, labels=[f"{r:.1f}" for r in radial_ticks],
                 angle=0, fontsize=8)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="gray", alpha=0.5)
    ax.spines["polar"].set_visible(False)

    # Remove angular labels (0, 90, 180, etc.)
    ax.set_xticks([])

    # Draw spokes
    for ang in angles[:-1]:
        ax.plot([ang, ang], [value_range[0], outer_edge_value],
               color="gray", lw=0.5, alpha=0.5, zorder=1)

    # Plot data polygon(s)
    if doses_data:
        # Multiple doses case: one polygon per dose, drawn in progressively darker
        # shades of ONE hue (base_color) — light = lowest dose, dark = highest.
        import matplotlib.colors as mcolors
        base_rgb = np.array(mcolors.to_rgb(base_color or "#1f77b4"))
        n_doses = len(doses_data)
        if n_doses == 1:
            dose_colors = [tuple(base_rgb)]
        else:
            light = base_rgb + (1.0 - base_rgb) * 0.70   # blend toward white
            dark = base_rgb * 0.42                        # blend toward black
            dose_colors = [tuple(light + (dark - light) * t)
                           for t in np.linspace(0.0, 1.0, n_doses)]
        legend_handles = []
        for (dose_val, dose_df), color in zip(doses_data.items(), dose_colors):
            # Calculate subgroup averages for this dose
            averages = {}
            for mg, subgroups in categorized_features.items():
                averages[mg] = {}
                for sg, features in subgroups.items():
                    if features:
                        avg_val = dose_df[features].mean(axis=1).mean()
                    else:
                        avg_val = 0.0
                    averages[mg][sg] = float(avg_val)

            stats = [averages[mg][sg] for mg in main_groups for sg in main_groups[mg]]
            stats += stats[:1]

            try:
                dose_lbl = f"{float(dose_val):g} µM"
            except (TypeError, ValueError):
                dose_lbl = f"{dose_val} µM"
            ax.plot(angles, stats, color=color, alpha=1.0, linewidth=2, zorder=3, label=dose_lbl)
            ax.fill(angles, stats, color=color, alpha=poly_alpha, zorder=2)
            legend_handles.append(mpatches.Patch(color=color, label=dose_lbl))

        # Add compartment color legend
        legend_handles.extend([mpatches.Patch(color=c, label=g) for c, g in zip(seg_colors, group_names)])
        ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.05, 0.5),
                 fontsize=9, frameon=False)
    else:
        # Single dose case
        averages = {}
        for mg, subgroups in categorized_features.items():
            averages[mg] = {}
            for sg, features in subgroups.items():
                if features:
                    avg_val = data[features].mean(axis=1).mean()
                else:
                    avg_val = 0.0
                averages[mg][sg] = float(avg_val)

        stats = [averages[mg][sg] for mg in main_groups for sg in main_groups[mg]]
        stats += stats[:1]

        ax.plot(angles, stats, color="#1f77b4", alpha=1.0, zorder=2, linewidth=2)
        ax.fill(angles, stats, color="#1f77b4", alpha=poly_alpha, zorder=2)

        # Legend for stains (main groups)
        legend_handles = [mpatches.Patch(color=c, label=g) for c, g in zip(seg_colors, group_names)]
        ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.05, 0.5),
                 fontsize=9, frameon=False)

    # Add abbreviation legend below the color channel legend
    abbrev_text = "Abbreviations:\n" + "\n".join([f"{k} = {v}" for k, v in sorted(ABBREVIATION_LEGEND.items())])
    fig.text(1.05, 0.20, abbrev_text, fontsize=8, family='monospace', ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Title with optional batch ID and dose
    title_parts = [str(display_name if display_name else compound_name)]
    if batch_id:
        title_parts.append(f"Batch: {batch_id}")
    if dose is not None:
        title_parts.append(f"Dose: {dose} µM")
    elif doses_data:
        title_parts.append("All Doses")
    title = " | ".join(title_parts)
    plt.title(title, fontsize=12, fontweight='bold')

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate radar plots for compound feature analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python radarplots.py --h5ad features/HOG_adata_L5.h5ad --compound "7-Ketocholesterol"
  python radarplots.py --h5ad features/HOG_adata_L5.h5ad --batch-id "BA_001"
  python radarplots.py --h5ad features/HOG_adata_L5.h5ad --compound "3-Epideoxycholic acid" --output ./plots
  python radarplots.py --csv well_df.csv --batch-id "BA_002" --output ./plots
        """
    )

    parser.add_argument("--h5ad", type=str, default=None,
                       help="Path to h5ad file with feature data")
    parser.add_argument("--csv", type=str, default=None,
                       help="Path to CSV file with feature data")
    parser.add_argument("--compound", type=str, default=None,
                       help="Compound name to plot (value in Metadata_compound column)")
    parser.add_argument("--batch-id", type=str, default=None,
                       help="Batch ID to plot (looks up compound in Metadata_batch_id)")
    parser.add_argument("--dose", type=float, default=None,
                       help="Dose in µM to filter by (e.g., 1.0, 3.16). Use --list-doses to see available.")
    parser.add_argument("--list-doses", action="store_true",
                       help="List available doses for a compound and exit")
    parser.add_argument("--output", type=str, default=None,
                       help="Directory to save output plot (default: no save)")
    parser.add_argument("--id-col", type=str, default="Metadata_compound",
                       help="Column name containing compound identifiers")
    parser.add_argument("--palette", type=str, default="Paired",
                       help="Color palette (seaborn palette name)")
    parser.add_argument("--value-range", type=float, nargs=2, default=None,
                       help="Min and max values for radial axis (default: -5 5)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print feature categorization details")
    parser.add_argument("--min-grit", type=float, default=None,
                       help="Minimum grit value to include (filters doses using metrics from --metrics-path)")
    parser.add_argument("--metrics-path", type=str, default=None,
                       help="compound_metrics.parquet holding a 'grit' column; may contain {cell_line}. Required with --min-grit")

    args = parser.parse_args()

    # Validate inputs
    if not args.h5ad and not args.csv:
        parser.error("Must provide either --h5ad or --csv")
    if not args.compound and not args.batch_id:
        parser.error("Must provide either --compound or --batch-id")

    # Load data
    try:
        df = load_data(csv_path=args.csv, h5ad_path=args.h5ad)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")

    # Resolve batch ID to compound if needed
    if args.batch_id:
        if "Metadata_batch_id" not in df.columns:
            print("Error: Metadata_batch_id column not found in data", file=sys.stderr)
            sys.exit(1)

        batch_compounds = df[df["Metadata_batch_id"] == args.batch_id]["Metadata_compound"].unique()
        if len(batch_compounds) == 0:
            print(f"Error: No data found for batch_id: {args.batch_id}", file=sys.stderr)
            sys.exit(1)

        if len(batch_compounds) > 1:
            print(f"\nWarning: Multiple compounds found in batch {args.batch_id}:")
            for i, c in enumerate(batch_compounds, 1):
                print(f"  {i}. {c}")
            args.compound = batch_compounds[0]
            print(f"Using first compound: {args.compound}\n")
        else:
            args.compound = batch_compounds[0]

        print(f"Batch ID {args.batch_id} → Compound: {args.compound}")

    # List available doses if requested
    if args.list_doses:
        if not args.compound:
            print("Error: Must specify --compound or --batch-id to list doses", file=sys.stderr)
            sys.exit(1)

        compound_data = df[df["Metadata_compound"] == args.compound]
        if "Metadata_conc_uM" not in df.columns:
            print("Error: Metadata_conc_uM column not found in data", file=sys.stderr)
            sys.exit(1)

        doses = sorted(compound_data["Metadata_conc_uM"].dropna().unique())
        print(f"\nAvailable doses for '{args.compound}':")
        for dose in doses:
            count = len(compound_data[compound_data["Metadata_conc_uM"] == dose])
            print(f"  {dose:8.4f} µM (n={count})")
        sys.exit(0)

    # Get feature columns
    metadata_cols = [col for col in df.columns if col.startswith("Metadata_")]
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    print(f"Found {len(feature_cols)} feature columns and {len(metadata_cols)} metadata columns")

    # Categorize features
    categorized_features = categorize_features(feature_cols, MAIN_GROUPS, verbose=args.verbose)

    # Determine cell line for grit filtering
    cell_lines = df[df[args.id_col] == args.compound]["Metadata_cell_line"].unique()
    cell_line = cell_lines[0] if len(cell_lines) > 0 else None

    # Filter by grit if specified
    grit_filter_doses = None
    if args.min_grit is not None and not args.metrics_path:
        parser.error("--min-grit requires --metrics-path")
    if args.min_grit is not None and cell_line:
        grit_metrics = load_grit_metrics(args.metrics_path, cell_line, args.min_grit)
        if grit_metrics is not None:
            # Create a set of (compound, dose) tuples that pass grit filter
            grit_filter_doses = set()
            for _, row in grit_metrics.iterrows():
                if row['Metadata_compound'] == args.compound:
                    grit_filter_doses.add(row['Metadata_conc_uM'])
            print(f"Grit filter: keeping doses {sorted(grit_filter_doses)}")

    # Filter by dose if specified
    compound_data = df[df[args.id_col] == args.compound]

    # Apply grit filter if applicable
    if grit_filter_doses is not None:
        compound_data = compound_data[compound_data['Metadata_conc_uM'].isin(grit_filter_doses)]
        if compound_data.empty:
            print(f"Error: No doses remain after grit filtering (min_grit={args.min_grit})", file=sys.stderr)
            sys.exit(1)

    doses_data = None
    dose_suffix = ""

    if args.dose is not None:
        if "Metadata_conc_uM" not in df.columns:
            print("Error: Metadata_conc_uM column not found in data", file=sys.stderr)
            sys.exit(1)

        # Filter to specific dose
        dose_data = compound_data[compound_data["Metadata_conc_uM"] == args.dose]
        if dose_data.empty:
            available = sorted(compound_data["Metadata_conc_uM"].dropna().unique())
            print(f"Error: No data found for dose {args.dose} µM", file=sys.stderr)
            print(f"Available doses: {available}", file=sys.stderr)
            sys.exit(1)

        print(f"Filtering to dose: {args.dose} µM")
        compound_data = dose_data
        dose_suffix = f"_{args.dose}uM"
    else:
        # If no dose specified, prepare data for all doses
        if "Metadata_conc_uM" in df.columns:
            unique_doses = sorted(compound_data["Metadata_conc_uM"].dropna().unique())
            if len(unique_doses) > 1:
                print(f"Plotting all {len(unique_doses)} doses: {unique_doses}")
                doses_data = {}
                for dose_val in unique_doses:
                    doses_data[dose_val] = compound_data[compound_data["Metadata_conc_uM"] == dose_val][feature_cols]
                dose_suffix = "_all_doses"

    # Determine output path using cell line
    if cell_line and not compound_data.empty:
        # Create output directory organized by cell line
        if args.output:
            output_dir = Path(args.output) / str(cell_line) / "outputs"
        else:
            output_dir = Path("./outputs") / str(cell_line) / "outputs"

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"radar_{args.compound}{dose_suffix}.png"
        print(f"Output directory: {output_dir}")
    else:
        print(f"Warning: No cell line data found for {args.compound}", file=sys.stderr)
        output_path = None

    # Create radar plot
    try:
        # Use provided range or default to (-5, 5)
        value_range = tuple(args.value_range) if args.value_range else (-5, 5)
        fig = create_radar_plot(
            df=compound_data,
            compound_name=args.compound,
            main_groups=MAIN_GROUPS,
            categorized_features=categorized_features,
            id_col=args.id_col,
            feature_cols=feature_cols,
            palette=args.palette,
            value_range=value_range,
            save_path=str(output_path) if output_path else None,
            batch_id=args.batch_id,
            dose=args.dose,
            doses_data=doses_data
        )
        plt.show()
        if output_path:
            print(f"Radar plot saved to: {output_path}")
        print(f"Radar plot created successfully for: {args.compound}")
    except Exception as e:
        print(f"Error creating plot: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
