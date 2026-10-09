#!/usr/bin/env python3
"""
Drop manually-reviewed outlier sites and recompute Level 3+ well-level
profiles without them.

Reads site_outliers.csv (written by the pipeline when config.site_level.run
is on — see site_outliers.py), takes the rows where you've set drop=True
after inspecting the CSV and the PCA/heatmap plots in <cell_line>/plots/06_site_level/,
and reruns the pipeline with those exact sites added to
config.manual_site_drop (the same mechanism already used for sites identified
by eye from QC images) — so Level 3 (and everything downstream) is recomputed
excluding them, with no new dropping logic.

Usage
-----
    python drop_outlier_sites.py --config configs/my_project.yaml \
        --cell-line U2OS --input /path/to/level_2_5.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import load_config_from_yaml
from .pipeline import CellPaintingPipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Project YAML config")
    ap.add_argument("--cell-line", required=True)
    ap.add_argument("--input", required=True, help="Level 2.5 CSV (same input used for the original run)")
    ap.add_argument("--outliers-csv", help="Defaults to <base_dir>/<cell_line>/<features_subdir>/site_outliers.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("drop_outlier_sites")

    config = load_config_from_yaml(args.config)

    outliers_csv = Path(args.outliers_csv) if args.outliers_csv else (
        Path(config.output.base_dir) / args.cell_line / config.output.features_subdir / "site_outliers.csv"
    )
    if not outliers_csv.exists():
        raise FileNotFoundError(f"{outliers_csv} not found — run the pipeline with site_level.run: true first")

    df = pd.read_csv(outliers_csv)
    if "drop" not in df.columns:
        raise ValueError(f"{outliers_csv} has no 'drop' column")
    to_drop = df[df["drop"] == True]  # noqa: E712 (explicit bool compare reads clearer here)
    if to_drop.empty:
        log.info("No rows marked drop=True in %s — nothing to do", outliers_csv)
        return

    m = config.metadata
    new_entries = [
        {"barcode": str(row[m.barcode_col]), "well": str(row[m.well_col]),
         "site": str(row[m.site_col]), "cell_line": args.cell_line}
        for _, row in to_drop.iterrows()
        if m.barcode_col in row and m.well_col in row and m.site_col in row
    ]
    log.info("Dropping %d site(s) flagged in %s", len(new_entries), outliers_csv.name)

    config.manual_site_drop.enabled = True
    config.manual_site_drop.sites = list(config.manual_site_drop.sites) + new_entries

    pipeline = CellPaintingPipeline(config)
    output_paths = pipeline.run(args.input, args.cell_line)

    log.info("Recomputed Level 3+ profiles for %s without %d outlier site(s):", args.cell_line, len(new_entries))
    for level, path in output_paths.items():
        log.info("  %s -> %s", level, path)


if __name__ == "__main__":
    main()
