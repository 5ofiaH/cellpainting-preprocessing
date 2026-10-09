"""
Standalone Level 2.5a/2.5b: site-level normalization + feature selection
(no site→well aggregation) from an existing level_2_5.csv.

Applies the same pre-processing pipeline.run does before Level 3 (metadata
harmonization, manual site drop, drop unannotated sites), then
normalization.normalize_and_select_sites. Writes level_2_5a.parquet and
level_2_5b.parquet (+ site_outliers.csv) next to the input, or to --out-dir.

Usage:
    python run_site_level.py --config config_envtox_u2os_v2.yaml \
        --input /path/to/features_XXXX/level_2_5.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import load_config_from_yaml
from .pipeline import CellPaintingPipeline
from . import normalization, io_utils, site_outliers


class _NoSummary:
    def add_step(self, *a, **k):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True, help="Level 2.5 site-level CSV")
    ap.add_argument("--out-dir", default=None, help="default: directory of --input")
    ap.add_argument("--cell-line", default="U2OS")
    ap.add_argument("--no-outliers", action="store_true", help="skip site_outliers.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("run_site_level")

    config = load_config_from_yaml(args.config)
    pipe = CellPaintingPipeline(config)
    out_dir = Path(args.out_dir or Path(args.input).parent)

    log.info("Loading %s", args.input)
    df = pd.read_csv(args.input, low_memory=False).infer_objects()
    log.info("  %d sites x %d columns", *df.shape)

    df = pipe._harmonize_metadata(df)
    df = pipe._apply_manual_site_drop(df, args.cell_line, _NoSummary())
    pert_col = config.metadata.pert_type_col
    n0 = len(df)
    df = df[df[pert_col].notna()].copy()
    log.info("Dropped %d unannotated sites (no %s)", n0 - len(df), pert_col)

    df_a, meta_a, df_b, meta_b, n_low = normalization.normalize_and_select_sites(df, config)
    log.info("Dropped %d low-count sites", n_low)
    pa = io_utils.save_level_parquet(df_a, meta_a, out_dir, "level_2_5a")
    pb = io_utils.save_level_parquet(df_b, meta_b, out_dir, "level_2_5b")
    log.info("Wrote %s", pa)
    log.info("Wrote %s", pb)

    if not args.no_outliers:
        df_out = site_outliers.compute_site_outliers(df_b, config)
        p = out_dir / "site_outliers.csv"
        df_out.to_csv(p, index=False)
        log.info("Wrote %s (%d flagged / %d evaluable)", p,
                 int(df_out["outlier_flag"].sum()), int(df_out["evaluable"].sum()))


if __name__ == "__main__":
    main()
