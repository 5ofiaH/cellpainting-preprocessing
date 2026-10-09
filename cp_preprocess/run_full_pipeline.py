#!/usr/bin/env python3
"""
Full Cell Painting Analysis Pipeline

Orchestrates complete workflow:
  1. Load from database (or cache) → Level 2.5
  2. Normalize & select features → Level 4b with metrics
  3. Batch correction → Level 4c
  4. Consensus → Level 5

Replaces: 01_data_prep.py + 02_experiment_review.py + 03_cellpainting_pipeline.py

Usage
-----
    python run_full_pipeline.py --config configs/my_project.yaml [--cell-line U2OS]
    python run_full_pipeline.py --config configs/my_project.yaml --skip-data-loading
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config_from_yaml
from .data_loading import load_and_prepare_all_cell_lines
from .pipeline import CellPaintingPipeline


def setup_logging(level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main entry point for full pipeline."""
    parser = argparse.ArgumentParser(
        description="Full Cell Painting Analysis Pipeline (01 + 02-05)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file (e.g., configs/my_project.yaml)",
    )
    parser.add_argument(
        "--cell-line",
        help="Process specific cell line only (optional)",
    )
    parser.add_argument(
        "--skip-data-loading",
        action="store_true",
        help="Skip 01_data_prep step (use pre-computed Level 2.5)",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Run data loading (01) only, skip preprocessing (02-05)",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Run preprocessing (02-05) only, skip data loading (01)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger("pipeline")

    # Load config
    logger.info(f"Loading config from {args.config}...")
    config = load_config_from_yaml(args.config)

    if not config.output.base_dir:
        config.output.base_dir = str(Path.cwd())

    logger.info(f"Project: {config.project_name}")

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 1: Data Loading (01_data_prep equivalent)
    # ─────────────────────────────────────────────────────────────────────────────

    data_by_cellline = {}

    if not args.preprocess_only and not args.skip_data_loading:
        logger.info("\n" + "="*70)
        logger.info("STEP 1: DATA LOADING (01_data_prep)")
        logger.info("="*70)

        try:
            logger.info("Loading Level 2.5 from database/cache...")
            data_by_cellline = load_and_prepare_all_cell_lines(config)

            if not data_by_cellline:
                logger.error("No data loaded!")
                sys.exit(1)

            logger.info(f"✓ Loaded {len(data_by_cellline)} cell line(s)")

            # Save Level 2.5 CSVs
            for cell_line, df_l2 in data_by_cellline.items():
                out_path = (
                    Path(config.output.base_dir) /
                    cell_line /
                    config.output.features_subdir /
                    "level_2_5.csv"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                df_l2.to_csv(out_path, index=False)
                logger.info(f"  Saved {out_path.relative_to(config.output.base_dir)}")

        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 2: Preprocessing (02-05 equivalent)
    # ─────────────────────────────────────────────────────────────────────────────

    if args.data_only:
        logger.info("Data loading complete (--data-only flag set)")
        return

    logger.info("\n" + "="*70)
    logger.info("STEP 2: PREPROCESSING (02-05)")
    logger.info("="*70)

    # Auto-discover Level 2.5 if not loaded in step 1
    if not data_by_cellline:
        logger.info("Auto-discovering Level 2.5 files...")
        base_dir = Path(config.output.base_dir)
        cell_lines = config.metadata.cell_lines or []
        if not cell_lines:
            cell_lines = [d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]

        for cell_line in cell_lines:
            level2_5_path = (
                base_dir / cell_line / config.output.features_subdir / "level_2_5.csv"
            )
            if level2_5_path.exists():
                data_by_cellline[cell_line] = str(level2_5_path)

    # Convert DataFrames to file paths for consistent handling
    for cell_line in list(data_by_cellline.keys()):
        if not isinstance(data_by_cellline[cell_line], str):
            out_path = (
                Path(config.output.base_dir) /
                cell_line /
                config.output.features_subdir /
                "level_2_5.csv"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data_by_cellline[cell_line].to_csv(out_path, index=False)
            data_by_cellline[cell_line] = str(out_path)
            logger.info(f"  Converted DataFrame to {out_path.relative_to(config.output.base_dir)}")

    if not data_by_cellline:
        logger.error("No Level 2.5 files found!")
        sys.exit(1)

    # Filter to specific cell line if requested
    if args.cell_line:
        if args.cell_line not in data_by_cellline:
            logger.error(f"Cell line {args.cell_line} not found")
            sys.exit(1)
        data_by_cellline = {args.cell_line: data_by_cellline[args.cell_line]}

    # Run preprocessing pipeline
    pipeline = CellPaintingPipeline(config)
    failed_lines = []

    for cell_line, data_path in data_by_cellline.items():
        try:
            logger.info(f"\n{'─'*70}")
            logger.info(f"Processing: {cell_line}")
            logger.info(f"{'─'*70}")

            # Handle both DataFrame and file path
            if isinstance(data_path, str):
                data_path = str(data_path)
            else:
                # It's a DataFrame, save temporarily
                temp_path = (
                    Path(config.output.base_dir) /
                    cell_line /
                    config.output.features_subdir /
                    "level_2_5.csv"
                )
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                data_path.to_csv(temp_path, index=False)
                data_path = str(temp_path)

            output_paths = pipeline.run(data_path, cell_line)

            logger.info(f"✓ {cell_line} completed")
            for level, path in output_paths.items():
                logger.info(f"  {level}: {Path(path).name}")

        except Exception as e:
            logger.error(f"✗ {cell_line} failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed_lines.append(cell_line)

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Processed: {len(data_by_cellline) - len(failed_lines)}/{len(data_by_cellline)} cell lines")

    if failed_lines:
        logger.error(f"Failed: {', '.join(failed_lines)}")
        sys.exit(1)
    else:
        logger.info("✓ All steps completed successfully")


if __name__ == "__main__":
    main()
