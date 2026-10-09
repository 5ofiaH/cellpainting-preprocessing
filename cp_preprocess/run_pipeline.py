#!/usr/bin/env python3
"""
Main entry point for the Cell Painting Preprocessing Pipeline.

This script orchestrates the complete workflow:
  Level 2 → 3 → 4a → 4b → 4c → 5

All project-specific settings are configured via YAML config files.
No code modifications are needed for different projects.

Usage
-----
    python run_pipeline.py --config config_envtox_u2os.yaml [--cell-line U2OS]
    python run_pipeline.py --config config_dnt_hog.yaml

Running part of the pipeline
----------------------------
Steps, in order:
    level3   Level 2.5 → 3   viability, (opt.) site level, site → well aggregation
    level4a  Level 3 → 4a    per-plate normalization
    level4b  Level 4a → 4b   feature selection + viability attach
    level4c  Level 4b → 4c   batch correction (+ row/column fixes)
    level5   Level 4b/4c → 5 consensus (both variants)
    metrics  Level 4b        mAP / GRIT / consistency (+ profile_metrics,
                             mahalanobis if enabled), attached to Level 5
    plots                    plots from the files in the features folder

    --from-step STEP / --to-step STEP   run a contiguous slice (default: all)
    --only STEP                         run one step
    --level2-5 FILE                     Level 2.5 file to start from (csv/parquet;
                                        {cell_line} allowed) instead of the one
                                        in the features folder
    --input FILE                        input for the first step (csv/parquet);
                                        or LEVEL=FILE, repeatable, for any level
    --input-dir DIR                     earlier run's features folder to read
                                        missing inputs from ({cell_line} allowed;
                                        default: the config's features folder)
    --plots qc,metrics                  plot groups to (re)generate

The same can be set in the config's run_steps: section; flags override it.

Inputs not produced in the run are read from <input-dir>/<level>.parquet.
Outputs always go to the config's features folder; an input read from
elsewhere is copied there under its standard name (an existing file is kept
as <level>.replaced_<timestamp>.parquet). Tip: set a new
output.features_subdir/plots_subdir to keep the original run untouched.

    # edited Level 3 → rerun normalization onwards
    python run_pipeline.py --config cfg.yaml --cell-line U2OS \\
        --from-step level4a --input my_level_3_edited.parquet
    # only mAP / GRIT on the existing Level 4b
    python run_pipeline.py --config cfg.yaml --only metrics
    # only the QC plots
    python run_pipeline.py --config cfg.yaml --only plots --plots qc

Example config file: config_base.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import load_config_from_yaml, PipelineConfig
from .pipeline import CellPaintingPipeline


def setup_logging(level=logging.INFO):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_input_files(config: PipelineConfig, require_level2: bool = True,
                     input_dir: str = None) -> dict:
    """
    Locate Level 2.5 input files for each cell line.

    Parameters
    ----------
    config : PipelineConfig
    require_level2 : bool
        If False (run starts after level3), cell lines without a
        level_2_5.csv are kept, with path None.
    input_dir : str, optional
        Fallback folder ({cell_line} allowed) searched when the features
        folder has no level_2_5.csv — e.g. a new dated run reading the
        Level 2.5 of an earlier run.

    Returns
    -------
    dict
        {cell_line_name: path_to_level2_5_csv or None}
    """
    input_files = {}

    # Look for level_2_5.csv in each cell line folder
    base_dir = Path(config.output.base_dir)

    cell_lines = config.metadata.cell_lines or []
    if not cell_lines:
        # Auto-discover from subdirectories
        for subdir in base_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("_"):
                cell_lines.append(subdir.name)

    for cell_line in cell_lines:
        level2_5_path = (
            base_dir / cell_line / config.output.features_subdir / "level_2_5.csv"
        )
        if not level2_5_path.exists() and input_dir:
            alt = Path(input_dir.format(cell_line=cell_line)) / "level_2_5.csv"
            if alt.exists():
                level2_5_path = alt
        if level2_5_path.exists():
            input_files[cell_line] = str(level2_5_path)
        elif not require_level2:
            input_files[cell_line] = None
        else:
            logging.warning(
                f"No level_2_5.csv found for {cell_line} at {level2_5_path}"
                + ("" if input_dir else " — for a new (e.g. dated) output folder, set "
                   "run_steps.input_dir / --input-dir to the folder that has it"))

    return input_files


def parse_inputs(values, from_step: str) -> dict:
    """--input values → {level_name: path}. A bare FILE is the input of the
    first step; LEVEL=FILE names the level ("level_3", "level_4b", ...)."""
    first = CellPaintingPipeline.STEP_INPUTS[from_step]
    out = {}
    for v in values:
        level, sep, path = v.partition("=")
        if not sep:
            if not first:
                raise ValueError(f"--input FILE: step '{from_step}' reads no level file; "
                                 "use LEVEL=FILE")
            level, path = first[0], v
        out[level.strip()] = path
    return out


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cell Painting Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file (e.g., config_envtox_u2os.yaml)",
    )
    parser.add_argument(
        "--cell-line",
        help="Process specific cell line only (optional)",
    )
    steps = list(CellPaintingPipeline.STEPS)
    parser.add_argument("--from-step", choices=steps,
                        help="First step to run (default: config run_steps.from_step, else level3)")
    parser.add_argument("--to-step", choices=steps,
                        help="Last step to run (default: config run_steps.to_step, else plots)")
    parser.add_argument("--only", choices=steps,
                        help="Run just this step (same as --from-step X --to-step X)")
    parser.add_argument("--level2-5", dest="level2_5", metavar="FILE",
                        help="Level 2.5 file (csv/parquet) for the level3 step, instead of "
                             "<features>/level_2_5.csv ({cell_line} is replaced). "
                             "Same as --input level_2_5=FILE.")
    parser.add_argument("--input", action="append", default=[], metavar="[LEVEL=]FILE",
                        help="Input file for the first step, or LEVEL=FILE (e.g. "
                             "level_4b=edited.parquet). Repeatable.")
    parser.add_argument("--input-dir",
                        help="Folder with an earlier run's level files ({cell_line} "
                             "is replaced). Default: the config's features folder.")
    parser.add_argument("--plots",
                        help="Plot groups for the plots step: qc, batch, controls, "
                             "overview, metrics, site_level (default: all)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger("pipeline")

    # Load config
    logger.info(f"Loading config from {args.config}...")
    config = load_config_from_yaml(args.config)

    if not config.output.base_dir:
        config.output.base_dir = str(Path.cwd())

    # Step selection: command line overrides config.run_steps.
    rs = config.run_steps
    from_step = args.only or args.from_step or rs.from_step
    to_step = args.only or args.to_step or rs.to_step
    for name, val in (("from_step", from_step), ("to_step", to_step)):
        if val not in steps:
            parser.error(f"run_steps.{name} '{val}' is not one of {steps}")
    if steps.index(from_step) > steps.index(to_step):
        parser.error(f"from-step {from_step} comes after to-step {to_step}")
    try:
        explicit_inputs = {**(rs.input_files or {}), **parse_inputs(args.input, from_step)}
    except ValueError as e:
        parser.error(str(e))
    level2_5 = args.level2_5 or rs.level2_5_file
    if level2_5:
        explicit_inputs["level_2_5"] = level2_5
    input_dir = args.input_dir or rs.input_dir
    plot_groups = args.plots.split(",") if args.plots else rs.plot_groups

    logger.info(f"Project: {config.project_name}")
    logger.info(f"Output directory: {config.output.base_dir}")

    # Find input files
    needs_level2 = from_step == "level3" and "level_2_5" not in explicit_inputs
    logger.info("Searching for Level 2.5 input files...")
    input_files = find_input_files(config, require_level2=needs_level2, input_dir=input_dir)

    if not input_files:
        logger.error("No Level 2.5 input files found!" if needs_level2 else "No cell lines found!")
        sys.exit(1)

    logger.info(f"Found {len(input_files)} cell line(s): {list(input_files.keys())}")

    # Filter to specific cell line if requested
    if args.cell_line:
        if args.cell_line not in input_files:
            logger.error(f"Cell line {args.cell_line} not found in input files")
            sys.exit(1)
        input_files = {args.cell_line: input_files[args.cell_line]}
    if len(input_files) > 1 and any("{cell_line}" not in p for p in explicit_inputs.values()):
        logger.error("Input files belong to one cell line: add --cell-line "
                     "(or put {cell_line} in the path)")
        sys.exit(1)
    for cl in input_files:
        for level, p in explicit_inputs.items():
            if not Path(p.format(cell_line=cl)).exists():
                logger.error(f"Input file for {level} not found: {p.format(cell_line=cl)}")
                sys.exit(1)
    if from_step != steps[0] or to_step != steps[-1]:
        logger.info(f"Steps: {from_step} → {to_step}")

    # Run pipeline
    pipeline = CellPaintingPipeline(config)

    failed_lines = []
    for cell_line, input_path in input_files.items():
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing: {cell_line}")
            logger.info(f"{'='*70}")

            output_paths = pipeline.run(
                input_path, cell_line,
                from_step=from_step, to_step=to_step,
                input_files={k: v.format(cell_line=cell_line) for k, v in explicit_inputs.items()},
                input_dir=input_dir.format(cell_line=cell_line) if input_dir else None,
                plot_groups=plot_groups,
            )

            logger.info(f"✓ {cell_line} completed successfully")
            for level, path in output_paths.items():
                logger.info(f"  {level}: {path}")

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
    logger.info(f"Processed: {len(input_files) - len(failed_lines)}/{len(input_files)} cell lines")

    if failed_lines:
        logger.error(f"Failed: {', '.join(failed_lines)}")
        sys.exit(1)
    else:
        logger.info("✓ All cell lines completed successfully")


if __name__ == "__main__":
    main()
