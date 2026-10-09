"""
Core pipeline orchestration.
Manages data flow from Level 2 → Level 5 with batch correction and metrics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import PipelineConfig

from . import normalization, batch_correction, consensus, metrics, viability, io_utils, radar_wrapper, site_outliers
from . import profile_metrics as pm
from . import mahalanobis_distance as mdist
from .run_summary import RunSummary

logger = logging.getLogger(__name__)


class CellPaintingPipeline:
    """Main pipeline orchestrator."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logger
        # Remember the *raw*, project-specific compound/conc column names as
        # given in the config. `_harmonize_metadata()` overwrites
        # `config.metadata.compound_col`/`conc_col` with the canonical
        # "Metadata_compound"/"Metadata_conc_uM" names for the duration of a
        # `run()` call (several steps later in the same call rely on the
        # config already pointing at the canonical names). Since one config
        # instance is reused across multiple cell lines in a loop, we must
        # restore the original raw names at the start of each `run()` so the
        # second and subsequent cell lines still get renamed instead of
        # silently skipping harmonization (the rename guard checks
        # `compound_col != "Metadata_compound"`).
        self._raw_compound_col = config.metadata.compound_col
        self._raw_conc_col = config.metadata.conc_col

    # ─────────────────────────────────────────────────────────────────────────
    # Resumable steps. A run executes STEPS[from_step : to_step] in order; each
    # step takes its input from the previous step in memory, or — when the run
    # starts at that step — from disk (an explicit input file, or the level
    # file in input_dir, default = this run's features folder). Every step
    # writes its own output as soon as it finishes.
    # ─────────────────────────────────────────────────────────────────────────
    STEPS = ("level3", "level4a", "level4b", "level4c", "level5", "metrics", "plots")
    # Level files each step reads when it was not produced earlier in the run.
    STEP_INPUTS = {
        "level3": ("level_2_5",),
        "level4a": ("level_3",),
        "level4b": ("level_4a",),
        "level4c": ("level_4b",),
        "level5": ("level_4c", "level_4b"),
        "metrics": ("level_4b",),  # + level_4c / level_5_* if present
        "plots": (),
    }
    # Columns _step_metrics attaches to Level 5 — stripped before re-attaching
    # so a metrics-only rerun replaces them instead of creating _x/_y copies.
    L5_METRIC_COLS = (
        "Metadata_mAP", "Metadata_mAP_corrected_p", "Metadata_mAP_p",
        "Metadata_grit", "Metadata_grit_std",
        "Metadata_Viability_pct", "Metadata_frac_cytotoxic",
    )
    VIABILITY_COLS = (
        "Metadata_Viability_pct", "Metadata_Viability_flag",
        "Metadata_Viability_n_sites", "Metadata_frac_sites_cytotoxic",
        "Metadata_Viability_low_confidence", "Metadata_dmso_ref_outlier_sites",
    )

    def run(
        self,
        input_level2_path: str = None,
        cell_line: str = None,
        from_step: str = "level3",
        to_step: str = "plots",
        input_files: Dict[str, str] = None,
        input_dir: str = None,
        plot_groups=None,
    ) -> Dict[str, Path]:
        """
        Run the pipeline (or a contiguous slice of it) for a cell line.

        Parameters
        ----------
        input_level2_path : str, optional
            Path to the Level 2.5 CSV (needed only when the level3 step runs).
        cell_line : str
            Cell line name (used for output naming)
        from_step, to_step : str
            First / last step to run, from STEPS. Default: the whole pipeline.
        input_files : dict, optional
            {level_name: path} — explicit input files (csv or parquet), e.g.
            {"level_3": "my_edited_level_3.parquet"}. Override input_dir.
        input_dir : str, optional
            Folder holding the level files of an earlier run, used for any
            input not given in input_files. Default: this run's features folder.
        plot_groups : list of str, optional
            Restrict the plots step to these subfolders (e.g. ["01_qc"]).

        Returns
        -------
        dict
            Paths to the files written by this run.
        """
        if from_step not in self.STEPS or to_step not in self.STEPS:
            raise ValueError(f"from_step/to_step must be one of {self.STEPS}")
        i0, i1 = self.STEPS.index(from_step), self.STEPS.index(to_step)
        if i0 > i1:
            raise ValueError(f"from_step '{from_step}' comes after to_step '{to_step}'")
        steps = self.STEPS[i0:i1 + 1]
        resumed = from_step != "level3"

        start_time = datetime.now()
        # Restore the raw compound/conc column names before harmonizing this
        # cell line's data. `self.config` is shared/reused across cell lines
        # in a loop, and `_harmonize_metadata()` permanently rewrites these
        # fields to the canonical names on the first call — without this
        # reset, every cell line after the first would skip renaming.
        self.config.metadata.compound_col = self._raw_compound_col
        self.config.metadata.conc_col = self._raw_conc_col
        self._viability_sites = None
        self._viability_wells = None
        features_dir, plots_dir = self._setup_output_dirs(cell_line)

        inputs = dict(input_files or {})
        if input_level2_path and "level_2_5" not in inputs:
            inputs["level_2_5"] = str(input_level2_path)
        S = {
            "out_dir": features_dir,
            "input_dir": Path(input_dir) if input_dir else features_dir,
            "inputs": inputs,
            "loaded": {},   # level -> source path, for the run summary
            "meta": {},     # level -> metadata frame returned by a step
            "outputs": {},  # level -> written path
        }

        # Traceable, always-updated run summary (RUN_SUMMARY.md + run_summary.json).
        # Written into features_dir (not the cell_line root) so a dated
        # features_subdir keeps a rerun's summary from overwriting an older run's.
        # A resumed run appends to the existing summary instead of replacing it.
        summary = RunSummary(features_dir, cell_line, resume=resumed)
        summary.set_meta(
            project=self.config.project_name,
            normalize_method=self.config.normalization.method,
            batch_correction=self.config.batch_correction.run,
            consensus=self.config.consensus.operation,
        )

        try:
            if resumed or to_step != "plots":
                self.logger.info("[%s] Running steps: %s", cell_line, " → ".join(steps))
            # Load explicitly given inputs up front: fail fast on a bad path, and
            # copy them into the features folder before any step runs.
            for level in list(inputs):
                if level != "level_2_5":
                    self._load_level(S, level)
            if resumed:
                summary.add_step(
                    f"Resumed run: {' → '.join(steps)}",
                    metrics={f"input_{k}": v for k, v in S["loaded"].items()} or {"inputs": "from disk"},
                    notes=f"Steps before '{from_step}' were not rerun; their outputs were "
                          f"read from {S['input_dir']} (or the explicit input files listed).",
                )

            for step in steps:
                getattr(self, f"_step_{step}")(S, cell_line, summary, plot_groups=plot_groups)

            self._write_pipeline_summary(cell_line, S, features_dir)

            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info("[%s] ✓ Complete (%.1fs)", cell_line, elapsed)

        except Exception as e:
            import traceback
            self.logger.error("[%s] ✗ Failed: %s", cell_line, type(e).__name__)
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            raise

        return S["outputs"]

    # ── input / output helpers for resumable steps ─────────────────────────────

    @staticmethod
    def _n_feat(df):
        return len([c for c in df.columns if not c.startswith("Metadata_")])

    @staticmethod
    def _meta_of(df):
        return df[[c for c in df.columns if c.startswith("Metadata_")]].copy()

    @staticmethod
    def _find_level_file(folder: Path, level: str):
        for ext in (".parquet", ".csv"):
            p = Path(folder) / f"{level}{ext}"
            if p.exists():
                return p
        return None

    def _read_table(self, path, level: str) -> pd.DataFrame:
        path = Path(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if level == "level_2_5":
            return pd.read_csv(path, engine="python").infer_objects()
        return pd.read_csv(path).infer_objects()

    def _load_level(self, S: Dict, level: str, required: bool = True):
        """Return a level table from memory, an explicit input, or input_dir.

        A table read from outside the features folder is also saved there under
        its standard name, so the folder stays self-contained for the plots and
        for later resumes. An existing file that would be replaced is kept as
        <level>.replaced_<timestamp>.parquet."""
        if level in S:
            return S[level]
        path = S["inputs"].get(level) or self._find_level_file(S["input_dir"], level)
        if path is None or not Path(path).exists():
            if not required:
                return None
            raise FileNotFoundError(
                f"Input for {level} not found (looked for {path or Path(S['input_dir']) / (level + '.parquet')}). "
                f"Run the earlier step, or pass it with --input {level}=PATH / --input-dir."
            )
        self.logger.info("  Reading %s from %s", level, path)
        df = self._harmonize_metadata(self._read_table(path, level))
        S[level] = df
        S["loaded"][level] = str(path)

        if level == "level_2_5":
            # Too large to copy: symlink it so the 01_qc site plots find it. Any
            # other level_2_5.csv/.parquet in the folder is moved aside (a link
            # is just removed), so the plots use the file this run read.
            src = Path(path).resolve()
            link = Path(S["out_dir"]) / f"level_2_5{Path(path).suffix}"
            for old in (Path(S["out_dir"]) / "level_2_5.csv", Path(S["out_dir"]) / "level_2_5.parquet"):
                if not (old.exists() or old.is_symlink()) or old.resolve() == src:
                    continue
                if old.is_symlink():
                    old.unlink()
                else:
                    backup = old.with_name(f"level_2_5.replaced_{datetime.now():%Y%m%d-%H%M%S}{old.suffix}")
                    old.rename(backup)
                    self.logger.info("  Kept previous %s as %s", old.name, backup.name)
            if not link.exists() and not link.is_symlink():
                try:
                    link.symlink_to(src)
                    self.logger.info("  Linked %s → %s", link, path)
                except OSError as e:
                    self.logger.warning("  Could not link %s into %s: %s", path, S["out_dir"], e)
            return df

        dest = Path(S["out_dir"]) / f"{level}.parquet"
        if Path(path).resolve() != dest.resolve():
            if dest.exists():
                backup = dest.with_name(f"{level}.replaced_{datetime.now():%Y%m%d-%H%M%S}.parquet")
                dest.rename(backup)
                self.logger.info("  Kept previous %s as %s", dest.name, backup.name)
            S["outputs"][level] = io_utils.save_level_parquet(df, None, S["out_dir"], level)
        return df

    def _save_level(self, S: Dict, level: str, df: pd.DataFrame, metadata: pd.DataFrame = None) -> None:
        S[level] = df
        if metadata is not None:
            S["meta"][level] = metadata
        S["outputs"][level] = io_utils.save_level_parquet(df, metadata, S["out_dir"], level)

    def _get_meta(self, S: Dict, level: str) -> pd.DataFrame:
        if level in S["meta"]:
            return S["meta"][level]
        return self._meta_of(self._load_level(S, level))

    # ── steps ──────────────────────────────────────────────────────────────────

    def _prepare_level_2_5(self, df_l2: pd.DataFrame, cell_line: str, summary: RunSummary) -> pd.DataFrame:
        """Pre-aggregation cleanup of the site-level table: manual site drop and
        removal of unannotated wells (harmonization happens on load)."""
        # Manually drop sites identified by eye (e.g. from QC images showing
        # very low cell count) that automated image_qc thresholds missed.
        # Applies here — right after Level 2.5 is loaded — so it covers both
        # run_full_pipeline.py (fresh load) and run_pipeline.py (reading an
        # existing level_2_5.csv) the same way.
        df_l2 = self._apply_manual_site_drop(df_l2, cell_line, summary)

        # Drop wells with no experimental annotation (imaged but absent from
        # the metadata file, e.g. empty/border wells). They have no pert_type,
        # cannot serve as a normalization reference or a treatment, and would
        # otherwise form a phantom NaN compound×dose profile at Level 5.
        pert_col = self.config.metadata.pert_type_col
        if pert_col in df_l2.columns:
            n0 = len(df_l2)
            df_l2 = df_l2[df_l2[pert_col].notna()].copy()
            n_drop = n0 - len(df_l2)
            if n_drop:
                self.logger.info(
                    "[%s] Dropped %d unannotated site rows (no %s)",
                    cell_line, n_drop, pert_col,
                )
        return df_l2

    def _compute_viability(self, S: Dict, df_sites: pd.DataFrame, cell_line: str, summary: RunSummary) -> None:
        """Site-level viability → well rollup; saves viability_sites.parquet."""
        viability_result = viability.compute_viability(df_sites, self.config, log=self.logger)
        self._viability_sites = viability_result["sites"]
        self._viability_wells = viability_result["wells"]
        summary.add_step(
            "Viability (site level → wells)",
            metrics=viability_result["stats"],
            notes="Per-site count vs a cleaned per-plate DMSO reference (low-count DMSO "
                  "outliers dropped via robust Z), rolled up to wells by median-of-ratios. "
                  "Wells below threshold are flagged, not dropped.",
        )

        # Site-level viability table (keys + per-site % / flags) for plotting.
        via_sites = self._viability_sites
        if via_sites is not None and not via_sites.empty:
            candidates = [
                self.config.metadata.barcode_col, self.config.metadata.well_col,
                self.config.metadata.site_col, self.config.metadata.pert_type_col,
                self.config.metadata.compound_col, self.config.metadata.conc_col,
                self.config.metrics.viability_site_count_col,
                "Metadata_Count_nuclei", "Metadata_Cells_per_well_total",
                "Metadata_site_viability_pct", "Metadata_site_viability_flag",
                "Metadata_dmso_ref_outlier",
            ]
            # Dedupe while preserving order (viability_site_count_col may already
            # be "Metadata_Count_nuclei", which would otherwise appear twice).
            seen = set()
            keep = [c for c in candidates
                    if c in via_sites.columns and not (c in seen or seen.add(c))]
            via_path = Path(S["out_dir"]) / "viability_sites.parquet"
            via_sites[keep].to_parquet(via_path, index=False)
            S["outputs"]["viability_sites"] = str(via_path)
            self.logger.info("[%s] Wrote site-level viability → %s", cell_line, via_path.name)

    def _resume_viability(self, S: Dict, cell_line: str, summary: RunSummary) -> None:
        """Viability for a run that did not start at level3: recompute it (with
        the current config thresholds) from viability_sites.parquet, else from
        Level 2.5. If neither exists, the input's own viability columns are kept."""
        path = self._find_level_file(S["input_dir"], "viability_sites")
        if path is not None:
            self.logger.info("[%s] Recomputing viability from %s", cell_line, path)
            df_sites = self._harmonize_metadata(pd.read_parquet(path))
        elif self._load_level(S, "level_2_5", required=False) is not None:
            self.logger.info("[%s] Recomputing viability from Level 2.5", cell_line)
            df_sites = self._prepare_level_2_5(S["level_2_5"], cell_line, summary)
        else:
            self.logger.warning(
                "[%s] No viability_sites.parquet or level_2_5 in %s — keeping any viability "
                "columns already in the input", cell_line, S["input_dir"])
            return
        self._compute_viability(S, df_sites, cell_line, summary)

    def _step_level3(self, S: Dict, cell_line: str, summary: RunSummary, **_) -> None:
        # ─────────────────────────────────────────────────────────────────────
        # Level 2.5 → Level 3: Aggregate sites to wells
        # ─────────────────────────────────────────────────────────────────────
        self.logger.info("[%s] Loading Level 2.5...", cell_line)
        # Harmonize the project's compound/concentration columns to the
        # canonical names used across normalization, consensus and metrics
        # (done by _load_level).
        df_l2 = self._prepare_level_2_5(self._load_level(S, "level_2_5"), cell_line, summary)

        # Record data-loading / image-QC accounting (from data_loading's summary)
        summary.add_step(
            "Data loading (Level 2.5)",
            metrics=self._load_qc_drop_summary(cell_line) or {
                "sites_loaded": len(df_l2),
                "features": self._n_feat(df_l2),
            },
            notes="Site-level profiles after compartment merge + image QC. "
                  "See _image_qc/ for per-site flags and drop accounting.",
        )

        # Extract ID columns from metadata if configured
        self.logger.info("[%s] Checking for ID columns in metadata...", cell_line)
        self._extract_id_columns(df_l2)

        # ─────────────────────────────────────────────────────────────────────
        # Viability (SITE level): computed here, before site→well aggregation,
        # so per-site cell counts drive the toxicity call and low-count sites
        # are NOT pre-dropped (a low count IS the toxicity signal). Rolled up to
        # wells by median-of-ratios and merged onto the well levels downstream.
        # ─────────────────────────────────────────────────────────────────────
        self.logger.info("[%s] Computing site-level viability...", cell_line)
        self._compute_viability(S, df_l2, cell_line, summary)

        # ─────────────────────────────────────────────────────────────────────
        # Level 2.5a / 2.5b (optional, opt-in via config.site_level.run):
        # site-level normalization + feature selection, WITHOUT site→well
        # aggregation. Site-level analogue of Level 4a/4b — one row per
        # imaging site — so outlier sites within a compound×dose replicate
        # group can be inspected before Level 3 smooths them into the well
        # median/mean.
        # ─────────────────────────────────────────────────────────────────────
        if getattr(self.config, "site_level", None) and self.config.site_level.run:
            self._run_site_level(S, df_l2, cell_line, summary)

        self.logger.info("[%s] ▶ Level 3: Aggregating sites → wells...", cell_line)
        n_sites_in = len(df_l2)
        df_l3, metadata_l3 = normalization.aggregate_sites_to_wells(
            df_l2,
            config=self.config,
            cell_line=cell_line,
        )
        self.logger.info("[%s] ✓ Level 3 complete: %d wells", cell_line, len(df_l3))
        summary.add_step(
            "Level 3 — site → well aggregation",
            metrics={
                "sites_in": n_sites_in,
                "low_count_sites_dropped": getattr(
                    normalization.aggregate_sites_to_wells, "last_n_dropped_lowcount", 0),
                "wells_out": len(df_l3),
                "features": self._n_feat(df_l3),
                "operation": self.config.aggregation.operation,
            },
            notes=f"Sites with <{self.config.aggregation.min_dmso_cells_per_site} DMSO / "
                  f"<{self.config.aggregation.min_cells_per_site} treatment cells dropped before aggregation.",
        )
        self._save_level(S, "level_3", df_l3, metadata_l3)
        S.pop("level_2_5", None)  # free the (large) site-level table

    def _run_site_level(self, S: Dict, df_l2: pd.DataFrame, cell_line: str, summary: RunSummary) -> None:
        out_dir = S["out_dir"]
        self.logger.info("[%s] ▶ Level 2.5a/2.5b: site-level normalization + feature selection...", cell_line)
        try:
            df_l25a, meta_l25a, df_l25b, meta_l25b, n_dropped_site = normalization.normalize_and_select_sites(
                df_l2, config=self.config,
            )
            S["outputs"]["level_2_5a"] = io_utils.save_level_parquet(
                df_l25a, meta_l25a, out_dir, "level_2_5a"
            )
            S["outputs"]["level_2_5b"] = io_utils.save_level_parquet(
                df_l25b, meta_l25b, out_dir, "level_2_5b"
            )
            self.logger.info("[%s] ✓ Level 2.5a/2.5b complete: %d sites", cell_line, len(df_l25b))
            summary.add_step(
                "Level 2.5a/2.5b — site-level normalization + feature selection",
                metrics={
                    "sites_in": len(df_l2),
                    "sites_dropped_lowcount": n_dropped_site,
                    "sites_out": len(df_l25b),
                    "features_2_5a": self._n_feat(df_l25a),
                    "features_2_5b": self._n_feat(df_l25b),
                },
                notes="Site-level (pre-aggregation) analogue of Level 4a/4b — same "
                      "per-plate MAD normalization and feature_select, one row per "
                      "imaging site instead of one row per well. For inspecting/"
                      "flagging outlier sites before Level 3 aggregation.",
            )

            # Flag candidate outlier sites within each compound x dose
            # group (distance-to-median + replicate correlation). Writes
            # a CSV for manual review; nothing is dropped automatically.
            self.logger.info("[%s] Flagging site-level outliers...", cell_line)
            try:
                df_outliers = site_outliers.compute_site_outliers(df_l25b, self.config)
                outliers_path = Path(out_dir) / "site_outliers.csv"
                df_outliers.to_csv(outliers_path, index=False)
                S["outputs"]["site_outliers"] = str(outliers_path)
                summary.add_step(
                    "Site outlier flags (Level 2.5b)",
                    metrics={
                        "sites_evaluated": int(df_outliers["evaluable"].sum()),
                        "sites_flagged": int(df_outliers["outlier_flag"].sum()),
                    },
                    notes="Per compound x dose: distance-to-group-median and median "
                          "replicate correlation, both MAD-robust Z-scored. Review "
                          f"{outliers_path.name}, set drop=True on rows to remove, then "
                          "run drop_outlier_sites.py to recompute well-level profiles.",
                )
            except Exception as e:
                self.logger.warning("[%s] Site outlier flagging failed: %s", cell_line, e)
                summary.add_step("Site outlier flags (Level 2.5b)",
                                  metrics={"status": "failed", "error": str(e)})
        except Exception as e:
            self.logger.warning("[%s] Level 2.5a/2.5b failed: %s", cell_line, e)
            summary.add_step(
                "Level 2.5a/2.5b — site-level normalization + feature selection",
                metrics={"status": "failed", "error": str(e)},
            )

    def _step_level4a(self, S: Dict, cell_line: str, summary: RunSummary, **_) -> None:
        # ─────────────────────────────────────────────────────────────────────
        # Level 4a: Per-plate MAD normalization
        # ─────────────────────────────────────────────────────────────────────
        df_l3 = self._load_level(S, "level_3")
        self.logger.info("[%s] ▶ Level 4a: Per-plate normalization...", cell_line)
        n_feat_before_norm = self._n_feat(df_l3)
        df_l4a, metadata_l4a = normalization.normalize_per_plate(
            df_l3,
            self._get_meta(S, "level_3"),
            config=self.config,
        )
        self.logger.info("[%s] ✓ Level 4a complete", cell_line)
        dmso_center_stats = getattr(normalization.normalize_per_plate, "last_dmso_center_stats", {}) or {}
        summary.add_step(
            "Level 4a — per-plate MAD normalization",
            metrics={
                "features_in": n_feat_before_norm,
                "features_out": self._n_feat(df_l4a),
                "invariant_features_dropped": n_feat_before_norm - self._n_feat(df_l4a),
                "method": self.config.normalization.method,
                **{k: v for k, v in dmso_center_stats.items() if k != "dmso_center_per_plate_mean"},
            },
            notes="Normalized to DMSO per plate; globally-invariant features (MAD≈0 on DMSO) dropped. "
                  "DMSO-centering check verifies post-normalization DMSO mean feature value ≈ 0 per plate.",
        )
        self._save_level(S, "level_4a", df_l4a, metadata_l4a)

    def _step_level4b(self, S: Dict, cell_line: str, summary: RunSummary, **_) -> None:
        # ─────────────────────────────────────────────────────────────────────
        # Level 4b: Feature selection + QC
        # ─────────────────────────────────────────────────────────────────────
        df_l4a = self._load_level(S, "level_4a")
        self.logger.info("[%s] ▶ Level 4b: Feature selection...", cell_line)
        n_feat_before_fs = self._n_feat(df_l4a)
        df_l4b, metadata_l4b = normalization.feature_select(
            df_l4a,
            self._get_meta(S, "level_4a"),
            config=self.config,
        )
        self.logger.info("[%s] ✓ Level 4b complete: %d features retained", cell_line, self._n_feat(df_l4b))
        summary.add_step(
            "Level 4b — feature selection",
            metrics={
                "features_in": n_feat_before_fs,
                "features_out": self._n_feat(df_l4b),
                "features_dropped": n_feat_before_fs - self._n_feat(df_l4b),
                "operations": ", ".join(self.config.feature_selection.operations),
                "wells": len(df_l4b),
            },
            notes="pycytominer feature_select (variance, NA, correlation, blocklist, outliers).",
        )

        # Attach the well-level viability rollup (computed at site level)
        # AS METADATA on df_l4b so it is saved and carried through ComBat into
        # Level 4c. Merge on the well key (Barcode × Well).
        if self._viability_wells is None:
            self._resume_viability(S, cell_line, summary)
        self.logger.info("[%s] Attaching viability rollup to Level 4b...", cell_line)
        df_l4b = df_l4b.reset_index(drop=True)
        via_wells = self._viability_wells
        bc, well = self.config.metadata.barcode_col, self.config.metadata.well_col
        if via_wells is not None and not via_wells.empty and {bc, well}.issubset(df_l4b.columns) \
                and {bc, well}.issubset(via_wells.columns):
            take = [bc, well] + [c for c in self.VIABILITY_COLS if c in via_wells.columns]
            # Replace (not duplicate) viability columns an edited input may carry.
            df_l4b = df_l4b.drop(columns=[c for c in take[2:] if c in df_l4b.columns])
            df_l4b = df_l4b.merge(via_wells[take], on=[bc, well], how="left")
        else:
            self.logger.warning("[%s] Could not merge viability on well key — skipping", cell_line)
        self._save_level(S, "level_4b", df_l4b, self._meta_of(df_l4b))

    def _step_level4c(self, S: Dict, cell_line: str, summary: RunSummary, **_) -> None:
        # ─────────────────────────────────────────────────────────────────────
        # Level 4c: batch correction (optional)
        # ─────────────────────────────────────────────────────────────────────
        df_l4b = self._load_level(S, "level_4b")
        metadata_l4b = self._get_meta(S, "level_4b")
        if self.config.batch_correction.run:
            bc_method = self.config.batch_correction.method
            self.logger.info("[%s] ▶ Level 4c: batch correction (%s)...", cell_line, bc_method)
            df_l4c, metadata_l4c = batch_correction.apply_batch_correction(
                df_l4b,
                metadata_l4b.copy(),
                config=self.config,
            )
            self.logger.info("[%s] ✓ Level 4c complete", cell_line)
            summary.add_step(
                "Level 4c — batch correction",
                metrics={
                    "method": bc_method,
                    "wells": len(df_l4c),
                    "features": self._n_feat(df_l4c),
                    "batch_variable": self.config.batch_correction.plate_col,
                },
                notes=f"{bc_method} correction of plate effects in feature space.",
            )
        else:
            self.logger.info("[%s] Skipping batch correction...", cell_line)
            df_l4c = df_l4b.copy()
            metadata_l4c = metadata_l4b.copy()
            summary.add_step("Level 4c — batch correction", metrics={"status": "skipped"})

        # Well row / column covariate regression (method-independent), applied
        # on L4c after any batch method so row/column effects are removed with
        # ComBat, Harmony or Spherize alike. No-op when batch_effect_fix is
        # "barcode" (plate only).
        cov_mode = getattr(self.config.batch_correction, "batch_effect_fix", "barcode")
        if cov_mode and cov_mode != "barcode":
            self.logger.info("[%s] Regressing out row/column covariates (%s)...", cell_line, cov_mode)
            n_feat_pre = self._n_feat(df_l4c)
            df_l4c = batch_correction.regress_out_covariates(df_l4c, self.config, self.logger)
            metadata_l4c = self._meta_of(df_l4c)
            summary.add_step(
                "Row/column covariate regression (L4c)",
                metrics={"wells": len(df_l4c), "features": n_feat_pre, "batch_effect_fix": cov_mode},
                notes="Least-squares removal of well row/column positional effects; "
                      "runs for any batch-correction method (was previously ComBat-only).",
            )
            # NOTE: the dedicated before/after row-column-fix QC plots
            # (plot_row_column_fix_qc / _embeddings) lived only in the
            # now-removed plotting.py and have no plotting_analysis.py
            # equivalent yet. plot_spatial_effects() (plotting_analysis.py,
            # wired into run_analysis_plots.py's 01_qc step) still shows the
            # per-plate row/column PC1 diagnostic on L4b.
            self.logger.info(
                "[%s] Row/column fix QC plot skipped (plotting.py removed; "
                "see plot_spatial_effects in 01_qc/ for a similar diagnostic)",
                cell_line,
            )

        # Within-plate column positional de-trend (control-based), applied on
        # L4c so metrics + consensus use the corrected features.
        if getattr(self.config.batch_correction, "regress_out_column", False):
            self.logger.info("[%s] Regressing out within-plate column effect...", cell_line)
            n_feat_pre = self._n_feat(df_l4c)
            df_l4c = batch_correction.regress_out_column_effect(df_l4c, self.config, self.logger)
            metadata_l4c = self._meta_of(df_l4c)
            summary.add_step(
                "Column de-trend (L4c)",
                metrics={"wells": len(df_l4c), "features": n_feat_pre,
                         "reference": self.config.metrics.negcon_value,
                         "min_negcon_per_column": self.config.batch_correction.column_min_dmso},
                notes="Within-plate column gradient removed using the negcon (DMSO) column "
                      "offsets; features only, metadata unchanged.",
            )
        self._save_level(S, "level_4c", df_l4c, metadata_l4c)

    def _step_level5(self, S: Dict, cell_line: str, summary: RunSummary, **_) -> None:
        # ─────────────────────────────────────────────────────────────────────
        # Level 5: MODZ/Median consensus profiles — TWO variants are produced so
        # batch-correction's effect on the consensus can be compared directly:
        #   • level_5_batch_corrected     — consensus of Level 4c (post-correction)
        #   • level_5_no_batch_correction — consensus of Level 4b (pre-correction)
        # When batch_correction.run is False the two are identical by construction.
        # mAP / GRIT / viability are attached afterwards by the metrics step.
        # ─────────────────────────────────────────────────────────────────────
        df_l4c = self._load_level(S, "level_4c")
        df_l4b = self._load_level(S, "level_4b")
        self.logger.info("[%s] ▶ Level 5: Computing consensus (batch-corrected, from L4c)...", cell_line)
        df_l5, metadata_l5 = consensus.compute_consensus(
            df_l4c,
            self._get_meta(S, "level_4c"),
            config=self.config,
        )
        self.logger.info("[%s] ✓ Level 5 (batch-corrected) complete: %d consensus profiles", cell_line, len(df_l5))

        self.logger.info("[%s] ▶ Level 5: Computing consensus (NON batch-corrected, from L4b)...", cell_line)
        df_l5_nobc, metadata_l5_nobc = consensus.compute_consensus(
            df_l4b,
            self._get_meta(S, "level_4b"),
            config=self.config,
        )
        self.logger.info("[%s] ✓ Level 5 (non batch-corrected) complete: %d consensus profiles",
                         cell_line, len(df_l5_nobc))
        summary.add_step(
            "Level 5 — consensus profiles",
            metrics={
                "wells_in": len(df_l4c),
                "consensus_profiles": len(df_l5),
                "features": self._n_feat(df_l5),
                "operation": self.config.consensus.operation,
                "variants": "batch_corrected (L4c) + no_batch_correction (L4b)",
            },
            notes="One profile per compound × concentration. Two variants: "
                  "batch-corrected (from Level 4c) and non-batch-corrected (from Level 4b).",
        )
        # Level 5 written under explicit, self-describing names so the batch
        # correction status of each consensus is unambiguous.
        self._save_level(S, "level_5_batch_corrected", df_l5, metadata_l5)
        self._save_level(S, "level_5_no_batch_correction", df_l5_nobc, metadata_l5_nobc)

    def _step_metrics(self, S: Dict, cell_line: str, summary: RunSummary, **_) -> None:
        # ─────────────────────────────────────────────────────────────────────
        # Compute metrics on Level 4b (feature-selected, PRE batch correction)
        # mAP and GRIT are quality/phenotype measures of the replicate structure
        # and are computed before ComBat/Harmony/Spherize so they reflect the
        # unmixed per-well signal rather than a batch model's residuals.
        # Note: Metrics are computed but not plotted here to keep pipeline clean.
        # Use run_analysis_plots.py to generate analysis plots after completion.
        # ─────────────────────────────────────────────────────────────────────
        features_dir = S["out_dir"]
        df_l4b = self._load_level(S, "level_4b")
        self.logger.info("[%s] Computing metrics on Level 4b (phenotypic activity, consistency, GRIT)...", cell_line)
        metrics_data = metrics.compute_all_metrics(
            df_l4b,
            config=self.config,
        )
        summary.add_step(
            "Metrics (mAP / GRIT / consistency)",
            metrics=self._summarize_metrics(metrics_data),
            notes="Phenotypic activity (mAP), GRIT and consistency computed on Level 4b "
                  "(feature-selected, pre-batch-correction) data.",
        )
        # Full metric tables, saved alongside the features.
        for data_key, filename in (("activity_map_df", "metrics_map.parquet"),
                                   ("grit_df", "metrics_grit.parquet")):
            mdf = metrics_data.get(data_key)
            if mdf is not None and not getattr(mdf, "empty", True):
                mdf.to_parquet(Path(features_dir) / filename, index=False)
                S["outputs"][filename[:-8]] = str(Path(features_dir) / filename)

        # ─────────────────────────────────────────────────────────────────────
        # Standalone profile_metrics.py (opt-in): biosimilarity (Pearson-based
        # profile matching) + activity detection, on the same Level 4b table.
        # Independent module — writes its own CSV/PNG outputs, does not feed
        # anything back into the pipeline dataframes.
        # ─────────────────────────────────────────────────────────────────────
        if getattr(self.config, "profile_metrics", None) and self.config.profile_metrics.run:
            self._run_profile_metrics(df_l4b, cell_line, features_dir, summary)

        # ─────────────────────────────────────────────────────────────────────
        # Standalone mahalanobis_distance.py (opt-in): OASIS-style per-plate MD
        # from the DMSO centroid. Defaults to Level 4b (feature-selected,
        # pre-batch-correction); config.mahalanobis.level = "4c" runs it on the
        # batch-corrected table instead. Independent module — writes its own
        # CSV, does not feed anything back into the pipeline dataframes.
        # ─────────────────────────────────────────────────────────────────────
        if getattr(self.config, "mahalanobis", None) and self.config.mahalanobis.run:
            md_input = self._load_level(S, "level_4c") if self.config.mahalanobis.level == "4c" else df_l4b
            self._run_mahalanobis(md_input, cell_line, features_dir, summary)

        # Attach mAP / GRIT (computed on L4b) and well viability to BOTH Level 5
        # variants by compound×dose, and rewrite them.
        for level, wells_level in (("level_5_batch_corrected", "level_4c"),
                                   ("level_5_no_batch_correction", "level_4b")):
            df_l5 = self._load_level(S, level, required=False)
            if df_l5 is None:
                self.logger.warning("[%s] %s not found — metric tables saved, not attached", cell_line, level)
                continue
            df_l5 = df_l5.drop(columns=[c for c in self.L5_METRIC_COLS if c in df_l5.columns])
            df_wells = self._load_level(S, wells_level, required=False)
            df_l5 = self._attach_metrics_to_level5(
                df_l5, metrics_data, features_dir, df_wells=df_wells, save_tables=False)
            self._save_level(S, level, df_l5, self._meta_of(df_l5))

    def _step_plots(self, S: Dict, cell_line: str, summary: RunSummary, plot_groups=None, **_) -> None:
        # ─────────────────────────────────────────────────────────────────────
        # Generate plots through the single shared entry point, so plots are
        # identical whether produced inline or via run_analysis_plots.py. Reads
        # every level from the features folder.
        # ─────────────────────────────────────────────────────────────────────
        self.logger.info("[%s] Generating plots...", cell_line)
        try:
            from .run_analysis_plots import generate_plots
            generate_plots(self.config.output.base_dir, cell_line, self.config, groups=plot_groups)
        except Exception as e:
            self.logger.warning(f"  Plot generation failed: {e}")

        plots_subdir_name = self.config.output.plots_subdir
        summary.add_step("Plots", metrics={
            "status": "generated",
            "groups": ", ".join(plot_groups) if plot_groups else "all",
            "location": f"{cell_line}/{plots_subdir_name}/{{01_qc,02_batch,03_controls,04_overview}}"})

    def _attach_metrics_to_level5(
        self,
        df_l5: pd.DataFrame,
        metrics_data: Dict,
        out_dir: Path,
        df_wells: pd.DataFrame = None,
        save_tables: bool = True,
    ) -> pd.DataFrame:
        """
        Merge phenotypic activity (mAP), GRIT and (aggregated) viability onto the
        Level 5 profiles by compound×dose, and (optionally) save the full metric
        tables. ``save_tables`` is set False on the second call so the shared
        metric parquets are written only once.

        All added columns are prefixed with Metadata_ so they are stored as
        annotations rather than being mistaken for morphological features.
        """
        key = [c for c in ["Metadata_compound", "Metadata_conc_uM"] if c in df_l5.columns]
        if not key:
            self.logger.warning("  Level 5 has no compound/dose key — skipping metric merge")
            return df_l5

        specs = [
            ("activity_map_df", "metrics_map.parquet",
             {"mean_average_precision": "Metadata_mAP",
              "corrected_p_value": "Metadata_mAP_corrected_p",
              "p_value": "Metadata_mAP_p"}),
            ("grit_df", "metrics_grit.parquet",
             {"grit": "Metadata_grit",
              "grit_std": "Metadata_grit_std"}),
        ]

        for data_key, filename, col_map in specs:
            mdf = metrics_data.get(data_key)
            if mdf is None or getattr(mdf, "empty", True):
                self.logger.info("  No %s available to attach", data_key)
                continue

            # Save the full metric table (once — skipped on the second L5 variant)
            if save_tables:
                mdf.to_parquet(Path(out_dir) / filename, index=False)

            if not set(key).issubset(mdf.columns):
                continue
            take = {src: dst for src, dst in col_map.items() if src in mdf.columns}
            if not take:
                continue
            merge_cols = key + list(take.keys())
            metric_slice = mdf[merge_cols].drop_duplicates(subset=key).rename(columns=take)
            df_l5 = df_l5.merge(metric_slice, on=key, how="left")
            self.logger.info("  Attached %s → L5 (%s)", data_key, list(take.values()))

        # Aggregate well-level viability to compound×dose (median % + fraction flagged)
        if df_wells is not None and set(key).issubset(df_wells.columns) \
                and "Metadata_Viability_pct" in df_wells.columns:
            via = (
                df_wells.groupby(key, observed=True)
                .agg(
                    Metadata_Viability_pct=("Metadata_Viability_pct", "median"),
                    Metadata_frac_cytotoxic=("Metadata_Viability_flag", "mean"),
                )
                .reset_index()
            )
            df_l5 = df_l5.merge(via, on=key, how="left")
            self.logger.info("  Attached viability → L5 (Metadata_Viability_pct, Metadata_frac_cytotoxic)")

        return df_l5

    def _apply_manual_site_drop(self, df: pd.DataFrame, cell_line: str, summary: RunSummary) -> pd.DataFrame:
        """Drop specific sites listed in config.manual_site_drop.sites (identified
        by eye from QC images, e.g. visibly very low cell count that automated
        image_qc thresholds didn't flag)."""
        cfg = self.config.manual_site_drop
        if not cfg.enabled or not cfg.sites:
            return df

        m = self.config.metadata
        barcode_col, well_col, site_col = m.barcode_col, m.well_col, m.site_col
        missing_cols = [c for c in (barcode_col, well_col, site_col) if c not in df.columns]
        if missing_cols:
            self.logger.warning(
                "[%s] manual_site_drop: columns %s not found in Level 2.5, skipping",
                cell_line, missing_cols,
            )
            return df

        # Entries with a "cell_line" key only apply to that cell line; entries
        # without one apply to every cell line in the project.
        entries = [
            e for e in cfg.sites
            if not e.get("cell_line") or str(e.get("cell_line")) == str(cell_line)
        ]
        if not entries:
            return df

        mask = pd.Series(False, index=df.index)
        matched_counts = []
        for entry in entries:
            # Only constrain on keys actually present in the entry. This lets
            # "site" (± "cell_line") be given alone, matching that site number
            # across every barcode/well for the cell line — e.g. {"site": "3",
            # "cell_line": "U2OS"} drops all U2OS site-3 rows, not one specific
            # plate/well's site 3.
            entry_mask = pd.Series(True, index=df.index)
            if "barcode" in entry:
                entry_mask &= (df[barcode_col].astype(str) == str(entry.get("barcode")))
            if "well" in entry:
                entry_mask &= (df[well_col].astype(str) == str(entry.get("well")))
            if "site" in entry:
                site_val = entry.get("site")
                if isinstance(site_val, (list, tuple, set)):
                    entry_mask &= df[site_col].astype(str).isin([str(s) for s in site_val])
                else:
                    entry_mask &= (df[site_col].astype(str) == str(site_val))
            matched_counts.append((entry, int(entry_mask.sum())))
            mask |= entry_mask

        n_drop = int(mask.sum())
        for entry, n in matched_counts:
            if n == 0:
                self.logger.warning("[%s] manual_site_drop: no rows matched %s", cell_line, entry)

        if n_drop:
            self.logger.info(
                "[%s] manual_site_drop: dropping %d row(s) for %d configured site(s)",
                cell_line, n_drop, len(entries),
            )
        df_out = df[~mask].copy()

        summary.add_step(
            "Manual site drop (Level 2.5)",
            metrics={"sites_configured": len(entries), "rows_dropped": n_drop},
            notes="Sites dropped by explicit config.manual_site_drop.sites entry "
                  "(identified from QC images / low cell count), before any other filtering.",
        )
        return df_out

    def _run_profile_metrics(self, df_l4b: pd.DataFrame, cell_line: str, features_dir: Path, summary: RunSummary) -> None:
        """Run the standalone profile_metrics.py (biosimilarity matching +
        activity detection) on Level 4b and write its outputs under
        <features_dir>/profile_metrics/. Never raises — failures are logged
        and recorded in the run summary so they don't abort the pipeline."""
        cfg = self.config.profile_metrics
        outdir = Path(features_dir) / "profile_metrics"
        outdir.mkdir(parents=True, exist_ok=True)

        m = self.config.metadata
        cols = pm.Columns(
            compound_col=m.compound_col,
            well_col=m.well_col,
            plate_col=m.barcode_col,
            pert_col=m.pert_type_col,
            negcon_value=self.config.metrics.negcon_value,
            poscon_value=self.config.metrics.poscon_value,
            dmso_compound=cfg.dmso_compound,
        )

        result = {}
        try:
            self.logger.info("[%s] Running profile_metrics (biosimilarity + activity)...", cell_line)
            df_clean, kept = pm.clean_features(df_l4b, cols, log=self.logger)
            self.logger.info("[%s] profile_metrics: %d wells x %d clean features", cell_line, len(df_clean), len(kept))

            if cfg.compute_activity:
                tbl = pm.activity_table(df_clean, cols, active_percentile=cfg.active_percentile)
                path = outdir / "profile_activity_table.csv"
                tbl.to_csv(path, index=False)
                result["activity_compounds"] = len(tbl)
                result["activity_active_count"] = int(tbl["active"].sum())
                self.logger.info("[%s]   %d/%d compounds active -> %s", cell_line, result["activity_active_count"], len(tbl), path.name)

            cons_df = None
            if cfg.compute_matching:
                cons_df = pm.consensus(df_clean, cols)
                bmat = pm.biosimilarity_matrix(cons_df, cols)
                path = outdir / "profile_biosimilarity_matrix.csv"
                bmat.to_csv(path)
                result["biosimilarity_compounds"] = bmat.shape[0]
                self.logger.info("[%s]   biosimilarity %dx%d -> %s", cell_line, *bmat.shape, path.name)

            if cfg.query_compound:
                if cons_df is None:
                    cons_df = pm.consensus(df_clean, cols)
                sim = pm.find_similar(cons_df, cfg.query_compound, threshold=cfg.similarity_threshold, cols=cols)
                sim.to_csv(outdir / f"profile_similar_to_{pm._slug(cfg.query_compound)}.csv", index=False)
                result["query_similar_hits"] = len(sim)
                poscons = pm._poscon_compounds(df_clean, cols)
                if poscons:
                    mc = pm.match_to_controls(cons_df, cfg.query_compound, poscons, cols=cols)
                    mc.to_csv(outdir / f"profile_match_controls_{pm._slug(cfg.query_compound)}.csv", index=False)
                if cfg.plot_profile:
                    overlay = [cfg.query_compound] + poscons
                    fig = pm.plot_zscore_profiles(
                        df_clean, overlay, cols=cols, top_n_var=cfg.top_n_var,
                        outpath=str(outdir / f"profile_zscore_{pm._slug(cfg.query_compound)}.png"),
                        title=f"Z-score profile - {cfg.query_compound} vs positive controls + DMSO",
                    )
                    import matplotlib.pyplot as plt
                    plt.close(fig)

            summary.add_step(
                "Profile metrics (biosimilarity + activity, Level 4b)",
                metrics=result,
                notes=f"Standalone profile_metrics.py, outputs in features/profile_metrics/.",
            )
        except Exception as e:
            self.logger.warning("[%s] profile_metrics failed: %s", cell_line, e)
            summary.add_step("Profile metrics (biosimilarity + activity, Level 4b)",
                              metrics={"status": "failed", "error": str(e)})

    def _run_mahalanobis(self, df_level: pd.DataFrame, cell_line: str, features_dir: Path, summary: RunSummary) -> None:
        """Run the standalone mahalanobis_distance.py (OASIS-style per-plate MD
        from the DMSO centroid) and write <features_dir>/mahalanobis_distances.csv.
        Never raises — failures are logged and recorded in the run summary so
        they don't abort the pipeline."""
        cfg = self.config.mahalanobis
        outdir = Path(features_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        m = self.config.metadata
        plate_col = self.config.batch_correction.plate_col
        # Canonical names pipeline.py harmonizes to at Level 3+ (NOT the raw
        # project-specific config.metadata.compound_col/conc_col).
        compound_col = "Metadata_compound"
        conc_col = "Metadata_conc_uM"
        pert_col = self.config.metrics.pert_type_col

        result = {}
        try:
            for col in (plate_col, compound_col, conc_col, pert_col):
                if col not in df_level.columns:
                    raise ValueError(f"Column '{col}' not found in Level {cfg.level}")

            self.logger.info("[%s] Running mahalanobis_distance (Level %s, OASIS-style)...", cell_line, cfg.level)
            df_md = mdist.compute_mahalanobis_for_df(
                df_level,
                plate_col=plate_col,
                compound_col=compound_col,
                conc_col=conc_col,
                pert_col=pert_col,
                negcon_value=self.config.metrics.negcon_value,
                poscon_value=self.config.metrics.poscon_value,
                compartments=cfg.compartments,
                channel_markers=mdist.DEFAULT_CHANNEL_MARKERS,
                cover_var=cfg.cover_var,
                min_dmso_wells=cfg.min_dmso_wells,
                drop_poscon=not cfg.keep_poscon,
                seed=cfg.seed,
            )
            path = outdir / "mahalanobis_distances.csv"
            df_md.to_csv(path, index=False)
            result["rows"] = len(df_md)
            result["categories"] = sorted(df_md["category"].unique().tolist())
            self.logger.info("[%s]   %d MD rows (%d categories) -> %s", cell_line, len(df_md), len(result["categories"]), path.name)

            summary.add_step(
                f"Mahalanobis distance (OASIS-style, Level {cfg.level})",
                metrics=result,
                notes=f"Standalone mahalanobis_distance.py, output in features/mahalanobis_distances.csv.",
            )
        except Exception as e:
            self.logger.warning("[%s] mahalanobis_distance failed: %s", cell_line, e)
            summary.add_step(f"Mahalanobis distance (OASIS-style, Level {cfg.level})",
                              metrics={"status": "failed", "error": str(e)})

    def _load_qc_drop_summary(self, cell_line: str) -> Dict:
        """Read the data-loading QC drop-accounting CSV, if present."""
        path = Path(self.config.output.base_dir) / "_image_qc" / f"qc_drop_summary_{cell_line}.csv"
        if not path.exists():
            return {}
        try:
            row = pd.read_csv(path).iloc[0].to_dict()
            row.pop("cell_line", None)
            return {k: (int(v) if isinstance(v, float) and float(v).is_integer() else v)
                    for k, v in row.items() if pd.notna(v)}
        except Exception:
            return {}

    def _summarize_metrics(self, metrics_data: Dict) -> Dict:
        """Compact numeric summary of the metric tables for the run summary."""
        out: Dict = {}
        mdf = metrics_data.get("activity_map_df")
        if mdf is not None and not getattr(mdf, "empty", True):
            out["mAP_compound_dose_groups"] = len(mdf)
            if "mean_average_precision" in mdf.columns:
                out["mAP_median"] = float(mdf["mean_average_precision"].median())
            if "below_corrected_p" in mdf.columns:
                out["mAP_active_pct"] = float(100 * mdf["below_corrected_p"].mean())
        gdf = metrics_data.get("grit_df")
        if gdf is not None and not getattr(gdf, "empty", True) and "grit" in gdf.columns:
            out["grit_median"] = float(gdf["grit"].median())
            out["grit_gt1_count"] = int((gdf["grit"] > 1).sum())
        return out

    def _harmonize_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename the project's compound / concentration columns to the canonical
        names used throughout the pipeline (Metadata_compound, Metadata_conc_uM).

        pycytominer metrics (mAP, GRIT) and the consensus step expect these
        canonical names. After renaming, the config is pointed at the canonical
        names so every downstream step is consistent.
        """
        rename = {}
        compound_col = self.config.metadata.compound_col
        conc_col = self.config.metadata.conc_col

        if compound_col in df.columns and compound_col != "Metadata_compound":
            rename[compound_col] = "Metadata_compound"
        if conc_col in df.columns and conc_col != "Metadata_conc_uM":
            rename[conc_col] = "Metadata_conc_uM"

        if rename:
            df = df.rename(columns=rename)
            self.logger.info("  Harmonized metadata columns: %s", rename)

        # Point config at the canonical names for all downstream steps.
        self.config.metadata.compound_col = "Metadata_compound"
        self.config.metadata.conc_col = "Metadata_conc_uM"
        return df

    def _extract_id_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Extract ID columns from metadata if configured.

        Returns a dict mapping ID type to the column name in the dataframe.
        If ID columns are specified in config but not found in data, logs warning.
        """
        id_columns = {}

        if self.config.metadata.qc_id_col and self.config.metadata.qc_id_col in df.columns:
            id_columns["qc_id"] = self.config.metadata.qc_id_col
            self.logger.info(f"  Found QC ID column: {self.config.metadata.qc_id_col}")
        elif self.config.metadata.qc_id_col:
            self.logger.warning(f"  QC ID column not found: {self.config.metadata.qc_id_col}")

        if self.config.metadata.feat_id_col and self.config.metadata.feat_id_col in df.columns:
            id_columns["feat_id"] = self.config.metadata.feat_id_col
            self.logger.info(f"  Found feature ID column: {self.config.metadata.feat_id_col}")
        elif self.config.metadata.feat_id_col:
            self.logger.warning(f"  Feature ID column not found: {self.config.metadata.feat_id_col}")

        if self.config.metadata.plate_acq_id_col and self.config.metadata.plate_acq_id_col in df.columns:
            id_columns["plate_acq_id"] = self.config.metadata.plate_acq_id_col
            self.logger.info(f"  Found plate acquisition ID column: {self.config.metadata.plate_acq_id_col}")
        elif self.config.metadata.plate_acq_id_col:
            self.logger.warning(f"  Plate acquisition ID column not found: {self.config.metadata.plate_acq_id_col}")

        return id_columns

    def _setup_output_dirs(self, cell_line: str) -> tuple:
        """Create and return output directories for cell line.

        Returns
        -------
        tuple
            (features_dir, plots_dir) paths
        """
        base = Path(self.config.output.base_dir)
        # Create both features and plots directories
        features_dir = base / cell_line / self.config.output.features_subdir
        features_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = base / cell_line / self.config.output.plots_subdir
        plots_dir.mkdir(parents=True, exist_ok=True)
        return features_dir, plots_dir

    def _write_pipeline_summary(
        self,
        cell_line: str,
        S: Dict,
        out_dir: Path,
    ) -> None:
        """Write pipeline run summary. Statistics/outputs from levels this run
        did not touch are kept from the existing file (resumed runs)."""
        # Written into out_dir (features_dir) itself, not its parent, so a dated
        # features_subdir keeps a rerun's summary from overwriting an older run's.
        summary_path = out_dir / "pipeline_summary.json"
        old = {}
        if summary_path.exists():
            try:
                old = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                old = {}

        stats = dict(old.get("statistics", {}))
        for key, level in (("level_3_wells", "level_3"), ("level_4b_wells", "level_4b"),
                           ("level_5_profiles", "level_5_batch_corrected")):
            if level in S:
                stats[key] = int(len(S[level]))
        if "level_4b" in S:
            l4b = S["level_4b"]
            stats["level_4b_viability_flagged"] = int(
                l4b["Metadata_Viability_flag"].sum() if "Metadata_Viability_flag" in l4b else 0
            )

        summary = {
            "cell_line": cell_line,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "normalize_method": self.config.normalization.method,
                "batch_correction": self.config.batch_correction.run,
                "consensus_operation": self.config.consensus.operation,
            },
            "statistics": stats,
            "outputs": {**old.get("outputs", {}), **{k: str(v) for k, v in S["outputs"].items()}},
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Summary saved to %s", summary_path)
