"""
Configuration schema and management for the cell painting pipeline.
All settings can be overridden via YAML configs per project.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database connection and query settings."""
    uri: str = ""
    feature_analysis_type: str = "cp-features"
    qc_analysis_type: str = "cp-qc"
    use_db: bool = False  # If False, use pre-computed parquets/CSVs


@dataclass
class DataLoadingConfig:
    """Data loading (01_data_prep) settings."""
    skip_loading: bool = False  # If True, assume Level 2.5 already exists
    load_compartments: List[str] = field(default_factory=lambda: ["nuclei", "cells", "cytoplasm"])
    merge_compartments: bool = True
    # Cache the expensive per-plate compartment merge to parquet. On re-run the
    # cached site-level table is loaded instead of re-reading/merging cell-level
    # parquets from the database results (minutes → seconds).
    use_cache: bool = True
    cache_subdir: str = "_feature_cache"
    refresh_cache: bool = False  # force recompute even if a cache file exists
    # Compartment file names in database/cache
    compartment_files: Dict[str, str] = field(default_factory=lambda: {
        "nuclei": "featICF_nuclei.parquet",
        "cells": "featICF_cells.parquet",
        "cytoplasm": "featICF_cytoplasm.parquet",
    })


@dataclass
class ImageQCConfig:
    """Image-level QC settings (blur/focus + low cell count site flagging)."""
    run: bool = True  # Query cp-qc analyses and compute image QC flags
    apply_filter: bool = True  # Drop flagged sites from features (else just flag)
    # ImageQuality metrics to check (robust Z-score across sites per plate)
    qc_metrics: List[str] = field(default_factory=lambda: [
        "FocusScore",
        "PowerLogLogSlope",
        "MaxIntensity",
        "MeanIntensity",
    ])
    z_threshold: float = 5.0  # robust |Z| cutoff for outlier sites
    min_nuclei_per_site: int = 20  # sites below this nuclei count are flagged
    count_col: str = "Count_nuclei"  # real per-site cell count column in QC parquet

    # ── DMSO (negcon) site QC ────────────────────────────────────────────────
    # DMSO sites define the normalization reference, so problematic DMSO sites
    # (abnormally low cell count) are flagged/dropped separately and more strictly.
    dmso_site_qc: bool = True  # run DMSO-specific site QC after metadata merge
    dmso_min_cells_per_site: int = 30  # DMSO sites below this count are dropped
    dmso_count_z_threshold: float = 3.5  # also drop DMSO sites with robust Z below -this
    drop_dmso_outliers: bool = True  # drop flagged DMSO sites (else just flag)


@dataclass
class ManualSiteDropConfig:
    """Manually drop specific imaging sites identified by eye (e.g. from QC
    images showing very low cell count) that automated image_qc thresholds
    didn't catch. Applied once, right after Level 2.5 is loaded — so it takes
    effect whether Level 2.5 came from run_full_pipeline's fresh load or from
    a pre-existing level_2_5.csv (run_pipeline.py's normal entry point)."""
    enabled: bool = False
    # Each entry identifies rows to drop. Keys match the columns named in
    # metadata.barcode_col / well_col / site_col (default: Metadata_Barcode,
    # Metadata_Well, Metadata_Site) — only the keys present in the entry are
    # constrained, so a "site"-only entry (optionally with "cell_line") drops
    # that site number everywhere, across every barcode/well. "site" may be a
    # single value or a list to drop several site numbers in one entry.
    # Optional "cell_line" key restricts the entry to one cell line (matched
    # against the cell_line argument the pipeline is run with); omit it to
    # apply to every cell line in the project.
    # Example:
    #   sites:
    #     - {barcode: "PB000123", well: "C04", site: "3"}                     # one site, all cell lines
    #     - {barcode: "PB000123", well: "C04", site: "7", cell_line: "U2OS"}  # one site, U2OS only
    #     - {site: [3, 6], cell_line: "U2OS"}  # drop every site 3 and 6 row, any barcode/well, U2OS only
    sites: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class MetadataConfig:
    """Metadata column names and locations."""
    file_path: str = ""
    sep: str = ","
    # Column name mappings (what they're called in your CSV)
    barcode_col: str = "Metadata_Barcode"
    well_col: str = "Metadata_Well"
    cell_line_col: str = "Metadata_CellLine"
    compound_col: str = "Metadata_Compound_Name"
    conc_col: str = "Metadata_conc_uM"
    pert_type_col: str = "Metadata_pert_type"
    project_col: str = "Metadata_Project"
    site_col: str = "Metadata_Site"

    # ID columns from metadata (optional: if present, used instead of overrides)
    qc_id_col: Optional[str] = None  # e.g., "Metadata_pipeline_id_qc"
    feat_id_col: Optional[str] = None  # e.g., "Metadata_pipeline_id_feat_texture"
    plate_acq_id_col: Optional[str] = None  # e.g., "Metadata_plate_acquisition_id"

    # Optional: harmonize cell line names
    cell_line_rename: Dict[str, str] = field(default_factory=dict)
    # Optional: cell lines to process (None = all)
    cell_lines: Optional[List[str]] = None
    plate_name_overrides: Dict[str, str] = field(default_factory=dict)
    plate_acq_id_overrides: Dict[str, int] = field(default_factory=dict)


@dataclass
class AggregationConfig:
    """Site → well aggregation settings."""
    operation: str = "median"  # "mean" or "median"
    min_cells_per_site: int = 20
    min_dmso_cells_per_site: int = 20
    min_cells_per_well: int = 50


@dataclass
class NormalizationConfig:
    """Level 4a MAD normalization settings."""
    method: str = "mad_robustize"  # mad_robustize | standardize | robustize | spherize
    samples_query: str = "Metadata_pert_type == 'negcon'"
    invariant_filter_reference_query: str = "Metadata_pert_type == 'negcon'"
    mad_threshold: float = 0.001
    clip_min: float = -10.0
    clip_max: float = 10.0
    # Post-normalization QC: DMSO wells should have ~0 mean feature value per
    # plate (that's what mad_robustize against the DMSO reference guarantees
    # when it succeeds). Flag plates where |mean| exceeds this tolerance.
    dmso_center_tolerance: float = 0.5

    # Blocklist features to drop
    blocklist_patterns: List[tuple] = field(default_factory=lambda: [
        ("Granularity_12", "_nucleus"),
        ("Correlation_Manders", "_nucleus"),
        ("Correlation_Manders", "_cytoplasm"),
        ("Correlation_RWC", "_nucleus"),
        ("Granularity_14", "_nucleus"),
        ("Granularity_15", "_nucleus"),
        ("Granularity_16", "_nucleus"),
        ("Granularity_3", "_cytoplasm"),
        ("Granularity_5", "_cytoplasm"),
        ("Granularity_4", "_cytoplasm"),
    ])


@dataclass
class FeatureSelectionConfig:
    """Level 4b feature selection settings."""
    operations: List[str] = field(default_factory=lambda: [
        "variance_threshold",
        "drop_na_columns",
        "correlation_threshold",
        "blocklist",
        "drop_outliers",
    ])


@dataclass
class BatchCorrectionConfig:
    """Level 4c batch correction settings."""
    run: bool = True
    # Which correction to apply. All operate in feature space and return a table
    # with the same metadata + feature columns, so downstream levels are agnostic:
    #   "spherize" — pycytominer ZCA whitening fit on the DMSO (negcon) reference,
    #                decorrelating the technical covariance shared by all wells
    #                (default — closest to the standard profiling-recipe/LINCS
    #                normalization philosophy: no cross-plate statistical model)
    #   "combat"   — scanpy ComBat: per-plate location/scale correction
    #   "harmony"  — harmonypy: iterative soft-kmeans batch integration (plate = batch)
    method: str = "spherize"
    plate_col: str = "Metadata_Barcode"  # batch variable (always the plate)

    # Well row/column positional effect removal, applied (method-independently)
    # after batch correction. The batch variable is always the plate (barcode);
    # this chooses which within-plate positional covariates are regressed out:
    #   "barcode"             — plate only, no row/column removal (default)
    #   "barcode_row"         — also remove well row (A–P) effect
    #   "barcode_column"      — also remove well column (1–24) effect
    #   "barcode_row_column"  — remove both row and column
    # Row/column are derived from the well id (config.metadata.well_col).
    # (Renamed from covariate_mode; the old key is still accepted, see loader.)
    batch_effect_fix: str = "barcode"

    min_dmso_per_plate: int = 3

    # Within-plate COLUMN positional de-trend (applied after ComBat, on L4c).
    # ComBat models plate as the batch but does not remove within-plate column
    # gradients (a systematic left→right drift across columns 1–24). This step
    # estimates each column's offset from the negcon (DMSO) wells — same biology
    # across all columns, so any column trend there is pure position — and
    # subtracts it from every well in that column. Features only; metadata intact.
    regress_out_column: bool = False
    column_min_dmso: int = 4  # min negcon wells in a column to estimate its offset


@dataclass
class ProfileMetricsConfig:
    """Standalone profile_metrics.py, wired into the run. Runs on Level 4b
    (well-level, feature-selected, PRE batch correction — already RobustMAD
    z-scores, which profile_metrics.py assumes and does not re-normalize).
    Writes its own CSV/PNG outputs directly into <cell_line>/features/profile_metrics/,
    independent of the metrics.py mAP/GRIT computation."""
    run: bool = False  # opt-in — off by default so existing configs are unaffected

    compute_activity: bool = True   # profile_activity_table.csv (Euclidean vs DMSO centroid)
    active_percentile: float = 95.0  # null (DMSO) percentile defining "active"

    compute_matching: bool = True   # profile_biosimilarity_matrix.csv (Pearson-based, per compound)

    # Optional: run find_similar()/match_to_controls() (+ a z-score profile plot
    # if plot_profile) for one compound of interest. Leave null to skip.
    query_compound: Optional[str] = None
    similarity_threshold: float = 90.0  # find_similar cutoff (biosimilarity %)
    plot_profile: bool = False
    top_n_var: int = 40  # features to plot if plot_profile and no explicit list

    # DMSO compound label (fallback identification alongside pert_type_col).
    dmso_compound: str = "DMSO"


@dataclass
class MahalanobisConfig:
    """Standalone mahalanobis_distance.py, wired into the run (opt-in). OASIS-style
    per-plate Mahalanobis distance from the plate's own DMSO centroid, using a
    global (across-plates) PCA + treatment-residual covariance model. Runs on
    Level 4b (feature-selected, pre-batch-correction) by default. Writes
    <features_dir>/mahalanobis_distances.csv, independent of metrics.py."""
    run: bool = False  # opt-in — off by default so existing configs are unaffected

    level: str = "4b"  # "4b" (feature-selected, pre-batch-correction) or "4c" (batch-corrected)
    cover_var: float = 0.95  # PCA explained-variance to retain
    min_dmso_wells: int = 4  # minimum DMSO wells per plate to compute MD
    keep_poscon: bool = False  # keep poscon rows in the output (default: dropped)
    compartments: List[str] = field(default_factory=lambda: ["Nuclei", "Cells", "Cytoplasm"])
    seed: int = 42


@dataclass
class SiteLevelConfig:
    """Optional Level 2.5a/2.5b outputs: per-plate MAD normalization (2.5a) +
    feature_select (2.5b) applied at the imaging SITE level, i.e. the site-level
    analogue of Level 4a/4b — one row per site instead of one row per well
    (no site→well aggregation). Off by default so existing runs are unaffected.
    Useful for inspecting/flagging outlier sites within a compound×dose group
    before they get smoothed into the Level 3 well median/mean."""
    run: bool = False


@dataclass
class ConsensusConfig:
    """Level 5 consensus settings."""
    operation: str = "median"  # "median" | "mean" | "modz" — one profile per compound x dose
    aggregation_method: str = "median"  # For post-consensus aggregation
    # Replicate group = one treatment. Left empty so grouping is derived from
    # metadata.compound_col + metadata.conc_col; set explicitly to override.
    replicate_cols: List[str] = field(default_factory=list)


@dataclass
class MetricsConfig:
    """Metrics computation settings."""
    # Phenotypic activity (mAP of replicates vs controls)
    compute_map: bool = True  # mAP via copairs (always computed if available)
    map_null_size: int = 500
    map_fdr_threshold: float = 0.05

    # Phenotypic consistency (mAP of compounds grouped by MOA/target)
    compute_consistency: bool = True  # Consistency via copairs
    consistency_col: str = "Metadata_moa"  # Column to group by (e.g., "Metadata_moa", "Metadata_target")
    consistency_null_size: int = 500
    consistency_fdr_threshold: float = 0.05

    # GRIT and other metrics
    compute_grit: bool = True  # GRIT via cytominer_eval
    compute_reproducibility: bool = True  # Replicate reproducibility via cytominer_eval
    grit_flag_threshold: float = 1.0  # Flag wells with GRIT > threshold (low quality)

    # Viability thresholds (flagging only, no filtering)
    viability_threshold_pct: float = 0.80  # Flag wells below this % of DMSO median cell count
    # Viability is computed at SITE level (count relative to the per-plate DMSO
    # reference) and rolled up to wells by median-of-ratios. The DMSO reference
    # is cleaned of low-count control sites before the median is taken.
    viability_site_count_col: str = "Metadata_Count_nuclei"  # measured per-site nuclei count
    viability_dmso_z_threshold: float = 3.0  # DMSO sites with robust Z < -this (low side) are dropped from the reference
    viability_min_sites: int = 2  # wells with fewer retained sites → flagged low-confidence (not dropped)

    # PCA settings (for UMAP/visualization)
    pca_variance_explained: float = 0.95

    # Control settings for metrics
    negcon_value: str = "negcon"
    poscon_value: str = "poscon"
    pert_type_col: str = "Metadata_pert_type"


@dataclass
class OutputConfig:
    """Output directory and file settings."""
    base_dir: str = ""  # Will be set to script dir if empty
    skip_existing: bool = True

    # Sub-folders created per cell line
    features_subdir: str = "features"
    qc_subdir: str = "qc"
    plots_subdir: str = "plots"


@dataclass
class RunStepsConfig:
    """Which slice of the pipeline to run (see run_pipeline.py --help).
    Steps, in order: level3, level4a, level4b, level4c, level5, metrics, plots.
    Command-line flags (--from-step, --to-step, --only, --level2-5, --input, --input-dir,
    --plots) override these."""
    from_step: str = "level3"
    to_step: str = "plots"
    # Folder with an earlier run's level files, read for any input not produced
    # in this run ({cell_line} is replaced). None = this run's features folder.
    input_dir: Optional[str] = None
    # Explicit input files, {level_name: path}, e.g. {"level_3": "edited.parquet"}.
    input_files: Dict[str, str] = field(default_factory=dict)
    # Level 2.5 file for the level3 step (csv/parquet, {cell_line} allowed).
    # None = <features>/level_2_5.csv. Same as input_files["level_2_5"].
    level2_5_file: Optional[str] = None
    # Plot groups for the plots step (qc, batch, controls, overview, metrics,
    # site_level). None = all.
    plot_groups: Optional[List[str]] = None


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    project_name: str = "default_project"

    # Sub-configs
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data_loading: DataLoadingConfig = field(default_factory=DataLoadingConfig)
    image_qc: ImageQCConfig = field(default_factory=ImageQCConfig)
    manual_site_drop: ManualSiteDropConfig = field(default_factory=ManualSiteDropConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    batch_correction: BatchCorrectionConfig = field(default_factory=BatchCorrectionConfig)
    site_level: SiteLevelConfig = field(default_factory=SiteLevelConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    profile_metrics: ProfileMetricsConfig = field(default_factory=ProfileMetricsConfig)
    mahalanobis: MahalanobisConfig = field(default_factory=MahalanobisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    run_steps: RunStepsConfig = field(default_factory=RunStepsConfig)

    # Control columns for analysis
    compound_col_map: Dict[str, str] = field(default_factory=dict)
    conc_col_map: Dict[str, str] = field(default_factory=dict)

    # Legacy run-control flags — NOT read by the pipeline (kept so older
    # configs still load). Use batch_correction.run and run_steps instead.
    run_batch_correction: bool = True
    run_from_level4_only: bool = False
    run_metrics_for_level5_only: bool = True


def load_config_from_yaml(yaml_path: str) -> PipelineConfig:
    """Load and merge config from YAML file with defaults."""
    cfg = PipelineConfig()

    if not yaml_path or not Path(yaml_path).exists():
        resolve_database_uri(cfg)
        return cfg

    with open(yaml_path) as f:
        overrides = yaml.safe_load(f) or {}

    # Back-compat: covariate_mode was renamed to batch_effect_fix. Remap the old
    # key so existing configs keep working instead of silently reverting to the
    # default (unknown keys are ignored by _update_dataclass).
    bc = overrides.get("batch_correction")
    if isinstance(bc, dict) and "covariate_mode" in bc and "batch_effect_fix" not in bc:
        import warnings
        warnings.warn(
            "batch_correction.covariate_mode is deprecated — rename it to "
            "batch_effect_fix.", DeprecationWarning, stacklevel=2,
        )
        bc["batch_effect_fix"] = bc.pop("covariate_mode")

    # Recursively update dataclass fields
    _update_dataclass(cfg, overrides)
    resolve_output_placeholders(cfg)
    resolve_database_uri(cfg)
    return cfg


def resolve_database_uri(cfg: PipelineConfig) -> None:
    """Keep credentials out of config files: expand ${VAR} references in
    database.uri, and fall back to the IMAGEDB_URI environment variable when
    the config leaves it empty."""
    import os
    uri = os.path.expandvars(cfg.database.uri or "")
    if not uri or "$" in uri:
        uri = os.environ.get("IMAGEDB_URI", "")
    cfg.database.uri = uri


def resolve_output_placeholders(cfg: PipelineConfig, now=None) -> None:
    """Fill {date} (YYYY-MM-DD) and {time} (HHMM) in the output sub-folder
    names, e.g. features_subdir: "features_{date}" → "features_2026-10-05".
    Resolved once when the config is loaded, so one run uses one stamp."""
    from datetime import datetime
    now = now or datetime.now()
    stamps = {"{date}": now.strftime("%Y-%m-%d"), "{time}": now.strftime("%H%M")}
    for name in ("features_subdir", "plots_subdir", "qc_subdir"):
        val = getattr(cfg.output, name)
        if isinstance(val, str):
            for k, v in stamps.items():
                val = val.replace(k, v)
            setattr(cfg.output, name, val)


def _update_dataclass(obj: Any, updates: Dict[str, Any]) -> None:
    """Recursively update dataclass fields from dict."""
    for key, value in updates.items():
        if not hasattr(obj, key):
            continue

        attr = getattr(obj, key)
        if hasattr(attr, "__dataclass_fields__"):
            # It's a nested dataclass
            _update_dataclass(attr, value)
        else:
            # Direct field update
            setattr(obj, key, value)


def config_to_dict(cfg: PipelineConfig) -> Dict[str, Any]:
    """Convert config to dict for serialization."""
    return asdict(cfg)
