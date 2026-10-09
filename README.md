# Cell Painting preprocessing pipeline

A config-driven pipeline that turns site-level CellProfiler features (Level 2.5)
into normalized, feature-selected, batch-corrected well profiles and per-compound
consensus profiles (Level 5). It also computes viability, mAP, GRIT, and QC plots.

**One codebase, one YAML file per project.** Everything project-specific (column
names, thresholds, normalization and batch-correction choices, output folders)
lives in the config, so you don't edit Python to run a new dataset.

| Level | What happens | File |
|-------|--------------|------|
| 2.5 | Input: one row per imaging **site** (features + metadata, after image QC) | `level_2_5.csv` |
| 3   | Sites → **wells** (median); site-level viability | `level_3.parquet` |
| 4a  | Per-plate normalization against DMSO (MAD robustize) | `level_4a.parquet` |
| 4b  | Feature selection (variance, correlation, blocklist, …), with **mAP + GRIT computed here** | `level_4b.parquet` |
| 4c  | Batch correction (spherize / ComBat / Harmony; optional) | `level_4c.parquet` |
| 5   | Consensus per **compound × dose**, written twice (with and without batch correction) | `level_5_*.parquet` |

See [`docs/PIPELINE_LEVELS.md`](docs/PIPELINE_LEVELS.md) for a diagram of the levels.

---

## Quick start

```bash
# 1. Get the code
git clone https://github.com/<your-user>/cellpainting-preprocessing.git
cd cellpainting-preprocessing

# 2. Install (in your own env / venv; editable so `git pull` updates it)
pip install -e .            # or: pip install -r requirements.txt

# 3. Database access (only needed to build Level 2.5 from the image DB)
export IMAGEDB_URI="postgresql://<user>:<password>@<host>/imagedb"

# 4. Make your project config
mkdir -p configs/local
cp configs/config_base.yaml configs/local/my_project.yaml
#    edit the lines marked [EDIT] (see "Configuring a project" below)

# 5. Run
python run_full_pipeline.py --config configs/local/my_project.yaml
```

`configs/local/` is git-ignored, so your project configs stay on your machine.
You can also keep a config next to your project's data and pass its path.

### Database access

The image-database connection string is **not** stored in this repository. Ask a
group member or check the group's internal docs for the read-only URI, then
either:

- export it in your shell (add the line to `~/.bashrc` to make it permanent):
  `export IMAGEDB_URI="postgresql://..."`, or
- reference it in your config: `database: { uri: "${IMAGEDB_URI}" }`.

If `database.uri` is empty, the pipeline reads `IMAGEDB_URI`. You only need the
database for step 0 (building Level 2.5). If you already have a Level 2.5 file,
you can skip it.

---

## Running

All commands are run from the repository root. After `pip install -e .`, the
same entry points are also available from anywhere as `cp-run-full-pipeline`,
`cp-run-pipeline` and `cp-run-plots`.

| You have… | Command |
|-----------|---------|
| Nothing yet: build from the image DB | `python run_full_pipeline.py --config CFG` |
| A Level 2.5 file | `python run_pipeline.py --config CFG [--cell-line U2OS]` |
| A finished run and only want plots | `python run_analysis_plots.py --output-dir <base_dir> --config CFG` |

### Running part of the pipeline

The steps, in order, are `level3`, `level4a`, `level4b`, `level4c`, `level5`,
`metrics`, `plots`. Any input a run doesn't produce itself is read from
`--input` / `--input-dir` (or from `run_steps:` in the config).

```bash
# Whole pipeline from a specific Level 2.5 file (csv or parquet)
python run_pipeline.py --config CFG --cell-line U2OS --level2-5 /path/to/level_2_5.parquet

# You edited Level 3 → rerun from normalization onwards
python run_pipeline.py --config CFG --cell-line U2OS --from-step level4a --input my_level_3.parquet

# Only recompute mAP / GRIT on an earlier run
python run_pipeline.py --config CFG --only metrics --input-dir "/path/{cell_line}/features_2026-09-23"

# Only the QC plots
python run_pipeline.py --config CFG --only plots --plots qc
```

`python run_pipeline.py --help` lists every option.

---

## Configuring a project

[`configs/config_base.yaml`](configs/config_base.yaml) lists **every** option
with its default and a comment. Copy it and change what you need; anything you
delete falls back to the default. Unknown or misspelled keys are silently
ignored, so copy key names exactly.

At minimum, set the lines marked `[EDIT]`:

```yaml
project_name: "my_project"

output:
  base_dir: "/path/to/my_project"     # where results are written
  features_subdir: "features_{date}"  # {date}/{time} → one folder per run day

metadata:
  file_path: "/path/to/metadata.csv"  # only needed for step 0
  cell_lines: ["U2OS"]
  barcode_col: "Metadata_Barcode"     # map these to YOUR column names
  well_col: "Metadata_Well"
  compound_col: "Metadata_compound"
  conc_col: "Metadata_conc_uM"
  # ...
```

Worked examples are in [`configs/examples/`](configs/examples/):
`envtox_u2os.yaml` (EnvTox, U2OS) and `dnt.yaml`.

### Common changes

```yaml
normalization:
  method: "mad_robustize"     # mad_robustize | standardize | robustize | spherize

batch_correction:
  run: true                   # false → Level 4c = Level 4b
  method: "spherize"          # spherize (default) | combat | harmony
  batch_effect_fix: "barcode" # opt-in row/column removal: barcode_row | barcode_column | barcode_row_column

consensus:
  operation: "median"         # median | modz | mean

feature_selection:
  operations: ["variance_threshold", "drop_na_columns", "correlation_threshold", "blocklist", "drop_outliers"]

metrics:
  compute_map: true
  compute_grit: true
```

The exact normalization rules (negative-control definition, plate-wise
behavior, excluded features) are in
[`docs/NORMALIZATION.md`](docs/NORMALIZATION.md).

---

## Outputs

```
<base_dir>/
├── _image_qc/                 image-QC flags + drop accounting (step 0)
├── _feature_cache/            cached compartment merge (step 0; makes reruns fast)
└── <cell_line>/
    ├── features_<date>/
    │   ├── RUN_SUMMARY.md     ← start here: what every step did, counts, drops
    │   ├── level_2_5.csv
    │   ├── level_3.parquet … level_4c.parquet
    │   ├── level_5_batch_corrected.parquet       consensus of Level 4c
    │   ├── level_5_no_batch_correction.parquet   consensus of Level 4b
    │   ├── metrics_map.parquet, metrics_grit.parquet, viability_sites.parquet
    │   └── run_summary.json, pipeline_summary.json
    └── plots_<date>/
        └── 01_qc/ 02_batch/ 03_controls/ 04_overview/ 05_metrics/ 06_site_level/
```

Each parquet holds `Metadata_*` columns first, then metric columns, then
features. To load a level into scanpy:
`python -m cp_preprocess.parquet_to_anndata --input level_4b.parquet --output level_4b.h5ad`.

---

## Repository layout

```
cellpainting-preprocessing/
├── run_full_pipeline.py      entry point: DB → Level 2.5 → … → Level 5
├── run_pipeline.py           entry point: Level 2.5 → … → Level 5 (+ metrics, plots)
├── run_analysis_plots.py     entry point: plots from an existing run
├── configs/
│   ├── config_base.yaml      template: every option, documented
│   ├── examples/             real project configs to copy from
│   └── local/                your own configs (git-ignored)
├── cp_preprocess/            the package
│   ├── config.py             config schema (dataclasses) + YAML loading
│   ├── data_loading.py       step 0: image DB → Level 2.5
│   ├── pipeline.py           orchestrator (Levels 3 → 5)
│   ├── normalization.py      Levels 3, 4a, 4b
│   ├── batch_correction.py   Level 4c
│   ├── consensus.py          Level 5
│   ├── viability.py          site-level cell-count viability
│   ├── metrics.py            mAP (copairs), GRIT (cytominer-eval)
│   ├── plotting_analysis.py  QC/metric plots (run_analysis_plots.py drives them)
│   ├── profile_metrics.py    standalone profile matching / activity (no pipeline imports)
│   └── …                     io_utils, run_summary, site_outliers, radarplots, …
└── docs/                     method notes (normalization, levels, metrics, Harmony)
```

## Documentation

| Doc | Read it for |
|-----|-------------|
| [PIPELINE_LEVELS.md](docs/PIPELINE_LEVELS.md) | What each level contains, as a diagram |
| [NORMALIZATION.md](docs/NORMALIZATION.md) | The exact normalization spec |
| [PHENOTYPIC_METRICS.md](docs/PHENOTYPIC_METRICS.md) | mAP / GRIT, and how to interpret them |
| [HARMONY_NOTES.md](docs/HARMONY_NOTES.md) | Harmony batch correction details |
| [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) | How data loading and preprocessing fit together |
| [PROFILE_METRICS_ANALYSIS.md](docs/PROFILE_METRICS_ANALYSIS.md) | Standalone `profile_metrics.py` usage |

---

## Troubleshooting

- **"No database URI configured"**: set `IMAGEDB_URI` (see [Database access](#database-access)).
- **"No Level 2.5 input files found"**: with dated folders, a new run day has no
  `level_2_5.csv` yet. Pass `--level2-5 FILE` or `--input-dir <earlier features folder>`.
- **A setting seems to have no effect**: check the key name against
  `config_base.yaml`, because misspelled keys are ignored.
- **ComBat / Harmony fails**: the pipeline falls back to uncorrected data with a
  warning. ComBat needs ≥2 plates with enough DMSO wells.
- **Metric columns are NaN**: the compound, pert-type or cell-count column was
  not found. Check the `metadata.*_col` mappings.

## Contributing

Bug fixes and new options are welcome. Please work on a branch and open a pull request.
To add a new option:

1. Add the field to the right dataclass in `cp_preprocess/config.py`.
2. Document it, with its default, in `configs/config_base.yaml`.
3. Use it in the relevant module.

Defaults should preserve current behavior, so existing configs keep giving the same results.

## Methodology

- Ewald et al. (2025), *Cell painting as a window into disease mechanism*
- Nyffeler et al. (2023), *Functional compound profiling with high-resolution…*
- Nyffeler et al. (2021), *A 384-well image-based platform for phenotypic drug screening*
- Kalinin et al. (2025), *A versatile information retrieval framework for evaluating
  profile strength and similarity*, Nat. Commun. 16, 5181 (mAP / copairs)

## Author

Sofia Hernandez (Uppsala University)
