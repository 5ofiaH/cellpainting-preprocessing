# Integration Guide: 01_data_prep + 02-05 Preprocessing

## Overview

This pipeline now includes **both** data loading (01) and preprocessing (02-05) as a unified, config-driven system.

```
run_full_pipeline.py
  ├─ Step 1: Data Loading (01_data_prep)
  │   └─ Query database → Level 2.5 CSV
  │
  └─ Step 2: Preprocessing (02-05)
      ├─ Level 3: Aggregate sites → wells
      ├─ Level 4a: Normalize
      ├─ Level 4b: Feature select + metrics
      ├─ Level 4c: Batch correct
      └─ Level 5: Consensus
```

## Why Two Entry Points?

### `run_pipeline.py` (Preprocessing Only)
- **When to use**: Level 2.5 already exists (01_data_prep already ran)
- **Input**: Level 2.5 CSV in each cell line folder
- **Output**: Levels 3, 4a, 4b, 4c, 5 parquets
- **Advantage**: Fast iteration on preprocessing logic (no database queries)

### `run_full_pipeline.py` (01 + 02-05)
- **When to use**: Running complete analysis from scratch
- **Input**: Database URI + metadata file
- **Output**: Levels 2.5, 3, 4a, 4b, 4c, 5
- **Advantage**: Single command for full pipeline

## Usage

### Preprocessing Only (Recommended for Iteration)

```bash
# Assumes Level 2.5 already exists at:
# <cell_line>/features/level_2_5.csv

python run_pipeline.py --config configs/my_project.yaml
```

### Full Pipeline (Data Loading + Preprocessing)

```bash
# Loads from database, creates Level 2.5, then preprocesses

python run_full_pipeline.py --config configs/my_project.yaml
```

### Specific Operations

```bash
# Data loading only (01_data_prep)
python run_full_pipeline.py --config configs/my_project.yaml --data-only

# Preprocessing only (02-05), auto-discover Level 2.5
python run_full_pipeline.py --config configs/my_project.yaml --preprocess-only

# Preprocessing only, skip data loading
python run_pipeline.py --config configs/my_project.yaml
```

## Configuration

### Enable/Disable Data Loading

```yaml
data_loading:
  skip_loading: false  # Set to true to skip database queries
  load_compartments:
    - "nuclei"
    - "cells"
    - "cytoplasm"
  compartment_files:
    nuclei: "featICF_nuclei.parquet"
    cells: "featICF_cells.parquet"
    cytoplasm: "featICF_cytoplasm.parquet"

database:
  uri: "postgresql://user:pass@host/database"
  feature_analysis_type: "cp-features"
  qc_analysis_type: "cp-qc"
  use_db: true  # Set to false to use local cache
```

### Skip Data Loading

If Level 2.5 already exists:

```yaml
data_loading:
  skip_loading: true
```

Then run:
```bash
python run_full_pipeline.py --config configs/my_project.yaml --skip-data-loading
```

Or just use the simpler command:
```bash
python run_pipeline.py --config configs/my_project.yaml
```

## Why I Didn't Include 01 Initially

### Reasoning

1. **Scope**: You specifically asked to optimize **02-05**. I focused on that first.

2. **Different Architecture**
   - 02-05: Stateless transformations (CSV → CSV/parquet)
   - 01: Stateful, requires external DB, multiple file I/O, caching

3. **Separation of Concerns**
   - 01 = Data Ingestion (specific to your DB schema, might not work for others)
   - 02-05 = Data Processing (generic, portable across projects)

4. **Assumed Working**: Looking at your code, 01_data_prep seemed stable and debugged. 02-05 had active issues (ComBat, metrics computation).

5. **Portability**: 01 requires PostgreSQL + specific DB schema. Harder to make GitHub-ready without credentials.

### Why I Should Have Included It

**You're right** — for a truly unified workflow:
- Users expect one command to run complete analysis
- Config-driven approach should apply to ALL steps, not just 02-05
- Easier to track data lineage (database → L2.5 → L5)
- Reproducibility improves when everything is versioned together

---

## Now Both Are Included

### `data_loading.py`
- `DataLoader` class: Load from database or cache
- Handles compartment merging (nuclei + cells + cytoplasm)
- Placeholder for image-level QC (TODO)
- **Status**: Skeleton ready, database connection needs implementation per your DB schema

### `run_full_pipeline.py`
- CLI orchestrating both steps
- Can run data loading only, preprocessing only, or full pipeline
- Auto-discovers Level 2.5 if step 1 skipped
- Structured logging for both steps

### Integration Points

1. **Config**: Single YAML defines both steps
2. **Output**: Level 2.5 saved after step 1, fed to step 2 automatically
3. **Logging**: Unified, timestamped output for entire pipeline
4. **Error Handling**: Graceful fallbacks at both stages

---

## Next: Implementing Database Loading

`data_loading.py` has a placeholder. To make it fully functional, you need to implement:

```python
def _load_features(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Load features from database."""
    # 1. Connect to database via self.config.database.uri
    # 2. For each plate in metadata, query features
    # 3. Merge compartments (nuclei + cells + cytoplasm)
    # 4. Return combined feature matrix
```

This is your specific business logic (knows your DB schema, table names, etc.). I can help implement this once you provide:
- Table names in your database
- Query structure for CellProfiler results
- How compartments are stored/accessed

For now, `run_full_pipeline.py` can still work if you:
1. Pre-compute parquets from your database
2. Place them in a cache directory
3. Set `use_db: false` in config

---

## Benefits of This Architecture

✅ **Two-stage pipeline** (data loading separate from preprocessing)
✅ **Flexible**: Use full pipeline OR preprocessing only
✅ **Config-driven**: All settings in YAML
✅ **Modular**: Each module has one responsibility
✅ **Testable**: Can test data loading independently
✅ **Portable**: Preprocessing logic works with any data source
✅ **Documented**: Clear why and when to use each approach

---

## Example: EnvTox Workflow

### First Run (Full Pipeline)

```bash
cp config_envtox_u2os.yaml /path/to/envtox/
cd /path/to/envtox/

# Load from database + preprocess
python run_full_pipeline.py \
  --config config_envtox_u2os.yaml \
  --cell-line U2OS
```

This creates:
- `U2OS/features/level_2_5.csv` (from database)
- `U2OS/features/level_3.parquet` … `level_4c.parquet` plus two Level 5 variants:
  `level_5_batch_corrected.parquet` and `level_5_no_batch_correction.parquet` (preprocessed)

### Iterate on Preprocessing

```bash
# Edit preprocessing settings in config
nano config_envtox_u2os.yaml

# Re-run preprocessing only (no database queries)
python run_pipeline.py \
  --config config_envtox_u2os.yaml \
  --cell-line U2OS
```

Fast feedback loop without re-querying the database.

---

## Summary

**Before**: Created 02-05 pipeline (preprocessing only)

**Now**: Added 01 (data loading) + full orchestration

**Entry points**:
- `run_pipeline.py` — preprocessing only (fast)
- `run_full_pipeline.py` — full pipeline from database (complete)

**Your input needed**: Database connection logic in `data_loading.py`
