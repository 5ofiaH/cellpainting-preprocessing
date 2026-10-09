# Phenotypic Activity and Consistency Metrics

This document describes the phenotypic activity and consistency metrics now integrated into the preprocessing workflow.

## Overview

The pipeline now automatically computes two key phenotypic metrics on Level 4b (feature-selected, PRE batch correction) data:

1. **Phenotypic Activity (mAP)** — How distinguishable is each perturbation from controls?
2. **Phenotypic Consistency (mAP by MOA/Target)** — Do compounds with the same mechanism have similar phenotypes?

Both metrics use Mean Average Precision (mAP) from the `copairs` library, following the methodology in:
> Kalinin et al., Nature Communications 16, 5181 (2025)

## Phenotypic Activity

### What it measures
Phenotypic activity assesses how well a perturbation's phenotype separates from negative controls (DMSO). It answers: "Is this compound's effect detectable in the cell painting phenotype?"

### Implementation
1. **Replicate-level AP**: Calculates average precision for each well, measuring how similar it is to other wells of the same compound/dose vs DMSO controls
2. **Compound-level mAP**: Averages AP scores across all replicates of each compound/dose combination
3. **P-values**: Permutation testing with FDR correction (threshold: p < 0.05)

### Output
- **Columns in results**:
  - `mean_average_precision`: Raw mAP score (0-1)
  - `mean_normalized_average_precision`: Normalized by expected random distribution
  - `corrected_p_value`: FDR-adjusted p-value
  - `-log10(p-value)`: Log-transformed p-value for plotting

- **Files saved**:
  - `{cell_line}_phenotypic_activity.png` — Scatter plots of mAP vs -log10(p-value)
  - `{cell_line}_activity_map.csv` — Full results table (optional, via pipeline config)

### Example interpretation
```
Phenotypically active = 72.3% of compounds
- Compounds with mAP > 0.5 and p < 0.05 show strong, statistically significant activity
- Compounds with mAP < 0.3 are indistinguishable from DMSO
- Low p-value indicates robust detection across replicates
```

### Configuration
In your YAML config:
```yaml
metrics:
  compute_map: true  # Always computed if copairs available
  map_null_size: 500  # Permutations for p-value calculation
  map_fdr_threshold: 0.05  # Significance threshold
```

## Phenotypic Consistency

### What it measures
Phenotypic consistency tests whether compounds with the same target/MOA have phenotypically similar effects. It answers: "Do compounds that act the same way produce similar cell phenotypes?"

### Implementation
1. **Filter to active compounds**: Only compounds with significant phenotypic activity (p < 0.05) are included
2. **Aggregate by compound**: Take median of all features for each compound (reduces noise)
3. **Group by MOA/Target**: Define pairs of compounds sharing the same MOA or target
4. **Positive pairs**: Compounds within the same MOA/target group
5. **Negative pairs**: Compounds from different MOA/target groups
6. **mAP calculation**: Average precision for retrieving compounds from the same group
7. **P-values**: Permutation testing with FDR correction

### Output
- **Columns in results**:
  - `{consistency_col}`: The grouping variable (e.g., `Metadata_moa` value)
  - `mean_average_precision`: mAP score for that group (0-1)
  - `mean_normalized_average_precision`: Normalized mAP
  - `corrected_p_value`: FDR-adjusted p-value
  - `-log10(p-value)`: Log-transformed p-value

- **Files saved**:
  - `{cell_line}_phenotypic_consistency.png` — Scatter plots of mAP vs -log10(p-value)
  - `{cell_line}_consistency_map.csv` — Full results table (optional)

### Example interpretation
```
Phenotypically consistent = 35.2% of target groups
- "Proteasome inhibitors" (mAP=0.75, p=0.001): Strong consistency
- "PI3K inhibitors" (mAP=0.42, p=0.08): Weak consistency (not significant)
- "Kinase inhibitors" (mAP=0.15, p=0.6): No phenotypic consistency

Interpretation: Proteasome inhibitors produce similar phenotypes (as expected),
but broad kinase inhibitors do not (due to diverse downstream pathways).
```

### Configuration
In your YAML config:
```yaml
metrics:
  compute_consistency: true  # Enable consistency computation
  consistency_col: "Metadata_moa"  # Column to group by
                                   # Or use "Metadata_target", etc.
  consistency_null_size: 500  # Permutations for p-value
  consistency_fdr_threshold: 0.05  # Significance threshold
```

### Multi-label handling
If your grouping column contains pipe-separated values (e.g., `EGFR|HER2`), the pipeline automatically uses multi-label average precision. This allows:
- A compound to belong to multiple MOAs simultaneously
- Proper counting of positive pairs (compounds sharing ANY target)

## Usage in Pipeline

### Default behavior
The metrics are automatically computed during the pipeline run:

```python
from cp_preprocess.pipeline import CellPaintingPipeline
from cp_preprocess.config import load_config_from_yaml

config = load_config_from_yaml("config.yaml")
pipeline = CellPaintingPipeline(config)

# Metrics computed automatically on Level 4b (pre batch correction)
output_paths = pipeline.run("level_2_5.csv", "A549")
```

### Accessing results
Results are saved as PNG plots in the `plots/` directory:
- `{cell_line}_phenotypic_activity.png`
- `{cell_line}_phenotypic_consistency.png`

To also save full result tables, add output handling in your config or pipeline extension.

### Custom analysis
To access metrics programmatically:

```python
from cp_preprocess.metrics import compute_phenotypic_activity, compute_phenotypic_consistency

# Compute activity
ap_df, map_df = compute_phenotypic_activity(df_level4b, config)

# Get active compounds
active = map_df[map_df["corrected_p_value"] < 0.05]
print(f"Active compounds: {len(active)}/{len(map_df)}")

# Compute consistency
consistency_df = compute_phenotypic_consistency(df_level4b, map_df, config)
consistent = consistency_df[consistency_df["corrected_p_value"] < 0.05]
print(f"Consistent groups: {len(consistent)}/{len(consistency_df)}")
```

## References

**Paper**: Kalinin, A.A., et al. (2025). A versatile information retrieval framework for evaluating profile strength and similarity. *Nature Communications* 16, 5181. https://doi.org/10.1038/s41467-025-60306-2

**Library**: https://github.com/cytomining/copairs

**Citation**: 
```bibtex
@article{Kalinin2025copairs,
  title={A versatile information retrieval framework for evaluating profile strength and similarity},
  author={Kalinin, Alexey A. and Arevalo, Juan and Serrano, Erik and others},
  journal={Nature Communications},
  volume={16},
  pages={5181},
  year={2025}
}
```

## Troubleshooting

### "copairs not installed"
Install via: `pip install copairs`

### "No phenotypically active compounds found"
- Check that your data has meaningful compound effects
- Verify metadata columns (`Metadata_compound`, `Metadata_conc_uM`) are present
- Try increasing `map_null_size` for more robust p-values

### "consistency_col not configured"
Set in your YAML:
```yaml
metrics:
  consistency_col: "Metadata_moa"
```

### Unexpected low mAP scores
- Check feature selection quality (Level 4b should have 200-400 features)
- Verify DMSO controls are properly labeled
- Metrics computed on Level 4b (pre batch correction), so they reflect the raw replicate structure; attached to both Level 5 variants (batch-corrected + non-batch-corrected)

## Integration with existing code

### Changes to config.py
Added fields to `MetricsConfig`:
- `compute_consistency: bool` — Enable/disable consistency computation
- `consistency_col: str` — Column to group by (default: "Metadata_moa")
- `consistency_null_size: int` — Permutations for p-values
- `consistency_fdr_threshold: float` — Significance threshold

### Changes to metrics.py
New functions:
- `compute_phenotypic_activity()` — Replicate + compound-level mAP
- `compute_phenotypic_consistency()` — mAP for grouped compounds
- `compute_map()` — Legacy wrapper (compound-level only)

Updated functions:
- `compute_all_metrics()` — Now returns dict with activity and consistency dataframes

### Changes to plotting.py
New functions:
- `plot_phenotypic_activity()` — 4-panel plot (raw/normalized, replicate/compound)
- `plot_phenotypic_consistency()` — 2-panel plot (raw/normalized mAP)

### Changes to pipeline.py
- Calls `compute_phenotypic_activity()` and `compute_phenotypic_consistency()`
- Generates plots automatically
- Output directory set up early for all plot functions
