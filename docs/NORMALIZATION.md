# Normalization — explicit specification

This document states exactly what `normalization.py::normalize_per_plate()`
(Level 4a) does, so it can be checked against the standard
[pycytominer](https://pycytominer.readthedocs.io/) /
[profiling-recipe](https://github.com/cytomining/profiling-recipe) /
[LINCS](https://github.com/broadinstitute/lincs-cell-painting/blob/master/profiles/README.md)
normalization recipe without reading the code.

## Normalization method

`pycytominer.normalize(method="mad_robustize")`, run independently per plate
(see **Strata** below). `mad_robustize` rescales each feature by the median
and MAD (median absolute deviation) of the reference population (see
**Negative-control definition**), which is the standard pycytominer/LINCS
Level 4a normalization method. `mad_robustize_epsilon` is set to
`normalization.mad_threshold` (default `0.001`).

After `pycytominer.normalize()` returns:
- Any `inf`/`-inf`/`NaN` produced by a zero-MAD feature is replaced with `0.0`.
- Values are clipped to `[normalization.clip_min, normalization.clip_max]`
  (default `[-10, 10]`).

Neither the inf/NaN fixup nor the clipping is part of pycytominer's own
`normalize()` — both are added here to keep downstream steps (ComBat,
consensus, metrics) numerically stable; they do not change which population
defines the normalization reference.

## Negative-control definition

A well is a negative control (DMSO) when
`row[metadata.pert_type_col] == metrics.negcon_value`
(default column `Metadata_pert_type`, default value `"negcon"`). This is the
single definition of "DMSO" used everywhere in the pipeline: as the
`mad_robustize` reference population (`normalization.samples_query`, default
`"Metadata_pert_type == 'negcon'"`), as the reference for invariant-feature
filtering (`normalization.invariant_filter_reference_query`, same default),
and as the fit population for spherize/ZCA batch correction.

## Strata / plate-wise behavior

Normalization is **plate-wise**: `normalize_per_plate()` groups rows by
`metadata.barcode_col` (default `Metadata_Barcode`) and calls
`pycytominer.normalize()` once per plate group, so every plate is normalized
only against its own DMSO wells — no cross-plate pooling at Level 4a. If a
plate's normalization call fails (e.g. too few DMSO wells), that plate's raw
values are kept unchanged and a warning is logged; it is not silently dropped.

Aggregation (Level 3, site → well) and consensus (Level 5) also use strata,
but different ones: Level 3 groups by well identity (barcode + well) plus any
metadata column constant within a well; Level 5 groups by
`metadata.compound_col` + `metadata.conc_col` (compound × concentration) by
default.

## Features excluded from normalization

Two exclusion steps exist, at different levels, and they are not the same
list:

1. **Before Level 4a runs** — globally-invariant features are dropped.
   `mad_raw` is computed on the raw (pre-normalization) DMSO reference
   population; any feature with `MAD ≤ normalization.mad_threshold` there
   (i.e. it doesn't vary even across DMSO wells) is removed before
   `pycytominer.normalize()` is called at all. This prevents divide-by-zero
   MAD from corrupting an otherwise-informative feature block.
2. **At Level 4b** (`feature_select()`, not Level 4a) — the
   `normalization.blocklist_patterns` list (fixed CellProfiler feature-name
   fragments known to be noisy: certain Granularity/Correlation channels) is
   removed via pycytominer's `feature_select(operation="blocklist")`, together
   with variance/correlation/NA/outlier-based selection. Deferring blocklist
   removal to 4b (rather than 4a) matches the reference pipeline this
   project's CP_MS/BA template follows.

## Preprocessing before normalization

Everything below happens before `normalize_per_plate()` is ever called:

1. Load Level 2.5 (site-level features + metadata after CellProfiler
   compartment merge).
2. Image-level QC (`image_qc`): blur/focus outlier sites and low
   nuclei-count sites are flagged/dropped; DMSO sites get an additional,
   stricter QC pass (`dmso_site_qc`) since they define the normalization
   reference.
3. Manual site drop (`manual_site_drop`, opt-in): specific sites identified
   by eye from QC images are removed.
4. Rows with no experimental annotation (`pert_type` missing) are dropped —
   they can be neither a normalization reference nor a treatment.
5. Site-level viability is computed (does not filter anything — flag only).
6. Per-site cell-count filtering (`aggregation.min_dmso_cells_per_site` /
   `min_cells_per_site`), then site → well aggregation (Level 3, median or
   mean per `aggregation.operation`).

Only after all of the above does Level 4a normalization run on the resulting
well-level table.

## DMSO-centering validation

After normalization, `normalize_per_plate()` calls
`_validate_dmso_centering()`, which checks — per plate — that the mean
feature value across that plate's DMSO wells is within
`normalization.dmso_center_tolerance` (default `0.5`) of `0`. This is the
direct, checkable consequence of `mad_robustize` against the DMSO reference
having worked: DMSO is the reference, so DMSO should end up centered at 0.
Results are logged, folded into the run summary's Level 4a step, and plotted
per-plate by `plotting_analysis.plot_dmso_normalization_check()` (see
`<cell_line>/plots/01_qc/`).

## Consensus (Level 5)

`consensus.py::compute_consensus()` calls `pycytominer.consensus()` directly.
`ConsensusConfig.operation` defaults to `"median"` — pycytominer's own
default — and is unchanged by any customization in this pipeline. `"modz"`
and `"mean"` remain available via config but are not the default.

## Metadata annotation, Level 4 → Level 5

Every level's `metadata_lN` table is the full set of `Metadata_`-prefixed
columns present in that level's dataframe — nothing is annotated in from an
external file via a separate `annotate()` step, because Level 2.5 already
carries all metadata columns from the source database/CSV. This metadata
survives every subsequent transform:

- Level 4a/4b: metadata columns pass through `pycytominer.normalize()` /
  `feature_select()` untouched (`meta_features=` argument).
- Level 4c: ComBat/Harmony are run through an AnnData object whose `.obs` is
  the metadata table; Spherize uses pycytominer's own `meta_features=`
  passthrough. All three return metadata unchanged.
- Level 5: `pycytominer.consensus()` itself keeps only the replicate
  (compound × dose) and feature columns, so `compute_consensus()` explicitly
  re-merges every metadata column that is constant within a replicate group
  (pert_type, cell line, project, ID columns, viability, …) back onto the
  consensus table, so both `level_5_batch_corrected.parquet` and
  `level_5_no_batch_correction.parquet` stay self-describing. mAP/GRIT
  (computed on Level 4b) are attached afterward, by compound × dose, as
  additional `Metadata_`-prefixed columns.
