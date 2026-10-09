# Profile-Level Analysis: Biosimilarity, Nearest Neighbors & Significance

This documents the two exploratory notebooks that sit on top of `profile_metrics.py`
(a standalone, no-pipeline-imports module — see `CLAUDE.md`). Neither notebook
modifies `profile_metrics.py`, `metrics.py`, `config.py`, or any parquet file; both
read Level 4b (well-level, pre-batch-correction) and, where noted, Level 5 outputs.

- `profile_metrics_explorer.ipynb` — day-to-day compound lookup tool.
- `nn_significance_test.ipynb` — sandbox that adds rigor (batch-aware identity,
  viability gating, significance testing) on top of the same underlying metrics,
  written explicitly as a **test before deciding whether to fold anything into
  the main pipeline**.

Both notebooks require `%matplotlib inline` to be re-asserted AFTER importing
`profile_metrics` — that module calls `matplotlib.use("Agg")` at import time (so
its CLI works headless), which silently kills inline plot rendering otherwise.

## 1. Core metrics (`profile_metrics.py`)

| Metric | Function | What it measures | Range |
|---|---|---|---|
| Activity | `activity_scores` / `activity_table` | Per-well Euclidean distance to the DMSO centroid; a compound is "active" if its median well score exceeds a percentile cutoff of the DMSO-well null distribution (default P90) | unbounded (magnitude) |
| Biosimilarity | `biosimilarity` / `biosimilarity_matrix` | Pearson correlation between two compounds' **consensus** (median-of-replicates) profiles, floored at 0 and scaled to a percent | 0–100 (shape only, scale-invariant) |
| Concordance | `concordance_matrix` / `find_concordant` | Lin's Concordance Correlation Coefficient (CCC) between consensus profiles — penalizes magnitude/scale mismatch as well as shape mismatch, unlike Pearson | 0–100 (shape **and** magnitude) |
| Nearest neighbor | `find_similar` | Compounds ≥ a biosimilarity threshold to a query, one-directional | — |

**Why both biosimilarity and concordance exist**: a compound can be "biosimilar"
(same trend/shape) to a query while responding at a much weaker or stronger
amplitude — concordance catches that; biosimilarity alone would score both the
same. Use `find_similar` (Pearson) for "does it trend the same way" and
`find_concordant` (CCC) for "does it respond with a similar magnitude too."

**Channel/compartment grouping** (`CHANNEL_KEYWORDS`, used by the z-score profile
and paired-biosimilarity plots): features are bucketed into DNA / ER / RNA / AGP /
Mito / Morphology by substring match on the feature name. `illumCONC` = Concanavalin
A = **ER** stain; `illumPHAAndWGA` = Phalloidin+WGA = **AGP** channel (actin/Golgi/
plasma-membrane, imaged mostly in the Cytoplasm compartment) — both keywords had
to be added; without them, these features silently fell into a catch-all "Other"
bucket instead of their real channel.

## 2. `profile_metrics_explorer.ipynb` — compound lookup tool

Set `L4B_PATH` and `QUERY_COMPOUND`, run all cells. Produces, per query:

1. **Activity table** — is the query (and every compound) active vs. DMSO.
2. **Biosimilarity matches** (`find_similar`) + **concordance matches**
   (`find_concordant`), shown side by side so a biosimilar-but-not-concordant
   hit (same trend, different magnitude — e.g. a partial phenocopy) is visible.
3. **Nearest-neighbor heatmap** — top-N most biosimilar compounds to the query.
4. **Hierarchical clustering** — average-linkage on `1 - biosimilarity/100`,
   cut into `N_CLUSTERS`; dendrogram truncated to the last 60 merged clusters
   when there are too many compounds to render legibly (full assignments still
   go to `profile_clusters.csv`).
5. **Z-score profile plot** — one line **per compound** (the per-compound
   median-of-replicates consensus, i.e. the same aggregation logic as the
   pipeline's Level 5 step, computed locally from L4b — not literally the L5
   parquet), not per well, so it isn't noisy spaghetti.
6. **Paired biosimilarity heatmap** — query vs. its top hit, one row per
   compound, features grouped/labeled by channel.

All tables/figures are also written to `OUTDIR` for reuse outside the notebook.

## 3. `nn_significance_test.ipynb` — sandbox: batch-aware identity + significance

This notebook exists because compound **names alone are not a safe identity
key** in this dataset, and because raw biosimilarity numbers have no built-in
notion of "is this actually better than chance." It layers the following on
top of the Section 1 metrics, in order:

### 3a. Viability gating (Step 1b)
Drops any non-DMSO well with `Metadata_Viability_pct < 50` (site-level,
DMSO-normalized, computed by `viability.py`) BEFORE anything else runs — this
is not done anywhere else in `profile_metrics.py`. Concretely, this mattered a
lot: one Etoposide source batch (`C0934`) was 100% below-threshold (mean
viability 36.5%) and disappeared entirely once filtered; a second batch
(`C0691`) went from 17/20 reciprocal neighbors down to 2/20 once its dying
wells were removed, revealing that its earlier "clean" neighbor list was
largely a shared cytotoxicity/cell-death signature, not an Etoposide-specific
phenotype. Always check whether a similarity result survives viability
filtering before trusting it.

### 3b. Batch-aware identity (Step 2)
Compound identity = **`"<canonical name> [<Metadata_batch_id>]"`**, never just
the name. Casing/whitespace variants of the same name (e.g. `Etoposide` vs
`ETOPOSIDE`) are folded to one **display** name (majority-casing by well
count) — confirmed safe via matching `Metadata_DTXSID` — but different
`Metadata_batch_id` values are ALWAYS kept as separate analysis entities, since
they come from different vendors/projects even when the compound name and
DTXSID match. A same-name compound spanning multiple batches (e.g. Etoposide,
Rotenone, Bisphenol A, 26 total in this dataset) is split into that many rows
in the consensus table; a dedicated "cross-batch consistency" check (Step 7b)
reports whether those batches actually agree with each other (raw Pearson +
reciprocal-neighbor status) — Etoposide's batches did (pre-viability-filter);
Bisphenol A's two batches did not (37.3% raw Pearson, well below chance),
which is itself a useful red flag about that compound's data.

### 3c. Two null distributions (Steps 4–5)
- **Null A** (DMSO-DMSO pairwise, well-level) — a reproducibility reference,
  NOT used as the significance threshold. Do not compare it directly to Null B
  without accounting for aggregation level (see caveat below).
- **Null B** (permuted-label / chance baseline) — compound labels are shuffled
  across wells, then aggregated into consensus profiles the **same way** the
  real data is (so it's the correct apples-to-apples null for consensus-level
  biosimilarity), and pairwise Pearson is computed on those. The 95th
  percentile of Null B is the significance threshold used everywhere below.
- **Caveat**: Null A can come out LOWER than Null B. This is not an assay
  quality problem — it's because Null A is well-level (single replicate vs.
  single replicate, full well-to-well noise) while Null B is consensus-level
  (median-of-many-replicates vs. median-of-many-replicates, noise already
  averaged out). Only Null B is a fair chance baseline for consensus-level
  comparisons.

### 3d. Reciprocal nearest-neighbor + "defensible neighbor" flag (Steps 7–8)
A neighbor is **reciprocal** if the query is also in that neighbor's own
top-N list (not just one-directional, unlike `find_similar`). **Defensible**
= reciprocal AND above the Null B 95th-percentile threshold — the strictest
combination, requiring both a mutual relationship and a real, above-chance
similarity score (computed on the RAW/unclipped Pearson matrix, not the
0-floored `bmat` — clipping is applied only when rendering heatmaps).

### 3e. mAP — read, not recomputed (Step 6)
`metrics.py`'s `compute_phenotypic_activity` (copairs) already runs as part of
the real pipeline and writes `Metadata_mAP` / `Metadata_mAP_p` /
`Metadata_mAP_corrected_p` onto `level_5_no_batch_correction.parquet` and
`level_5_batch_corrected.parquet`, keyed by compound × dose. This notebook
reads those columns for the test queries instead of recomputing them.
**Caveat**: that precomputed mAP is keyed by the OLD, non-batch-aware compound
name, so it still conflates different batches under one name — the same gap
Section 3b fixes for biosimilarity has not been fixed for mAP.

### 3f. MOA cross-check (Step 3b in the notebook)
Joined from `SPECS/input/specs.csv` by compound **NAME** (case-folded) — that
file has no batch_id or DTXSID column, so NAME is the only usable key. This
confirms (trivially, since MOA is a name-level property) that casing/batch
variants of the same compound share the same MOA, and reports real coverage:
only ~18% of this dataset's ~1000 compounds match `specs.csv` at all, since
it's a drug-repurposing library, not built for environmental chemicals. Treat
MOA coloring in the embeddings as a partial view.

### 3g. Embeddings (Step 10)
PCA, t-SNE, and UMAP on the same batch-aware consensus matrix used everywhere
else — no new data, just a different lens. Colored by (a) cluster (reusing
the same average-linkage hierarchical clustering as Section 2, not a new
method) and (b) MOA (sparse, per 3f). Test-query entities are outlined and
labeled on all three projections so you can see at a glance whether same-name
different-batch entities land near each other or far apart.

## 4. What's NOT yet fixed anywhere except this sandbox

- Compound identity being name-only (not batch-aware) is still true
  everywhere else: `consensus()`, `biosimilarity_matrix()`, `find_similar()`,
  `find_concordant()`, `compute_phenotypic_activity()`'s `pos_sameby`, and the
  separate SPECS k-NN script (`specs_search/scripts/nearest_neighbors.py`).
- Viability gating is not applied anywhere in `profile_metrics.py`.
- Reciprocal-NN / defensible-neighbor logic does not exist outside this
  sandbox notebook.

If any of the above gets adopted into the main pipeline, `nn_significance_test.ipynb`
Steps 2, 6, and 7-8 are the reference implementation to port from.
