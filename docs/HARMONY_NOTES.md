# Harmony batch correction — RESOLVED

## Status: FIXED (2026-07-20)
`batch_correction.apply_harmony()` is wired in and dispatched via
`config.batch_correction.method: "harmony"`. All three methods now verified on
synthetic 2-plate data (negcon cross-plate gap): combat 2.75→0.53,
spherize 2.47→0.67, harmony 2.47→1.35.

Two fixes were applied (see below for the original diagnosis):
1. `nclust = max(2, min(N//30, 100))` — floors the auto value so it can't
   collapse to 1 and crash on small N.
2. Orientation-robust `Z_corr`: this PyTorch fork returns (n_wells, n_features)
   (classic harmonypy returns the transpose); the code now picks whichever
   orientation matches the input instead of always transposing.

## Error observed
Installed lib: `harmonypy 0.2.0` — a **PyTorch** fork ("Running Harmony (PyTorch
on cuda)"), not the classic CPU harmonypy. On a tiny synthetic set (32 wells):

```
✗ Harmony FAILED (TypeError: object of type 'float' has no len())
```

Auto `nclust = min(N/30, 100)` collapsed to **1** for the tiny N, and a scalar
was passed where the code expects an array → `len(float)`. The wrapper caught it
and returned Level 4b unchanged (safe, but no correction applied).

## Signature (this fork)
```
run_harmony(data_mat, meta_data, vars_use, theta=None, lamb=None, sigma=0.1,
            nclust=None, tau=0, block_size=0.05, max_iter_harmony=10, ...,
            device=None)
```
So the call `run_harmony(X, obs, vars_use=[plate_col])` is correct; the problem
is only the auto-`nclust` on small N.

## Next steps
1. Pass `nclust` explicitly and floor it, e.g. `nclust=max(2, min(N//30, 100))`,
   in `apply_harmony`. This should fix the tiny-N crash.
2. Real SPECS L4b has hundreds/thousands of wells, so `nclust` auto is likely
   fine there — confirm on the real 6k table before assuming the guard is enough.
3. Consider `device="cpu"` for reproducibility if GPU non-determinism matters.
4. Harmony runs directly in feature space (Z_corr.T → wells×features); if that is
   unstable on many correlated features, switch to PCA → harmony → inverse_transform.

## Verified working
- `combat`  — negcon plate gap 2.75 → 0.53
- `spherize`— negcon plate gap 2.75 → 0.66
- `harmony` — pending the `nclust` guard above
