# TASK-R1-09-FIX-01 — Semantic Identity and Global Validation BCE

This fix preserves the frozen Pen TCN architecture, optimizer, threshold, feature data, targets, splits, and LightGBM baseline while correcting two experiment-integrity details.

`semantic_run_hash` now excludes `artifact_hashes`, checkpoint container bytes, environment artifact bytes, and other volatile fields. It still includes the source lineage, configuration snapshot hash, normalization hash, architecture, model state hash, predictions, metrics, diagnostics, comparison, seed, device, threshold, best epoch, and best validation loss. `manifest_integrity_hash` is now written to `run_manifest.json` after `run_hash` and covers the complete manifest metadata except its own field. Reload validates both hashes separately; artifact hashes remain available for tamper detection.

Validation and training history BCE reporting now use global masked aggregation:

```text
sum(BCE over real timesteps and all labels) / total valid label positions
```

The optimizer still minimizes each batch's masked mean BCE, so this correction changes reporting and early-stopping measurement without changing the batch gradient objective. Unequal-batch and padding regression tests cover the distinction.

The fixed run remains capped at 200 epochs. If `best_epoch` equals 200, the run reached the fixed max-epoch boundary and early stopping did not trigger before the cap; the epoch cap is not increased as part of this fix.
