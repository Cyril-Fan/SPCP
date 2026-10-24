# Semantic and class-wise analysis

This directory contains the seed-level metrics and class/confusion records used
to plot the semantic-preservation analyses. It reproduces the figures from those
records; it does not recompute text embeddings from checkpoints or reproduce model
training. No dataset images or trained checkpoints are included.

## Inputs and statistical units

- `semantic_seed_metrics.csv`: directional and normalized squared length errors,
  averaged over outgoing neighbors within a class and then equally over classes.
- `class_seed_metrics.csv` and `confusion_counts.csv`: class-level outcomes and
  confusion counts for Full and NoS (w/o S), with seeds 1, 2 and 3.
- The other CSVs contain accuracy, paired differences and summaries from those runs.

Means and sample standard deviations summarize training seeds. Class-wise
Full-minus-NoS differences are paired within each seed; confusion entries are
row-normalized before comparison. The new-class graph is for analysis only.
Removing S removes both A and E, so the comparison does not isolate a causal
effect of directional or length distortion on HM. Recognition changes are
dataset-dependent: NWPU-RESISC45 new-class accuracy rises by 5.30 points and
PatternNet falls by 3.67 points.

## Plotting

Use a separate plotting environment satisfying `scripts/requirements.txt`.
These Matplotlib/pandas/NumPy versions are independent of the older PyTorch
training environment. Install Times New Roman for matching publication text
metrics. The scripts also document their font fallback behavior.

From this directory:

```text
python scripts/make_semantic_figures.py --preview
python scripts/make_classwise.py --preview
```

The commands create `figures/` and `qa/`. `--preview` still exports vector PDFs
and preview/QA files; omitting it additionally exports SVG, EPS, PNG and TIFF.
Outputs keep the stable filenames used in `docs/figures/`; their numeric prefixes
are asset names, not manuscript version numbers. Inspect any regenerated outputs
before replacing the published assets. No GPU or new training run is required.
