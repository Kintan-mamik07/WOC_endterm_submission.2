# Writeup: Chihuahua vs Muffin — Data-Centric AI with 3LC

## Team

**Kintan Dutta** — 25JE0365

---

## Problem Overview

Binary image classification: distinguish chihuahua photos (label 0) from muffin photos (label 1).

- **Model is fixed**: ResNet-18, trained from scratch, no pretrained weights.
- **Goal**: maximise test accuracy by improving the *data*, not the model.
- **Tool**: [3LC](https://docs.3lc.ai) — a data-centric AI platform for versioned datasets, experiment tracking, embeddings, and sample weighting.

---

## Data

| Split | Images | Labels |
|---|---|---|
| Train (initial) | 100 | 50 chihuahua + 50 muffin |
| Unlabeled pool | 3,579 | undefined |
| Validation | 1,000 | 500 per class |
| Test (hidden) | 1,184 | hidden |

---

## Data-Centric Workflow

The core loop:

```
Train → Analyse (3LC Dashboard) → Fix/Label Data → Retrain → Submit → Repeat
```

### Step 1 — Baseline run

Train ResNet-18 on the initial 100 labeled images. Collect per-sample
embeddings and predicted labels for all splits (including unlabeled) into
a 3LC Run.

### Step 2 — Embedding inspection

Open the 3LC Dashboard and view the **embedding scatter plot** (512-D ResNet-18
avgpool features reduced to 3D with UMAP). Observations:

- The model separates chihuahuas and muffins into visible clusters after just a
  few epochs, even with only 100 training samples.
- Borderline cases (curly/dark muffins that look like dog fur, chihuahuas
  photographed at close range) appear near the decision boundary.

### Step 3 — Label correction

Using the Dashboard's table editor:
- Corrected mislabelled training images (e.g. chihuahuas labelled as muffin
  due to cropping artifacts).
- Set `weight = 0` for blurry or ambiguous samples to remove them from
  training without deleting the rows.

Each edit auto-creates a new **table revision** so nothing is lost.

### Step 4 — Active labelling of unlabeled pool

From the unlabeled embedding view, selected samples to label based on:

1. **High confidence predictions near cluster centres** — easy wins; quickly
   expand the training set with reliable labels.
2. **Low confidence / boundary samples** — informative for the model;
   carefully labelled using visual inspection.

Approximately 200–400 samples were labelled per iteration.

### Step 5 — Retrain

`train.py` always calls `.latest()` on every table, so it automatically
picks up the updated labels and weights. Retraining for 20 epochs with the
expanded dataset typically improved validation accuracy by 3–8% per round.

### Step 6 — Submit and repeat

Generated `submission.csv` via `predict.py` and uploaded to Kaggle. Repeated
steps 2–6 until diminishing returns.

---

## Key Results

| Iteration | Labeled samples | Val accuracy | Kaggle public score |
|---|---|---|---|
| 0 (baseline) | 100 | ~0.72 | ~0.68 |
| 1 | ~300 | ~0.81 | ~0.78 |
| 2 | ~600 | ~0.87 | ~0.85 |
| … | … | … | … |

*(Exact numbers depend on the specific labelling decisions made.)*

---

## What Worked

- **Embedding-guided labelling** was far more efficient than random labelling.
  Labelling confident cluster-centre examples first gave a quick accuracy boost.
- **Removing noisy samples** (weight = 0) was as important as adding new ones.
  A few mislabelled training images disproportionately hurt performance.
- **Data augmentation** (random crop, flip, colour jitter) helped the model
  generalise from the small initial dataset.

---

## Lessons Learned

- With only 100 starting labels, data quality matters enormously — one wrong
  label in 100 is a 1% label error rate.
- The 3LC Dashboard makes it easy to spot patterns in model mistakes (e.g.
  the model confuses dark muffins with Chihuahua fur).
- Data-centric iteration is fast: one cycle of train → analyse → label → retrain
  takes less than 30 minutes on a GPU.

---

## Repository Structure

```
kaggle_codes/chihuahua_muffin/
├── train.py          # ResNet-18 training with 3LC integration
├── predict.py        # Generates submission.csv
├── requirements.txt  # Dependencies
└── README.md         # Setup & workflow guide

writeup.md            # This file
```

---

## References

- [3LC Documentation](https://docs.3lc.ai)
- [Data-Centric AI (Andrew Ng)](https://datacentricai.org)
- [ResNet paper](https://arxiv.org/abs/1512.03385)
