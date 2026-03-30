# Chihuahua vs Muffin — Data-Centric Image Classifier

Binary image classifier (chihuahua = 0, muffin = 1) built with **ResNet-18**
trained **from scratch** (no pretrained weights) as part of the
[3LC Data-Centric AI Hackathon](https://www.kaggle.com/competitions/chihuahua-or-muffin).

The twist: instead of tuning the model we improve our **data** using the
[3LC](https://docs.3lc.ai) platform — visualising embeddings, correcting
labels, and strategically labelling unlabeled images.

---

## Project layout

```
chihuahua_muffin/
├── train.py          # Training script (ResNet-18 + 3LC)
├── predict.py        # Generates submission.csv from test images
├── requirements.txt  # Python dependencies
└── README.md         # This file

data/                 # Put Kaggle data here (not committed)
├── train/
│   ├── chihuahua/
│   └── muffin/
├── val/
│   ├── chihuahua/
│   └── muffin/
├── unlabeled/        # Images without labels (label = "undefined")
└── test/             # Hidden test images (filenames = submission IDs)
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download competition data

From the Kaggle competition page, download and unzip the data into the
`data/` directory following the layout above.

### 3. First training run

```bash
python train.py --data-root data --epochs 20
```

This will:
- Create (or reuse) 3LC Tables for `train`, `val`, and `unlabeled`.
- Train ResNet-18 from scratch for 20 epochs.
- Save the best checkpoint as `best_model.pt`.
- Collect per-sample embeddings, predicted labels, and confidences into
  a 3LC Run.

### 4. Analyse in the 3LC Dashboard

```bash
3lc service   # starts the dashboard at http://localhost:5000
```

Open the Dashboard, select the **Chihuahua-Muffin** project, and:
- View the **embedding scatter plot** to find clusters.
- Spot mislabelled training images and correct them.
- Use model confidence to pick the **most informative unlabeled samples**
  to label (active learning).
- Set `weight = 0` for noisy or duplicate samples to exclude them.

Every edit creates a new **table revision** — training always uses the
latest revision automatically.

### 5. Generate predictions and submit

```bash
python predict.py --model-path best_model.pt --test-dir data/test
```

Upload the generated `submission.csv` to Kaggle.

### 6. Iterate

Repeat steps 3 → 5, each time improving the dataset rather than the model.

---

## Key 3LC concepts used

| Concept | What it does |
|---|---|
| **Tables** | Versioned datasets — every edit creates a new revision |
| **Runs** | Experiment tracking with per-sample metrics |
| **Sample weights** | `weight=0` deactivates a sample; `weight=1` includes it |
| **Embeddings** | 512-D ResNet-18 avgpool features, visualised in 3D in the Dashboard |

---

## Model architecture

**ResNet-18** (no pretrained weights, trained from scratch)

- Input: 224 × 224 RGB images
- Backbone: standard ResNet-18
- Head: `Linear(512 → 2)` for binary classification
- Training: Adam + CosineAnnealingLR, data augmentation (random crop, flip, colour jitter)

---

## Classes

| Label | Class |
|---|---|
| 0 | chihuahua |
| 1 | muffin |

---

## Competition rules (reminder)

- Model: ResNet-18 only — no other architectures.
- No pretrained weights.
- Training data: only the provided dataset — no external image data.
- 3LC is required for the data-centric workflow.
