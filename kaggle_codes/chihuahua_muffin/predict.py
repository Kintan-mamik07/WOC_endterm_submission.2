"""
predict.py — Generate submission.csv for the Chihuahua-vs-Muffin Kaggle competition.

Usage
-----
    python predict.py --model-path best_model.pt --test-dir data/test

The script writes submission.csv in the current directory, ready to upload
to the Kaggle leaderboard.

Expected test directory layout
-------------------------------
data/test/
  <id>.jpg  (or .png)  — one image per row in the submission
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants (must match train.py)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
CLASSES = ["chihuahua", "muffin"]  # index 0 → chihuahua, 1 → muffin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_val_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(num_classes: int = 2):
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(model_path: str, device: torch.device):
    model = build_model()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, test_dir: str, device: torch.device):
    """
    Returns a list of (image_id, predicted_label) tuples sorted by image_id.

    image_id is the stem of the filename (e.g. "00001" for "00001.jpg").
    predicted_label is 0 (chihuahua) or 1 (muffin).
    """
    transform = get_val_transform()
    test_dir = Path(test_dir)

    # Collect all image paths
    image_paths = sorted(
        [p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {test_dir}")

    results = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        logits = model(tensor)
        pred = logits.argmax(dim=1).item()
        results.append((img_path.stem, pred))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission for chihuahua-vs-muffin")
    parser.add_argument("--model-path", default="best_model.pt", help="Path to saved model weights")
    parser.add_argument("--test-dir", default="data/test", help="Directory containing test images")
    parser.add_argument("--output", default="submission.csv", help="Output CSV file path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Loading model from {args.model_path}…")
    model = load_model(args.model_path, device)

    print(f"Running predictions on {args.test_dir}…")
    results = predict(model, args.test_dir, device)

    output_path = Path(args.output)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for image_id, label in results:
            writer.writerow([image_id, label])

    print(f"Wrote {len(results)} predictions → {output_path}")
    print("Upload submission.csv to Kaggle to see your score!")


if __name__ == "__main__":
    main()
