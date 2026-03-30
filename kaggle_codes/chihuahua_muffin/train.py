"""
train.py — Chihuahua vs Muffin image classifier using ResNet-18 + 3LC.

Data-centric workflow
---------------------
1. Run this script to train on the current table revision and collect
   per-sample metrics + embeddings.
2. Open the 3LC Dashboard, inspect embeddings, correct labels, add
   unlabeled samples, and adjust sample weights.
3. Re-run this script — it automatically picks up the `.latest()`
   revision of every table.
4. Check your Kaggle score after running predict.py.
5. Repeat.

Expected directory layout
--------------------------
data/
  train/
    chihuahua/   (jpg/png images)
    muffin/      (jpg/png images)
  unlabeled/     (all unlabeled images — label column = "undefined")
  val/
    chihuahua/
    muffin/
  test/          (test images — filenames match submission IDs)
"""

import argparse
import os
import time
from pathlib import Path

import tlc
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES = ["chihuahua", "muffin"]  # index 0 → chihuahua, 1 → muffin
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMBEDDING_DIM = 512  # ResNet-18 penultimate layer

PROJECT_NAME = "Chihuahua-Muffin"
TRAIN_TABLE_NAME = "train"
VAL_TABLE_NAME = "val"
UNLABELED_TABLE_NAME = "unlabeled"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train chihuahua-vs-muffin ResNet-18 with 3LC")
    p.add_argument("--data-root", default="data", help="Root folder containing train/val/unlabeled/test sub-dirs")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-name", default=None, help="Optional 3LC run name (auto-generated if omitted)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# 3LC table helpers
# ---------------------------------------------------------------------------

def build_or_load_image_table(folder: str, table_name: str, project_name: str) -> tlc.Table:
    """Return the latest revision of a 3LC Table backed by *folder*.

    On the very first run the table is created (table_name="initial").
    On subsequent runs, ``if_exists="reuse"`` returns the existing table and
    ``.latest()`` resolves to whatever revision the user last saved in the
    Dashboard, so edits are automatically picked up.
    """
    folder = str(Path(folder).resolve())
    table = tlc.Table.from_image_folder(
        folder,
        project_name=project_name,
        dataset_name=table_name,
        table_name="initial",
        if_exists="reuse",
    )
    # Always train on the most recent revision (picks up Dashboard edits)
    return table.latest()


def build_or_load_unlabeled_table(folder: str, project_name: str) -> tlc.Table:
    """Return the latest revision of the unlabeled 3LC Table.

    Images in this folder have no ground-truth label; 3LC stores them with
    ``label = "undefined"`` so they can be labelled in the Dashboard.
    """
    folder = str(Path(folder).resolve())
    table = tlc.Table.from_image_folder(
        folder,
        project_name=project_name,
        dataset_name=UNLABELED_TABLE_NAME,
        table_name="initial",
        if_exists="reuse",
        default_label="undefined",
    )
    return table.latest()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 2):
    """ResNet-18 trained from scratch (no pretrained weights)."""
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# 3LC metrics collection (embeddings + per-sample predictions)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_metrics(model, table: tlc.Table, transform, device, run: tlc.Run, split: str):
    """
    Collect per-sample embeddings, predicted class, and confidence and
    write them into the 3LC Run so they appear in the Dashboard.
    """
    model.eval()

    # Hook to capture penultimate (avgpool) activations
    embeddings_list = []
    def _hook(module, inp, out):
        embeddings_list.append(out.flatten(1).cpu())

    handle = model.avgpool.register_forward_hook(_hook)

    dataset = tlc.TLCDataset(table, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_preds = []
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())  # prob of muffin
        all_labels.extend(labels.tolist())

    handle.remove()
    embeddings = torch.cat(embeddings_list, dim=0).numpy()

    # Build metrics dict expected by 3LC
    metrics = {
        tlc.PREDICTED_LABEL: all_preds,
        "muffin_confidence": all_probs,
        tlc.EMBEDDINGS: embeddings.tolist(),
    }
    if all_labels and any(l != -1 for l in all_labels):
        metrics[tlc.LABEL] = all_labels

    run.add_metrics_data(
        table=table,
        metrics=metrics,
        split=split,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)
    data_root = Path(args.data_root)

    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # 1. Build / load 3LC tables                                          #
    # ------------------------------------------------------------------ #
    print("Loading 3LC tables…")
    train_table = build_or_load_image_table(
        str(data_root / "train"), TRAIN_TABLE_NAME, PROJECT_NAME
    )
    val_table = build_or_load_image_table(
        str(data_root / "val"), VAL_TABLE_NAME, PROJECT_NAME
    )

    unlabeled_path = data_root / "unlabeled"
    collect_unlabeled = unlabeled_path.exists()
    if collect_unlabeled:
        unlabeled_table = build_or_load_unlabeled_table(str(unlabeled_path), PROJECT_NAME)

    # ------------------------------------------------------------------ #
    # 2. Torch datasets / loaders from 3LC tables (honours sample weights)#
    # ------------------------------------------------------------------ #
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    # 3LC dataset respects the weight column — zero-weight rows are skipped
    train_dataset = tlc.TLCDataset(train_table, transform=train_transform)
    val_dataset = tlc.TLCDataset(val_table, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # ------------------------------------------------------------------ #
    # 3. Model, optimiser, loss                                           #
    # ------------------------------------------------------------------ #
    model = build_model(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ #
    # 4. 3LC Run                                                          #
    # ------------------------------------------------------------------ #
    run_name = args.run_name or f"resnet18_{int(time.time())}"
    run = tlc.Run.from_url(
        run_name=run_name,
        project_name=PROJECT_NAME,
        description="ResNet-18 trained from scratch on chihuahua-muffin",
        parameters={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "device": str(device),
        },
        tables={
            "train": train_table,
            "val": val_table,
        },
    )

    # ------------------------------------------------------------------ #
    # 5. Training loop                                                    #
    # ------------------------------------------------------------------ #
    best_val_acc = 0.0
    best_model_state = None

    print(f"\nStarting training for {args.epochs} epochs…")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # Log scalars to 3LC Run
        run.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Save best model weights                                          #
    # ------------------------------------------------------------------ #
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model_path = Path("best_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved best model → {model_path}")

    # ------------------------------------------------------------------ #
    # 7. Collect per-sample metrics + embeddings into the 3LC Run        #
    # ------------------------------------------------------------------ #
    print("\nCollecting per-sample metrics for 3LC Dashboard…")
    collect_metrics(model, train_table, val_transform, device, run, split="train")
    collect_metrics(model, val_table, val_transform, device, run, split="val")
    if collect_unlabeled:
        print("Collecting metrics for unlabeled set (for active-labeling)…")
        collect_metrics(model, unlabeled_table, val_transform, device, run, split="unlabeled")

    run.finalize()
    print("\nRun finalized. Open the 3LC Dashboard to inspect embeddings and label samples.")
    print("Then retrain with:  python train.py")


if __name__ == "__main__":
    main()
