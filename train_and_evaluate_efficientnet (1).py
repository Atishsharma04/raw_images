"""
train_and_evaluate_efficientnet.py
-------------------------------------
Trains and evaluates a lightweight EfficientNet-B0 classifier for building
damage assessment (No Damage / Minor Damage / Major Damage / Destroyed)
on cropped building images.

Same structure and outputs as train_and_evaluate_mobilenetv3.py -- run
this on the same three arms (raw_context_crops, bbox_crops, sam_crops)
to get the MobileNetV3-vs-EfficientNet-B0 comparison your PDF's Stage IV
and RQ3 require.

Matches this folder layout:

    <ROOT>\\train_images_labels_targets\\train\\sam_crops\\ (or bbox_crops, raw_context_crops)
        destroyed\\
        major-damage\\
        minor-damage\\
        no-damage\\
        un-classified\\        <- excluded by default (not a real damage class)

    <ROOT>\\test_images_labels_targets\\test\\sam_crops\\ (matching structure)

A validation split is automatically carved out of the train set
(stratified by class), OR you can pass --val_dir to use an explicit
physical validation folder instead (e.g. from split_classification_train_val.py).

Produces:
    outputs/<run_name>/
        best_model.pth
        history.json
        history_curves.png          (loss / accuracy / F1 vs epoch)
        confusion_matrix.png
        per_class_f1_bar.png
        test_metrics.json
        results_table.csv           (summary table: accuracy, precision, recall, F1 per class + macro)

Usage:
    python train_and_evaluate_efficientnet.py ^
        --train_dir "D:\\segmentation\\train_images_labels_targets\\train\\sam_crops" ^
        --test_dir  "D:\\segmentation\\test_images_labels_targets\\test\\sam_crops" ^
        --run_name sam_arm_efficientnet --epochs 12

Run with --help to see all options.
"""

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, just saving PNGs
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

DEFAULT_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train + evaluate EfficientNet-B0 for building damage classification")
    parser.add_argument("--train_dir", type=str, required=True,
                         help="Path to sam_crops folder containing train class subfolders")
    parser.add_argument("--test_dir", type=str, required=True,
                         help="Path to sam_crops folder containing test class subfolders")
    parser.add_argument("--run_name", type=str, default="efficientnet_run")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--classes", type=str, nargs="+", default=DEFAULT_CLASSES,
                         help="Which class subfolders to use (excludes un-classified by default)")
    parser.add_argument("--model_variant", type=str, default="b0", choices=["b0"],
                         help="Kept for compatibility with the MobileNetV3 script's CLI structure; only b0 is used")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_dir", type=str, default=None,
                         help="Optional path to a SEPARATE physical validation folder (class subfolders, "
                              "same format as train_dir). If provided, this is used directly instead of "
                              "carving a validation split out of train_dir in-memory.")
    parser.add_argument("--val_split", type=float, default=0.15,
                         help="Fraction of train set carved out for validation (only used if --val_dir is NOT provided)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to last_checkpoint.pth to resume training from")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class DamageCropDataset(Dataset):
    """Scans only the specified class subfolders (ignores anything else,
    e.g. un-classified, unless explicitly included in `classes`)."""

    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.samples = []

        for cls in classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                print(f"[warn] Class folder not found, skipping: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(IMG_EXTS):
                    self.samples.append((str(cls_dir / fname), self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root_dir} for classes {classes}")

        print(f"[data] {root_dir} -> {len(self.samples)} images across {len(classes)} classes")
        for cls in classes:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[cls])
            print(f"        {cls:15s}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_labels(self):
        return [label for _, label in self.samples]


class TransformSubset(Dataset):
    """Wraps a Subset so train/val split from the same base dataset can use
    different transforms (train augmentation vs plain val preprocessing)."""

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.base_dataset.samples[self.indices[idx]]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf


def build_dataloaders(args):
    train_tf, eval_tf = build_transforms(args.img_size)

    test_ds = DamageCropDataset(args.test_dir, args.classes, transform=eval_tf)

    if args.val_dir is not None:
        # Explicit physical validation folder provided -- use it directly,
        # no in-memory splitting needed.
        print(f"[data] Using explicit validation folder: {args.val_dir}")
        train_full_ds = DamageCropDataset(args.train_dir, args.classes, transform=None)
        val_ds_raw = DamageCropDataset(args.val_dir, args.classes, transform=eval_tf)

        train_indices = list(range(len(train_full_ds)))
        train_ds = TransformSubset(train_full_ds, train_indices, train_tf)
        val_ds = val_ds_raw  # already has eval_tf applied via its own transform param
        train_labels = train_full_ds.get_labels()

        print(f"[data] Train: {len(train_ds)} images | Val (separate folder): {len(val_ds)} images")

    else:
        # No explicit val_dir -- carve validation split out of train_dir in-memory.
        full_train_ds = DamageCropDataset(args.train_dir, args.classes, transform=None)
        labels = full_train_ds.get_labels()
        indices = list(range(len(full_train_ds)))

        train_idx, val_idx = train_test_split(
            indices, test_size=args.val_split, stratify=labels, random_state=args.seed
        )

        train_ds = TransformSubset(full_train_ds, train_idx, train_tf)
        val_ds = TransformSubset(full_train_ds, val_idx, eval_tf)
        train_labels = [labels[i] for i in train_idx]

        print(f"[data] Train/Val split: {len(train_ds)} / {len(val_ds)} (val_split={args.val_split})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_labels


def compute_class_weights(train_labels, num_classes, device):
    """Computed directly from the label list (no image decoding needed --
    iterating the DataLoader here would needlessly decode every image)."""
    counts = torch.zeros(num_classes)
    for label in train_labels:
        counts[label] += 1
    counts = torch.clamp(counts, min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(variant, num_classes, freeze_backbone, device):
    # variant kept for CLI compatibility with the MobileNetV3 script, but
    # only "b0" is used here (EfficientNet-B0 -- the lightweight variant
    # specified in your PDF alongside MobileNetV3)
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model.to(device)


# --------------------------------------------------------------------------
# Train / Eval loops
# --------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train=True, desc=""):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    torch.set_grad_enabled(train)
    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1_macro, all_preds, all_labels


def evaluate_full(model, loader, criterion, device, class_names):
    loss, acc, f1_macro, preds, labels = run_epoch(model, loader, criterion, None, device, train=False)

    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)

    metrics = {
        "loss": loss,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision,
        "recall_macro": recall,
        "per_class_precision": {class_names[i]: float(per_class_precision[i]) for i in range(len(class_names))},
        "per_class_recall": {class_names[i]: float(per_class_recall[i]) for i in range(len(class_names))},
        "per_class_f1": {class_names[i]: float(per_class_f1[i]) for i in range(len(class_names))},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return metrics


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_history_curves(history, out_path, run_name):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
    axes[0].set_title(f"{run_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="Train Acc")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="Val Acc")
    axes[1].set_title(f"{run_name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [h["train_f1"] for h in history], label="Train F1 (macro)")
    axes[2].plot(epochs, [h["val_f1"] for h in history], label="Val F1 (macro)")
    axes[2].set_title(f"{run_name} - F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, out_path, run_name):
    cm = np.array(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{run_name} - Confusion Matrix (row-normalized)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)

    fig.colorbar(im, ax=ax, label="Row-normalized fraction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_class_bar(metrics, class_names, out_path, run_name):
    x = np.arange(len(class_names))
    width = 0.25

    precision = [metrics["per_class_precision"][c] for c in class_names]
    recall = [metrics["per_class_recall"][c] for c in class_names]
    f1 = [metrics["per_class_f1"][c] for c in class_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"{run_name} - Per-Class Test Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results_table(metrics, class_names, out_path, run_name):
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "class", "precision", "recall", "f1", "accuracy"])
        for c in class_names:
            writer.writerow([run_name, c,
                              round(metrics["per_class_precision"][c], 4),
                              round(metrics["per_class_recall"][c], 4),
                              round(metrics["per_class_f1"][c], 4),
                              ""])  # accuracy is a single overall number, not per-class
        writer.writerow([run_name, "MACRO_AVG",
                          round(metrics["precision_macro"], 4),
                          round(metrics["recall_macro"], 4),
                          round(metrics["f1_macro"], 4),
                          ""])
        writer.writerow([run_name, "OVERALL_ACCURACY", "", "", "", round(metrics["accuracy"], 4)])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Using device: {device}")
    if device.type == "cuda":
        print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = args.classes
    print(f"[data] Using classes: {class_names}")

    train_loader, val_loader, test_loader, train_labels = build_dataloaders(args)
    num_classes = len(class_names)

    print(f"[model] Building EfficientNet-B0 (pretrained ImageNet)...")
    model = build_model(args.model_variant, num_classes, args.freeze_backbone, device)

    if args.use_class_weights:
        print("[data] Computing class weights (inverse frequency)...")
        class_weights = compute_class_weights(train_labels, num_classes, device)
        print(f"[data] Class weights: {dict(zip(class_names, class_weights.cpu().numpy().round(3)))}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_f1 = -1.0
    epochs_no_improve = 0
    history = []
    start_epoch = 1

    if args.resume is not None and Path(args.resume).exists():
        print(f"[resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["best_val_f1"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        history = ckpt["history"]
        print(f"[resume] Resuming from epoch {start_epoch}, best_val_f1={best_val_f1:.4f}")

    print(f"[train] Starting training for up to {args.epochs} epochs (from epoch {start_epoch})...")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            desc=f"Epoch {epoch}/{args.epochs} [train]"
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, val_loader, criterion, None, device, train=False,
            desc=f"Epoch {epoch}/{args.epochs} [val]"
        )

        scheduler.step(val_f1)
        elapsed = time.time() - t0

        print(f"[epoch {epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
              f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            ckpt_path = out_dir / "best_model.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "val_f1": val_f1,
                "args": vars(args),
            }, ckpt_path)
            print(f"[checkpoint] New best model saved (val_f1={val_f1:.4f}) -> {ckpt_path}")
        else:
            epochs_no_improve += 1

        # Always save a resumable checkpoint after every epoch, so an
        # interrupted run (crash, power loss, Ctrl+C) can pick back up
        # with --resume instead of starting over from epoch 1.
        last_ckpt_path = out_dir / "last_checkpoint.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_names": class_names,
            "epoch": epoch,
            "best_val_f1": best_val_f1,
            "epochs_no_improve": epochs_no_improve,
            "history": history,
            "args": vars(args),
        }, last_ckpt_path)

        if epochs_no_improve >= args.patience:
            print(f"[early stop] No val_f1 improvement for {args.patience} epochs. Stopping.")
            break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history_curves(history, out_dir / "history_curves.png", args.run_name)

    # -----------------------------------------------------------------
    # Final test evaluation using best checkpoint
    # -----------------------------------------------------------------
    print("[test] Loading best checkpoint for final test evaluation...")
    ckpt = torch.load(out_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_full(model, test_loader, criterion, device, class_names)

    print("\n" + "=" * 60)
    print(f"TEST RESULTS ({args.run_name})")
    print("=" * 60)
    print(f"Accuracy      : {test_metrics['accuracy']:.4f}")
    print(f"F1 (macro)    : {test_metrics['f1_macro']:.4f}")
    print(f"Precision     : {test_metrics['precision_macro']:.4f}")
    print(f"Recall        : {test_metrics['recall_macro']:.4f}")
    print("\nPer-class F1:")
    for cls, f1 in test_metrics["per_class_f1"].items():
        print(f"  {cls:20s}: {f1:.4f}")
    print("\nFull classification report:")
    print(test_metrics["classification_report"])

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    plot_confusion_matrix(test_metrics["confusion_matrix"], class_names,
                           out_dir / "confusion_matrix.png", args.run_name)
    plot_per_class_bar(test_metrics, class_names,
                        out_dir / "per_class_f1_bar.png", args.run_name)
    save_results_table(test_metrics, class_names,
                        out_dir / "results_table.csv", args.run_name)

    print(f"\n[done] All results, graphs, and tables saved to: {out_dir}")
    print("  - best_model.pth")
    print("  - history.json / history_curves.png")
    print("  - test_metrics.json")
    print("  - confusion_matrix.png")
    print("  - per_class_f1_bar.png")
    print("  - results_table.csv")


if __name__ == "__main__":
    main()
