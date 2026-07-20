"""
split_classification_train_val.py
-------------------------------------
Physically splits a class-labeled image folder (e.g. your yolo_crops\\train,
sam_crops, or bbox_crops) into TWO SEPARATE folders on disk: train/ and val/.

This is the classification-dataset equivalent of split_yolo_train_val.py --
same idea, but for ImageFolder-style class subfolders instead of YOLO's
images/labels pairs.

Before:
    <SOURCE_DIR>/
        no-damage/
        minor-damage/
        major-damage/
        destroyed/
        un-classified/

After:
    <OUTPUT_DIR>/
        train/
            no-damage/
            minor-damage/
            major-damage/
            destroyed/
            un-classified/
        val/
            no-damage/
            minor-damage/
            major-damage/
            destroyed/
            un-classified/

Split is stratified per class (same % held out from each damage category),
so class balance is preserved in both train and val.

Your original <SOURCE_DIR> is left untouched -- files are copied, not moved.

Usage:
    python split_classification_train_val.py ^
        --source_dir "D:\\yoloboundingboxes\\yolo_crops\\train" ^
        --output_dir "D:\\yoloboundingboxes\\yolo_crops_split" ^
        --val_ratio 0.15
"""

import shutil
import random
import argparse
from pathlib import Path

from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def get_args():
    parser = argparse.ArgumentParser(description="Split a class-labeled crop folder into train/val")
    parser.add_argument("--source_dir", type=str, required=True,
                         help="Folder containing class subfolders to split")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Where to write train/ and val/ subfolders")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", default=False,
                         help="Move files instead of copying (saves disk space, removes them from source_dir)")
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class subfolders found in {source_dir}")

    print(f"[info] Found {len(class_dirs)} class folders: {[d.name for d in class_dirs]}")

    transfer_fn = shutil.move if args.move else shutil.copy2
    action_word = "Moving" if args.move else "Copying"

    total_train = 0
    total_val = 0

    for class_dir in class_dirs:
        cls_name = class_dir.name
        files = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        random.shuffle(files)

        val_count = int(len(files) * args.val_ratio)
        val_files = files[:val_count]
        train_files = files[val_count:]

        train_out = output_dir / "train" / cls_name
        val_out = output_dir / "val" / cls_name
        train_out.mkdir(parents=True, exist_ok=True)
        val_out.mkdir(parents=True, exist_ok=True)

        for f in tqdm(train_files, desc=f"{action_word} {cls_name} -> train", leave=False):
            transfer_fn(str(f), str(train_out / f.name))
        for f in tqdm(val_files, desc=f"{action_word} {cls_name} -> val", leave=False):
            transfer_fn(str(f), str(val_out / f.name))

        print(f"  {cls_name:15s}: {len(train_files)} train / {len(val_files)} val")
        total_train += len(train_files)
        total_val += len(val_files)

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"Train: {total_train} images -> {output_dir / 'train'}")
    print(f"Val  : {total_val} images -> {output_dir / 'val'}")


if __name__ == "__main__":
    main()
