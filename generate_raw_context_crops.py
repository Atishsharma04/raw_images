"""
generate_raw_context_crops.py
---------------------------------
Generates the "Original / Raw Building Images" arm of your three-arm
comparison -- crops centered on each building's ground-truth location,
but WITH generous surrounding context included (unlike your tight
bbox_crops arm). This represents the baseline "no preprocessing" case
your PDF describes: classifying directly from imagery where surrounding
roads, vegetation, neighboring buildings, and shadows are still present.

Uses the SAME ground-truth polygons as your bbox and SAM arms (via
Shapely), so all three arms are generated independently from the same
source labels -- just with different amounts of preprocessing applied:

    raw_context_crops  -> bounding box + generous padding (least processed)
    bbox_crops         -> tight bounding box, no padding
    sam_crops          -> precise building mask, background removed (most processed)

Usage:
    python generate_raw_context_crops.py ^
        --images_dir "D:\\segmentation\\train_images_labels_targets\\train\\images" ^
        --labels_dir "D:\\segmentation\\train_images_labels_targets\\train\\labels" ^
        --output_dir "D:\\segmentation\\train_images_labels_targets\\train\\raw_context_crops" ^
        --padding_ratio 0.5

--padding_ratio 0.5 means each side of the crop is expanded by 50% of the
building's own width/height (e.g. a 40x40 building box becomes roughly an
80x80 crop, centered on the building, capturing surrounding context).
"""

import json
import argparse
from pathlib import Path

from PIL import Image
from shapely import wkt
from tqdm import tqdm

VALID_CLASSES = {"no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"}


def get_args():
    parser = argparse.ArgumentParser(description="Generate raw/context-included building crops from ground-truth polygons")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--padding_ratio", type=float, default=0.5,
                         help="Fraction of the building's own width/height to add as padding on each side")
    parser.add_argument("--min_box_size", type=int, default=5)
    parser.add_argument("--image_ext", type=str, default=".png")
    return parser.parse_args()


def find_post_disaster_labels(labels_dir):
    all_labels = sorted(Path(labels_dir).glob("*.json"))
    post_labels = [p for p in all_labels if "post_disaster" in p.stem]
    skipped = len(all_labels) - len(post_labels)
    if skipped > 0:
        print(f"[info] Skipping {skipped} pre-disaster label files (no damage class available)")
    return post_labels


def main():
    args = get_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    for cls in VALID_CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    label_files = find_post_disaster_labels(labels_dir)
    print(f"[info] Found {len(label_files)} post-disaster label files")
    print(f"[info] Padding ratio: {args.padding_ratio} (each side expanded by this fraction of box width/height)")

    total_buildings = 0
    saved_buildings = 0
    skipped_small = 0
    skipped_bad_geom = 0
    missing_images = 0

    class_counts = {cls: 0 for cls in VALID_CLASSES}

    for label_path in tqdm(label_files, desc="Generating raw context crops"):
        image_stem = label_path.stem
        image_path = images_dir / f"{image_stem}{args.image_ext}"

        if not image_path.exists():
            alt_found = False
            for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
                alt_path = images_dir / f"{image_stem}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    alt_found = True
                    break
            if not alt_found:
                missing_images += 1
                continue

        with open(label_path, "r") as f:
            label_data = json.load(f)

        features = label_data.get("features", {}).get("xy", [])
        if not features:
            continue

        image = None

        for feat in features:
            props = feat.get("properties", {})
            if props.get("feature_type") != "building":
                continue

            subtype = props.get("subtype", "un-classified")
            if subtype not in VALID_CLASSES:
                subtype = "un-classified"

            total_buildings += 1

            try:
                polygon = wkt.loads(feat["wkt"])
                if not polygon.is_valid or polygon.is_empty:
                    skipped_bad_geom += 1
                    continue
                minx, miny, maxx, maxy = polygon.bounds
            except Exception:
                skipped_bad_geom += 1
                continue

            box_w = maxx - minx
            box_h = maxy - miny
            if box_w < args.min_box_size or box_h < args.min_box_size:
                skipped_small += 1
                continue

            pad_x = box_w * args.padding_ratio
            pad_y = box_h * args.padding_ratio

            if image is None:
                image = Image.open(image_path).convert("RGB")

            padded_minx = max(0, int(minx - pad_x))
            padded_miny = max(0, int(miny - pad_y))
            padded_maxx = min(image.width, int(maxx + pad_x))
            padded_maxy = min(image.height, int(maxy + pad_y))

            crop = image.crop((padded_minx, padded_miny, padded_maxx, padded_maxy))

            building_id = props.get("uid", f"b{total_buildings}")
            out_name = f"{image_stem}_{building_id}.png"
            out_path = output_dir / subtype / out_name
            crop.save(out_path)

            class_counts[subtype] += 1
            saved_buildings += 1

    print("\n" + "=" * 60)
    print("RAW CONTEXT CROP GENERATION SUMMARY")
    print("=" * 60)
    print(f"Label files processed : {len(label_files)}")
    print(f"Total buildings found : {total_buildings}")
    print(f"Crops saved           : {saved_buildings}")
    print(f"Skipped (too small)   : {skipped_small}")
    print(f"Skipped (bad geometry): {skipped_bad_geom}")
    print(f"Missing source images : {missing_images}")
    print("\nPer-class counts:")
    for cls, count in class_counts.items():
        print(f"  {cls:15s}: {count}")
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
