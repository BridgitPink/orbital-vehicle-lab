from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import random
import shutil
import re

import cv2


TRAIN_LIST = Path("C:/Users/bridg/orbital-vehicle-lab/data/raw/cowc_detection/COWC_train_list_detection.txt")
TEST_LIST = Path("C:/Users/bridg/orbital-vehicle-lab/data/raw/cowc_detection/COWC_test_list_detection.txt")

PATCH_ROOT = Path("C:/Users/bridg/orbital-vehicle-lab/data/raw/cowc_detection")
OUT_ROOT = Path("C:/Users/bridg/orbital-vehicle-lab/data/processed/cowc_yolo_patch")

SEED = 42
VAL_RATIO = 0.1

CLASS_ID = 0
CLASS_NAME = "car"
BOX_SIZE_PX = 16


def parse_list_file(list_path: Path) -> list[tuple[str, int]]:
    if not list_path.exists():
        raise FileNotFoundError(f"Missing list file: {list_path}")

    items: list[tuple[str, int]] = []

    # utf-8-sig strips BOM automatically
    for raw_line in list_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        rel_path = parts[0].replace("\\", "/").strip()
        try:
            label = int(parts[-1])
        except ValueError:
            continue

        items.append((rel_path, label))

    return items


def normalize_name(name: str) -> str:
    name = name.replace("\\", "/").strip()
    name = re.sub(r"/+", "/", name)

    # Remove BOM if somehow still present
    name = name.lstrip("\ufeff")

    # Fix occasional 5-digit year typo like 20077 -> 2007
    name = re.sub(r"(19|20)(\d{2})\d(?=[^0-9])", r"\1\2", name)

    return name


def sanitize_filename(name: str) -> str:
    """
    Keep filenames Windows-safe and predictable.
    """
    name = name.lstrip("\ufeff")
    name = re.sub(r'[^A-Za-z0-9._-]+', "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def build_file_index(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]], dict[str, list[Path]]]:
    exact_rel_map: dict[str, Path] = {}
    basename_map: dict[str, list[Path]] = defaultdict(list)
    normalized_basename_map: dict[str, list[Path]] = defaultdict(list)

    for p in root.rglob("*.png"):
        rel = p.relative_to(root).as_posix()
        rel_norm = normalize_name(rel)
        exact_rel_map[rel_norm] = p

        basename_map[p.name].append(p)
        normalized_basename_map[normalize_name(p.name)].append(p)

    return exact_rel_map, basename_map, normalized_basename_map


def resolve_image_path(
    rel_path: str,
    exact_rel_map: dict[str, Path],
    basename_map: dict[str, list[Path]],
    normalized_basename_map: dict[str, list[Path]],
) -> Path:
    rel_norm = normalize_name(rel_path)

    # 1) exact relative path
    if rel_norm in exact_rel_map:
        return exact_rel_map[rel_norm]

    # 2) exact basename
    basename = Path(rel_norm).name
    candidates = basename_map.get(basename, [])
    if len(candidates) == 1:
        return candidates[0]

    # 3) normalized basename
    basename_norm = normalize_name(basename)
    candidates = normalized_basename_map.get(basename_norm, [])
    if len(candidates) == 1:
        return candidates[0]

    # 4) narrow by city and split if possible
    parts = Path(rel_norm).parts
    city = parts[0] if len(parts) > 0 else ""
    split = parts[1] if len(parts) > 1 else ""

    narrowed = [
        p for p in candidates
        if city in p.as_posix() and split in p.as_posix()
    ]
    if len(narrowed) == 1:
        return narrowed[0]

    raise FileNotFoundError(
        f"Could not resolve image path.\n"
        f"Requested: {rel_path}\n"
        f"Normalized: {rel_norm}\n"
        f"Basename candidates: {len(candidates)}"
    )


def make_output_dirs(root: Path) -> None:
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def clear_output_dir(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def yolo_center_box_from_image_shape(width: int, height: int, box_size_px: int) -> tuple[float, float, float, float]:
    bw = min(box_size_px, width) / width
    bh = min(box_size_px, height) / height
    xc = 0.5
    yc = 0.5
    return xc, yc, bw, bh


def write_label_file_from_source(label_path: Path, label: int, src_image_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)

    if label == 0:
        label_path.write_text("", encoding="utf-8")
        return

    image = cv2.imread(str(src_image_path))
    if image is None:
        raise ValueError(f"Failed to read source image: {src_image_path}")

    h, w = image.shape[:2]
    xc, yc, bw, bh = yolo_center_box_from_image_shape(w, h, BOX_SIZE_PX)

    label_text = f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
    label_path.write_text(label_text, encoding="utf-8")


def safe_output_stem(rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").lstrip("\ufeff")
    stem = Path(rel).stem
    parent = Path(rel).parent.as_posix().replace("/", "__")
    combined = f"{parent}__{stem}"
    return sanitize_filename(combined)


def copy_and_label(
    items: list[tuple[str, int]],
    split_name: str,
    exact_rel_map: dict[str, Path],
    basename_map: dict[str, list[Path]],
    normalized_basename_map: dict[str, list[Path]],
) -> Counter:
    stats = Counter()
    missing_examples = []

    for idx, (rel_path, label) in enumerate(items, start=1):
        try:
            src_img = resolve_image_path(
                rel_path,
                exact_rel_map,
                basename_map,
                normalized_basename_map,
            )
        except FileNotFoundError:
            stats["missing"] += 1
            if len(missing_examples) < 10:
                missing_examples.append(rel_path)
            continue

        out_stem = safe_output_stem(rel_path)
        dst_img = OUT_ROOT / "images" / split_name / f"{out_stem}{src_img.suffix.lower()}"
        dst_lbl = OUT_ROOT / "labels" / split_name / f"{out_stem}.txt"

        # Read source image for label creation first
        write_label_file_from_source(dst_lbl, label, src_img)

        # Then copy image
        shutil.copy2(src_img, dst_img)

        stats["total"] += 1
        if label == 1:
            stats["positive"] += 1
        else:
            stats["negative"] += 1

        if idx % 25000 == 0:
            print(f"[{split_name}] processed {idx}/{len(items)}")

    if missing_examples:
        print(f"\n[{split_name}] first missing examples:")
        for ex in missing_examples:
            print(f"  {ex}")

    return stats


def split_train_val(
    train_items: list[tuple[str, int]],
    val_ratio: float,
    seed: int
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    rng = random.Random(seed)
    items = train_items[:]
    rng.shuffle(items)

    n_val = int(len(items) * val_ratio)
    val_items = items[:n_val]
    train_items_final = items[n_val:]
    return train_items_final, val_items


def write_dataset_yaml(path: Path) -> None:
    yaml_text = (
        f"path: {OUT_ROOT.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n"
        f"  {CLASS_ID}: {CLASS_NAME}\n"
    )
    path.write_text(yaml_text, encoding="utf-8")


def main() -> None:
    print("Parsing official COWC split files...")
    train_items = parse_list_file(TRAIN_LIST)
    test_items = parse_list_file(TEST_LIST)

    print(f"Train items from official list: {len(train_items)}")
    print(f"Test items from official list : {len(test_items)}")

    print("\nIndexing extracted PNG files...")
    exact_rel_map, basename_map, normalized_basename_map = build_file_index(PATCH_ROOT)
    print(f"Indexed PNG files: {len(exact_rel_map)}")

    clear_output_dir(OUT_ROOT)
    make_output_dirs(OUT_ROOT)

    train_items_final, val_items = split_train_val(train_items, VAL_RATIO, SEED)

    print(f"\nFinal train items: {len(train_items_final)}")
    print(f"Validation items : {len(val_items)}")
    print(f"Test items       : {len(test_items)}")

    train_stats = copy_and_label(
        train_items_final, "train",
        exact_rel_map, basename_map, normalized_basename_map
    )
    val_stats = copy_and_label(
        val_items, "val",
        exact_rel_map, basename_map, normalized_basename_map
    )
    test_stats = copy_and_label(
        test_items, "test",
        exact_rel_map, basename_map, normalized_basename_map
    )

    write_dataset_yaml(OUT_ROOT / "dataset.yaml")

    print("\nDone.")
    print(f"Saved YOLO patch dataset to: {OUT_ROOT.resolve()}")

    print("\nSplit summary:")
    print(
        f"train -> total={train_stats['total']} "
        f"pos={train_stats['positive']} neg={train_stats['negative']} "
        f"missing={train_stats['missing']}"
    )
    print(
        f"val   -> total={val_stats['total']} "
        f"pos={val_stats['positive']} neg={val_stats['negative']} "
        f"missing={val_stats['missing']}"
    )
    print(
        f"test  -> total={test_stats['total']} "
        f"pos={test_stats['positive']} neg={test_stats['negative']} "
        f"missing={test_stats['missing']}"
    )
    print(f"\nYAML: {(OUT_ROOT / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    main()