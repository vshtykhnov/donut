#!/usr/bin/env python
"""Utilities to prepare data for Donut OCR training."""
import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def find_gt(img: Path) -> Path | None:
    for ext in (".json", ".jsonc"):
        gt = img.with_suffix(ext)
        if gt.exists():
            return gt
    return None


def prepare_cards(cards: Path, out: Path, train_ratio: float, val_ratio: float, move: bool) -> None:
    random.seed(42)
    pairs: list[tuple[Path, Path]] = []
    for img in cards.iterdir():
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        gt = find_gt(img)
        if gt is None:
            print(f"[WARN] ground truth for '{img.name}' not found – skipping")
            continue
        pairs.append((img, gt))
    if not pairs:
        print("No card/ground truth pairs found – nothing to do")
        return

    train_img = out / "images" / "train"
    val_img = out / "images" / "val"
    test_img = out / "images" / "test"
    train_gt = out / "ground_truth" / "train"
    val_gt = out / "ground_truth" / "val"
    test_gt = out / "ground_truth" / "test"
    for d in (train_img, val_img, test_img, train_gt, val_gt, test_gt):
        d.mkdir(parents=True, exist_ok=True)

    random.shuffle(pairs)
    split1 = int(len(pairs) * train_ratio)
    split2 = int(len(pairs) * (train_ratio + val_ratio))
    train_pairs = pairs[:split1]
    val_pairs = pairs[split1:split2]
    test_pairs = pairs[split2:]

    def transfer(src: Path, dst: Path):
        if move:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

    for img, gt in train_pairs:
        transfer(img, train_img / img.name)
        transfer(gt, train_gt / gt.name)

    for img, gt in val_pairs:
        transfer(img, val_img / img.name)
        transfer(gt, val_gt / gt.name)

    for img, gt in test_pairs:
        transfer(img, test_img / img.name)
        transfer(gt, test_gt / gt.name)

    print(
        f"Done! Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Donut dataset utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Prepare Donut dataset from cards")
    p_prep.add_argument("--cards", default="cards", help="Source cards directory")
    p_prep.add_argument("--out", default="donut_dataset", help="Output dataset root")
    p_prep.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training split ratio",
    )
    p_prep.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio",
    )
    p_prep.add_argument("--move", action="store_true", help="Move files instead of copy")
    p_prep.set_defaults(
        func=lambda a: prepare_cards(Path(a.cards), Path(a.out), a.train_ratio, a.val_ratio, a.move)
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
