"""NWPU VHR-10 -> COCO json with a seeded image-level train/test split."""
import argparse
import glob
import json
import os
import random
import re
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmrnet.data.coco_writer import CocoWriter  # noqa: E402

NWPU_CLASSES = ["airplane", "ship", "storage tank", "baseball diamond", "tennis court",
                "basketball court", "ground track field", "harbor", "bridge", "vehicle"]
LINE = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*,\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*,\s*(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="folder with 'positive image set', 'negative image set', 'ground truth'")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-negatives", action="store_true", help="include the 150 negative images")
    args = ap.parse_args()

    items = []
    for path in sorted(glob.glob(os.path.join(args.root, "positive image set", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        boxes = []
        with open(os.path.join(args.root, "ground truth", f"{stem}.txt"), encoding="utf-8") as f:
            for line in f:
                m = LINE.search(line)
                if m:
                    x1, y1, x2, y2, c = m.groups()
                    boxes.append((NWPU_CLASSES[int(c) - 1], float(x1), float(y1), float(x2), float(y2)))
        items.append((os.path.relpath(path, args.root), boxes))
    if args.with_negatives:
        for path in sorted(glob.glob(os.path.join(args.root, "negative image set", "*.jpg"))):
            items.append((os.path.relpath(path, args.root), []))

    random.Random(args.seed).shuffle(items)
    n_test = int(round(len(items) * args.test_ratio))
    splits = {"test": items[:n_test], "train": items[n_test:]}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "nwpu_split.json"), "w", encoding="utf-8") as f:
        json.dump({k: [i[0] for i in v] for k, v in splits.items()}, f, indent=1)

    for name, subset in splits.items():
        writer = CocoWriter(NWPU_CLASSES)
        for rel, boxes in subset:
            w, h = Image.open(os.path.join(args.root, rel)).size
            iid = writer.add_image(rel, w, h)
            for cls, x1, y1, x2, y2 in boxes:
                writer.add_box(iid, cls, x1, y1, x2, y2)
        writer.save(os.path.join(args.out_dir, f"nwpu_{name}.json"))


if __name__ == "__main__":
    main()
