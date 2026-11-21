"""DOTA-v1.0 -> 800x800 non-overlapping tiles with horizontal boxes in COCO json."""
import argparse
import glob
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmrnet.data.coco_writer import CocoWriter, clip_box_to_tile, tile_origins  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court",
    "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter",
    "roundabout", "soccer-ball-field", "swimming-pool",
]


def read_label_file(path, skip_difficult):
    boxes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue                                          # header lines (imagesource, gsd)
            xs = [float(v) for v in parts[0:8:2]]
            ys = [float(v) for v in parts[1:8:2]]
            cls = parts[8]
            difficult = len(parts) > 9 and parts[9] == "1"
            if cls not in DOTA_CLASSES or (difficult and skip_difficult):
                continue
            boxes.append((cls, min(xs), min(ys), max(xs), max(ys)))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="split folder containing images/ and labelTxt/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True, help="output json name, e.g. dota_train")
    ap.add_argument("--tile", type=int, default=800)
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--skip-difficult", action="store_true")
    ap.add_argument("--keep-empty", action="store_true")
    args = ap.parse_args()

    label_dir = next(os.path.join(args.root, d) for d in ["labelTxt", "labelTxt-v1.0"]
                     if os.path.isdir(os.path.join(args.root, d)))
    tile_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(tile_dir, exist_ok=True)
    writer = CocoWriter(DOTA_CLASSES)

    images = sorted(glob.glob(os.path.join(args.root, "images", "*.png")))
    for n, path in enumerate(images):
        stem = os.path.splitext(os.path.basename(path))[0]
        boxes = read_label_file(os.path.join(label_dir, f"{stem}.txt"), args.skip_difficult)
        img = Image.open(path).convert("RGB")
        W, H = img.size
        for y0 in tile_origins(H, args.tile):
            for x0 in tile_origins(W, args.tile):
                kept = []
                for cls, *b in boxes:
                    c = clip_box_to_tile(b, x0, y0, args.tile, args.min_visible)
                    if c is not None:
                        kept.append((cls, c))
                if not kept and not args.keep_empty:
                    continue
                tile = Image.new("RGB", (args.tile, args.tile))
                tile.paste(img.crop((x0, y0, min(x0 + args.tile, W), min(y0 + args.tile, H))), (0, 0))
                fname = f"{stem}_{x0}_{y0}.png"
                tile.save(os.path.join(tile_dir, fname))
                iid = writer.add_image(os.path.join(args.name, fname), args.tile, args.tile)
                for cls, c in kept:
                    writer.add_box(iid, cls, *c)
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(images)} images tiled")
    writer.save(os.path.join(args.out_dir, f"{args.name}.json"))


if __name__ == "__main__":
    main()
