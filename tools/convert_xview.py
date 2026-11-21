"""xView (geojson) -> 800x800 non-overlapping tiles in COCO json, with an image-level
train/test split of the labelled training images."""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmrnet.data.coco_writer import CocoWriter, clip_box_to_tile, tile_origins  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

XVIEW_CLASSES = {
    11: "Fixed-wing Aircraft", 12: "Small Aircraft", 13: "Cargo Plane", 15: "Helicopter",
    17: "Passenger Vehicle", 18: "Small Car", 19: "Bus", 20: "Pickup Truck", 21: "Utility Truck",
    23: "Truck", 24: "Cargo Truck", 25: "Truck w/Box", 26: "Truck Tractor", 27: "Trailer",
    28: "Truck w/Flatbed", 29: "Truck w/Liquid", 32: "Crane Truck", 33: "Railway Vehicle",
    34: "Passenger Car", 35: "Cargo Car", 36: "Flat Car", 37: "Tank car", 38: "Locomotive",
    40: "Maritime Vessel", 41: "Motorboat", 42: "Sailboat", 44: "Tugboat", 45: "Barge",
    47: "Fishing Vessel", 49: "Ferry", 50: "Yacht", 51: "Container Ship", 52: "Oil Tanker",
    53: "Engineering Vehicle", 54: "Tower crane", 55: "Container Crane", 56: "Reach Stacker",
    57: "Straddle Carrier", 59: "Mobile Crane", 60: "Dump Truck", 61: "Haul Truck",
    62: "Scraper/Tractor", 63: "Front loader/Bulldozer", 64: "Excavator", 65: "Cement Mixer",
    66: "Ground Grader", 71: "Hut/Tent", 72: "Shed", 73: "Building", 74: "Aircraft Hangar",
    76: "Damaged Building", 77: "Facility", 79: "Construction Site", 83: "Vehicle Lot",
    84: "Helipad", 86: "Storage Tank", 89: "Shipping container lot", 91: "Shipping Container",
    93: "Pylon", 94: "Tower",
}


def load_boxes(geojson_path):
    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)
    per_image = defaultdict(list)
    for feat in data["features"]:
        p = feat["properties"]
        t = int(p["type_id"])
        if t not in XVIEW_CLASSES:
            continue
        x1, y1, x2, y2 = (float(v) for v in p["bounds_imcoords"].split(","))
        if x2 > x1 and y2 > y1:
            per_image[p["image_id"]].append((XVIEW_CLASSES[t], x1, y1, x2, y2))
    return per_image


def tile_split(image_ids, per_image, args, split):
    tile_dir = os.path.join(args.out_dir, f"xview_{split}")
    os.makedirs(tile_dir, exist_ok=True)
    writer = CocoWriter(list(XVIEW_CLASSES.values()))
    for n, image_id in enumerate(image_ids):
        path = os.path.join(args.images, image_id)
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        W, H = img.size
        stem = os.path.splitext(image_id)[0]
        for y0 in tile_origins(H, args.tile):
            for x0 in tile_origins(W, args.tile):
                kept = []
                for cls, *b in per_image[image_id]:
                    c = clip_box_to_tile(b, x0, y0, args.tile, args.min_visible)
                    if c is not None:
                        kept.append((cls, c))
                if not kept and not (args.keep_empty_test and split == "test"):
                    continue
                tile = Image.new("RGB", (args.tile, args.tile))
                tile.paste(img.crop((x0, y0, min(x0 + args.tile, W), min(y0 + args.tile, H))), (0, 0))
                fname = f"{stem}_{x0}_{y0}.jpg"
                tile.save(os.path.join(tile_dir, fname), quality=95)
                iid = writer.add_image(os.path.join(f"xview_{split}", fname), args.tile, args.tile)
                for cls, c in kept:
                    writer.add_box(iid, cls, *c)
        if (n + 1) % 50 == 0:
            print(f"[{split}] {n + 1}/{len(image_ids)} images tiled")
    writer.save(os.path.join(args.out_dir, f"xview_{split}.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with xView train_images/*.tif")
    ap.add_argument("--geojson", required=True, help="xView_train.geojson")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=800)
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--keep-empty-test", action="store_true")
    args = ap.parse_args()

    per_image = load_boxes(args.geojson)
    ids = sorted(per_image)
    random.Random(args.seed).shuffle(ids)
    n_test = int(round(len(ids) * args.test_ratio))
    split = {"test": sorted(ids[:n_test]), "train": sorted(ids[n_test:])}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "xview_split.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    for name in ("train", "test"):
        tile_split(split[name], per_image, args, name)


if __name__ == "__main__":
    main()
