"""DIOR (horizontal boxes, VOC xml) -> COCO json. Images stay in place (800x800)."""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmrnet.data.coco_writer import CocoWriter  # noqa: E402

DIOR_CLASSES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge", "chimney", "dam",
    "Expressway-Service-area", "Expressway-toll-station", "golffield", "groundtrackfield", "harbor",
    "overpass", "ship", "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill",
]


def find_ann_dir(root):
    for cand in ["Annotations/Horizontal Bounding Boxes", "Annotations/HBB", "Annotations"]:
        p = os.path.join(root, cand)
        if os.path.isdir(p) and any(f.endswith(".xml") for f in os.listdir(p)):
            return p
    raise FileNotFoundError("DIOR horizontal-box annotation folder not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="DIOR root folder")
    ap.add_argument("--splits", nargs="+", default=["trainval", "test"],
                    help="names in ImageSets/Main (trainval = train + val)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ann_dir = find_ann_dir(args.root)
    img_dirs = [d for d in ["JPEGImages-trainval", "JPEGImages-test", "JPEGImages"]
                if os.path.isdir(os.path.join(args.root, d))]
    lookup = {cls.lower(): cls for cls in DIOR_CLASSES}

    for split in args.splits:
        parts = ["train", "val"] if split == "trainval" else [split]
        ids = []
        for p in parts:
            with open(os.path.join(args.root, "ImageSets", "Main", f"{p}.txt"), encoding="utf-8") as f:
                ids += [l.strip() for l in f if l.strip()]

        writer = CocoWriter(DIOR_CLASSES)
        for image_id in ids:
            rel = next((os.path.join(d, f"{image_id}.jpg") for d in img_dirs
                        if os.path.exists(os.path.join(args.root, d, f"{image_id}.jpg"))), None)
            if rel is None:
                print(f"missing image {image_id}")
                continue
            tree = ET.parse(os.path.join(ann_dir, f"{image_id}.xml")).getroot()
            size = tree.find("size")
            w = int(size.find("width").text) if size is not None else 800
            h = int(size.find("height").text) if size is not None else 800
            iid = writer.add_image(rel, w, h)
            for obj in tree.iter("object"):
                name = lookup[obj.find("name").text.strip().lower()]
                bb = obj.find("bndbox")
                x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
                writer.add_box(iid, name, x1, y1, x2, y2)
        writer.save(os.path.join(args.out_dir, f"dior_{split}.json"))


if __name__ == "__main__":
    main()
