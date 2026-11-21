import json
import os


class CocoWriter:
    """Accumulates images and horizontal boxes and writes a COCO-format json file."""

    def __init__(self, class_names):
        self.categories = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]
        self.cat_id = {n: i + 1 for i, n in enumerate(class_names)}
        self.images, self.annotations = [], []

    def add_image(self, file_name, width, height):
        image_id = len(self.images) + 1
        self.images.append({"id": image_id, "file_name": file_name.replace("\\", "/"),
                            "width": int(width), "height": int(height)})
        return image_id

    def add_box(self, image_id, class_name, x1, y1, x2, y2):
        w, h = float(x2) - float(x1), float(y2) - float(y1)
        if w <= 0 or h <= 0:
            return
        self.annotations.append({"id": len(self.annotations) + 1, "image_id": image_id,
                                 "category_id": self.cat_id[class_name],
                                 "bbox": [float(x1), float(y1), w, h], "area": w * h, "iscrowd": 0})

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"images": self.images, "annotations": self.annotations,
                       "categories": self.categories}, f)
        print(f"{path}: {len(self.images)} images, {len(self.annotations)} boxes, "
              f"{len(self.categories)} classes")


def clip_box_to_tile(box, x0, y0, size, min_visible):
    """Clips an absolute box to a tile; returns tile-relative box or None."""
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    cx1, cy1 = max(x1, x0), max(y1, y0)
    cx2, cy2 = min(x2, x0 + size), min(y2, y0 + size)
    if cx2 <= cx1 or cy2 <= cy1 or area <= 0:
        return None
    if (cx2 - cx1) * (cy2 - cy1) < min_visible * area:
        return None
    return cx1 - x0, cy1 - y0, cx2 - x0, cy2 - y0


def tile_origins(length, size):
    """Non-overlapping tile origins; the last tile is padded when the image is smaller."""
    return list(range(0, max(length, 1), size))
