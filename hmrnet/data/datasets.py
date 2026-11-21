import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from .label_space import normalize_name

Image.MAX_IMAGE_PIXELS = None


class CocoDetection(Dataset):
    """COCO-format detection dataset that returns unified labels and its dataset index.

    Every converted benchmark (DIOR, DOTA tiles, xView tiles, NWPU) is stored in this
    format by the scripts in tools/.
    """

    def __init__(self, ann_file, img_root, dataset_index, label_map, transforms=None,
                 exclude_labels=None, image_filter=None, keep_empty=True):
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.img_root = img_root
        self.dataset_index = dataset_index
        self.label_map = label_map
        self.transforms = transforms
        self.categories = {c["id"]: c["name"] for c in data["categories"]}

        anns = defaultdict(list)
        for a in data["annotations"]:
            anns[a["image_id"]].append(a)

        exclude = set(exclude_labels or [])
        self.images, self.anns = [], {}
        for img in data["images"]:
            items = [a for a in anns.get(img["id"], []) if a["category_id"] in label_map]
            labels = {label_map[a["category_id"]] for a in items}
            if image_filter == "drop_excluded" and labels & exclude:
                continue
            if image_filter == "only_excluded" and (not labels or not labels <= exclude):
                continue
            if exclude and image_filter != "only_excluded":
                items = [a for a in items if label_map[a["category_id"]] not in exclude]
            if not items and not keep_empty:
                continue
            self.images.append(img)
            self.anns[img["id"]] = items

    def __len__(self):
        return len(self.images)

    def load_image(self, idx):
        info = self.images[idx]
        img = Image.open(os.path.join(self.img_root, info["file_name"])).convert("RGB")
        return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx):
        info = self.images[idx]
        image = self.load_image(idx)
        boxes, labels = [], []
        for a in self.anns[info["id"]]:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.label_map[a["category_id"]])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(info["id"]),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target, self.dataset_index


def collate_fn(batch):
    images, targets, ds = zip(*batch)
    return list(images), list(targets), torch.tensor(ds, dtype=torch.long)


def dataset_index(cfg, name):
    names = [d["name"] for d in cfg["datasets"]]
    return names.index(name)


def unseen_labels(cfg, label_space):
    z = cfg.get("zsd", {})
    return [label_space.label_of(n) for n in z.get("unseen_classes", [])] if z.get("enabled") else []


def build_dataset(cfg, label_space, name, split, transforms=None):
    ds_cfg = next(d for d in cfg["datasets"] if d["name"] == name)
    idx = dataset_index(cfg, name)
    exclude = unseen_labels(cfg, label_space)
    if split == "train":
        image_filter = "drop_excluded" if exclude else None
        return CocoDetection(ds_cfg["train_ann"], ds_cfg["train_img"], idx, label_space.dataset_maps[name],
                             transforms, exclude_labels=exclude, image_filter=image_filter, keep_empty=False)
    image_filter = "only_excluded" if exclude else None
    return CocoDetection(ds_cfg["test_ann"], ds_cfg["test_img"], idx, label_space.dataset_maps[name],
                         None, exclude_labels=exclude, image_filter=image_filter)


def build_train_set(cfg, label_space, transforms):
    parts = [build_dataset(cfg, label_space, n, "train", transforms) for n in cfg["train_datasets"]]
    return ConcatDataset(parts), parts


def class_names_of(dataset, label_space):
    return sorted({normalize_name(n) for n in dataset.categories.values()}, key=label_space.label_of)
