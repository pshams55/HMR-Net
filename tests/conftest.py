import json
import os
import random
import sys

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hmrnet.utils import load_config, merge_dicts  # noqa: E402


def make_coco_dataset(root, name, classes, n_images, size=128, seed=0, fixed_labels=None):
    rng = random.Random(seed)
    img_dir = os.path.join(root, name)
    os.makedirs(img_dir, exist_ok=True)
    images, anns = [], []
    for i in range(n_images):
        arr = (np.random.RandomState(seed + i).rand(size, size, 3) * 255).astype(np.uint8)
        fn = f"{i}.png"
        Image.fromarray(arr).save(os.path.join(img_dir, fn))
        images.append({"id": i + 1, "file_name": fn, "width": size, "height": size})
        for _ in range(rng.randint(1, 3)):
            x, y = rng.randint(0, size - 48), rng.randint(0, size - 48)
            w, h = rng.randint(10, 40), rng.randint(10, 40)
            cat = fixed_labels[i] if fixed_labels and i in fixed_labels else rng.randint(1, len(classes))
            anns.append({"id": len(anns) + 1, "image_id": i + 1, "category_id": cat,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    ann_file = os.path.join(root, f"{name}.json")
    with open(ann_file, "w") as f:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": i + 1, "name": c} for i, c in enumerate(classes)]}, f)
    return {"name": name, "train_ann": ann_file, "train_img": img_dir, "test_ann": ann_file, "test_img": img_dir}


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    """Three small synthetic datasets with overlapping class names and a tiny model."""
    root = str(tmp_path_factory.mktemp("data"))
    a = make_coco_dataset(root, "A", ["plane", "storage-tank", "ship"], 6, seed=1)
    b = make_coco_dataset(root, "B", ["airplane", "storagetank", "vehicle", "windmill"], 6, seed=2,
                          fixed_labels={0: 4, 1: 4})
    c = make_coco_dataset(root, "C", ["ship", "vehicle"], 4, seed=3)
    base = load_config(os.path.join(ROOT, "configs", "hmrnet_multidomain.yaml"))
    return merge_dicts(base, {
        "datasets": [a, b, c],
        "train_datasets": ["A", "B"],
        "test_datasets": ["A", "B", "C"],
        "work_dir": os.path.join(root, "work"),
        "model": {"input_size": 128, "pretrained_backbone": False, "score_thresh": 0.0,
                  "domain_routing": {"num_experts": 3, "expert_hidden": 32},
                  "scene_routing": {"max_experts": 4, "expert_hidden": 32}},
        "augment": {"scale_range": [0.9, 1.1]},
        "train": {"embedding_init_images": 4, "num_workers": 0},
    })


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
