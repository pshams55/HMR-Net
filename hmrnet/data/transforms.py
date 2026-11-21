import random

import torch
import torch.nn.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            w = image.shape[-1]
            image = image.flip(-1)
            b = target["boxes"].clone()
            b[:, [0, 2]] = w - target["boxes"][:, [2, 0]]
            target["boxes"] = b
        return image, target


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            h = image.shape[-2]
            image = image.flip(-2)
            b = target["boxes"].clone()
            b[:, [1, 3]] = h - target["boxes"][:, [3, 1]]
            target["boxes"] = b
        return image, target


class RandomRotate90:
    """Rotation by a random multiple of 90 degrees (keeps horizontal boxes exact)."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        k = random.randint(1, 3)
        boxes = target["boxes"]
        for _ in range(k):
            h, w = image.shape[-2:]
            image = torch.rot90(image, 1, dims=(-2, -1))          # counter-clockwise
            x1, y1, x2, y2 = boxes.unbind(-1)
            boxes = torch.stack([y1, w - x2, y2, w - x1], dim=-1)
        target["boxes"] = boxes
        return image, target


class RandomScaleCrop:
    """Random rescaling followed by a crop or pad back to a fixed square size."""

    def __init__(self, size=800, scale_range=(0.8, 1.2), min_visible=0.5):
        self.size = size
        self.scale_range = scale_range
        self.min_visible = min_visible

    def __call__(self, image, target):
        s = random.uniform(*self.scale_range)
        h, w = image.shape[-2:]
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        image = F.interpolate(image[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
        boxes = target["boxes"] * torch.tensor([nw / w, nh / h, nw / w, nh / h])

        ox = random.randint(0, max(0, nw - self.size))
        oy = random.randint(0, max(0, nh - self.size))
        image = image[:, oy:oy + self.size, ox:ox + self.size]
        ch, cw = image.shape[-2:]
        if ch < self.size or cw < self.size:
            image = F.pad(image, (0, self.size - cw, 0, self.size - ch))

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxes = boxes - torch.tensor([ox, oy, ox, oy], dtype=boxes.dtype)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, cw)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, ch)
        new_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = (new_area > 1) & (new_area >= self.min_visible * area.clamp_min(1e-6))
        target["boxes"] = boxes[keep]
        target["labels"] = target["labels"][keep]
        return image, target


def build_train_transforms(cfg):
    a = cfg.get("augment", {})
    ts = []
    if a.get("hflip", 0.5) > 0:
        ts.append(RandomHorizontalFlip(a.get("hflip", 0.5)))
    if a.get("vflip", 0.5) > 0:
        ts.append(RandomVerticalFlip(a.get("vflip", 0.5)))
    if a.get("rot90", 0.5) > 0:
        ts.append(RandomRotate90(a.get("rot90", 0.5)))
    if a.get("scale_range"):
        ts.append(RandomScaleCrop(cfg["model"].get("input_size", 800), tuple(a["scale_range"])))
    return Compose(ts)
