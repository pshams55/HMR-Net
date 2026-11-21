import copy

import pytest
import torch

from hmrnet.data.transforms import RandomHorizontalFlip, RandomRotate90, RandomScaleCrop, RandomVerticalFlip


def _image_with_box():
    img = torch.zeros(3, 40, 60)
    img[:, 5:15, 10:30] = 1.0
    target = {"boxes": torch.tensor([[10.0, 5.0, 30.0, 15.0]]), "labels": torch.tensor([1])}
    return img, target


def _box_from_pixels(img):
    ys, xs = torch.nonzero(img[0] > 0.5, as_tuple=True)
    return torch.tensor([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float()


@pytest.mark.parametrize("transform", [RandomHorizontalFlip(1.0), RandomVerticalFlip(1.0), RandomRotate90(1.0)])
def test_boxes_follow_pixels(transform):
    for _ in range(6):
        img, target = _image_with_box()
        out_img, out_t = transform(img, copy.deepcopy(target))
        assert torch.allclose(out_t["boxes"][0], _box_from_pixels(out_img))


def test_rotate90_swaps_size():
    img = torch.zeros(3, 40, 60)
    out, _ = RandomRotate90(1.0)(img, {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)})
    assert out.shape[-2:] in {(60, 40), (40, 60)}


def test_scale_crop_fixed_size_and_valid_boxes():
    for _ in range(10):
        img, target = _image_with_box()
        out, t = RandomScaleCrop(32, (0.8, 1.5))(img, copy.deepcopy(target))
        assert out.shape[-2:] == (32, 32)
        assert len(t["boxes"]) == len(t["labels"])
        if len(t["boxes"]):
            b = t["boxes"]
            assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
            assert b.min() >= 0 and b.max() <= 32
