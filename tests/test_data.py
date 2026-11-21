import torch

from hmrnet.data import LabelSpace, build_dataset, build_train_set, build_train_transforms
from hmrnet.data.datasets import unseen_labels
from hmrnet.engine import HeadLabels
from hmrnet.utils import merge_dicts


def test_label_space_merges_aliases(tiny_cfg):
    ls = LabelSpace.from_config(tiny_cfg)
    assert ls.names.count("airplane") == 1
    assert ls.names.count("storage tank") == 1
    assert ls.dataset_maps["A"][1] == ls.dataset_maps["B"][1]     # plane == airplane
    assert ls.dataset_maps["A"][3] == ls.dataset_maps["C"][1]     # ship == ship


def test_dataset_items(tiny_cfg):
    ls = LabelSpace.from_config(tiny_cfg)
    train, parts = build_train_set(tiny_cfg, ls, build_train_transforms(tiny_cfg))
    assert len(train) == sum(len(p) for p in parts)
    img, target, ds_index = train[0]
    assert img.shape[0] == 3 and img.max() <= 1.0
    assert target["boxes"].shape[1] == 4 and target["labels"].min() >= 1
    assert ds_index in (0, 1)


def test_zero_shot_split(tiny_cfg):
    cfg = merge_dicts(tiny_cfg, {"zsd": {"enabled": True, "unseen_classes": ["windmill"]}})
    ls = LabelSpace.from_config(cfg)
    unseen = unseen_labels(cfg, ls)
    train = build_dataset(cfg, ls, "B", "train")
    for items in train.anns.values():
        assert all(train.label_map[a["category_id"]] not in unseen for a in items)
    test = build_dataset(cfg, ls, "B", "test")
    assert len(test) == 2
    for items in test.anns.values():
        assert {test.label_map[a["category_id"]] for a in items} <= set(unseen)


def test_head_label_mapping():
    h = HeadLabels([2, 5, 7])
    x = torch.tensor([2, 7, 5])
    assert torch.equal(h.encode(x), torch.tensor([1, 3, 2]))
    assert torch.equal(h.decode(h.encode(x)), x)
