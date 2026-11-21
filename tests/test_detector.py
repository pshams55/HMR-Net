import os

import pytest
import torch
from torch.utils.data import DataLoader

from hmrnet.data import build_dataset, build_train_set, build_train_transforms, collate_fn
from hmrnet.engine import build_model_and_labels, init_dataset_embeddings, predict
from hmrnet.models.text_encoder import PromptBank
from hmrnet.data.label_space import LabelSpace
from hmrnet.utils import merge_dicts


def _one_step(model, loader, head, stage, prompts=None):
    model.set_stage(stage)
    model.train()
    images, targets, ds = next(iter(loader))
    pe = prompts.batch_embeddings(targets, "cpu") if prompts is not None and stage >= 3 else None
    for t in targets:
        t["labels"] = head.encode(t["labels"])
    losses = model(images, targets, dataset_ids=ds, prompt_embeds=pe)
    loss = sum(losses.values())
    loss.backward()
    return losses


@pytest.mark.parametrize("stage,expected", [
    (1, set()),
    (2, {"loss_route", "loss_distill"}),
    (3, {"loss_route", "loss_distill", "loss_coh"}),
])
def test_losses_per_stage(tiny_cfg, stage, expected):
    model, ls, head, _ = build_model_and_labels(tiny_cfg)
    train, _ = build_train_set(tiny_cfg, ls, build_train_transforms(tiny_cfg))
    loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_fn)
    losses = _one_step(model, loader, head, stage)
    detection = {"loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg"}
    assert set(losses) == detection | expected
    assert all(torch.isfinite(v) for v in losses.values())


def test_detr_head(tiny_cfg):
    cfg = merge_dicts(tiny_cfg, {"model": {"detector": "detr",
                                           "detr": {"enc_layers": 1, "dec_layers": 1, "num_queries": 10,
                                                    "dim_feedforward": 64}}})
    model, ls, head, _ = build_model_and_labels(cfg)
    for i in (0, 1):
        model.domain.embedding.set_embedding(i, torch.rand(256))
    train, _ = build_train_set(cfg, ls, build_train_transforms(cfg))
    loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_fn)
    losses = _one_step(model, loader, head, 3)
    assert {"loss_ce", "loss_bbox", "loss_giou", "loss_route", "loss_distill", "loss_coh"} <= set(losses)
    assert model.detr.query.weight.grad is not None
    model.eval()
    img = train[0][0]
    det = model([img], dataset_ids=torch.tensor([0]))[0]
    assert det["boxes"].shape == (10 * len(head.labels), 4)       # top-k over (query, class) pairs
    assert (det["labels"] >= 1).all() and (det["labels"] <= len(head.labels)).all()
    h, w = img.shape[-2:]
    assert det["boxes"][:, [0, 2]].max() <= w + 1e-3 and det["boxes"][:, [1, 3]].max() <= h + 1e-3


def test_held_out_dataset_inference(tiny_cfg):
    model, ls, head, _ = build_model_and_labels(tiny_cfg)
    test_c = build_dataset(tiny_cfg, ls, "C", "test")
    init_dataset_embeddings(model, {2: test_c}, "cpu", max_images=4, freeze=True, num_workers=0)
    assert bool(model.domain.embedding.frozen[2])
    model.set_stage(3)
    res = predict(model, DataLoader(test_c, batch_size=2, collate_fn=collate_fn), "cpu", head=head)
    assert res and all(len(r["bbox"]) == 4 for r in res)
    _, routing = model([test_c[0][0]], dataset_ids=torch.tensor([2]), return_routing=True)
    assert {"domain_expert", "num_regions", "partition"} <= set(routing)


def test_prompt_head_and_cem(tiny_cfg, tmp_path):
    ls = LabelSpace.from_config(tiny_cfg)
    bank = {PromptBank.class_prompt(n): torch.randn(512) for n in ls.names}
    bank_path = os.path.join(tmp_path, "bank.pt")
    torch.save(bank, bank_path)
    cfg = merge_dicts(tiny_cfg, {
        "model": {"head": "prompt", "cem": {"enabled": True, "prompt_bank": bank_path, "phrases_file": None}},
        "zsd": {"enabled": True, "unseen_classes": ["windmill"]},
    })
    model, ls, head, prompts = build_model_and_labels(cfg)
    assert ls.label_of("windmill") not in head.labels
    for i in (0, 1):
        model.domain.embedding.set_embedding(i, torch.rand(256))
    train, _ = build_train_set(cfg, ls, None)
    loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_fn)
    losses = _one_step(model, loader, head, 3, prompts)
    assert "loss_align" in losses
