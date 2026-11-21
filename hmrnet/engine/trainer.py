import json
import math
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from ..data import LabelSpace, PromptSampler, build_dataset, build_train_set, build_train_transforms, collate_fn
from ..data.datasets import dataset_index, unseen_labels
from ..models import HMRNet
from ..models.text_encoder import PromptBank
from ..utils import init_distributed, is_main, reduce_dict, set_seed
from .embeddings import init_dataset_embeddings


class HeadLabels:
    """Maps unified labels to classifier indices and back.

    With the prompt-guided head under the zero-shot protocol the classifier only knows
    the seen classes, so labels are remapped to a contiguous range.
    """

    def __init__(self, labels):
        self.labels = list(labels)
        size = max(self.labels) + 1 if self.labels else 1
        self.to_head = torch.zeros(size, dtype=torch.long)
        for i, l in enumerate(self.labels):
            self.to_head[l] = i + 1
        self.from_head = torch.tensor([0] + self.labels, dtype=torch.long)

    def encode(self, labels):
        return self.to_head.to(labels.device)[labels]

    def decode(self, labels):
        return self.from_head.to(labels.device)[labels]


def build_model_and_labels(cfg):
    label_space = LabelSpace.from_config(cfg)
    unseen = unseen_labels(cfg, label_space)
    seen = [l for l in range(1, len(label_space) + 1) if l not in unseen]
    prompt_head = cfg["model"].get("head", "closed") == "prompt"
    head = HeadLabels(seen if prompt_head else range(1, len(label_space) + 1))

    bank, prompts = None, None
    if prompt_head or cfg["model"].get("cem", {}).get("enabled", False):
        c = cfg["model"]["cem"]
        bank = PromptBank.load(c["prompt_bank"])
        prompts = PromptSampler(label_space, bank, max_prompts=c.get("max_alignments", 16),
                                phrases_file=c.get("phrases_file"), seen_labels=seen,
                                template=c.get("template"))
        cfg["model"]["text_dim"] = bank.dim

    model = HMRNet(len(head.labels), len(cfg["datasets"]), cfg)
    if prompt_head:
        model.set_class_prompts(prompts.class_embeddings(head.labels))
    return model, label_space, head, prompts


def stage_for_epoch(cfg, epoch):
    s = cfg["train"]["stages"]
    if epoch < s["domain_start"]:
        return 1
    if epoch < s["scene_start"]:
        return 2
    return 3


def save_checkpoint(path, model, optimizer, scheduler, epoch, extra):
    core = model.module if hasattr(model, "module") else model
    torch.save({"model": core.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, **extra}, path)


def train(cfg, resume=None):
    distributed, rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42) + rank)
    tc = cfg["train"]
    work_dir = cfg["work_dir"]
    os.makedirs(work_dir, exist_ok=True)

    model, label_space, head, prompts = build_model_and_labels(cfg)
    model.to(device)
    if is_main():
        with open(os.path.join(work_dir, "label_space.json"), "w", encoding="utf-8") as f:
            json.dump(label_space.to_dict(), f, indent=2)

    train_set, _ = build_train_set(cfg, label_space, build_train_transforms(cfg))
    sampler = DistributedSampler(train_set, shuffle=True) if distributed else RandomSampler(train_set)
    loader = DataLoader(train_set, batch_size=tc["batch_size_per_gpu"], sampler=sampler,
                        num_workers=tc.get("num_workers", 4), collate_fn=collate_fn,
                        pin_memory=True, drop_last=True)

    params = [p for p in model.parameters() if p.requires_grad]
    if tc.get("optimizer", "sgd") == "adamw":
        optimizer = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-4))
    else:
        optimizer = torch.optim.SGD(params, lr=tc["lr"], momentum=tc.get("momentum", 0.9),
                                    weight_decay=tc.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=tc["lr_steps"],
                                                     gamma=tc.get("lr_gamma", 0.1))

    start_epoch, emb_ready = 0, False
    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        emb_ready = ckpt.get("embeddings_initialized", False)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                                        find_unused_parameters=True)
    core = model.module if distributed else model

    use_amp = tc.get("amp", False) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log_path = os.path.join(work_dir, "train_log.jsonl")
    extra_ckpt = lambda: {"embeddings_initialized": emb_ready, "head_labels": head.labels,
                          "label_space": label_space.to_dict(), "config": cfg}

    for epoch in range(start_epoch, tc["epochs"]):
        stage = stage_for_epoch(cfg, epoch)
        if stage >= 2 and not emb_ready:
            plain = {dataset_index(cfg, n): build_dataset(cfg, label_space, n, "train")
                     for n in cfg["train_datasets"]}
            init_dataset_embeddings(model, plain, device, max_images=tc.get("embedding_init_images", 500),
                                    num_workers=tc.get("num_workers", 4))
            emb_ready = True
        core.set_stage(stage)
        model.train()
        if distributed:
            sampler.set_epoch(epoch)

        t0 = time.time()
        for it, (images, targets, ds_ids) in enumerate(loader):
            global_iter = epoch * len(loader) + it
            warmup = tc.get("warmup_iters", 0)
            if global_iter <= warmup and warmup > 0:
                factor = tc.get("warmup_factor", 0.001)
                scale = factor + (1 - factor) * global_iter / warmup
                for g in optimizer.param_groups:
                    g["lr"] = g["initial_lr"] * scale
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            prompt_embeds = None
            if stage >= 3 and core.cem is not None:
                prompt_embeds = prompts.batch_embeddings(targets, device)
            for t in targets:
                t["labels"] = head.encode(t["labels"])

            with torch.autocast(device_type=device.type, enabled=use_amp):
                losses = model(images, targets, dataset_ids=ds_ids.to(device), prompt_embeds=prompt_embeds)
            loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, iter {it}: {losses}")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tc.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            if it % tc.get("log_interval", 50) == 0:
                logged = reduce_dict(losses)
                if is_main():
                    rec = {"epoch": epoch, "iter": it, "stage": stage, "lr": optimizer.param_groups[0]["lr"],
                           "loss": sum(logged.values()), **logged, "time": round(time.time() - t0, 1)}
                    print(json.dumps(rec), flush=True)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")

        scheduler.step()
        if is_main():
            save_checkpoint(os.path.join(work_dir, "last.pth"), model, optimizer, scheduler, epoch, extra_ckpt())
            if (epoch + 1) % tc.get("save_interval", 20) == 0:
                save_checkpoint(os.path.join(work_dir, f"epoch_{epoch + 1}.pth"), model, optimizer, scheduler,
                                epoch, extra_ckpt())

    if is_main():
        save_checkpoint(os.path.join(work_dir, "final.pth"), model, optimizer, scheduler, tc["epochs"] - 1,
                        extra_ckpt())
