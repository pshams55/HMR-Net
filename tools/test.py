import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmrnet.data import build_dataset, collate_fn  # noqa: E402
from hmrnet.data.datasets import dataset_index, unseen_labels  # noqa: E402
from hmrnet.engine import (HeadLabels, build_model_and_labels, detection_metrics, expert_usage,  # noqa: E402
                           init_dataset_embeddings, predict, zsd_metrics)
from hmrnet.utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate HMR-Net")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", nargs="*", default=None, help="override test_datasets")
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expert-usage", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_space, head, prompts = build_model_and_labels(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.set_stage(args.stage)
    model.to(device).eval()

    zsd = cfg.get("zsd", {}).get("enabled", False)
    names = args.datasets or cfg["test_datasets"]
    report = {}
    for name in names:
        ds = build_dataset(cfg, label_space, name, "test")
        idx = dataset_index(cfg, name)
        if name not in cfg["train_datasets"] and args.stage >= 2:
            # Unseen target domain: embedding from unlabeled target images, kept fixed.
            init_dataset_embeddings(model, {idx: ds}, device,
                                    max_images=cfg["train"].get("embedding_init_images", 500),
                                    freeze=True, num_workers=args.workers)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn)

        if zsd:
            unseen = unseen_labels(cfg, label_space)
            eval_head = HeadLabels(unseen)
            model.set_class_prompts(prompts.class_embeddings(unseen).to(device))
            results = predict(model, loader, device, head=eval_head)
            metrics = zsd_metrics(ds, results)
            model.set_class_prompts(prompts.class_embeddings(head.labels).to(device))
        else:
            valid = set(label_space.dataset_maps[name].values())
            results = predict(model, loader, device, head=head, valid_labels=valid)
            metrics = detection_metrics(ds, results)

        report[name] = metrics
        print(name, " ".join(f"{k}={v:.2f}" for k, v in metrics.items()), flush=True)

        if args.expert_usage and args.stage >= 2:
            counts = expert_usage(model, loader, device, len(cfg["datasets"]))
            report[name]["expert_usage"] = counts[idx].tolist()
            print(f"  domain expert usage: {counts[idx].tolist()}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
