"""Routing visualisations.

  usage       domain-expert usage per dataset (heatmap)
  mds         2-D MDS projection of routed image descriptors, coloured by domain expert
  partitions  scene routing units overlaid on images
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmrnet.data import build_dataset, collate_fn  # noqa: E402
from hmrnet.data.datasets import dataset_index  # noqa: E402
from hmrnet.engine import build_model_and_labels, init_dataset_embeddings  # noqa: E402
from hmrnet.models.domain_routing import DomainExpertRouting  # noqa: E402
from hmrnet.utils import load_config  # noqa: E402


def load(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_space, _, _ = build_model_and_labels(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.set_stage(3)
    return cfg, model.to(device).eval(), label_space, device


def loaders(cfg, model, label_space, device, args):
    out = {}
    for name in args.datasets or cfg["test_datasets"]:
        ds = build_dataset(cfg, label_space, name, "test")
        if name not in cfg["train_datasets"]:
            init_dataset_embeddings(model, {dataset_index(cfg, name): ds}, device, max_images=200,
                                    freeze=True, num_workers=0)
        if args.max_images and len(ds) > args.max_images:
            ds = Subset(ds, list(range(args.max_images)))
        out[name] = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
    return out


@torch.no_grad()
def collect(model, loader, device):
    """Per-image routed descriptor (GAP of routed reference level) and assigned expert."""
    feats_out, experts = [], []
    for images, _, ds_ids in loader:
        images = [img.to(device) for img in images]
        il, _ = model.detector.transform(images)
        feats = model.detector.backbone(il.tensors)
        routed, aux = model.domain(feats, ds_ids.to(device))
        feats_out.append(DomainExpertRouting.descriptor(routed, model.domain.ref_level).cpu())
        experts.append(aux["assign"].cpu())
    return torch.cat(feats_out).numpy(), torch.cat(experts).numpy()


def plot_usage(cfg, model, label_space, device, args):
    names, rows = [], []
    E = model.domain.num_experts
    for name, loader in loaders(cfg, model, label_space, device, args).items():
        _, experts = collect(model, loader, device)
        rows.append(np.bincount(experts, minlength=E) / max(len(experts), 1))
        names.append(name)
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * E, 0.8 + 0.5 * len(names)))
    im = ax.imshow(np.array(rows), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(E), [f"E{e + 1}" for e in range(E)])
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Domain expert")
    fig.colorbar(im, ax=ax, label="Fraction of images")
    fig.tight_layout()
    return fig


def plot_mds(cfg, model, label_space, device, args):
    from sklearn.manifold import MDS

    X, experts, sets = [], [], []
    for name, loader in loaders(cfg, model, label_space, device, args).items():
        f, e = collect(model, loader, device)
        X.append(f)
        experts.append(e)
        sets += [name] * len(e)
    X, experts = np.concatenate(X), np.concatenate(experts)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Y = MDS(n_components=2, random_state=0, normalized_stress="auto").fit_transform(X)
    fig, ax = plt.subplots(figsize=(5, 4))
    markers = "osD^vP*X"
    for i, name in enumerate(dict.fromkeys(sets)):
        idx = np.array([s == name for s in sets])
        sc = ax.scatter(Y[idx, 0], Y[idx, 1], c=experts[idx], cmap="tab10", vmin=0, vmax=9,
                        marker=markers[i % len(markers)], s=14, label=name)
    ax.legend(title="Dataset", fontsize=8)
    ax.set_title("Routed image descriptors (MDS), colour = domain expert")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_partitions(cfg, model, label_space, device, args):
    figs = []
    cmap = plt.get_cmap("tab10")
    for name, loader in loaders(cfg, model, label_space, device, args).items():
        for images, _, ds_ids in loader:
            dev_images = [img.to(device) for img in images]
            image_list, _ = model.detector.transform(dev_images)
            padded_h = image_list.tensors.shape[-2]
            _, routing = model(dev_images, dataset_ids=ds_ids.to(device), return_routing=True)
            for img, part, R, (rh, rw) in zip(images, routing["partition"].cpu(), routing["num_regions"].cpu(),
                                              image_list.image_sizes):
                h, w = img.shape[-2:]
                stride = padded_h / part.shape[0]
                part = part[: int(np.ceil(rh / stride)), : int(np.ceil(rw / stride))]   # drop padding
                part = F.interpolate(part[None, None].float(), size=(h, w), mode="nearest")[0, 0].long()
                color = torch.tensor([cmap(int(k) % 10)[:3] for k in range(model.scene.E)])[part]
                overlay = 0.55 * img.permute(1, 2, 0) + 0.45 * color
                fig, axs = plt.subplots(1, 2, figsize=(7, 3.6))
                axs[0].imshow(img.permute(1, 2, 0))
                axs[0].set_title(name)
                axs[1].imshow(overlay.clamp(0, 1))
                axs[1].set_title(f"routing units (R = {int(R)})")
                for a in axs:
                    a.axis("off")
                fig.tight_layout()
                figs.append(fig)
                if len(figs) >= args.max_figures:
                    return figs
    return figs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["usage", "mds", "partitions"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--max-figures", type=int, default=6)
    ap.add_argument("--out", default="outputs/routing")
    args = ap.parse_args()

    cfg, model, label_space, device = load(args)
    os.makedirs(args.out, exist_ok=True)
    if args.mode == "partitions":
        for i, fig in enumerate(plot_partitions(cfg, model, label_space, device, args)):
            fig.savefig(os.path.join(args.out, f"partition_{i}.png"), dpi=120)
    else:
        fn = plot_usage if args.mode == "usage" else plot_mds
        fn(cfg, model, label_space, device, args).savefig(os.path.join(args.out, f"{args.mode}.png"), dpi=150)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
