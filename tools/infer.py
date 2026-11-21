import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmrnet.data.datasets import dataset_index  # noqa: E402
from hmrnet.engine import build_model_and_labels  # noqa: E402
from hmrnet.models.text_encoder import FrozenCLIPTextEncoder, PromptBank  # noqa: E402
from hmrnet.utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run HMR-Net on images, optionally with free-text prompts")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--dataset", required=True, help="dataset name whose embedding is used for routing")
    parser.add_argument("--prompts", default=None,
                        help="comma-separated class names or phrases (prompt-guided head only)")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_space, head, prompts = build_model_and_labels(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.set_stage(3)
    model.to(device).eval()

    names = [label_space.names[l - 1] for l in head.labels]
    if args.prompts:
        if not model.prompt_head:
            raise SystemExit("--prompts needs a model trained with model.head: prompt")
        texts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        queries = [p if len(p.split()) > 2 else PromptBank.class_prompt(p, prompts.template) for p in texts]
        encoder = None
        if any(q not in prompts.bank.embeddings for q in queries):
            encoder = FrozenCLIPTextEncoder(cfg["model"]["cem"].get("clip_model", "openai/clip-vit-base-patch32"))
        model.set_class_prompts(prompts.bank.get(queries, encoder).to(device))
        names = texts

    ds_id = torch.tensor([dataset_index(cfg, args.dataset)], device=device)
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.images:
        img = Image.open(path).convert("RGB")
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div(255).to(device)
        with torch.no_grad():
            det = model([x], dataset_ids=ds_id)[0]
        draw = ImageDraw.Draw(img)
        for b, s, l in zip(det["boxes"].tolist(), det["scores"].tolist(), det["labels"].tolist()):
            if s < args.score_thr:
                continue
            draw.rectangle(b, outline=(255, 0, 0), width=2)
            draw.text((b[0] + 2, b[1] + 2), f"{names[l - 1]} {s:.2f}", fill=(255, 255, 0))
        out = os.path.join(args.out_dir, os.path.basename(path))
        img.save(out)
        print(out)


if __name__ == "__main__":
    main()
