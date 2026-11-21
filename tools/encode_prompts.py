import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmrnet.data import LabelSpace  # noqa: E402
from hmrnet.models.text_encoder import DEFAULT_TEMPLATE, FrozenCLIPTextEncoder, PromptBank  # noqa: E402
from hmrnet.utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Encode class prompts and phrases with the frozen CLIP text encoder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--extra", nargs="*", default=[], help="additional prompts to cache")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    c = cfg["model"]["cem"]
    template = c.get("template") or DEFAULT_TEMPLATE
    label_space = LabelSpace.from_config(cfg)

    prompts = [PromptBank.class_prompt(n, template) for n in label_space.names]
    if c.get("phrases_file"):
        with open(c["phrases_file"], "r", encoding="utf-8") as f:
            for phrases in json.load(f).values():
                prompts.extend(phrases)
    for name in cfg.get("zsd", {}).get("unseen_classes", []):
        prompts.append(PromptBank.class_prompt(name, template))
    prompts.extend(args.extra)
    prompts = list(dict.fromkeys(prompts))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = FrozenCLIPTextEncoder(c.get("clip_model", "openai/clip-vit-base-patch32"), device)
    embeds = encoder.encode(prompts)
    out = args.out or c["prompt_bank"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({p: e for p, e in zip(prompts, embeds)}, out)
    print(f"saved {len(prompts)} prompt embeddings ({embeds.shape[1]}-d) to {out}")


if __name__ == "__main__":
    main()
