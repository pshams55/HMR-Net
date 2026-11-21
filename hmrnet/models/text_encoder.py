import torch

DEFAULT_TEMPLATE = "an aerial image of a {}."


class FrozenCLIPTextEncoder:
    """Frozen CLIP text encoder. Embeddings are usually cached to disk with
    tools/encode_prompts.py so that training does not need to load CLIP."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        from transformers import CLIPModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, texts, batch_size=256):
        out = []
        for i in range(0, len(texts), batch_size):
            tok = self.tokenizer(texts[i:i + batch_size], padding=True, return_tensors="pt").to(self.device)
            out.append(self.model.get_text_features(**tok).float().cpu())
        return torch.cat(out, dim=0)


class PromptBank:
    """Maps prompt strings to cached text embeddings."""

    def __init__(self, embeddings):
        self.embeddings = embeddings                              # dict: prompt -> (D,) tensor
        self.dim = next(iter(embeddings.values())).numel() if embeddings else 512

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, map_location="cpu"))

    def get(self, prompts, encoder=None):
        missing = [p for p in prompts if p not in self.embeddings]
        if missing:
            if encoder is None:
                raise KeyError(f"No cached embedding for prompts: {missing[:5]}")
            for p, e in zip(missing, encoder.encode(missing)):
                self.embeddings[p] = e
        return torch.stack([self.embeddings[p] for p in prompts], dim=0)

    @staticmethod
    def class_prompt(name, template=DEFAULT_TEMPLATE):
        return template.format(name.replace("_", " ").replace("-", " "))
