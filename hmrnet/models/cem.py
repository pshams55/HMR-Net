import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class ConditionalExpertModule(nn.Module):
    """Aligns RU-level visual embeddings with text prompt embeddings (Eqs. 17-20).

    P_v and P_t project both modalities into a shared space; the RU-prompt assignment is
    the one-to-one matching that maximises the total similarity (Hungarian algorithm).
    """

    def __init__(self, vis_dim, text_dim, embed_dim=512, max_alignments=16):
        super().__init__()
        self.proj_v = nn.Linear(vis_dim, embed_dim)
        self.proj_t = nn.Linear(text_dim, embed_dim)
        self.max_alignments = max_alignments

    def similarity(self, v, t):
        v = F.normalize(self.proj_v(v), dim=-1)
        t = F.normalize(self.proj_t(t), dim=-1)
        return v @ t.t()                                          # S_{r,m}, Eq. (18)

    @staticmethod
    def hungarian(S):
        """pi_hat = argmax_pi sum S * pi over one-to-one assignments (Eq. 20)."""
        rows, cols = linear_sum_assignment(-S.detach().float().cpu().numpy())
        return torch.as_tensor(rows, device=S.device), torch.as_tensor(cols, device=S.device)

    def alignment_loss(self, ru_embed, ru_valid, prompt_embeds):
        """L_align = -sum_{r,m} S_{r,m} pi_hat_{r,m} (Eq. 19), averaged over matched pairs.

        ru_embed: (B, E_S, C); ru_valid: (B, E_S) bool; prompt_embeds: list of (M_b, D).
        """
        losses = []
        for b, t in enumerate(prompt_embeds):
            if t is None or t.numel() == 0:
                continue
            v = ru_embed[b][ru_valid[b]]
            if v.numel() == 0:
                continue
            t = t[: self.max_alignments]
            S = self.similarity(v, t)
            rows, cols = self.hungarian(S)
            losses.append(-S[rows, cols].mean())
        if not losses:
            return ru_embed.sum() * 0.0
        return torch.stack(losses).mean()

    @torch.no_grad()
    def match(self, ru_embed, ru_valid, prompt_embeds):
        """Inference helper: returns, per image, a list of (ru_index, prompt_index, score)."""
        results = []
        for b, t in enumerate(prompt_embeds):
            valid_idx = ru_valid[b].nonzero(as_tuple=True)[0]
            if t is None or t.numel() == 0 or valid_idx.numel() == 0:
                results.append([])
                continue
            S = self.similarity(ru_embed[b][valid_idx], t[: self.max_alignments])
            rows, cols = self.hungarian(S)
            results.append([(int(valid_idx[r]), int(c), float(S[r, c])) for r, c in zip(rows, cols)])
        return results


class PromptGuidedPredictor(nn.Module):
    """Box predictor whose classifier compares RoI features with text embeddings.

    Class logits are cosine similarities between projected RoI features and the current
    class prompt embeddings, plus a learned background embedding. Swapping the prompt
    embeddings at inference changes the label space without retraining. Box regression
    is class-agnostic so that it also applies to unseen classes.
    """

    def __init__(self, in_channels, text_dim, embed_dim=512, init_logit_scale=1 / 0.07):
        super().__init__()
        self.proj_roi = nn.Linear(in_channels, embed_dim)
        self.proj_t = nn.Linear(text_dim, embed_dim)
        self.background = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale)))
        self.bbox_pred = nn.Linear(in_channels, 4)
        self.register_buffer("class_text", torch.zeros(0, text_dim), persistent=False)

    def set_class_embeddings(self, text_embeds):
        """text_embeds: (C, text_dim), row c is the prompt embedding of foreground class c+1."""
        self.class_text = text_embeds.detach().to(self.background.device)

    @property
    def num_classes(self):
        return self.class_text.shape[0] + 1

    def forward(self, x):
        x = x.flatten(start_dim=1)
        v = F.normalize(self.proj_roi(x), dim=-1)
        t = F.normalize(self.proj_t(self.class_text), dim=-1)
        bg = F.normalize(self.background, dim=-1).unsqueeze(0)
        weights = torch.cat([bg, t], dim=0)                       # index 0 is background
        scores = self.logit_scale.exp().clamp(max=100) * v @ weights.t()
        deltas = self.bbox_pred(x).repeat(1, weights.shape[0])
        return scores, deltas
