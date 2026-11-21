import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import ConvExpert


class SceneExpertRouting(nn.Module):
    """Second routing stage: split each feature map into routing units (RUs) and send
    each RU to a scene expert (Eqs. 10-16).

    The partition is computed on the reference pyramid level and resized (nearest) to the
    other levels, so all levels share the same spatial routing.
    """

    def __init__(self, channels, max_experts=8, min_regions=2, theta_init=1.0,
                 adaptive=True, fixed_regions=6, ref_level="2", expert_hidden=None):
        super().__init__()
        self.E = max_experts
        self.min_regions = min_regions
        self.adaptive = adaptive
        self.fixed_regions = fixed_regions
        self.ref_level = ref_level
        self.theta = nn.Parameter(torch.tensor(float(theta_init)))
        self.omega = nn.Conv2d(channels, max_experts, kernel_size=3, padding=1)
        self.experts = nn.ModuleList([ConvExpert(channels, hidden_channels=expert_hidden) for _ in range(max_experts)])

    @staticmethod
    def feature_entropy(f, eps=1e-8):
        """Channel entropy of non-negative activations, normalised by log(c') (Eq. 10)."""
        s = f.clamp_min(0).sum(dim=(2, 3))
        p = s / s.sum(dim=1, keepdim=True).clamp_min(eps)
        ent = -(p * torch.log(p.clamp_min(eps))).sum(dim=1)
        return ent / math.log(f.shape[1])

    def num_regions(self, f):
        """R = round(2 + sigmoid(theta_R * entropy) * (E_S - 2)) (Eq. 11)."""
        if not self.adaptive:
            return torch.full((f.shape[0],), self.fixed_regions, dtype=torch.long, device=f.device)
        ent = self.feature_entropy(f)
        r = self.min_regions + torch.sigmoid(self.theta * ent) * (self.E - self.min_regions)
        return torch.round(r).long().clamp(self.min_regions, self.E)

    @staticmethod
    def coherence_loss(P):
        """Squared difference between routing scores of 4-neighbouring pixels (Eq. 16)."""
        dh = (P[:, :, 1:, :] - P[:, :, :-1, :]).pow(2).sum(dim=1)
        dw = (P[:, :, :, 1:] - P[:, :, :, :-1]).pow(2).sum(dim=1)
        return dh.mean() + dw.mean()

    def partition(self, f):
        B = f.shape[0]
        R = self.num_regions(f)
        P = F.softmax(self.omega(f), dim=1)                      # (B, E_S, h, w), Eq. (12)

        # Keep the top-R experts ranked by their global score sum_{x,y} P(x, y, r).
        order = P.sum(dim=(2, 3)).argsort(dim=1, descending=True)
        rank = torch.arange(self.E, device=f.device).expand(B, -1)
        active = torch.zeros(B, self.E, dtype=torch.bool, device=f.device)
        active.scatter_(1, order, rank < R.unsqueeze(1))

        P_active = P * active[:, :, None, None]
        r_xy = P_active.argmax(dim=1)                             # Eq. (13)
        hard = F.one_hot(r_xy, self.E).permute(0, 3, 1, 2).to(P.dtype)
        soft = P_active / P_active.sum(dim=1, keepdim=True).clamp_min(1e-8)
        M = hard + soft - soft.detach()                           # straight-through masks, Eq. (14)
        return P, M, hard, active, R

    def forward(self, feats, compute_coherence=False):
        f_ref = feats[self.ref_level]
        P, M, hard, active, R = self.partition(f_ref)
        used = active & (hard.sum(dim=(2, 3)) > 0)                # RUs that own at least one pixel

        fused = OrderedDict()
        ru_embed = None
        for name, x in feats.items():
            M_l = M if x.shape[-2:] == M.shape[-2:] else F.interpolate(M, size=x.shape[-2:], mode="nearest")
            out = x.clone()                                        # residual path of the fusion Phi
            for e in range(self.E):
                idx = used[:, e].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                o = self.experts[e](x[idx]).to(out.dtype)
                out[idx] = out[idx] + M_l[idx, e:e + 1] * o
                if name == self.ref_level:
                    if ru_embed is None:
                        ru_embed = x.new_zeros(x.shape[0], self.E, x.shape[1])
                    # Gamma: masked average pooling of the expert output inside the RU.
                    m = hard[idx, e:e + 1]
                    ru_embed[idx, e] = (o * m).sum(dim=(2, 3)) / m.sum(dim=(2, 3)).clamp_min(1.0)
            fused[name] = out

        if ru_embed is None:
            ru_embed = f_ref.new_zeros(f_ref.shape[0], self.E, f_ref.shape[1])

        aux = {"num_regions": R, "active": used, "ru_embed": ru_embed, "partition": hard.argmax(dim=1)}
        if compute_coherence:
            aux["loss_coh"] = self.coherence_loss(P)
        return fused, aux
