from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import ConvExpert


class DatasetEmbedding(nn.Module):
    """Latent embedding z_i per dataset.

    Rows are initialised from average-pooled backbone features of each dataset and then
    trained jointly. Rows marked as frozen (e.g. a held-out target dataset) receive no
    gradient and keep the value computed from unlabeled target images.
    """

    def __init__(self, num_datasets, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_datasets, dim))
        self.register_buffer("frozen", torch.zeros(num_datasets, dtype=torch.bool))
        self.weight.register_hook(self._mask_grad)

    def _mask_grad(self, grad):
        if self.frozen.any():
            grad = grad.clone()
            grad[self.frozen] = 0
        return grad

    @torch.no_grad()
    def set_embedding(self, index, value, freeze=False):
        self.weight[index].copy_(value.to(self.weight))
        self.frozen[index] = bool(freeze)

    def forward(self, dataset_ids):
        return self.weight[dataset_ids]


class DomainRouter(nn.Module):
    """Routing function G([u, z]) of Eq. (5).

        x = LN([u, z]);  h = ReLU(x W1 + b1);  g = softmax((x W_skip + h W2 + b2) / tau)
    """

    def __init__(self, feat_dim, emb_dim, num_experts, hidden_dim=256, tau=1.0):
        super().__init__()
        in_dim = feat_dim + emb_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.skip = nn.Linear(in_dim, num_experts, bias=False)
        self.tau = tau

    def forward(self, u, z):
        x = self.norm(torch.cat([u, z], dim=-1))
        h = F.relu(self.fc1(x))
        logits = self.skip(x) + self.fc2(h)
        return F.softmax(logits / self.tau, dim=-1)


class DomainExpertRouting(nn.Module):
    """First routing stage: image-level top-1 assignment to one of E_G domain experts.

    The routing decision is made once per image from the reference pyramid level and the
    selected expert is applied to every pyramid level. Cross-expert knowledge distillation
    (CEKD, Eqs. 7-9) is computed on the reference level.
    """

    def __init__(self, channels, num_datasets, num_experts=6, emb_dim=None, router_hidden=256,
                 tau=1.0, distill_temperature=2.0, ref_level="3", top_k=1, expert_hidden=None):
        super().__init__()
        emb_dim = emb_dim or channels
        self.num_experts = num_experts
        self.ref_level = ref_level
        self.T = distill_temperature
        self.top_k = top_k
        self.embedding = DatasetEmbedding(num_datasets, emb_dim)
        self.router = DomainRouter(channels, emb_dim, num_experts, router_hidden, tau)
        self.experts = nn.ModuleList([ConvExpert(channels, hidden_channels=expert_hidden) for _ in range(num_experts)])

    @staticmethod
    def descriptor(feats, ref_level):
        """psi(v): global average pooling of the reference level."""
        return feats[ref_level].mean(dim=(2, 3))

    def _apply_selected(self, x, assign, gate):
        """Run each image through its assigned expert only.

        The output is multiplied by gate / gate.detach() (value 1) so that the router
        receives a gradient while each expert only sees its own samples.
        """
        out = torch.zeros_like(x)
        scale = (gate / gate.detach()).view(-1, 1, 1, 1)
        for e in assign.unique().tolist():
            idx = (assign == e).nonzero(as_tuple=True)[0]
            out[idx] = self.experts[e](x[idx]).to(out.dtype)
        return out * scale

    def forward(self, feats, dataset_ids, compute_distill=False):
        u = self.descriptor(feats, self.ref_level)
        z = self.embedding(dataset_ids)
        probs = self.router(u, z)                          # (B, E_G)

        if self.top_k == 1:
            gate, assign = probs.max(dim=1)
            routed = OrderedDict()
            for name, x in feats.items():
                routed[name] = self._apply_selected(x, assign, gate)
        else:
            # Top-k / soft routing variants used in the routing-strategy ablation.
            k = self.num_experts if self.top_k <= 0 else self.top_k
            topv, topi = probs.topk(k, dim=1)
            weights = topv / topv.sum(dim=1, keepdim=True)
            assign = topi[:, 0]
            routed = OrderedDict()
            for name, x in feats.items():
                acc = torch.zeros_like(x)
                for slot in range(k):
                    out = torch.zeros_like(x)
                    for e in topi[:, slot].unique().tolist():
                        idx = (topi[:, slot] == e).nonzero(as_tuple=True)[0]
                        out[idx] = self.experts[e](x[idx]).to(out.dtype)
                    acc = acc + out * weights[:, slot].view(-1, 1, 1, 1)
                routed[name] = acc

        aux = {"probs": probs, "assign": assign}
        if compute_distill:
            aux["loss_distill"] = self.cekd_loss(feats[self.ref_level], routed[self.ref_level])
        return routed, aux

    def cekd_loss(self, x, student_out):
        """KL(p_teacher || p_student) with softmax over the channel dimension (Eqs. 7-9)."""
        with torch.no_grad():
            ensemble = torch.stack([expert(x) for expert in self.experts], dim=0).mean(dim=0)
            p_teacher = F.softmax(ensemble / self.T, dim=1)
        log_p_student = F.log_softmax(student_out / self.T, dim=1)
        kl = (p_teacher * (torch.log(p_teacher.clamp_min(1e-8)) - log_p_student)).sum(dim=1)
        return kl.mean()

    @staticmethod
    def routing_balance_loss(probs):
        """L_route = sum_e (p_bar_e - 1/E_G)^2 (Eq. 22), p_bar_e averaged over the batch."""
        p_bar = probs.mean(dim=0)
        return ((p_bar - 1.0 / probs.shape[1]) ** 2).sum()
