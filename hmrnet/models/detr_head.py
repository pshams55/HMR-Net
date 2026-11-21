import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou


def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


class SinePositionEmbedding(nn.Module):
    def __init__(self, dim=256, temperature=10000):
        super().__init__()
        self.half = dim // 2
        self.temperature = temperature

    def forward(self, mask):
        not_mask = ~mask
        y = not_mask.cumsum(1, dtype=torch.float32)
        x = not_mask.cumsum(2, dtype=torch.float32)
        eps, scale = 1e-6, 2 * math.pi
        y = y / (y[:, -1:, :] + eps) * scale
        x = x / (x[:, :, -1:] + eps) * scale
        dim_t = torch.arange(self.half, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.half)
        px = x[..., None] / dim_t
        py = y[..., None] / dim_t
        px = torch.stack([px[..., 0::2].sin(), px[..., 1::2].cos()], dim=4).flatten(3)
        py = torch.stack([py[..., 0::2].sin(), py[..., 1::2].cos()], dim=4).flatten(3)
        return torch.cat([py, px], dim=3).permute(0, 3, 1, 2)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou

    @torch.no_grad()
    def forward(self, logits, boxes, targets):
        out = []
        prob = logits.softmax(-1)
        for b, t in enumerate(targets):
            if t["labels"].numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                out.append((empty, empty))
                continue
            c = -self.cost_class * prob[b][:, t["labels"]]
            c = c + self.cost_bbox * torch.cdist(boxes[b], t["boxes_cxcywh"], p=1)
            c = c - self.cost_giou * generalized_box_iou(box_cxcywh_to_xyxy(boxes[b]),
                                                         box_cxcywh_to_xyxy(t["boxes_cxcywh"]))
            rows, cols = linear_sum_assignment(c.cpu().numpy())
            out.append((torch.as_tensor(rows, device=logits.device), torch.as_tensor(cols, device=logits.device)))
        return out


class DETRHead(nn.Module):
    """Query-based detection head on the routed feature map o.

    Transformer encoder-decoder with object queries, set prediction trained with Hungarian
    matching (classification + L1 + GIoU). Labels use index 0 as "no object".
    """

    def __init__(self, in_channels, num_classes, hidden_dim=256, num_queries=100, nheads=8,
                 enc_layers=6, dec_layers=6, dim_feedforward=2048, dropout=0.1, eos_coef=0.1,
                 loss_ce=1.0, loss_bbox=5.0, loss_giou=2.0):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.pos = SinePositionEmbedding(hidden_dim)
        self.transformer = nn.Transformer(hidden_dim, nheads, enc_layers, dec_layers, dim_feedforward,
                                          dropout, batch_first=True)
        self.query = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = HungarianMatcher(loss_ce, loss_bbox, loss_giou)
        self.weights = {"ce": loss_ce, "bbox": loss_bbox, "giou": loss_giou}
        self.num_classes = num_classes
        empty = torch.ones(num_classes + 1)
        empty[0] = eos_coef
        self.register_buffer("class_weight", empty)

    def forward(self, feat, mask):
        src = self.input_proj(feat)
        pos = self.pos(mask)
        B = src.shape[0]
        memory_in = (src + pos).flatten(2).transpose(1, 2)
        q = self.query.weight.unsqueeze(0).expand(B, -1, -1)
        hs = self.transformer(memory_in, q, src_key_padding_mask=mask.flatten(1),
                              memory_key_padding_mask=mask.flatten(1))
        return self.class_embed(hs), self.bbox_embed(hs).sigmoid()

    def loss(self, logits, boxes, targets):
        idx = self.matcher(logits, boxes, targets)
        target_classes = torch.zeros(logits.shape[:2], dtype=torch.long, device=logits.device)
        src_boxes, tgt_boxes = [], []
        for b, (qi, ti) in enumerate(idx):
            target_classes[b, qi] = targets[b]["labels"][ti]
            src_boxes.append(boxes[b, qi])
            tgt_boxes.append(targets[b]["boxes_cxcywh"][ti])
        loss_ce = F.cross_entropy(logits.transpose(1, 2), target_classes, self.class_weight)
        src_boxes, tgt_boxes = torch.cat(src_boxes), torch.cat(tgt_boxes)
        n = max(tgt_boxes.shape[0], 1)
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / n
        loss_giou = (1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                                        box_cxcywh_to_xyxy(tgt_boxes)))).sum() / n
        return {"loss_ce": self.weights["ce"] * loss_ce,
                "loss_bbox": self.weights["bbox"] * loss_bbox,
                "loss_giou": self.weights["giou"] * loss_giou}

    @torch.no_grad()
    def postprocess(self, logits, boxes, image_sizes, top_k=100):
        prob = logits.softmax(-1)[..., 1:]
        results = []
        for b, (h, w) in enumerate(image_sizes):
            scores, flat = prob[b].flatten().topk(min(top_k, prob[b].numel()))
            q, labels = flat // self.num_classes, flat % self.num_classes + 1
            xyxy = box_cxcywh_to_xyxy(boxes[b, q]) * torch.tensor([w, h, w, h], device=boxes.device)
            results.append({"boxes": xyxy, "scores": scores, "labels": labels})
        return results
