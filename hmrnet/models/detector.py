import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .cem import ConditionalExpertModule, PromptGuidedPredictor
from .detr_head import DETRHead, box_xyxy_to_cxcywh
from .domain_routing import DomainExpertRouting
from .scene_routing import SceneExpertRouting


class HMRNet(nn.Module):
    """Hierarchical Modular Routing Network on an R50-FPN backbone.

    backbone -> domain expert routing -> scene expert routing -> detection head

    The detection head is Faster R-CNN (RPN + RoI heads, default) or a DETR-style
    query-based head (``model.detector: detr``); both consume the same routed features.

    Training stages (set with ``set_stage``):
        1: shared backbone + detection head only (L_det)
        2: + domain routing and CEKD (L_route, L_distill)
        3: + scene routing and CEM (L_coh, L_align)
    """

    def __init__(self, num_classes, num_datasets, cfg):
        super().__init__()
        m = cfg["model"]
        weights = ResNet50_Weights.IMAGENET1K_V1 if m.get("pretrained_backbone", True) else None
        backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=weights,
                                       trainable_layers=m.get("trainable_layers", 3))
        C = backbone.out_channels

        self.detector = FasterRCNN(
            backbone,
            num_classes=num_classes + 1,
            min_size=m.get("input_size", 800),
            max_size=m.get("input_size", 800),
            box_nms_thresh=m.get("nms_thresh", 0.5),
            box_detections_per_img=m.get("detections_per_img", 100),
            box_score_thresh=m.get("score_thresh", 0.05),
        )
        if m.get("init", "imagenet") == "coco":
            self._load_coco_detector()

        text_dim = m.get("text_dim", 512)
        self.prompt_head = m.get("head", "closed") == "prompt"
        if self.prompt_head:
            in_ch = self.detector.roi_heads.box_predictor.cls_score.in_features
            self.detector.roi_heads.box_predictor = PromptGuidedPredictor(
                in_ch, text_dim, embed_dim=m.get("joint_dim", 512))

        d = m["domain_routing"]
        self.domain = DomainExpertRouting(
            C, num_datasets,
            num_experts=d.get("num_experts", 6),
            emb_dim=d.get("embedding_dim", C),
            router_hidden=d.get("router_hidden", 256),
            tau=d.get("tau", 1.0),
            distill_temperature=d.get("distill_temperature", 2.0),
            ref_level=str(d.get("ref_level", "3")),
            top_k=d.get("top_k", 1),
            expert_hidden=d.get("expert_hidden"),
        )

        s = m["scene_routing"]
        self.scene = SceneExpertRouting(
            C,
            max_experts=s.get("max_experts", 8),
            min_regions=s.get("min_regions", 2),
            theta_init=s.get("theta_init", 1.0),
            adaptive=s.get("adaptive", True),
            fixed_regions=s.get("fixed_regions", 6),
            ref_level=str(s.get("ref_level", "2")),
            expert_hidden=s.get("expert_hidden"),
        )

        c = m.get("cem", {})
        self.cem = None
        if c.get("enabled", False):
            self.cem = ConditionalExpertModule(C, text_dim, embed_dim=m.get("joint_dim", 512),
                                               max_alignments=c.get("max_alignments", 16))

        self.head_type = m.get("detector", "faster_rcnn")
        self.detections_per_img = m.get("detections_per_img", 100)
        if self.head_type == "detr":
            if self.prompt_head:
                raise ValueError("model.head: prompt is only available with the Faster R-CNN detector")
            dc = dict(m.get("detr", {}))
            self.detr_level = str(dc.pop("level", "3"))
            self.detr = DETRHead(C, num_classes, **dc)
            # Only the shared transform, backbone and routing of the Faster R-CNN wrapper are used.
            self.detector.rpn = nn.Identity()
            self.detector.roi_heads = nn.Identity()

        lw = cfg["loss_weights"]
        self.lambdas = {k: float(lw[k]) for k in ("route", "distill", "align", "coh")}
        self.stage = 3

    def _load_coco_detector(self):
        """Initialise backbone, FPN, RPN and box head from torchvision's COCO Faster R-CNN.

        The class-specific predictor is left randomly initialised.
        """
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

        state = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=True)
        state = {k: v for k, v in state.items() if not k.startswith("roi_heads.box_predictor")}
        self.detector.load_state_dict(state, strict=False)

    def set_stage(self, stage):
        self.stage = int(stage)

    def set_class_prompts(self, text_embeds):
        """Prompt-guided head only: set the (C, D) class prompt embeddings."""
        self.detector.roi_heads.box_predictor.set_class_embeddings(text_embeds)

    @torch.no_grad()
    def pooled_backbone_features(self, images):
        """Average-pooled backbone features at the domain-routing reference level."""
        images, _ = self.detector.transform(images)
        feats = self.detector.backbone(images.tensors)
        return DomainExpertRouting.descriptor(feats, self.domain.ref_level)

    def forward(self, images, targets=None, dataset_ids=None, prompt_embeds=None, return_routing=False):
        original_sizes = [img.shape[-2:] for img in images]
        image_list, targets = self.detector.transform(images, targets)
        feats = self.detector.backbone(image_list.tensors)

        routing, extra = {}, {}
        if self.stage >= 2:
            if dataset_ids is None:
                raise ValueError("dataset_ids are required once domain routing is active")
            feats, d_aux = self.domain(feats, dataset_ids, compute_distill=self.training)
            routing["domain_probs"] = d_aux["probs"]
            routing["domain_expert"] = d_aux["assign"]
            if self.training:
                extra["loss_route"] = self.lambdas["route"] * self.domain.routing_balance_loss(d_aux["probs"])
                extra["loss_distill"] = self.lambdas["distill"] * d_aux["loss_distill"]

        if self.stage >= 3:
            feats, s_aux = self.scene(feats, compute_coherence=self.training)
            routing["num_regions"] = s_aux["num_regions"]
            routing["scene_active"] = s_aux["active"]
            routing["partition"] = s_aux["partition"]
            routing["ru_embed"] = s_aux["ru_embed"]
            if self.training:
                extra["loss_coh"] = self.lambdas["coh"] * s_aux["loss_coh"]
                if self.cem is not None and prompt_embeds is not None:
                    extra["loss_align"] = self.lambdas["align"] * self.cem.alignment_loss(
                        s_aux["ru_embed"], s_aux["active"], prompt_embeds)

        if self.head_type == "detr":
            return self._detr_forward(feats, image_list, targets, original_sizes, routing, extra, return_routing)

        proposals, rpn_losses = self.detector.rpn(image_list, feats, targets)
        detections, det_losses = self.detector.roi_heads(feats, proposals, image_list.image_sizes, targets)

        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(det_losses)
            losses.update(extra)
            return losses

        detections = self.detector.transform.postprocess(detections, image_list.image_sizes, original_sizes)
        if return_routing:
            return detections, routing
        return detections

    def _detr_forward(self, feats, image_list, targets, original_sizes, routing, extra, return_routing):
        x = feats[self.detr_level]
        B, _, H, W = image_list.tensors.shape
        pad = torch.ones(B, H, W, dtype=torch.bool, device=x.device)
        for b, (h, w) in enumerate(image_list.image_sizes):
            pad[b, :h, :w] = False
        mask = F.interpolate(pad[None].float(), size=x.shape[-2:], mode="nearest")[0].bool()
        logits, boxes = self.detr(x, mask)

        if self.training:
            for t, (h, w) in zip(targets, image_list.image_sizes):
                scale = torch.tensor([w, h, w, h], dtype=t["boxes"].dtype, device=t["boxes"].device)
                t["boxes_cxcywh"] = box_xyxy_to_cxcywh(t["boxes"]) / scale
            losses = self.detr.loss(logits, boxes, targets)
            losses.update(extra)
            return losses

        detections = self.detr.postprocess(logits, boxes, image_list.image_sizes,
                                           top_k=self.detections_per_img)
        detections = self.detector.transform.postprocess(detections, image_list.image_sizes, original_sizes)
        if return_routing:
            return detections, routing
        return detections
