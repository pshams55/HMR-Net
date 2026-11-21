import contextlib
import io

import numpy as np
import torch

# Size ranges used in the paper: S < 32x32, 32x32 <= M < 64x64, L >= 64x64 (after resizing).
AREA_RANGES = [[0, 1e10], [0, 32 ** 2], [32 ** 2, 64 ** 2], [64 ** 2, 1e10]]
AREA_LABELS = ["all", "small", "medium", "large"]


def _coco_gt(dataset):
    images, annotations, cats = [], [], set()
    ann_id = 1
    for info in dataset.images:
        images.append({"id": info["id"], "width": info["width"], "height": info["height"]})
        for a in dataset.anns[info["id"]]:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            label = dataset.label_map[a["category_id"]]
            cats.add(label)
            annotations.append({"id": ann_id, "image_id": info["id"], "bbox": [x, y, w, h],
                                "area": w * h, "category_id": label, "iscrowd": 0})
            ann_id += 1
    return {"images": images, "annotations": annotations,
            "categories": [{"id": c} for c in sorted(cats)]}


@torch.no_grad()
def predict(model, loader, device, head=None, valid_labels=None):
    """Runs the model and returns COCO-style results with unified label ids."""
    model.eval()
    results = []
    for images, targets, ds_ids in loader:
        images = [img.to(device) for img in images]
        outputs = model(images, dataset_ids=ds_ids.to(device))
        for t, o in zip(targets, outputs):
            boxes = o["boxes"].cpu()
            scores = o["scores"].cpu()
            labels = o["labels"].cpu()
            if head is not None:
                labels = head.decode(labels)
            for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                if valid_labels is not None and l not in valid_labels:
                    continue
                results.append({"image_id": int(t["image_id"]), "category_id": int(l),
                                "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": float(s)})
    return results


def _cocoeval(gt_dict, results, iou_thrs=None, max_dets=(1, 10, 100)):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        dt = gt.loadRes(results) if results else COCO()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.areaRng = AREA_RANGES
        ev.params.areaRngLbl = AREA_LABELS
        ev.params.maxDets = list(max_dets)
        if iou_thrs is not None:
            ev.params.iouThrs = np.array(iou_thrs)
        ev.evaluate()
        ev.accumulate()
    return ev


def _mean_valid(x):
    x = x[x > -1]
    return float(x.mean()) * 100 if x.size else 0.0


def detection_metrics(dataset, results):
    """AP, AP50, AP75, APS, APM, APL and their average (as reported in Tables I-VIII)."""
    if not results:
        return {k: 0.0 for k in ["AP", "AP50", "AP75", "APS", "APM", "APL", "Avg"]}
    ev = _cocoeval(_coco_gt(dataset), results)
    p = ev.eval["precision"]                                      # [T, R, K, A, M]
    m = len(ev.params.maxDets) - 1
    out = {
        "AP": _mean_valid(p[:, :, :, 0, m]),
        "AP50": _mean_valid(p[0, :, :, 0, m]),
        "AP75": _mean_valid(p[5, :, :, 0, m]),
        "APS": _mean_valid(p[:, :, :, 1, m]),
        "APM": _mean_valid(p[:, :, :, 2, m]),
        "APL": _mean_valid(p[:, :, :, 3, m]),
    }
    out["Avg"] = float(np.mean(list(out.values())))
    return out


def zsd_metrics(dataset, results):
    """Recall@100 at IoU 0.4/0.5/0.6 and mAP at IoU 0.5 on unseen-only test images."""
    if not results:
        return {"RE@100_0.4": 0.0, "RE@100_0.5": 0.0, "RE@100_0.6": 0.0, "mAP": 0.0}
    gt = _coco_gt(dataset)
    ev = _cocoeval(gt, results, iou_thrs=[0.4, 0.5, 0.6], max_dets=(100,))
    r = ev.eval["recall"]                                         # [T, K, A, M]
    p = ev.eval["precision"]                                      # [T, R, K, A, M]
    return {
        "RE@100_0.4": _mean_valid(r[0, :, 0, 0]),
        "RE@100_0.5": _mean_valid(r[1, :, 0, 0]),
        "RE@100_0.6": _mean_valid(r[2, :, 0, 0]),
        "mAP": _mean_valid(p[1, :, :, 0, 0]),
    }


@torch.no_grad()
def expert_usage(model, loader, device, num_datasets):
    """Counts of domain-expert assignments per dataset (cf. Figs. 13-14)."""
    model.eval()
    E = model.domain.num_experts
    counts = torch.zeros(num_datasets, E, dtype=torch.long)
    for images, _, ds_ids in loader:
        images = [img.to(device) for img in images]
        _, routing = model(images, dataset_ids=ds_ids.to(device), return_routing=True)
        if "domain_expert" not in routing:
            break
        for d, e in zip(ds_ids.tolist(), routing["domain_expert"].cpu().tolist()):
            counts[d, e] += 1
    return counts
