import pytest

from hmrnet.data import LabelSpace, build_dataset
from hmrnet.utils import merge_dicts

pytest.importorskip("pycocotools")

from hmrnet.engine.evaluator import detection_metrics, zsd_metrics  # noqa: E402


def _perfect(ds):
    return [{"image_id": img["id"], "category_id": ds.label_map[a["category_id"]], "bbox": a["bbox"], "score": 1.0}
            for img in ds.images for a in ds.anns[img["id"]]]


def test_perfect_predictions(tiny_cfg):
    ls = LabelSpace.from_config(tiny_cfg)
    ds = build_dataset(tiny_cfg, ls, "A", "test")
    m = detection_metrics(ds, _perfect(ds))
    assert m["AP"] == pytest.approx(100.0)
    assert m["AP50"] == pytest.approx(100.0)
    assert m["AP75"] == pytest.approx(100.0)


def test_shifted_boxes_lower_ap75_before_ap50(tiny_cfg):
    ls = LabelSpace.from_config(tiny_cfg)
    ds = build_dataset(tiny_cfg, ls, "A", "test")
    preds = _perfect(ds)
    for p in preds:
        x, y, w, h = p["bbox"]
        p["bbox"] = [x + 0.15 * w, y, w, h]                        # IoU about 0.74
    m = detection_metrics(ds, preds)
    assert m["AP50"] == pytest.approx(100.0)
    assert m["AP75"] < 100.0


def test_no_predictions():
    assert detection_metrics(None, [])["AP"] == 0.0


def test_zsd_metrics_recall(tiny_cfg):
    cfg = merge_dicts(tiny_cfg, {"zsd": {"enabled": True, "unseen_classes": ["windmill"]}})
    ls = LabelSpace.from_config(cfg)
    ds = build_dataset(cfg, ls, "B", "test")
    m = zsd_metrics(ds, _perfect(ds))
    assert m["RE@100_0.5"] == pytest.approx(100.0)
    assert m["mAP"] == pytest.approx(100.0)
