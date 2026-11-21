# HMR-Net: Hierarchical Modular Routing for Cross-Domain Object Detection in Aerial Images


## Installation

```bash
conda create -n hmrnet python=3.10 -y
conda activate hmrnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data preparation

All benchmarks are converted to COCO json with horizontal boxes. DOTA and xView are cut into
non-overlapping 800x800 tiles. DIOR and NWPU keep their images and are resized/padded by the model.

```bash
# DIOR (uses ImageSets/Main/{train,val,test}.txt)
python tools/convert_dior.py --root data/DIOR --out-dir data/coco

# DOTA-v1.0 (train for training, val for testing)
python tools/convert_dota.py --root data/DOTA/train --out-dir data/coco --name dota_train
python tools/convert_dota.py --root data/DOTA/val   --out-dir data/coco --name dota_val --keep-empty

# xView (labelled training images split 80/20 at image level; split saved to xview_split.json)
python tools/convert_xview.py --images data/xView/train_images --geojson data/xView/xView_train.geojson \
    --out-dir data/coco --test-ratio 0.2

# NWPU VHR-10 (split saved to nwpu_split.json)
python tools/convert_nwpu.py --root data/NWPU-VHR-10 --out-dir data/coco --test-ratio 0.3 --with-negatives
```

Paths in `configs/hmrnet_multidomain.yaml` match this layout. Class names from all datasets are
merged into one label space; `label_aliases` controls which names are treated as the same class.

## Training

Training runs in three stages (`train.stages` in the config):

| Stage | Epochs (default) | Active parts | Losses |
|---|---|---|---|
| 1 | 0-19 | shared backbone + detection head | L_det |
| 2 | 20-59 | + domain routing, CEKD | + λ1 L_route + λ2 L_distill |
| 3 | 60-199 | + scene routing, CEM | + λ4 L_coh + λ3 L_align |

Dataset embeddings are initialised from average-pooled backbone features when stage 2 starts.

```bash
# 4 GPUs, batch size 8, SGD (lr 1e-3, momentum 0.9, wd 1e-4), 200 epochs, lr x0.1 at 120 and 160
torchrun --nproc_per_node 4 tools/train.py --config configs/hmrnet_multidomain.yaml

# single GPU
python tools/train.py --config configs/hmrnet_multidomain.yaml

# resume
python tools/train.py --config configs/hmrnet_multidomain.yaml --resume work_dirs/hmrnet_multidomain/last.pth
```

## Evaluation

**Per-dataset evaluation** (one model trained on all four datasets):

```bash
python tools/test.py --config configs/hmrnet_multidomain.yaml \
    --checkpoint work_dirs/hmrnet_multidomain/final.pth --expert-usage --out results.json
```



**Cross-domain generalisation** (leave one dataset out):

```bash
torchrun --nproc_per_node 4 tools/train.py --config configs/loo_dior.yaml
python tools/test.py --config configs/loo_dior.yaml --checkpoint work_dirs/loo_dior/final.pth
```

The held-out dataset is not used in training, including its embedding. At test time its
embedding is computed from unlabeled target images and kept fixed.

**Open-category (zero-shot) detection**:

```bash
# 1. set zsd.unseen_classes in configs/zsd_dior.yaml
# 2. cache CLIP text embeddings of all class prompts and phrases
python tools/encode_prompts.py --config configs/zsd_dior.yaml
# 3. train and evaluate (RE@100 at IoU 0.4/0.5/0.6 and mAP on unseen-only test images)
torchrun --nproc_per_node 4 tools/train.py --config configs/zsd_dior.yaml
python tools/test.py --config configs/zsd_dior.yaml --checkpoint work_dirs/zsd_dior/final.pth
```

## Inference with free-text prompts

```bash
python tools/infer.py --config configs/zsd_dior.yaml --checkpoint work_dirs/zsd_dior/final.pth \
    --dataset DIOR --images demo/scene.jpg --prompts "windmill, solar panel beside car"
```

Prompts that are not in the cached bank are encoded with the frozen CLIP text encoder.

## Proposal-based vs query-based head

The routed feature map can also be given to a DETR-style transformer head (object queries, Hungarian
set loss with classification, L1 and GIoU terms) instead of Faster R-CNN:

```bash
torchrun --nproc_per_node 4 tools/train.py --config configs/hmrnet_detr.yaml
```


## Tests

```bash
pip install pytest
pytest tests -q
```

## If you find it interesting, please cite the work

```bash
@article{shamsolmoali2026hmr,
  title={HMR-Net: Hierarchical Modular Routing for Cross-Domain Object Detection in Aerial Images},
  author={Shamsolmoali, Pourya and Zareapoor, Masoumeh and Felsberg, Michael and Pears, Nick and Zhou, Huiyu and Lu, Yue},
  journal={arXiv preprint arXiv:2604.18866},
  year={2026}
}
```
