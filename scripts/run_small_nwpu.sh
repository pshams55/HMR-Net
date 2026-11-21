#!/usr/bin/env bash
# Small-scale reference run on NWPU VHR-10: download, convert, train, evaluate, demo, routing plots.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw
if [ ! -d "data/NWPU VHR-10 dataset" ]; then
  curl -L -o data/raw/nwpu_vhr10.zip \
    "https://huggingface.co/datasets/torchgeo/vhr10/resolve/main/NWPU%20VHR-10%20dataset.zip"
  python -c "import zipfile; zipfile.ZipFile('data/raw/nwpu_vhr10.zip').extractall('data')"
fi

python tools/convert_nwpu.py --root "data/NWPU VHR-10 dataset" --out-dir data/coco \
    --test-ratio 0.3 --with-negatives --seed 0

python tools/train.py --config configs/small_nwpu.yaml
python tools/test.py --config configs/small_nwpu.yaml --checkpoint work_dirs/small_nwpu/final.pth \
    --expert-usage --out work_dirs/small_nwpu/metrics.json
python tools/infer.py --config configs/small_nwpu.yaml --checkpoint work_dirs/small_nwpu/final.pth \
    --dataset NWPU --images demo/images/*.jpg --score-thr 0.5 --out-dir demo/outputs
python tools/visualize_routing.py partitions --config configs/small_nwpu.yaml \
    --checkpoint work_dirs/small_nwpu/final.pth --max-figures 4 --out demo/routing
