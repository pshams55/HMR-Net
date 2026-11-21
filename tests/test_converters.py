import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")


def _run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, script), *map(str, args)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _image(path, w, h):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)).save(path)


def _boxes_by_file(j):
    names = {im["id"]: im["file_name"] for im in j["images"]}
    return {(names[a["image_id"]], tuple(round(v) for v in a["bbox"])) for a in j["annotations"]}


def test_dota_tiling(tmp_path):
    root = tmp_path / "dota"
    _image(root / "images" / "P0001.png", 1000, 900)
    os.makedirs(root / "labelTxt")
    (root / "labelTxt" / "P0001.txt").write_text(
        "imagesource:GoogleEarth\ngsd:0.1\n"
        "100 100 200 100 200 150 100 150 plane 0\n"
        "780 300 860 300 860 340 780 340 ship 0\n"
        "850 850 950 850 950 890 850 890 small-vehicle 1\n")
    _run("convert_dota.py", "--root", root, "--out-dir", tmp_path / "out", "--name", "dota_train")
    j = json.loads((tmp_path / "out" / "dota_train.json").read_text())
    boxes = _boxes_by_file(j)
    assert ("dota_train/P0001_0_0.png", (100, 100, 100, 50)) in boxes
    assert ("dota_train/P0001_800_0.png", (0, 300, 60, 40)) in boxes          # 75% visible -> kept
    assert not any(f.endswith("_0_0.png") and b[0] == 780 for f, b in boxes)  # 25% visible -> dropped
    assert ("dota_train/P0001_800_800.png", (50, 50, 100, 40)) in boxes       # padded edge tile
    assert all(im["width"] == 800 and im["height"] == 800 for im in j["images"])


def test_nwpu(tmp_path):
    root = tmp_path / "nwpu"
    os.makedirs(root / "ground truth")
    for i in range(1, 5):
        _image(root / "positive image set" / f"{i:03d}.jpg", 300, 200)
        (root / "ground truth" / f"{i:03d}.txt").write_text("(10,20),(60,80),1\n(100, 50),(150,90),10\n\n")
    _run("convert_nwpu.py", "--root", root, "--out-dir", tmp_path / "out", "--test-ratio", "0.25")
    j = json.loads((tmp_path / "out" / "nwpu_train.json").read_text())
    assert len(j["images"]) == 3 and len(j["annotations"]) == 6
    assert {a["bbox"][2] for a in j["annotations"]} == {50.0}


def test_dior(tmp_path):
    root = tmp_path / "dior"
    _image(root / "JPEGImages-trainval" / "00001.jpg", 800, 800)
    _image(root / "JPEGImages-test" / "11726.jpg", 800, 800)
    ann = root / "Annotations" / "Horizontal Bounding Boxes"
    os.makedirs(ann)
    os.makedirs(root / "ImageSets" / "Main")
    xml = ("<annotation><size><width>800</width><height>800</height></size><object><name>{}</name>"
           "<bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>70</ymax></bndbox></object></annotation>")
    (ann / "00001.xml").write_text(xml.format("Expressway-Service-area"))
    (ann / "11726.xml").write_text(xml.format("windmill"))
    (root / "ImageSets" / "Main" / "train.txt").write_text("00001\n")
    (root / "ImageSets" / "Main" / "val.txt").write_text("")
    (root / "ImageSets" / "Main" / "test.txt").write_text("11726\n")
    _run("convert_dior.py", "--root", root, "--out-dir", tmp_path / "out")
    j = json.loads((tmp_path / "out" / "dior_test.json").read_text())
    assert j["images"][0]["file_name"] == "JPEGImages-test/11726.jpg"
    assert j["annotations"][0]["bbox"] == [10, 20, 100, 50]
    assert len(j["categories"]) == 20


def test_xview(tmp_path):
    root = tmp_path / "xview"
    for k in range(3):
        _image(root / "imgs" / f"{k}.tif", 900, 850)
    feats = [{"properties": {"bounds_imcoords": "10,10,60,40", "image_id": f"{k}.tif", "type_id": 18}}
             for k in range(3)]
    feats.append({"properties": {"bounds_imcoords": "10,10,60,40", "image_id": "0.tif", "type_id": 75}})
    (root / "g.geojson").write_text(json.dumps({"features": feats}))
    _run("convert_xview.py", "--images", root / "imgs", "--geojson", root / "g.geojson",
         "--out-dir", tmp_path / "out", "--test-ratio", "0.34")
    tr = json.loads((tmp_path / "out" / "xview_train.json").read_text())
    te = json.loads((tmp_path / "out" / "xview_test.json").read_text())
    assert len(tr["annotations"]) == 2 and len(te["annotations"]) == 1
    assert len(tr["categories"]) == 60
