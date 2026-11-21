import json
import re


def normalize_name(name):
    name = name.strip().lower()
    name = re.sub(r"[_\-]+", " ", name)
    return re.sub(r"\s+", " ", name)


def read_categories(ann_file):
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(c["id"], c["name"]) for c in data["categories"]]


class LabelSpace:
    """Unified label space over several datasets.

    Category names are normalised and optionally merged through an alias table
    (e.g. {"plane": "airplane"}). Unified labels start at 1; 0 is background.
    """

    def __init__(self, dataset_categories, aliases=None):
        aliases = {normalize_name(k): normalize_name(v) for k, v in (aliases or {}).items()}
        self.names = []
        index = {}
        self.dataset_maps = {}
        for ds_name, cats in dataset_categories.items():
            mapping = {}
            for cat_id, cat_name in cats:
                n = normalize_name(cat_name)
                n = aliases.get(n, n)
                if n not in index:
                    index[n] = len(self.names) + 1
                    self.names.append(n)
                mapping[cat_id] = index[n]
            self.dataset_maps[ds_name] = mapping
        self.index = index

    @classmethod
    def from_config(cls, cfg):
        cats = {}
        only = cfg.get("label_space_datasets")
        for ds in cfg["datasets"]:
            if only and ds["name"] not in only:
                continue
            ann = ds.get("train_ann") or ds.get("test_ann")
            cats[ds["name"]] = read_categories(ann)
        return cls(cats, cfg.get("label_aliases"))

    def __len__(self):
        return len(self.names)

    def label_of(self, name):
        n = normalize_name(name)
        return self.index[n]

    def to_dict(self):
        return {"names": self.names, "dataset_maps": {k: {str(a): b for a, b in v.items()}
                                                      for k, v in self.dataset_maps.items()}}
