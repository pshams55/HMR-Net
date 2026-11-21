import copy
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("_base_", None)
    if base:
        parent = load_config(os.path.join(os.path.dirname(path), base))
        cfg = merge_dicts(parent, cfg)
    return cfg


def merge_dicts(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_dict(d):
    if not dist.is_initialized():
        return {k: float(v) for k, v in d.items()}
    keys = sorted(d.keys())
    vals = torch.stack([d[k].detach().float() for k in keys])
    dist.all_reduce(vals)
    vals /= dist.get_world_size()
    return {k: float(v) for k, v in zip(keys, vals)}
