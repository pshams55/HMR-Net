import torch
from torch.utils.data import DataLoader, Subset

from ..data.datasets import collate_fn


@torch.no_grad()
def dataset_mean_feature(model, dataset, device, max_images=500, batch_size=8, num_workers=4):
    """Average of the pooled backbone features over (a subset of) a dataset.

    Labels are not used, so the same routine initialises the embedding of an unseen
    target dataset from its unlabeled images.
    """
    if max_images and len(dataset) > max_images:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(dataset), generator=g)[:max_images].tolist()
        dataset = Subset(dataset, idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn)
    was_training = model.training
    model.eval()
    total, count = None, 0
    for images, _, _ in loader:
        feats = model.pooled_backbone_features([img.to(device) for img in images])
        s = feats.sum(dim=0)
        total = s if total is None else total + s
        count += feats.shape[0]
    model.train(was_training)
    return total / max(count, 1)


def init_dataset_embeddings(model, datasets, device, max_images=500, freeze=False, **kw):
    """datasets: dict {dataset_index: Dataset}."""
    core = model.module if hasattr(model, "module") else model
    for index, ds in datasets.items():
        mean = dataset_mean_feature(core, ds, device, max_images=max_images, **kw)
        core.domain.embedding.set_embedding(index, mean, freeze=freeze)
