from collections import OrderedDict

import pytest
import torch

from hmrnet.models.domain_routing import DatasetEmbedding, DomainExpertRouting, DomainRouter


def _feats(B=4, C=16):
    return OrderedDict([("2", torch.rand(B, C, 16, 16)), ("3", torch.rand(B, C, 8, 8))])


def test_router_outputs_distribution():
    router = DomainRouter(16, 8, num_experts=5, hidden_dim=32)
    p = router(torch.randn(7, 16), torch.randn(7, 8))
    assert p.shape == (7, 5)
    assert torch.allclose(p.sum(-1), torch.ones(7), atol=1e-5)


def test_temperature_sharpens_routing():
    router = DomainRouter(16, 8, num_experts=5)
    u, z = torch.randn(3, 16), torch.randn(3, 8)
    router.tau = 0.1
    sharp = router(u, z).max(-1).values
    router.tau = 10.0
    smooth = router(u, z).max(-1).values
    assert (sharp >= smooth).all()


def test_top1_routing_shapes_and_assignment():
    m = DomainExpertRouting(16, num_datasets=2, num_experts=3, emb_dim=16, ref_level="3")
    feats = _feats()
    out, aux = m(feats, torch.tensor([0, 1, 0, 1]), compute_distill=True)
    assert set(out) == set(feats)
    for k in feats:
        assert out[k].shape == feats[k].shape
    assert aux["assign"].shape == (4,)
    assert torch.equal(aux["assign"], aux["probs"].argmax(1))
    assert aux["loss_distill"] >= 0


def test_each_expert_only_gets_its_own_gradient():
    torch.manual_seed(0)
    m = DomainExpertRouting(16, num_datasets=1, num_experts=3, emb_dim=16, ref_level="3")
    feats = _feats(B=2)
    out, aux = m(feats, torch.tensor([0, 0]))
    sum(v.sum() for v in out.values()).backward()
    used = set(aux["assign"].tolist())
    for e, expert in enumerate(m.experts):
        g = expert.body[0].weight.grad
        if e in used:
            assert g is not None and g.abs().sum() > 0
        else:
            assert g is None or g.abs().sum() == 0
    assert m.router.fc2.weight.grad.abs().sum() > 0
    assert m.embedding.weight.grad.abs().sum() > 0


def test_frozen_embedding_receives_no_gradient():
    emb = DatasetEmbedding(3, 4)
    emb.set_embedding(2, torch.ones(4), freeze=True)
    emb(torch.tensor([0, 1, 2])).sum().backward()
    assert emb.weight.grad[2].abs().sum() == 0
    assert emb.weight.grad[0].abs().sum() > 0
    assert torch.equal(emb.weight[2].detach(), torch.ones(4))


def test_balance_loss():
    uniform = torch.full((8, 4), 0.25)
    assert DomainExpertRouting.routing_balance_loss(uniform).item() == pytest.approx(0.0)
    collapsed = torch.zeros(8, 4)
    collapsed[:, 0] = 1
    assert DomainExpertRouting.routing_balance_loss(collapsed).item() == pytest.approx(0.75)


@pytest.mark.parametrize("top_k", [0, 2])
def test_soft_and_top2_routing(top_k):
    m = DomainExpertRouting(16, num_datasets=1, num_experts=3, emb_dim=16, ref_level="3", top_k=top_k)
    feats = _feats(B=3)
    out, _ = m(feats, torch.zeros(3, dtype=torch.long))
    assert all(out[k].shape == feats[k].shape for k in feats)
