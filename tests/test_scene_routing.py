from collections import OrderedDict

import torch

from hmrnet.models.scene_routing import SceneExpertRouting


def _feats(B=3, C=16):
    return OrderedDict([("1", torch.relu(torch.randn(B, C, 32, 32))),
                        ("2", torch.relu(torch.randn(B, C, 16, 16))),
                        ("3", torch.relu(torch.randn(B, C, 8, 8)))])


def test_region_count_bounds():
    m = SceneExpertRouting(16, max_experts=8, min_regions=2)
    for theta in (-50.0, 0.0, 1.0, 50.0):
        m.theta.data.fill_(theta)
        R = m.num_regions(torch.relu(torch.randn(5, 16, 10, 10)))
        assert ((R >= 2) & (R <= 8)).all()


def test_entropy_is_normalised():
    f = torch.ones(2, 16, 4, 4)                                   # uniform over channels
    assert torch.allclose(SceneExpertRouting.feature_entropy(f), torch.ones(2), atol=1e-5)
    g = torch.zeros(2, 16, 4, 4)
    g[:, 0] = 1                                                   # one active channel
    assert torch.allclose(SceneExpertRouting.feature_entropy(g), torch.zeros(2), atol=1e-5)


def test_fixed_region_count():
    m = SceneExpertRouting(16, max_experts=8, adaptive=False, fixed_regions=4)
    assert (m.num_regions(torch.rand(3, 16, 8, 8)) == 4).all()


def test_partition_is_a_valid_assignment():
    m = SceneExpertRouting(16, max_experts=6)
    P, M, hard, active, R = m.partition(torch.relu(torch.randn(4, 16, 12, 12)))
    assert torch.allclose(P.sum(1), torch.ones(4, 12, 12), atol=1e-5)
    assert (hard.sum(1) == 1).all()                               # every pixel in exactly one RU
    assert (active.sum(1) == R).all()                             # top-R experts kept
    assert ((hard.sum((2, 3)) > 0) <= active).all()               # pixels only go to active experts
    assert torch.allclose(M, hard)                                # straight-through value equals hard mask


def test_coherence_loss():
    const = torch.full((2, 4, 6, 6), 0.25)
    assert SceneExpertRouting.coherence_loss(const).item() == 0.0
    noisy = torch.softmax(torch.randn(2, 4, 6, 6), dim=1)
    assert SceneExpertRouting.coherence_loss(noisy).item() > 0


def test_forward_shapes_and_gradients():
    m = SceneExpertRouting(16, max_experts=4, ref_level="2")
    feats = _feats()
    out, aux = m(feats, compute_coherence=True)
    for k in feats:
        assert out[k].shape == feats[k].shape
    assert aux["ru_embed"].shape == (3, 4, 16)
    assert aux["partition"].shape == (3, 16, 16)
    (sum(v.sum() for v in out.values()) + aux["loss_coh"]).backward()
    assert m.omega.weight.grad.abs().sum() > 0
