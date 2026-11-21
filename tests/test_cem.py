import itertools

import torch

from hmrnet.models.cem import ConditionalExpertModule, PromptGuidedPredictor


def test_hungarian_matches_brute_force():
    torch.manual_seed(0)
    for _ in range(20):
        S = torch.randn(3, 5)
        rows, cols = ConditionalExpertModule.hungarian(S)
        best = max(sum(S[r, c] for r, c in zip(range(3), perm)) for perm in itertools.permutations(range(5), 3))
        assert torch.isclose(S[rows, cols].sum(), best)
        assert len(set(cols.tolist())) == len(cols)               # one-to-one


def test_alignment_loss_trains_projections():
    cem = ConditionalExpertModule(16, 12, embed_dim=8)
    ru = torch.randn(2, 4, 16)
    valid = torch.tensor([[True, True, False, True], [True, False, False, False]])
    prompts = [torch.randn(5, 12), torch.randn(3, 12)]
    loss = cem.alignment_loss(ru, valid, prompts)
    assert -1.0 <= loss.item() <= 1.0
    loss.backward()
    assert cem.proj_v.weight.grad.abs().sum() > 0 and cem.proj_t.weight.grad.abs().sum() > 0


def test_alignment_loss_without_prompts_is_zero():
    cem = ConditionalExpertModule(16, 12)
    loss = cem.alignment_loss(torch.randn(2, 4, 16), torch.ones(2, 4, dtype=torch.bool), [None, None])
    assert loss.item() == 0.0


def test_max_alignments_limit():
    cem = ConditionalExpertModule(16, 12, max_alignments=2)
    res = cem.match(torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), [torch.randn(6, 12)])
    assert len(res[0]) == 2


def test_prompt_predictor_label_space_can_change():
    head = PromptGuidedPredictor(32, 12, embed_dim=8)
    head.set_class_embeddings(torch.randn(5, 12))
    scores, deltas = head(torch.randn(7, 32))
    assert scores.shape == (7, 6) and deltas.shape == (7, 24)
    head.set_class_embeddings(torch.randn(2, 12))                 # e.g. unseen classes at test time
    scores, deltas = head(torch.randn(7, 32))
    assert scores.shape == (7, 3) and deltas.shape == (7, 12)
    assert torch.equal(deltas[:, :4], deltas[:, 4:8])             # class-agnostic regression
