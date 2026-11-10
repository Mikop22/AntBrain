"""Verify camera/loss mathematics without performing any training updates."""

import math

import pytest
import torch
from torch.nn import functional as F

from ant_homing.level1 import convap


def _reference_loss(embeddings, labels, epsilon=0.1):
    """Independent scalar reference, deliberately separate from vectorized masks."""
    vectors = F.normalize(embeddings.detach().double(), dim=1).tolist()
    scores = [[sum(a * b for a, b in zip(x, y, strict=True)) for y in vectors] for x in vectors]
    losses = []
    for anchor, row in enumerate(scores):
        positives = [
            score for j, score in enumerate(row) if j != anchor and labels[j] == labels[anchor]
        ]
        negatives = [score for j, score in enumerate(row) if labels[j] != labels[anchor]]
        if not positives or not negatives:
            losses.append(0.0)
            continue
        selected_p = [s for s in positives if s < max(negatives) + epsilon]
        selected_n = [s for s in negatives if s > min(positives) - epsilon]
        if selected_p or selected_n:
            losses.append(
                math.log1p(sum(math.exp(-s) for s in selected_p))
                + math.log1p(sum(math.exp(50 * s) for s in selected_n)) / 50
            )
        else:
            losses.append(0.0)
    return sum(losses) / len(losses) if losses else 0.0


def test_camera_encoder_is_shared_and_does_not_cross_seams():
    torch.manual_seed(3)
    model = convap.build_model().eval()
    image = torch.rand(1, 3, 32, 128)
    with torch.no_grad():
        original = model.camera_features(image)
        changed = image.clone()
        changed[:, :, :, 31] = 0
        modified = model.camera_features(changed)
        assert torch.equal(original[:, 1:], modified[:, 1:])
        panels = list(image.split(32, dim=3))
        permutation = [2, 0, 3, 1]
        reordered = model.camera_features(torch.cat([panels[i] for i in permutation], dim=3))
        assert torch.equal(reordered, original[:, permutation])
        repeated = torch.cat([panels[0]] * 4, dim=3)
        repeated_features = model.camera_features(repeated)
        assert torch.equal(repeated_features[:, 0], repeated_features[:, 3])
        descriptor = convap.encode(model, image)
    assert descriptor.shape == (1, 128)
    assert torch.allclose(descriptor.norm(dim=1), torch.ones(1), atol=1e-6)
    assert sum(parameter.numel() for parameter in model.parameters()) == 91_328


def test_uniform_embeddings_have_expected_mined_loss():
    embeddings = torch.ones(4, 8, dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1])
    actual = convap.multi_similarity_loss(embeddings, labels)
    expected = math.log1p(math.exp(-1)) + math.log1p(2 * math.exp(50)) / 50
    assert actual.item() == pytest.approx(expected, abs=1e-12)
    positive, negative = convap._mined_pair_masks(
        F.normalize(embeddings, dim=1) @ F.normalize(embeddings, dim=1).T, labels, 0.1
    )
    assert not positive.diagonal().any()
    assert positive.sum().item() == 4 and negative.sum().item() == 8


def test_miner_and_all_anchor_reduction_match_independent_scalar_reference():
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.7, 0.7, 0.0],
            [0.9, 0.4, 0.0],
            [0.3, 0.95, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    expected = _reference_loss(embeddings, labels.tolist())
    actual = convap.multi_similarity_loss(embeddings, labels)
    assert actual.item() == pytest.approx(expected, abs=1e-12)
    # Pinned reference MeanReducer includes two inactive final-class anchors.
    reduced = convap.multi_similarity_loss(embeddings[:4], labels[:4])
    assert actual.item() == pytest.approx(reduced.item() * 4 / 6, abs=1e-12)


def test_pinned_generic_pair_guard_zeroes_a_single_directed_pair_of_each_kind():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.1, -math.sqrt(0.99)], [0.9, math.sqrt(0.19)]],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1])
    unit = F.normalize(embeddings, dim=1)
    positive, negative = convap._mined_pair_masks(unit @ unit.T, labels, 0.1)
    assert positive.sum().item() == negative.sum().item() == 1
    loss = convap.multi_similarity_loss(embeddings, labels)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(embeddings.grad, torch.zeros_like(embeddings))


@pytest.mark.parametrize("labels", [[0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 2, 3]])
def test_no_active_pairs_returns_differentiable_zero(labels):
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], requires_grad=True
    )
    loss = convap.multi_similarity_loss(embeddings, torch.tensor(labels))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(embeddings.grad, torch.zeros_like(embeddings))


def test_backward_is_finite_and_does_not_update_parameters():
    torch.manual_seed(4)
    model = convap.build_model()
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    batch = torch.rand(3, 4, 3, 32, 128)
    loss = convap.training_loss(model, batch)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert all(
        torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters()
    )


def test_invalid_images_and_batches_fail_cleanly():
    model = convap.build_model()
    for image in (
        torch.zeros(2, 3, 32, 32),
        torch.zeros(2, 3, 32, 128, dtype=torch.uint8),
        torch.full((2, 3, 32, 128), float("nan")),
        torch.full((2, 3, 32, 128), 1.1),
    ):
        with pytest.raises(ValueError):
            convap.encode(model, image)
    for batch in (
        torch.zeros(2, 3, 32, 128),
        torch.zeros(1, 4, 3, 32, 128),
        torch.zeros(2, 1, 3, 32, 128),
    ):
        with pytest.raises(ValueError):
            convap.training_loss(model, batch)
    assert convap.encode(model, torch.empty(0, 3, 32, 128)).shape == (0, 128)
