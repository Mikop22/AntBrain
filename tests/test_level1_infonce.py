"""Check the InfoNCE model and objective without performing any optimization."""

from math import log

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ant_homing.level1 import infonce


class _FirstPixels(nn.Module):
    """A transparent encoder for objective checks, not a fitted model."""

    def forward(self, images):
        return images.flatten(1)[:, :64]


def test_encoder_preserves_the_original_architecture_and_normalizes_descriptors():
    torch.manual_seed(4)
    model = infonce.build_model()
    assert [layer.out_channels for layer in model if isinstance(layer, nn.Conv2d)] == [12, 24, 32]
    assert sum(parameter.numel() for parameter in model.parameters()) == 42728
    descriptors = infonce.encode(model, torch.rand(3, 3, 32, 128))
    assert descriptors.shape == (3, 64)
    torch.testing.assert_close(torch.linalg.vector_norm(descriptors, dim=1), torch.ones(3))


def test_uniform_observations_have_log_number_of_places_loss():
    loss = infonce.training_loss(_FirstPixels(), torch.ones(5, 4, 3, 32, 128))
    assert loss.item() == pytest.approx(log(5))


def test_two_visit_loss_matches_the_existing_symmetric_formula():
    torch.manual_seed(8)
    batch = torch.rand(4, 2, 3, 32, 128)
    features = infonce.encode(_FirstPixels(), batch.reshape(8, 3, 32, 128)).reshape(4, 2, 64)
    logits = features[:, 0] @ features[:, 1].T / 0.1
    labels = torch.arange(4)
    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    torch.testing.assert_close(infonce.training_loss(_FirstPixels(), batch), expected)


def test_four_visits_average_both_pairs_and_preserve_positive_identity():
    torch.manual_seed(5)
    images = torch.rand(4, 3, 32, 128)
    batch = images[:, None].repeat(1, 4, 1, 1, 1)
    batch[:, 2:] = torch.rand(4, 2, 3, 32, 128)
    assert infonce.training_loss(_FirstPixels(), batch[:, :2]) < infonce.training_loss(
        _FirstPixels(), batch[:, 2:]
    )
    expected = (
        infonce.training_loss(_FirstPixels(), batch[:, :2])
        + infonce.training_loss(_FirstPixels(), batch[:, 2:])
    ) / 2
    torch.testing.assert_close(infonce.training_loss(_FirstPixels(), batch), expected)
    incorrect = batch.clone()
    incorrect[:, 1] = incorrect[:, 1].roll(1, dims=0)
    assert infonce.training_loss(_FirstPixels(), incorrect) > expected


def test_loss_has_finite_gradients_without_updating_any_weights():
    torch.manual_seed(2)
    model = infonce.build_model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    loss = infonce.training_loss(model, torch.rand(3, 4, 3, 32, 128))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "images, error",
    [
        (torch.zeros(0, 3, 32, 128), ValueError),
        (torch.zeros(2, 3, 32, 32), ValueError),
        (torch.zeros(2, 3, 32, 128, dtype=torch.uint8), TypeError),
        (torch.full((2, 3, 32, 128), float("nan")), ValueError),
        (torch.full((2, 3, 32, 128), 1.01), ValueError),
    ],
)
def test_encode_rejects_invalid_observations(images, error):
    with pytest.raises(error):
        infonce.encode(infonce.build_model(), images)


@pytest.mark.parametrize("shape", [(1, 4, 3, 32, 128), (3, 3, 3, 32, 128), (3, 4, 3, 16, 64)])
def test_loss_rejects_invalid_place_visit_batches(shape):
    with pytest.raises(ValueError, match="even V"):
        infonce.training_loss(infonce.build_model(), torch.zeros(shape))
