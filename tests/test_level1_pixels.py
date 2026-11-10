"""Check baseline invariances, retrieval semantics and image-only operation."""

import numpy as np
import pytest

from ant_homing.level1.pixels import encode, variants


def test_raw_descriptors_preserve_pixels_and_expected_distance():
    images = np.array([[[[0, 64, 255], [32, 128, 192]]]], dtype=np.uint8)
    actual = encode(images)
    expected = np.array([[0, 64, 255, 32, 128, 192]], dtype=np.float32) / 255.0
    np.testing.assert_array_equal(actual, expected)
    zero = encode(np.zeros_like(images))
    assert float(np.square(actual - zero).sum()) == pytest.approx(float(np.square(expected).sum()))
    assert actual.dtype == np.float32
    assert actual.nbytes == images.size * 4


def test_normalization_removes_positive_affine_brightness_changes_without_clipping():
    image = np.arange(24, dtype=np.uint8).reshape(1, 2, 4, 3) + 10
    changed = image * 3 + 7
    original_descriptor = encode(image, "normalized")
    changed_descriptor = encode(changed, "normalized")
    np.testing.assert_allclose(original_descriptor, changed_descriptor, atol=2e-6)
    assert not np.array_equal(encode(image, "raw"), encode(changed, "raw"))
    assert original_descriptor.mean() == pytest.approx(0, abs=1e-6)
    assert original_descriptor.std() == pytest.approx(1, abs=1e-6)


def test_each_descriptor_is_independent_of_other_queries_and_batch_chunking():
    rng = np.random.default_rng(4)
    images = rng.integers(0, 256, size=(5, 32, 128, 3), dtype=np.uint8)
    for variant in variants:
        batch = encode(images, variant)
        chunks = np.concatenate([encode(images[:2], variant), encode(images[2:], variant)])
        np.testing.assert_array_equal(batch, chunks)
        np.testing.assert_array_equal(batch[0], encode(images[:1], variant)[0])


def test_constants_and_empty_batches_are_finite():
    images = np.stack([np.full((32, 128, 3), level, dtype=np.uint8) for level in range(256)])
    normalized = encode(images, "normalized")
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized, np.zeros_like(normalized))
    for variant in variants:
        empty = encode(np.empty((0, 32, 128, 3), dtype=np.uint8), variant)
        assert empty.shape == (0, 12288)
        assert empty.dtype == np.float32


def test_encoding_preserves_input_and_accepts_noncontiguous_images():
    rng = np.random.default_rng(9)
    source = rng.integers(0, 256, size=(3, 32, 128, 3), dtype=np.uint8)
    before = source.copy()
    view = source[:, :, ::-1, :]
    assert not view.flags.c_contiguous
    for variant in variants:
        first, second = encode(view, variant), encode(view, variant)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, encode(view.copy(), variant))
        assert not np.shares_memory(first, source)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("variant", ["cosine", "RAW", None, []])
def test_unknown_variants_are_rejected(variant):
    with pytest.raises(ValueError, match="variant"):
        encode(np.zeros((1, 32, 128, 3), dtype=np.uint8), variant)


@pytest.mark.parametrize(
    "images",
    [
        np.zeros((32, 128, 3), dtype=np.uint8),
        np.zeros((1, 32, 128, 1), dtype=np.uint8),
        np.zeros((1, 0, 128, 3), dtype=np.uint8),
        np.zeros((1, 32, 128, 3), dtype=np.float32),
        np.full((1, 32, 128, 3), np.nan),
        [[[[0, 0, 0]]]],
    ],
)
def test_invalid_image_inputs_are_rejected(images):
    with pytest.raises(ValueError, match="uint8 RGB"):
        encode(images)
