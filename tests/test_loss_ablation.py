"""Pre-training ablation checks; no optimizer steps occur here."""

import json
from pathlib import Path

import pytest
import torch

from ant_homing.level1 import convap, infonce, infonce_ms, loss_ablation


def test_encoder_functions_and_initial_states_are_identical():
    assert infonce_ms.build_model is infonce.build_model
    assert infonce_ms.encode is infonce.encode
    assert infonce_ms.DESCRIPTOR_DIM == infonce.DESCRIPTOR_DIM == 64
    for seed in loss_ablation.SEEDS:
        assert loss_ablation.initial_state_hash(infonce, seed) == loss_ablation.initial_state_hash(
            infonce_ms, seed
        )


def test_ms_wrapper_matches_reviewed_loss_and_has_finite_gradients():
    torch.manual_seed(12)
    model = infonce.build_model()
    batch = torch.rand(3, 4, 3, 32, 128)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    actual = infonce_ms.training_loss(model, batch)
    embeddings = infonce.encode(model, batch.reshape(12, 3, 32, 128))
    expected = convap.multi_similarity_loss(
        embeddings, torch.arange(3).repeat_interleave(4), **convap.LOSS_CONFIG
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("shape", [(1, 4, 3, 32, 128), (2, 2, 3, 32, 128), (2, 4, 3, 16, 64)])
def test_ms_wrapper_rejects_wrong_ablation_inputs(shape):
    with pytest.raises(ValueError, match="Expected"):
        infonce_ms.training_loss(infonce.build_model(), torch.zeros(shape))


def gate_fixture(tmp_path, monkeypatch):
    fake_module = tmp_path / "src/ant_homing/level1/loss_ablation.py"
    monkeypatch.setattr(loss_ablation, "__file__", str(fake_module))
    sources = {}
    for name in loss_ablation.REVIEWED_SOURCE_PATHS | loss_ablation.EXTRA_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reviewed")
        sources[name] = loss_ablation.digest(path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "metadata.json").write_text("metadata")
    metadata = {
        "data_sha256": "arrays",
        "scene_seeds": {"test": list(range(400, 412))},
        "image_shape": [32, 128, 3],
        "observations_per_place": 4,
    }
    report = tmp_path / "review.md"
    report.write_text("PASS")
    record = {
        "approvals": dict.fromkeys(("loss", "training", "evaluation"), "PASS"),
        "reports": {
            role: {
                "path": "review.md",
                "sha256": loss_ablation.digest(report),
            }
            for role in ("loss", "training", "evaluation")
        },
        "dataset_sha256": "arrays",
        "metadata_sha256": loss_ablation.digest(data / "metadata.json"),
        "source_sha256": sources,
        "budget": {"seeds": [0, 1, 2], "epochs": 20, "learning_rate": 0.001},
    }
    reviews = tmp_path / "manifest.json"
    reviews.write_text(json.dumps(record))
    return reviews, data, metadata, record


def test_gate_accepts_complete_bound_review_and_rejects_mutations(tmp_path, monkeypatch):
    reviews, data, metadata, record = gate_fixture(tmp_path, monkeypatch)
    assert loss_ablation.verify_gate(reviews, data, metadata) == record
    (tmp_path / "review.md").write_text("changed")
    with pytest.raises(ValueError, match="report changed"):
        loss_ablation.verify_gate(reviews, data, metadata)
    (tmp_path / "review.md").write_text("PASS")
    record["approvals"].pop("loss")
    reviews.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Three passing"):
        loss_ablation.verify_gate(reviews, data, metadata)
    record["approvals"]["loss"] = "PASS"
    reviews.write_text(json.dumps(record))
    (tmp_path / "src/ant_homing/level1/infonce_ms.py").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        loss_ablation.verify_gate(reviews, data, metadata)


def test_gate_rejects_changed_metadata_and_old_test_scenes(tmp_path, monkeypatch):
    reviews, data, metadata, _ = gate_fixture(tmp_path, monkeypatch)
    metadata["scene_seeds"]["test"] = list(range(300, 312))
    with pytest.raises(ValueError, match="Fresh test"):
        loss_ablation.verify_gate(reviews, data, metadata)
    metadata["scene_seeds"]["test"] = list(range(400, 412))
    (data / "metadata.json").write_text("changed")
    with pytest.raises(ValueError, match="metadata changed"):
        loss_ablation.verify_gate(reviews, data, metadata)


def test_incomplete_training_is_rejected_before_checkpoint_or_test_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(loss_ablation, "load_dataset", lambda path: ({}, {"data_sha256": "d"}))
    monkeypatch.setattr(loss_ablation, "verify_gate", lambda *args: {})
    (tmp_path / "review-manifest.json").write_text("reviews")
    (tmp_path / "trained.json").write_text(json.dumps({"runs": []}))
    with pytest.raises(ValueError, match="Incomplete"):
        loss_ablation.evaluate(Path("unused"), tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def frozen_fixture(tmp_path, monkeypatch):
    metadata = {
        "data_sha256": "d",
        "scene_seeds": {"test": list(range(400, 412))},
        "image_shape": [32, 128, 3],
        "camera_convention": "compass",
    }
    data = tmp_path / "data"
    data.mkdir()
    (data / "metadata.json").write_text("metadata")
    (tmp_path / "review-manifest.json").write_text("reviews")
    review_hash = loss_ablation.digest(tmp_path / "review-manifest.json")
    report = {
        "frozen_at_utc": "frozen",
        "dataset_sha256": "d",
        "metadata_sha256": loss_ablation.digest(data / "metadata.json"),
        "review_manifest_sha256": review_hash,
        "runs": [],
    }
    for seed in loss_ablation.SEEDS:
        for method in loss_ablation.METHODS:
            directory = tmp_path / f"{method}-seed-{seed}"
            directory.mkdir()
            saved = {
                "module": method,
                "seed": seed,
                "epoch": 1,
                "dataset_sha256": "d",
                "review_manifest_sha256": review_hash,
                "scene_seeds": metadata["scene_seeds"],
                "image_shape": metadata["image_shape"],
                "camera_convention": "compass",
                "state_dict": {},
            }
            if seed == 2 and method == "infonce_ms":
                saved["camera_convention"] = "stale"
            torch.save(saved, directory / "model.pt")
            report["runs"].append(
                {
                    "method": method,
                    "seed": seed,
                    "selected_epoch": 1,
                    "threshold": 0.2,
                    "checkpoint_sha256": loss_ablation.digest(directory / "model.pt"),
                }
            )
    (tmp_path / "trained.json").write_text(json.dumps(report))
    (tmp_path / "trained.sha256").write_text(loss_ablation.digest(tmp_path / "trained.json"))
    monkeypatch.setattr(loss_ablation, "load_dataset", lambda path: ({}, metadata))
    monkeypatch.setattr(loss_ablation, "verify_gate", lambda *args: {})

    def forbidden_scoring(*args, **kwargs):
        pytest.fail("Test scoring occurred before all frozen provenance checks")

    monkeypatch.setattr(loss_ablation, "evaluate_split", forbidden_scoring)
    return data, report


def test_changed_threshold_is_rejected_before_test_scoring(tmp_path, monkeypatch):
    data, report = frozen_fixture(tmp_path, monkeypatch)
    report["runs"][0]["threshold"] = 99
    (tmp_path / "trained.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="changed frozen"):
        loss_ablation.evaluate(data, tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_bad_sixth_checkpoint_provenance_blocks_all_test_scoring(tmp_path, monkeypatch):
    data, _ = frozen_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="provenance differs"):
        loss_ablation.evaluate(data, tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
