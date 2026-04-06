import os
import json

import pytest

import utils.tensorflow_compat as tensorflow_compat


def test_resolve_model_path_prefers_manifest(tmp_path, monkeypatch):
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(
        '{"brain_classifier":{"path":"trained_models/tumor/latest.h5"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(manifest_path))
    tensorflow_compat.clear_model_cache()

    resolved_path = tensorflow_compat.resolve_model_path(
        "brain_model_new.h5",
        manifest_key="brain_classifier",
    )

    assert resolved_path.endswith(os.path.join("trained_models", "tumor", "latest.h5"))


def test_env_override_beats_manifest(tmp_path, monkeypatch):
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(
        '{"brain_classifier":{"path":"trained_models/tumor/latest.h5"}}',
        encoding="utf-8",
    )
    override_path = tmp_path / "override.h5"
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("BRAIN_MODEL_PATH", str(override_path))
    tensorflow_compat.clear_model_cache()

    resolved_path = tensorflow_compat.resolve_model_path(
        "brain_model_new.h5",
        env_key="BRAIN_MODEL_PATH",
        manifest_key="brain_classifier",
    )

    assert resolved_path == str(override_path)


def test_resolve_model_metadata_reads_manifest_metadata(tmp_path, monkeypatch):
    metadata_path = tmp_path / "brain_metadata.json"
    metadata_path.write_text('{"class_names":["glioma tumor","meningioma tumor","no tumor","pituitary tumor"]}', encoding="utf-8")
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "brain_classifier": {
                    "path": "trained_models/tumor/latest.h5",
                    "metadata_path": str(metadata_path),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(manifest_path))
    tensorflow_compat.clear_model_cache()

    metadata = tensorflow_compat.resolve_model_metadata("brain_classifier")

    assert metadata["class_names"][0] == "glioma tumor"


def test_safe_load_missing_model_returns_none(tmp_path):
    tensorflow_compat.clear_model_cache()
    assert tensorflow_compat.safe_load_keras_model(str(tmp_path / "missing_model.h5")) is None


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("1", True), ("true", True), ("yes", True), ("0", False), ("", False)],
)
def test_require_strict_model_loading(monkeypatch, raw_value, expected):
    monkeypatch.setenv("STRICT_MODEL_LOADING", raw_value)
    assert tensorflow_compat.require_strict_model_loading() is expected
