import json

from services import model_metrics


def test_get_model_metrics_prefers_metadata(tmp_path, monkeypatch):
    metadata_path = tmp_path / "tumor_metadata.json"
    metadata_path.write_text(
        json.dumps({"metrics": {"accuracy": 0.942, "precision": 0.913, "recall": 0.901, "f1_score": 0.907}}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(
        json.dumps({"brain_classifier": {"metadata_path": str(metadata_path)}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(manifest_path))
    model_metrics.clear_model_metrics_cache()
    metrics = model_metrics.get_model_metrics("brain_classifier")

    assert metrics["available"] is True
    assert metrics["source"] == "metadata"
    assert metrics["accuracy_label"] == "94.2%"
    assert metrics["precision_label"] == "91.3%"
    assert metrics["recall_label"] == "90.1%"
    assert metrics["f1_score_label"] == "90.7%"


def test_write_model_accuracy_registry_entry_round_trips(tmp_path, monkeypatch):
    accuracy_path = tmp_path / "model_accuracy.json"
    monkeypatch.setenv("MODEL_ACCURACY_PATH", str(accuracy_path))
    model_metrics.clear_model_metrics_cache()

    written_path = model_metrics.write_model_accuracy_registry_entry(
        "brain_classifier",
        {"accuracy": 0.951, "precision": 0.923, "recall": 0.917, "f1_score": 0.919},
    )
    metrics = model_metrics.get_model_metrics("brain_classifier")

    assert written_path == str(accuracy_path)
    assert metrics["source"] == "accuracy_registry"
    assert metrics["accuracy_label"] == "95.1%"
    assert metrics["precision_label"] == "92.3%"
    assert metrics["recall_label"] == "91.7%"
    assert metrics["f1_score_label"] == "91.9%"
