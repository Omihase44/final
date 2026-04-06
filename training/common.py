import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from services.enhancement import ImageEnhancementService
from services.model_metrics import write_model_accuracy_registry_entry
from utils.tensorflow_compat import DEFAULT_MODEL_MANIFEST_PATH

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - only exercised when TensorFlow is available
    raise RuntimeError("TensorFlow is required to run the training scripts.") from exc


@dataclass(frozen=True)
class Sample:
    path: str
    label: str


_ENHANCEMENT_SERVICE = ImageEnhancementService()


def ensure_directory(path: str) -> str:
    resolved_path = os.path.abspath(path)
    os.makedirs(resolved_path, exist_ok=True)
    return resolved_path


def _to_json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _enhance_training_image(image: np.ndarray) -> np.ndarray:
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(bgr_image, None, 4, 4, 7, 15)
    enhanced = _ENHANCEMENT_SERVICE.enhance(denoised)["enhanced_image"]
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)


def load_rgb_image(path: str, image_size: tuple[int, int], apply_enhancement: bool = True) -> np.ndarray:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        image_array = np.asarray(rgb_image, dtype=np.uint8)
        if apply_enhancement:
            image_array = _enhance_training_image(image_array)
        resized = cv2.resize(image_array, image_size, interpolation=cv2.INTER_CUBIC)
        return np.clip(resized.astype(np.float32) / 255.0, 0.0, 1.0)


def split_samples(samples: Sequence[Sample], validation_split: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    labels = [sample.label for sample in samples]
    train_samples, validation_samples = train_test_split(
        list(samples),
        test_size=validation_split,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    return train_samples, validation_samples


def build_dataset(
    samples: Sequence[Sample],
    class_names: Sequence[str],
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    apply_enhancement: bool = True,
):
    class_index = {label: index for index, label in enumerate(class_names)}
    num_classes = len(class_names)

    def generator():
        for sample in samples:
            image = load_rgb_image(sample.path, image_size, apply_enhancement=apply_enhancement)
            target = np.zeros(num_classes, dtype=np.float32)
            target[class_index[sample.label]] = 1.0
            yield image, target

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(max(len(samples), 1), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def with_augmentation(model, input_shape: tuple[int, int, int]):
    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.12)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomTranslation(0.05, 0.05)(x)
    x = layers.RandomContrast(0.15)(x)
    random_brightness_layer = getattr(layers, "RandomBrightness", None)
    if random_brightness_layer is not None:
        x = random_brightness_layer(0.12)(x)
    outputs = model(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{model.name}_with_augmentation")


def compile_multiclass_model(model, learning_rate: float = 1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_callbacks(best_model_path: str):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, mode="max"),
    ]


def write_training_metadata(metadata_path: str, payload: dict) -> None:
    ensure_directory(os.path.dirname(metadata_path))
    with open(metadata_path, "w", encoding="utf-8") as file_handle:
        json.dump(_to_json_compatible(payload), file_handle, indent=2)


def update_manifest_entry(manifest_key: str, model_path: str, metadata_path: str, framework: str = "keras") -> None:
    manifest_path = os.environ.get("MODEL_MANIFEST_PATH", "").strip() or DEFAULT_MODEL_MANIFEST_PATH
    resolved_manifest_path = os.path.abspath(manifest_path)
    ensure_directory(os.path.dirname(resolved_manifest_path))

    payload = {}
    if os.path.exists(resolved_manifest_path):
        with open(resolved_manifest_path, "r", encoding="utf-8") as file_handle:
            try:
                payload = json.load(file_handle)
            except json.JSONDecodeError:
                payload = {}
    if not isinstance(payload, dict):
        payload = {}

    payload[manifest_key] = {
        "path": model_path,
        "metadata_path": metadata_path,
        "framework": framework,
    }
    with open(resolved_manifest_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def write_model_accuracy_file(model_key: str, metrics: dict) -> str:
    return write_model_accuracy_registry_entry(model_key, metrics)


def evaluate_multiclass_model(
    model,
    samples: Sequence[Sample],
    class_names: Sequence[str],
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> dict:
    if not samples:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "sample_count": 0,
            "class_names": list(class_names),
        }

    dataset = build_dataset(
        samples,
        class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    raw_scores = np.asarray(model.predict(dataset, verbose=0))
    if raw_scores.ndim == 1:
        raw_scores = np.expand_dims(raw_scores, axis=0)

    y_true = np.asarray([class_names.index(sample.label) for sample in samples], dtype=np.int64)
    y_pred = np.argmax(raw_scores[:, : len(class_names)], axis=1)
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    per_class = {}
    for class_index, class_name in enumerate(class_names):
        class_precision, class_recall, class_f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[class_index],
            average=None,
            zero_division=0,
        )
        per_class[class_name] = {
            "precision": float(class_precision[0]) if len(class_precision) else 0.0,
            "recall": float(class_recall[0]) if len(class_recall) else 0.0,
            "f1_score": float(class_f1[0]) if len(class_f1) else 0.0,
            "support": int(support[0]) if len(support) else 0,
        }

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "sample_count": len(samples),
        "class_names": list(class_names),
        "per_class": per_class,
    }


def discover_samples_from_subdirectories(dataset_dir: str, allowed_labels: Optional[Iterable[str]] = None) -> list[Sample]:
    resolved_dir = os.path.abspath(dataset_dir)
    allowed_lookup = {label.lower(): label for label in allowed_labels} if allowed_labels else None
    samples: list[Sample] = []
    for subdirectory in sorted(Path(resolved_dir).iterdir()):
        if not subdirectory.is_dir():
            continue
        label_name = subdirectory.name
        if allowed_lookup is not None:
            label_name = allowed_lookup.get(subdirectory.name.lower())
            if label_name is None:
                continue
        for file_path in sorted(subdirectory.rglob("*")):
            if file_path.is_file():
                samples.append(Sample(path=str(file_path), label=str(label_name)))
    return samples


def discover_samples_from_csv(dataset_dir: str, csv_path: str) -> list[Sample]:
    resolved_dir = os.path.abspath(dataset_dir)
    resolved_csv_path = os.path.abspath(csv_path)
    samples: list[Sample] = []
    with open(resolved_csv_path, "r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            filename = str(
                row.get("filename")
                or row.get("file")
                or row.get("image")
                or row.get("path")
                or ""
            ).strip()
            label = str(
                row.get("label")
                or row.get("stage")
                or row.get("diagnosis")
                or row.get("class")
                or ""
            ).strip()
            if not filename or not label:
                continue
            sample_path = os.path.join(resolved_dir, filename)
            if os.path.exists(sample_path):
                samples.append(Sample(path=sample_path, label=label))
    return samples
