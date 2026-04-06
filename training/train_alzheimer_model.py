import argparse
import os
import sys
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.alzheimer_model import (
    ALZHEIMER_STAGE_CLASSES,
    build_alzheimer_staging_model,
    normalize_alzheimer_stage_label,
)
from training.common import (
    Sample,
    build_callbacks,
    build_dataset,
    compile_multiclass_model,
    discover_samples_from_csv,
    discover_samples_from_subdirectories,
    evaluate_multiclass_model,
    ensure_directory,
    split_samples,
    update_manifest_entry,
    with_augmentation,
    write_model_accuracy_file,
    write_training_metadata,
)


def discover_alzheimer_samples(dataset_dir: str, labels_csv=None) -> list[Sample]:
    samples = discover_samples_from_subdirectories(dataset_dir)
    if samples:
        normalized_samples = []
        for sample in samples:
            normalized_label = normalize_alzheimer_stage_label(sample.label)
            if normalized_label in ALZHEIMER_STAGE_CLASSES:
                normalized_samples.append(Sample(path=sample.path, label=normalized_label))
        if normalized_samples:
            return normalized_samples

    csv_path = labels_csv or os.path.join(dataset_dir, "labels.csv")
    if os.path.exists(csv_path):
        samples = discover_samples_from_csv(dataset_dir, csv_path)
        normalized_samples = []
        for sample in samples:
            normalized_label = normalize_alzheimer_stage_label(sample.label)
            if normalized_label in ALZHEIMER_STAGE_CLASSES:
                normalized_samples.append(Sample(path=sample.path, label=normalized_label))
        if normalized_samples:
            return normalized_samples

    raise ValueError(
        "No labeled Alzheimer dataset was found. Provide class subdirectories or a labels.csv file with filename,label columns."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the upgraded Alzheimer staging classifier.")
    parser.add_argument("--dataset-dir", default=os.path.join("dataset", "alz"))
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--output-dir", default=os.path.join("trained_models", "alz"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone", choices=("cnn", "mobilenetv2", "resnet50", "vgg16"), default="mobilenetv2")
    parser.add_argument("--fine-tune-layers", type=int, default=24)
    parser.add_argument("--weights", default="imagenet")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    samples = discover_alzheimer_samples(args.dataset_dir, labels_csv=args.labels_csv)
    if len(samples) < 4:
        raise ValueError("Not enough labeled Alzheimer samples were found to train the classifier.")

    train_samples, validation_samples = split_samples(samples, args.validation_split, args.seed)
    output_dir = ensure_directory(args.output_dir)
    image_size = (args.image_size, args.image_size)
    best_model_path = os.path.join(output_dir, "alz_classifier_best.h5")
    final_model_path = os.path.join(output_dir, "alz_classifier_final.h5")
    metadata_path = os.path.join(output_dir, "alz_classifier_metadata.json")
    metrics_path = os.path.join(output_dir, "model_metrics.json")

    base_model = build_alzheimer_staging_model(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=len(ALZHEIMER_STAGE_CLASSES),
        variant=args.backbone,
        weights=args.weights,
        fine_tune_layers=args.fine_tune_layers,
    )
    if base_model is None:
        raise RuntimeError("TensorFlow Keras is unavailable in the current environment.")
    model = compile_multiclass_model(
        with_augmentation(base_model, (args.image_size, args.image_size, 3)),
        learning_rate=args.learning_rate,
    )

    train_dataset = build_dataset(
        train_samples,
        ALZHEIMER_STAGE_CLASSES,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_dataset = build_dataset(
        validation_samples,
        ALZHEIMER_STAGE_CLASSES,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=build_callbacks(best_model_path),
        verbose=1,
    )
    keras_evaluation = model.evaluate(validation_dataset, verbose=0, return_dict=True)
    evaluation = evaluate_multiclass_model(
        model,
        validation_samples,
        ALZHEIMER_STAGE_CLASSES,
        image_size=image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if "loss" in keras_evaluation:
        evaluation["loss"] = round(float(keras_evaluation["loss"]), 4)
    model.save(final_model_path)
    write_training_metadata(metrics_path, evaluation)

    metadata = {
        "model_key": "alzheimer_classifier",
        "class_names": ALZHEIMER_STAGE_CLASSES,
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "image_size": list(image_size),
        "backbone": args.backbone,
        "weights": args.weights,
        "learning_rate": args.learning_rate,
        "target_accuracy": args.target_accuracy,
        "target_met": bool(evaluation.get("accuracy", 0.0) >= args.target_accuracy),
        "sample_count": len(samples),
        "class_distribution": dict(Counter(sample.label for sample in samples)),
        "train_count": len(train_samples),
        "validation_count": len(validation_samples),
        "metrics": evaluation,
        "history": history.history,
        "best_model_path": os.path.abspath(best_model_path),
        "final_model_path": os.path.abspath(final_model_path),
        "metrics_path": os.path.abspath(metrics_path),
    }
    write_training_metadata(metadata_path, metadata)
    write_model_accuracy_file("alzheimer_classifier", evaluation)

    if args.activate:
        update_manifest_entry(
            "alzheimer_classifier",
            os.path.abspath(final_model_path),
            os.path.abspath(metadata_path),
        )


if __name__ == "__main__":
    main()
