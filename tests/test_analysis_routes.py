import numpy as np

from routes import analysis_routes
from services.preprocessing import PreparedAnalysisInputs


class DummyPreprocessingService:
    def prepare(self, image_bytes=None, image=None):
        original = np.full((32, 32, 3), 80, dtype=np.uint8)
        enhanced = np.full((32, 32, 3), 120, dtype=np.uint8)
        classifier_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        segmentation_input = np.zeros((1, 128, 128, 1), dtype=np.float32)
        return PreparedAnalysisInputs(
            original_image=original,
            enhanced_image=enhanced,
            classifier_input=classifier_input,
            segmentation_input=segmentation_input,
            enhancement_backend="opencv",
            enhancement_steps=["gaussian_blur_light", "clahe", "sharpening_filter"],
        )


class DummyClassificationService:
    def classify(self, image, detection_type="combined", prepared_inputs=None):
        return {
            "tumor": {
                "detected": True,
                "classification": "glioma tumor",
                "tumor_type": "glioma tumor",
                "grade": "Grade IV",
                "tumor_stage": "Grade IV",
                "confidence": 0.91,
                "type_confidence": 0.9,
                "stage_confidence": 0.88,
                "model_metrics": {
                    "accuracy": 0.942,
                    "precision": 0.913,
                    "recall": 0.901,
                    "accuracy_label": "94.2%",
                    "precision_label": "91.3%",
                    "recall_label": "90.1%",
                },
            },
            "alzheimers": {
                "detected": False,
                "stage": "NonDemented",
                "confidence": 0.12,
                "model_metrics": {},
            },
        }


class DummySegmentationService:
    def segment(self, original_image, working_image, tumor_detected=True, prepared_input=None):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:22, 12:24] = 255
        overlay = original_image.copy()
        overlay[10:22, 12] = (0, 255, 0)
        return {
            "mask": mask,
            "display_mask": mask,
            "overlay": overlay,
            "bounding_box": {"x": 12, "y": 10, "width": 12, "height": 12},
            "backend": "heuristic",
            "contour_count": 1,
        }


class DummyVolumeService:
    def calculate(self, mask, pixel_spacing_x=1.0, pixel_spacing_y=1.0, slice_thickness=1.0):
        return {
            "tumor_volume_mm3": 22.75,
            "white_pixel_count": 144,
            "voxel_volume_mm3": 1.0,
        }


def test_analyze_medical_image_includes_stage_accuracy_and_overlay(monkeypatch):
    monkeypatch.setattr(analysis_routes, "get_preprocessing_service", lambda: DummyPreprocessingService())
    monkeypatch.setattr(analysis_routes, "get_classification_service", lambda: DummyClassificationService())
    monkeypatch.setattr(analysis_routes, "get_segmentation_service", lambda: DummySegmentationService())
    monkeypatch.setattr(analysis_routes, "get_volume_calculation_service", lambda: DummyVolumeService())
    monkeypatch.setattr(
        analysis_routes,
        "save_analysis_assets",
        lambda images: {
            "study_id": "study-1",
            "files": {
                "original_image": "original.png",
                "enhanced_image": "enhanced.png",
                "overlay_image": "overlay.png",
                "mask_image": "mask.png",
            },
        },
    )

    result = analysis_routes.analyze_medical_image(
        image_bytes=b"synthetic-image",
        detection_type="brain",
        include_encoded_images=False,
    )

    assert result["tumor"]["tumor_type"] == "glioma tumor"
    assert result["tumor"]["tumor_stage"] == "Grade IV"
    assert result["model_accuracy"] == "94.2%"
    assert result["model_accuracy_score"] == 0.942
    assert result["segmentation"]["segmentation_overlay"] == "overlay.png"
    assert result["segmentation"]["mask_quality"] == "clear"
    assert result["ai_clinical_insights"]["tumor_stage"] == "Grade IV"
    assert result["ai_clinical_insights"]["alzheimer_stage"] == "NonDemented"
    assert result["alzheimer_stage"] == "NonDemented"
    assert result["primary_result"]["tumor_stage"] == "Grade IV"
