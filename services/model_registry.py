from functools import lru_cache
from threading import Lock

from models.alzheimer_model import AlzheimerStagingModel
from models.segmentation_model import TumorSegmentationModel
from models.tumor_model import TumorGradingModel


class ModelRegistry:
    """Thread-safe lazy model registry used across inference services."""

    def __init__(self):
        self._lock = Lock()
        self._instances = {}

    def _get_or_create(self, key, factory):
        instance = self._instances.get(key)
        if instance is not None:
            return instance

        with self._lock:
            instance = self._instances.get(key)
            if instance is None:
                instance = factory()
                self._instances[key] = instance
        return instance

    def get_tumor_model(self) -> TumorGradingModel:
        return self._get_or_create("tumor_model", TumorGradingModel)

    def get_alzheimer_model(self) -> AlzheimerStagingModel:
        return self._get_or_create("alzheimer_model", AlzheimerStagingModel)

    def get_segmentation_model(self) -> TumorSegmentationModel:
        return self._get_or_create("segmentation_model", TumorSegmentationModel)


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
