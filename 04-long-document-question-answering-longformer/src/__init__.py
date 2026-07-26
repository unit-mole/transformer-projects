"""Long-document question answering package."""

from .config import InferenceConfig
from .inference_pipeline import LongDocumentQAPipeline

__all__ = ["InferenceConfig", "LongDocumentQAPipeline"]
