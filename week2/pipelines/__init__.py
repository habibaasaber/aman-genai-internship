"""
pipelines/__init__.py
=====================
Public API for the pipelines package.

Exports the shared ``PipelineResult`` dataclass and both pipeline classes
so callers can do:
    from pipelines import NaivePipeline, AdvancedPipeline, PipelineResult
"""

from pipelines.advanced_pipeline import AdvancedPipeline
from pipelines.naive_pipeline import NaivePipeline, PipelineResult

__all__ = [
    "NaivePipeline",
    "AdvancedPipeline",
    "PipelineResult",
]
