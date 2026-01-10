"""
Inference Module

Provides the pipeline manager for VTON inference.
"""

import os

from .pipeline_manager import PipelineManager, VTON_PROMPT_CN, VTON_PROMPT_EN
from .mock_pipeline import MockPipelineManager


def get_pipeline_manager(config):
    """
    Get the appropriate pipeline manager based on environment.
    
    Set GPU_SERVER_MOCK=1 to use mock mode.
    """
    if os.environ.get("GPU_SERVER_MOCK", "").lower() in ("1", "true", "yes"):
        print("🎭 Using MockPipelineManager (GPU_SERVER_MOCK=1)")
        return MockPipelineManager(config)
    else:
        return PipelineManager(config)


__all__ = [
    "PipelineManager",
    "MockPipelineManager",
    "get_pipeline_manager",
    "VTON_PROMPT_CN",
    "VTON_PROMPT_EN",
]
