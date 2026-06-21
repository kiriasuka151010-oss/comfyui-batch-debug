"""
ComfyUI Batch Debug Plugin
批量调试出图插件

A ComfyUI custom node pack for batch parameter sweeping,
comparison grid output, and scoring-oriented metadata export.

Nodes:
  - BatchDebugConfig:   Set up parameter sweep ranges
  - BatchDebugExecute:  Execute the batch generation loop
  - BatchDebugGridSave: Save individual images, labeled grid, and CSV metadata
"""

from .nodes import BatchDebugConfig, BatchDebugExecute, BatchDebugGridSave

NODE_CLASS_MAPPINGS = {
    "BatchDebugConfig": BatchDebugConfig,
    "BatchDebugExecute": BatchDebugExecute,
    "BatchDebugGridSave": BatchDebugGridSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchDebugConfig": "Batch Debug Config (批量调试配置)",
    "BatchDebugExecute": "Batch Debug Execute (批量调试执行)",
    "BatchDebugGridSave": "Batch Debug Grid Save (批量调试保存)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
