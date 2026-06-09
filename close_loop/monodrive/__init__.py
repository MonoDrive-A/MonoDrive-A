"""MonoDrive Carla 闭环推理子包。"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import MonoDriveAgent

__all__ = ["MonoDriveAgent"]


def __getattr__(name: str):
    if name == "MonoDriveAgent":
        from .agent import MonoDriveAgent

        return MonoDriveAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
