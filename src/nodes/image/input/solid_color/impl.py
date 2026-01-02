from typing import Dict, Any, Optional
from node_editor.node_def import ComputeLogic
import numpy as np


def hex_to_bgr(hex_color: str) -> tuple:
    """HEXカラーをBGRタプルに変換"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (0, 0, 255)  # デフォルトは赤
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCVはBGR形式


class SolidColorLogic(ComputeLogic):
    """単色画像を生成するノードロジック"""

    def __init__(self):
        self._cached_image: Optional[np.ndarray] = None
        self._last_width: int = 0
        self._last_height: int = 0
        self._last_color: str = ""

    def reset(self):
        """キャッシュをクリア"""
        self._cached_image = None
        self._last_width = 0
        self._last_height = 0
        self._last_color = ""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        width = int(properties.get("width", 640))
        height = int(properties.get("height", 360))
        color = properties.get("color", "#ff0000")

        # パラメータが変わった場合のみ再生成
        if (width != self._last_width or
            height != self._last_height or
            color != self._last_color):

            bgr = hex_to_bgr(color)
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[:] = bgr

            self._cached_image = image
            self._last_width = width
            self._last_height = height
            self._last_color = color

        return {"image_out": self._cached_image}
