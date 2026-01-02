from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2
import numpy as np


def parse_kernel_string(kernel_str: str) -> np.ndarray:
    """カーネル文字列をNumPy配列に変換"""
    try:
        values = [float(v.strip()) for v in kernel_str.split(",")]
        if len(values) != 9:
            return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        return np.array(values, dtype=np.float32).reshape(3, 3)
    except Exception:
        return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)


class Filter2DNodeLogic(ComputeLogic):
    """
    3x3カーネルで2Dフィルタリングを行うノードロジック。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        img = inputs.get("image")
        if img is None:
            return {"image": None}

        # カーネル文字列をパース
        kernel_str = properties.get("kernel", "0,0,0,0,1,0,0,0,0")
        kernel = parse_kernel_string(kernel_str)

        # 2Dフィルタリングを適用
        result = cv2.filter2D(img, -1, kernel)

        return {"image": result}
