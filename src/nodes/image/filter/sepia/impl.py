from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2
import numpy as np


class SepiaNodeLogic(ComputeLogic):
    """
    画像にセピア調エフェクトを適用するノードロジック。
    """

    # セピア変換行列
    SEPIA_MATRIX = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ], dtype=np.float32)

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        img = inputs.get("image")
        if img is None:
            return {"image": None}

        # セピア変換を適用
        result = cv2.transform(img, self.SEPIA_MATRIX)

        # 値を0-255にクリップ
        result = np.clip(result, 0, 255).astype(np.uint8)

        return {"image": result}
