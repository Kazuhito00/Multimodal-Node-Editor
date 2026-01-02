from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2
import numpy as np


class GammaNodeLogic(ComputeLogic):
    """
    ガンマ補正を行うノードロジック。
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

        gamma = float(properties.get("gamma", 1.0))

        # ガンマ補正テーブルを作成
        table = (np.arange(256) / 255) ** gamma * 255
        table = np.clip(table, 0, 255).astype(np.uint8)

        # ルックアップテーブルを適用
        result = cv2.LUT(img, table)

        return {"image": result}
