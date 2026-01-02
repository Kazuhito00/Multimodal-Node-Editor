"""RGB Adjustノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic

import cv2
import numpy as np


class RgbAdjustLogic(ComputeLogic):
    """
    画像のRGB（赤・緑・青）チャンネルを調整するノードロジック。
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

        # プロパティまたは入力ポートから値を取得
        r_add = inputs.get("r_add")
        if r_add is None:
            r_add = int(properties.get("r_add", 0))

        g_add = inputs.get("g_add")
        if g_add is None:
            g_add = int(properties.get("g_add", 0))

        b_add = inputs.get("b_add")
        if b_add is None:
            b_add = int(properties.get("b_add", 0))

        # 調整値がすべて0の場合は元画像を返す
        if r_add == 0 and g_add == 0 and b_add == 0:
            return {"image": img}

        # BGRチャンネルを分離
        b, g, r = cv2.split(img)

        # 各チャンネルを調整（0-255の範囲でクリップ）
        r = np.clip(r.astype(np.int16) + r_add, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.int16) + g_add, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.int16) + b_add, 0, 255).astype(np.uint8)

        # チャンネルを結合
        result = cv2.merge([b, g, r])

        return {"image": result}
