from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2
import numpy as np


class HsvNodeLogic(ComputeLogic):
    """
    画像のHSV（色相・彩度・明度）を調整するノードロジック。
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

        h_add = int(properties.get("h_add", 0))
        s_add = int(properties.get("s_add", 0))
        v_add = int(properties.get("v_add", 0))

        # 調整値がすべて0の場合は元画像を返す
        if h_add == 0 and s_add == 0 and v_add == 0:
            return {"image": img}

        # BGRからHSVに変換
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        # H（色相）: 0-179の範囲でラップアラウンド
        h = h.astype(np.int16) + h_add
        h[h < 0] += 180
        h[h > 179] -= 180
        h = h.astype(np.uint8)

        # S（彩度）: 0-255の範囲でクリップ
        s = np.clip(s.astype(np.int16) + s_add, 0, 255).astype(np.uint8)

        # V（明度）: 0-255の範囲でクリップ
        v = np.clip(v.astype(np.int16) + v_add, 0, 255).astype(np.uint8)

        # チャンネルを結合してBGRに戻す
        processed_hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(processed_hsv, cv2.COLOR_HSV2BGR)

        return {"image": result}
