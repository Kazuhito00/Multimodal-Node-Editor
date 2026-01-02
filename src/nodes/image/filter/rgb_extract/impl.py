"""RGB Extractノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic

import cv2
import numpy as np


class RgbExtractLogic(ComputeLogic):
    """
    画像から指定したRGBチャンネルを抽出するノードロジック。
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

        channel = int(properties.get("channel", 0))

        # BGRチャンネルを分離
        b, g, r = cv2.split(img)

        # 選択されたチャンネルを取得
        if channel == 0:
            # R チャンネル
            selected = r
        elif channel == 1:
            # G チャンネル
            selected = g
        else:
            # B チャンネル
            selected = b

        # グレースケールを3チャンネルに変換して返す
        result = cv2.cvtColor(selected, cv2.COLOR_GRAY2BGR)

        return {"image": result}
