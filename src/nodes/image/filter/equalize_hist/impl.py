from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2


class EqualizeHistNodeLogic(ComputeLogic):
    """
    ヒストグラム平坦化を行うノードロジック。
    HSVのV（明度）チャンネルに対してヒストグラム平坦化を適用。
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

        # BGRからHSVに変換
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # V（明度）チャンネルにヒストグラム平坦化を適用
        hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

        # HSVからBGRに戻す
        result = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return {"image": result}
