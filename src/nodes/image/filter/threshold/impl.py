from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2


class ThresholdNodeLogic(ComputeLogic):
    """
    画像に閾値処理を適用するノードロジック。
    """

    # アルゴリズムマッピング
    ALGORITHM_MAP = {
        0: cv2.THRESH_BINARY,
        1: cv2.THRESH_BINARY_INV,
        2: cv2.THRESH_TRUNC,
        3: cv2.THRESH_TOZERO,
        4: cv2.THRESH_TOZERO_INV,
        5: cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        6: cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
    }

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        img = inputs.get("image")
        if img is None:
            return {"image": None}

        algorithm = int(properties.get("algorithm", 0))
        threshold_value = int(properties.get("threshold", 128))

        # グレースケールに変換
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 閾値処理を適用
        thresh_type = self.ALGORITHM_MAP.get(algorithm, cv2.THRESH_BINARY)
        _, result = cv2.threshold(gray, threshold_value, 255, thresh_type)

        # 3チャンネルに戻す
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return {"image": result}
