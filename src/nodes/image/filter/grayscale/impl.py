from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2


class GrayscaleNodeLogic(ComputeLogic):
    """
    画像をグレースケールに変換するノードロジック。
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

        # グレースケールに変換し、3チャンネルに戻す
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return {"image": result}
