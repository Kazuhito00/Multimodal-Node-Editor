from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2


class FlipNodeLogic(ComputeLogic):
    """
    画像を水平/垂直反転するノードロジック。
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

        horizontal = properties.get("horizontal", True)
        vertical = properties.get("vertical", False)

        result = img

        # cv2.flip: 0=垂直, 1=水平, -1=両方
        if horizontal and vertical:
            result = cv2.flip(img, -1)
        elif horizontal:
            result = cv2.flip(img, 1)
        elif vertical:
            result = cv2.flip(img, 0)

        return {"image": result}
