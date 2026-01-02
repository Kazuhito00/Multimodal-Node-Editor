from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2


class BrightnessNodeLogic(ComputeLogic):
    """
    画像の輝度を調整するノードロジック。
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

        brightness = int(properties.get("brightness", 0))

        # 輝度調整（クリッピング付き）
        if brightness != 0:
            result = cv2.add(img, brightness)
        else:
            result = img

        return {"image": result}
