from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2


class ContrastNodeLogic(ComputeLogic):
    """
    画像のコントラストを調整するノードロジック。
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

        contrast = float(properties.get("contrast", 1.0))

        # コントラスト調整（中心を128として拡大/縮小）
        if contrast != 1.0:
            # 中心値(128)を基準にコントラスト調整
            result = cv2.convertScaleAbs(img, alpha=contrast, beta=128 * (1 - contrast))
        else:
            result = img

        return {"image": result}
