from typing import Dict, Any
from node_editor.node_def import ComputeLogic

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


class ResizeLogic(ComputeLogic):
    """画像リサイズノードロジック"""

    # OpenCV補間方法のマッピング
    INTERPOLATION_METHODS = {
        0: cv2.INTER_NEAREST if CV2_AVAILABLE else 0,
        1: cv2.INTER_LINEAR if CV2_AVAILABLE else 1,
        2: cv2.INTER_CUBIC if CV2_AVAILABLE else 2,
        3: cv2.INTER_AREA if CV2_AVAILABLE else 3,
        4: cv2.INTER_LANCZOS4 if CV2_AVAILABLE else 4,
    }

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {
                "image": None,
                "__error__": "opencv-python is not installed",
            }

        image = inputs.get("image")
        if image is None:
            return {"image": None}

        width = int(properties.get("width", 640))
        height = int(properties.get("height", 480))
        method_id = int(properties.get("method", 1))

        # 値の範囲チェック
        width = max(1, min(7680, width))
        height = max(1, min(4320, height))

        interpolation = self.INTERPOLATION_METHODS.get(method_id, cv2.INTER_LINEAR)

        try:
            resized = cv2.resize(image, (width, height), interpolation=interpolation)
            return {"image": resized}
        except Exception as e:
            return {
                "image": None,
                "__error__": str(e),
            }
