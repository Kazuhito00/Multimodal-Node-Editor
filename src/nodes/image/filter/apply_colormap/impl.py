from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2

# ドロップダウンの値からOpenCVカラーマップ定数へのマッピング
COLORMAP_OPTIONS = {
    0: cv2.COLORMAP_AUTUMN,
    1: cv2.COLORMAP_BONE,
    2: cv2.COLORMAP_JET,
    3: cv2.COLORMAP_WINTER,
    4: cv2.COLORMAP_RAINBOW,
    5: cv2.COLORMAP_OCEAN,
    6: cv2.COLORMAP_SUMMER,
    7: cv2.COLORMAP_SPRING,
    8: cv2.COLORMAP_COOL,
    9: cv2.COLORMAP_HSV,
    10: cv2.COLORMAP_PINK,
    11: cv2.COLORMAP_HOT,
    12: cv2.COLORMAP_PARULA,
    13: cv2.COLORMAP_MAGMA,
    14: cv2.COLORMAP_INFERNO,
    15: cv2.COLORMAP_PLASMA,
    16: cv2.COLORMAP_VIRIDIS,
    17: cv2.COLORMAP_CIVIDIS,
    18: cv2.COLORMAP_TWILIGHT,
    19: cv2.COLORMAP_TWILIGHT_SHIFTED,
    20: cv2.COLORMAP_TURBO,
    21: cv2.COLORMAP_DEEPGREEN,
}


class ApplyColormapNodeLogic(ComputeLogic):
    """
    グレースケール画像にカラーマップを適用するノードロジック。
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

        colormap_index = int(properties.get("colormap", 2))
        colormap = COLORMAP_OPTIONS.get(colormap_index, cv2.COLORMAP_JET)

        # グレースケールに変換
        if len(img.shape) == 2:
            gray = img
        elif img.shape[2] == 1:
            gray = img[:, :, 0]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # カラーマップを適用
        colored = cv2.applyColorMap(gray, colormap)

        return {"image": colored}
