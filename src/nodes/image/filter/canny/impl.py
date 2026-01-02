from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2


class CannyNodeLogic(ComputeLogic):
    """
    Cannyエッジ検出を実行するノードロジック。
    入力・出力ともにOpenCV画像（numpy配列）。Base64変換はcore.pyで自動処理。
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

        low_threshold = int(properties.get("low_threshold", 50))
        high_threshold = int(properties.get("high_threshold", 150))

        # グレースケールに変換
        if len(img.shape) == 2 or img.shape[2] == 1:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # 出力をBGRに変換（他ノードとの互換性のため）
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return {"image": edges_bgr}
