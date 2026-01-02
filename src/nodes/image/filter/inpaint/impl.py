from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2


class InpaintNodeLogic(ComputeLogic):
    """
    インペインティングで画像を修復するノードロジック。
    """

    # アルゴリズムマッピング
    ALGORITHM_MAP = {
        0: cv2.INPAINT_TELEA,
        1: cv2.INPAINT_NS,
    }

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        img = inputs.get("image")
        mask = inputs.get("mask")

        if img is None:
            return {"image": None}

        # マスクがない場合は元画像をそのまま返す
        if mask is None:
            return {"image": img}

        radius = int(properties.get("radius", 3))
        algorithm = int(properties.get("algorithm", 0))

        # マスクをグレースケールに変換
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        # マスクを二値化（0以外を255に）
        _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

        # マスクのサイズを画像に合わせる
        if mask_binary.shape[:2] != img.shape[:2]:
            mask_binary = cv2.resize(mask_binary, (img.shape[1], img.shape[0]))

        # インペインティング実行
        inpaint_method = self.ALGORITHM_MAP.get(algorithm, cv2.INPAINT_TELEA)
        result = cv2.inpaint(img, mask_binary, radius, inpaint_method)

        return {"image": result}
