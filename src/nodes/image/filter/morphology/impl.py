from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2
import numpy as np

# 操作の定数
MORPH_DILATE = 0
MORPH_ERODE = 1
MORPH_OPEN = 2
MORPH_CLOSE = 3


class MorphologyNodeLogic(ComputeLogic):
    """
    モルフォロジー演算（膨張、収縮、オープニング、クロージング）を適用するノード
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

        operation = int(properties.get("operation", MORPH_DILATE))
        kernel_size = int(properties.get("kernel_size", 3))
        iterations = int(properties.get("iterations", 1))

        # カーネルサイズを奇数に調整
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(1, min(31, kernel_size))

        # 構造化要素（カーネル）を作成
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )

        # モルフォロジー演算を適用
        if operation == MORPH_DILATE:
            result = cv2.dilate(img, kernel, iterations=iterations)

        elif operation == MORPH_ERODE:
            result = cv2.erode(img, kernel, iterations=iterations)

        elif operation == MORPH_OPEN:
            result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        elif operation == MORPH_CLOSE:
            result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        else:
            result = img

        return {"image": result}
