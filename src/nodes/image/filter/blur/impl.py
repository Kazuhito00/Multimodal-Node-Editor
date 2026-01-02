from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2

# アルゴリズムの定数
BLUR_BOX = 0
BLUR_GAUSSIAN = 1
BLUR_MEDIAN = 2
BLUR_BILATERAL = 3


class BlurNodeLogic(ComputeLogic):
    """
    画像にぼかしフィルタを適用するノードロジック。
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

        algorithm = int(properties.get("algorithm", BLUR_BOX))
        kernel_size = int(properties.get("kernel_size", 5))
        sigma = float(properties.get("sigma", 1.0))

        # カーネルサイズを奇数に調整（Median/Bilateralで必要）
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(1, min(49, kernel_size))

        if algorithm == BLUR_BOX:
            result = cv2.blur(img, (kernel_size, kernel_size))

        elif algorithm == BLUR_GAUSSIAN:
            result = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

        elif algorithm == BLUR_MEDIAN:
            # medianBlurはカーネルサイズが1より大きい奇数である必要がある
            ksize = max(3, kernel_size)
            if ksize % 2 == 0:
                ksize += 1
            result = cv2.medianBlur(img, ksize)

        elif algorithm == BLUR_BILATERAL:
            # bilateralFilter: d=カーネルサイズ, sigmaColor=sigma, sigmaSpace=sigma
            result = cv2.bilateralFilter(img, kernel_size, sigma * 10, sigma * 10)

        else:
            result = img

        return {"image": result}
