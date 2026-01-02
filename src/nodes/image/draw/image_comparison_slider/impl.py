"""
Image Comparison Slider ノードの実装。
2つの画像をスライダー位置で左右に分割して比較表示する。
"""
from typing import Dict, Any

import numpy as np
import cv2

from node_editor.node_def import ComputeLogic


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """画像をBGR形式に変換"""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def resize_to_match(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """画像を指定サイズにリサイズ"""
    h, w = image.shape[:2]
    if h == target_height and w == target_width:
        return image
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def create_comparison_image(
    image1: np.ndarray,
    image2: np.ndarray,
    position: float
) -> np.ndarray:
    """
    2つの画像を比較表示

    Args:
        image1: 左側の画像
        image2: 右側の画像
        position: スライダー位置（0.0〜1.0）

    Returns:
        比較画像
    """
    # BGR形式に変換
    img1 = ensure_bgr(image1)
    img2 = ensure_bgr(image2)

    # サイズを揃える（image1を基準）
    h, w = img1.shape[:2]
    img2 = resize_to_match(img2, h, w)

    # スライダー位置を計算
    split_x = int(w * position)
    split_x = max(0, min(w, split_x))

    # 結果画像を作成
    result = img2.copy()
    if split_x > 0:
        result[:, :split_x] = img1[:, :split_x]

    # スライダーラインを描画（白線）
    line_color = (255, 255, 255)
    cv2.line(result, (split_x, 0), (split_x, h), line_color, 2)

    return result


class ImageComparisonSliderLogic(ComputeLogic):
    """
    Image Comparison Sliderノードのロジック。
    2つの画像をスライダー位置で左右に分割して比較表示する。
    """

    def __init__(self):
        pass

    def reset(self):
        """ノードの状態をリセット"""
        pass

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        image1 = inputs.get("Image 1")
        image2 = inputs.get("Image 2")

        # どちらかの画像がない場合
        if image1 is None and image2 is None:
            return {"image": None}

        if image1 is None:
            return {"image": ensure_bgr(image2)}

        if image2 is None:
            return {"image": ensure_bgr(image1)}

        # キャンセルチェック
        if self.is_cancelled():
            self.clear_cancel()
            return {"image": None}

        # スライダー位置を取得
        position = float(properties.get("position", 0.5))
        position = max(0.0, min(1.0, position))

        # 比較画像を作成
        result = create_comparison_image(image1, image2, position)

        return {"image": result}
