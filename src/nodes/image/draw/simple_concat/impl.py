"""
Simple Concat ノードの実装。
2つの画像を水平方向または垂直方向に連結する。
"""
from typing import Dict, Any

import numpy as np
import cv2

from node_editor.node_def import ComputeLogic


# 方向定義
DIRECTION_HORIZONTAL = 0
DIRECTION_VERTICAL = 1


def resize_to_match_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """
    画像を指定幅にリサイズ（アスペクト比維持）

    Args:
        image: 入力画像
        target_width: 目標幅

    Returns:
        リサイズされた画像
    """
    h, w = image.shape[:2]
    if w == target_width:
        return image

    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_LINEAR)


def resize_to_match_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """
    画像を指定高さにリサイズ（アスペクト比維持）

    Args:
        image: 入力画像
        target_height: 目標高さ

    Returns:
        リサイズされた画像
    """
    h, w = image.shape[:2]
    if h == target_height:
        return image

    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_LINEAR)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    画像をBGR形式に変換

    Args:
        image: 入力画像

    Returns:
        BGR画像
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


class SimpleConcatLogic(ComputeLogic):
    """
    Simple Concatノードのロジック。
    2つの画像を水平または垂直方向に連結する。
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

        # BGR形式に変換
        img1 = ensure_bgr(image1)
        img2 = ensure_bgr(image2)

        # 方向取得
        direction = int(properties.get("direction", DIRECTION_HORIZONTAL))

        if direction == DIRECTION_HORIZONTAL:
            # 水平方向連結: 1枚目の高さに合わせる
            target_height = img1.shape[0]
            img2_resized = resize_to_match_height(img2, target_height)
            result = np.hstack([img1, img2_resized])
        else:
            # 垂直方向連結: 1枚目の幅に合わせる
            target_width = img1.shape[1]
            img2_resized = resize_to_match_width(img2, target_width)
            result = np.vstack([img1, img2_resized])

        return {"image": result}
