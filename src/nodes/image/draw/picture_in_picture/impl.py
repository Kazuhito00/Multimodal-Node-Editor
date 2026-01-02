"""
Picture In Picture ノードの実装。
Image 1を背景として、Image 2を指定領域に重ねて表示する。
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


class PictureInPictureLogic(ComputeLogic):
    """
    Picture In Pictureノードのロジック。
    Image 1を背景として、Image 2を指定領域（min_x, min_y, max_x, max_y）に重ねる。
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

        # どちらもない場合
        if image1 is None and image2 is None:
            return {"image": None}

        # Image 1のみの場合はそのまま返す
        if image2 is None:
            return {"image": ensure_bgr(image1)}

        # Image 2のみの場合はそのまま返す
        if image1 is None:
            return {"image": ensure_bgr(image2)}

        # キャンセルチェック
        if self.is_cancelled():
            self.clear_cancel()
            return {"image": None}

        # 正規化座標を取得（0.0〜1.0）
        min_x = float(properties.get("min_x", 0.7))
        min_y = float(properties.get("min_y", 0.7))
        max_x = float(properties.get("max_x", 0.9))
        max_y = float(properties.get("max_y", 0.9))

        # 値を0.0〜1.0にクランプ
        min_x = max(0.0, min(1.0, min_x))
        min_y = max(0.0, min(1.0, min_y))
        max_x = max(0.0, min(1.0, max_x))
        max_y = max(0.0, min(1.0, max_y))

        # min > maxの場合は入れ替え
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_y > max_y:
            min_y, max_y = max_y, min_y

        # BGR形式に変換
        img1 = ensure_bgr(image1)
        img2 = ensure_bgr(image2)

        # 結果画像を作成（Image 1をコピー）
        result = img1.copy()
        h, w = result.shape[:2]

        # ピクセル座標に変換
        x1 = int(min_x * w)
        y1 = int(min_y * h)
        x2 = int(max_x * w)
        y2 = int(max_y * h)

        # 領域サイズを計算
        region_w = x2 - x1
        region_h = y2 - y1

        # 領域が0以下の場合はImage 1を返す
        if region_w <= 0 or region_h <= 0:
            return {"image": result}

        # Image 2を領域サイズにリサイズ
        pip_image = cv2.resize(img2, (region_w, region_h), interpolation=cv2.INTER_LINEAR)

        # 領域に貼り付け
        result[y1:y2, x1:x2] = pip_image

        return {"image": result}
