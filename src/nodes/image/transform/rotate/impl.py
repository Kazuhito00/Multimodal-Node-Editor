"""Rotateノードの実装"""
from typing import Any, Dict

from node_editor.node_def import ComputeLogic

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

import numpy as np


def hex_to_bgr(hex_color: str) -> tuple:
    """HEXカラーをBGRタプルに変換"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)
    return (128, 128, 128)


class RotateLogic(ComputeLogic):
    """
    画像回転ノード。
    指定された角度で画像を回転させる。
    90度の倍数でない場合は余白をborder_colorで塗りつぶす。
    """

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

        angle = int(properties.get("angle", 0))
        border_color_hex = properties.get("border_color", "#808080")
        border_color = hex_to_bgr(border_color_hex)

        # 0度の場合は何もしない
        if angle == 0:
            return {"image": image}

        # 角度を-360〜360の範囲に正規化
        angle = angle % 360
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360

        try:
            height, width = image.shape[:2]

            # 90度の倍数の場合は高速なrotate関数を使用（時計回り）
            if angle == 90 or angle == -270:
                rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == -90 or angle == 270:
                rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180 or angle == -180:
                rotated = cv2.rotate(image, cv2.ROTATE_180)
            else:
                # 任意角度の回転（時計回りにするため角度を反転）
                center = (width / 2, height / 2)

                # 回転後に画像全体が収まるサイズを計算
                angle_rad = np.radians(abs(angle))
                cos_val = abs(np.cos(angle_rad))
                sin_val = abs(np.sin(angle_rad))
                new_width = int(width * cos_val + height * sin_val)
                new_height = int(width * sin_val + height * cos_val)

                # 回転行列を作成（中心を調整、時計回りにするため角度を反転）
                rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
                rotation_matrix[0, 2] += (new_width - width) / 2
                rotation_matrix[1, 2] += (new_height - height) / 2

                # 回転を適用
                rotated = cv2.warpAffine(
                    image,
                    rotation_matrix,
                    (new_width, new_height),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=border_color
                )

            return {"image": rotated}

        except Exception as e:
            return {
                "image": None,
                "__error__": str(e),
            }
