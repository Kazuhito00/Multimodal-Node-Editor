"""3D Rotateノードの実装"""
from math import pi
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


def deg_to_rad(deg: float) -> float:
    """度をラジアンに変換"""
    return deg * pi / 180.0


def get_rad(pitch: float, yaw: float, roll: float) -> tuple:
    """角度をラジアンに変換"""
    return (deg_to_rad(pitch), deg_to_rad(yaw), deg_to_rad(roll))


def get_projection_matrix(image, focal, pitch, yaw, roll, dx, dy, dz):
    """3次元射影変換行列を生成"""
    h, w = image.shape[:2]

    # 原点を画像中心に移動
    A1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1], [0, 0, 1]])

    # 回転行列（X軸・Y軸・Z軸）
    RX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch), 0],
            [0, np.sin(pitch), np.cos(pitch), 0],
            [0, 0, 0, 1],
        ]
    )

    RY = np.array(
        [
            [np.cos(yaw), 0, -np.sin(yaw), 0],
            [0, 1, 0, 0],
            [np.sin(yaw), 0, np.cos(yaw), 0],
            [0, 0, 0, 1],
        ]
    )

    RZ = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0, 0],
            [np.sin(roll), np.cos(roll), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # 回転行列の合成
    R = RX @ RY @ RZ

    # 平行移動行列
    T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])

    # 射影行列（3D→2D）
    A2 = np.array([[focal, 0, w / 2, 0], [0, focal, h / 2, 0], [0, 0, 1, 0]])

    # 全体の変換行列
    return A2 @ (T @ (R @ A1))


def rotate_3d(
    image,
    theta=0,
    phi=0,
    gamma=0,
    color=(128, 128, 128),
):
    """3D空間で画像を回転させる"""
    image_height, image_width = image.shape[:2]

    dx, dy, dz = 0, 0, 0
    pitch, yaw, roll = get_rad(theta, phi, gamma)
    d = np.sqrt(image_height**2 + image_width**2)
    focal = d / (2 * np.sin(roll) if np.sin(roll) != 0 else 1)
    dz_ = focal + dz

    # 射影変換行列を取得
    mat = get_projection_matrix(image, focal, pitch, yaw, roll, dx, dy, dz_)

    # 元画像の4隅の座標を取得
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(
        -1, 1, 2
    )

    # 射影変換での変換後の座標を計算
    transformed_corners = cv2.perspectiveTransform(corners, mat)

    # 新しい画像サイズ（見切れない範囲）
    x_coords = transformed_corners[:, 0, 0]
    y_coords = transformed_corners[:, 0, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))

    # オフセット行列（画像が切れないように中央に移動）
    offset_mat = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    final_mat = offset_mat @ mat

    # warpPerspectiveで画像変換（見切れ防止＆背景色反映）
    result = cv2.warpPerspective(
        image.copy(),
        final_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=color,
    )

    return result


class Rotate3dLogic(ComputeLogic):
    """
    3D回転ノード。
    Pitch、Yaw、Rollで3D空間における回転を適用する。
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

        # プロパティまたは入力ポートから値を取得
        pitch = inputs.get("pitch")
        if pitch is None:
            pitch = int(properties.get("pitch", 0))

        yaw = inputs.get("yaw")
        if yaw is None:
            yaw = int(properties.get("yaw", 0))

        roll = inputs.get("roll")
        if roll is None:
            roll = int(properties.get("roll", 0))

        border_color_hex = properties.get("border_color", "#808080")
        border_color = hex_to_bgr(border_color_hex)

        # すべて0の場合は何もしない
        if pitch == 0 and yaw == 0 and roll == 0:
            return {"image": image}

        try:
            result = rotate_3d(
                image,
                theta=pitch,
                phi=yaw,
                gamma=roll,
                color=border_color,
            )
            return {"image": result}

        except Exception as e:
            return {
                "image": None,
                "__error__": str(e),
            }
