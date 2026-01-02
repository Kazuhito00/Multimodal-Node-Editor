from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2
import numpy as np


def create_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ロール・ピッチ・ヨーから回転行列を生成"""
    roll = roll * np.pi / 180
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / 180

    # ロール（X軸回転）
    matrix_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)],
    ])

    # ピッチ（Y軸回転）
    matrix_pitch = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)],
    ])

    # ヨー（Z軸回転）
    matrix_yaw = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    matrix = np.dot(matrix_yaw, np.dot(matrix_pitch, matrix_roll))
    return matrix


def calculate_phi_and_theta(
    viewpoint: float,
    imagepoint: float,
    sensor_width: float,
    sensor_height: float,
    output_width: int,
    output_height: int,
    rotation_matrix: np.ndarray,
) -> tuple:
    """角度座標φ, θを算出"""
    width = np.arange(
        (-1) * sensor_width,
        sensor_width,
        sensor_width * 2 / output_width,
    )
    height = np.arange(
        (-1) * sensor_height,
        sensor_height,
        sensor_height * 2 / output_height,
    )

    ww, hh = np.meshgrid(width, height)

    point_distance = imagepoint - viewpoint
    if point_distance == 0:
        point_distance = 0.1

    a1 = ww / point_distance
    a2 = hh / point_distance
    b1 = -a1 * viewpoint
    b2 = -a2 * viewpoint

    a = 1 + (a1**2) + (a2**2)
    b = 2 * ((a1 * b1) + (a2 * b2))
    c = (b1**2) + (b2**2) - 1

    d = ((b**2) - (4 * a * c))**(1 / 2)

    x = (-b + d) / (2 * a)
    y = (a1 * x) + b1
    z = (a2 * x) + b2

    xd = rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2] * z
    yd = rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2] * z
    zd = rotation_matrix[2][0] * x + rotation_matrix[2][1] * y + rotation_matrix[2][2] * z

    phi = np.arcsin(zd)
    theta = np.arcsin(yd / np.cos(phi))

    xd[xd > 0] = 0
    xd[xd < 0] = 1
    yd[yd > 0] = np.pi
    yd[yd < 0] = -np.pi

    offset = yd * xd
    gain = -2 * xd + 1
    theta = gain * theta + offset

    return phi, theta


def create_remap_maps(
    phi: np.ndarray,
    theta: np.ndarray,
    input_width: int,
    input_height: int,
) -> tuple:
    """φ, θから固定小数点リマップ用マップを生成（高速化用）"""
    phi_map = (phi * input_height / np.pi + input_height / 2).astype(np.float32)
    theta_map = (theta * input_width / (2 * np.pi) + input_width / 2).astype(np.float32)

    # 固定小数点マップに変換（remap高速化）
    map1, map2 = cv2.convertMaps(theta_map, phi_map, cv2.CV_16SC2, nninterpolation=False)
    return map1, map2


def remap_image_fast(image: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    """固定小数点マップを使用した高速リマップ"""
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


class OmnidirectionalViewerNodeLogic(ComputeLogic):
    """
    正距円筒図法の360度画像をロール・ピッチ・ヨーで変換するノードロジック。
    """

    # センササイズ設定
    SENSOR_SIZE = 0.561

    def __init__(self):
        # キャッシュ用（インスタンス変数として保持）
        self._cache_key: str = ""
        self._cached_map1: np.ndarray = None
        self._cached_map2: np.ndarray = None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        img = inputs.get("image")
        if img is None:
            return {"image": None}

        pitch = int(properties.get("pitch", 0))
        yaw = int(properties.get("yaw", 0))
        roll = int(properties.get("roll", 0))
        image_point = float(properties.get("image_point", 0.0))
        output_width = int(properties.get("width", 640))
        output_height = int(properties.get("height", 360))

        input_height, input_width = img.shape[:2]

        # キャッシュキー（出力サイズ・入力サイズも含める）
        cache_key = f"{pitch}_{yaw}_{roll}_{image_point}_{output_width}_{output_height}_{input_width}_{input_height}"

        # パラメータまたは入力サイズが変わった場合のみマップを再計算
        if cache_key != self._cache_key:
            sensor_width = self.SENSOR_SIZE
            sensor_height = self.SENSOR_SIZE * (output_height / output_width)

            # 回転行列生成
            rotation_matrix = create_rotation_matrix(roll, pitch, yaw)

            # 角度座標φ, θ算出
            phi, theta = calculate_phi_and_theta(
                -1.0,
                image_point,
                sensor_width,
                sensor_height,
                output_width,
                output_height,
                rotation_matrix,
            )

            # 固定小数点マップを生成してキャッシュ
            self._cached_map1, self._cached_map2 = create_remap_maps(
                phi, theta, input_width, input_height
            )
            self._cache_key = cache_key

        # 高速リマップ実行
        result = remap_image_fast(img, self._cached_map1, self._cached_map2)

        return {"image": result}
