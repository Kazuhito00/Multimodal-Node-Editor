"""Perspectiveノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic

import cv2
import numpy as np


class PerspectiveLogic(ComputeLogic):
    """
    4つのコーナーポイントを使用して透視変換を適用するノードロジック。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        img = inputs.get("image")
        if img is None:
            return {"output": None}

        height, width = img.shape[:2]

        # 正規化座標をピクセル座標に変換
        x1 = float(properties.get("x1", 0.0)) * width
        y1 = float(properties.get("y1", 0.0)) * height
        x2 = float(properties.get("x2", 1.0)) * width
        y2 = float(properties.get("y2", 0.0)) * height
        x3 = float(properties.get("x3", 1.0)) * width
        y3 = float(properties.get("y3", 1.0)) * height
        x4 = float(properties.get("x4", 0.0)) * width
        y4 = float(properties.get("y4", 1.0)) * height

        output_width = int(properties.get("output_width", 640))
        output_height = int(properties.get("output_height", 480))

        # 変換元の4点（左上、右上、右下、左下の順）
        src_points = np.float32([
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ])

        # 変換先の4点（出力画像の4隅）
        dst_points = np.float32([
            [0, 0],
            [output_width, 0],
            [output_width, output_height],
            [0, output_height]
        ])

        # 透視変換行列を計算
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # 透視変換を適用
        result = cv2.warpPerspective(img, matrix, (output_width, output_height))

        return {"output": result}
