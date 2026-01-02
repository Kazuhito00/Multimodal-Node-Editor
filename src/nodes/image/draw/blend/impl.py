"""Blendノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic

import cv2
import numpy as np


def blend_add(img1, img2):
    """加算合成"""
    return cv2.add(img1, img2)


def blend_subtract(img1, img2):
    """減算合成"""
    return cv2.subtract(img1, img2)


def blend_multiply(img1, img2):
    """乗算合成"""
    img1_float = img1.astype(np.float32) / 255.0
    img2_float = img2.astype(np.float32) / 255.0
    result = img1_float * img2_float
    return (result * 255).astype(np.uint8)


def blend_screen(img1, img2):
    """スクリーン合成"""
    img1_float = img1.astype(np.float32) / 255.0
    img2_float = img2.astype(np.float32) / 255.0
    result = 1.0 - (1.0 - img1_float) * (1.0 - img2_float)
    return (result * 255).astype(np.uint8)


def blend_overlay(img1, img2):
    """オーバーレイ合成"""
    img1_float = img1.astype(np.float32) / 255.0
    img2_float = img2.astype(np.float32) / 255.0

    # オーバーレイ: 暗い部分は乗算、明るい部分はスクリーン
    mask = img1_float < 0.5
    result = np.where(
        mask,
        2 * img1_float * img2_float,
        1.0 - 2 * (1.0 - img1_float) * (1.0 - img2_float)
    )
    return (result * 255).astype(np.uint8)


def blend_darken(img1, img2):
    """比較（暗）"""
    return np.minimum(img1, img2)


def blend_lighten(img1, img2):
    """比較（明）"""
    return np.maximum(img1, img2)


def blend_difference(img1, img2):
    """差の絶対値"""
    return cv2.absdiff(img1, img2)


class BlendLogic(ComputeLogic):
    """
    2つの画像を様々なモードで合成するノードロジック。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        img1 = inputs.get("image1")
        img2 = inputs.get("image2")

        if img1 is None or img2 is None:
            return {"image": None}

        mode = int(properties.get("mode", 0))

        # 画像サイズを揃える（img1のサイズに合わせる）
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # チャンネル数を揃える
        if len(img1.shape) != len(img2.shape):
            if len(img1.shape) == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if len(img2.shape) == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # ブレンドモードに応じて合成
        blend_functions = {
            0: blend_add,
            1: blend_subtract,
            2: blend_multiply,
            3: blend_screen,
            4: blend_overlay,
            5: blend_darken,
            6: blend_lighten,
            7: blend_difference,
        }

        blend_func = blend_functions.get(mode, blend_add)
        result = blend_func(img1, img2)

        return {"image": result}
