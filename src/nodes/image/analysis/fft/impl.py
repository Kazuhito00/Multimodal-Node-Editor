"""
FFT ノードの実装。
入力画像に対してFFT（高速フーリエ変換）を行い、振幅スペクトルを可視化する。
"""
from typing import Dict, Any

import numpy as np
import cv2

from node_editor.node_def import ComputeLogic


def compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """
    画像のFFT振幅スペクトルを計算

    Args:
        image: グレースケール画像

    Returns:
        対数スケールで正規化された振幅スペクトル画像
    """
    # FFT計算
    f = np.fft.fft2(image.astype(np.float32))

    # 低周波成分を中心にシフト
    fshift = np.fft.fftshift(f)

    # 振幅スペクトル（対数スケール）
    magnitude = np.abs(fshift)
    magnitude = np.log1p(magnitude)

    # 0-255に正規化
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)

    return magnitude


class FftLogic(ComputeLogic):
    """
    FFTノードのロジック。
    入力画像からFFT振幅スペクトルを計算し、可視化画像を出力。
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
        image = inputs.get("image")
        if image is None:
            return {"image": None}

        # キャンセルチェック
        if self.is_cancelled():
            self.clear_cancel()
            return {"image": None}

        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # FFT振幅スペクトルを計算
        magnitude = compute_fft_magnitude(gray)

        # BGRに変換して出力（グレースケール表示）
        result = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

        return {"image": result}
