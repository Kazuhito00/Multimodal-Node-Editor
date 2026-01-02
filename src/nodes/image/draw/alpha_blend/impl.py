"""Alpha Blendノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic

import cv2


class AlphaBlendLogic(ComputeLogic):
    """
    2つの画像をアルファ、ベータ、ガンマで重み付け合成するノードロジック。
    出力 = alpha * image1 + beta * image2 + gamma
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

        # プロパティまたは入力ポートから値を取得
        alpha = inputs.get("alpha")
        if alpha is None:
            alpha = float(properties.get("alpha", 0.5))

        beta = inputs.get("beta")
        if beta is None:
            beta = float(properties.get("beta", 0.5))

        gamma = inputs.get("gamma")
        if gamma is None:
            gamma = float(properties.get("gamma", 0.0))

        # 画像サイズを揃える（img1のサイズに合わせる）
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # チャンネル数を揃える
        if len(img1.shape) != len(img2.shape):
            if len(img1.shape) == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if len(img2.shape) == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # アルファブレンド: dst = alpha * img1 + beta * img2 + gamma
        result = cv2.addWeighted(img1, alpha, img2, beta, gamma)

        return {"image": result}
