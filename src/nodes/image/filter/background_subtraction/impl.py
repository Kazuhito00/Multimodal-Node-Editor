from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2


class BackgroundSubtractionNodeLogic(ComputeLogic):
    """
    背景差分で前景を検出するノードロジック。
    """

    def _create_subtractor(self, algorithm: int):
        """アルゴリズムに応じたBackgroundSubtractorを作成"""
        if algorithm == 0:
            return cv2.createBackgroundSubtractorMOG2(
                history=1, varThreshold=16, detectShadows=False
            )
        else:
            return cv2.createBackgroundSubtractorKNN(
                history=1, dist2Threshold=400, detectShadows=False
            )

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        img = inputs.get("image")
        background = inputs.get("background")

        if img is None:
            return {"image": None}

        # 背景がない場合は元画像をそのまま返す
        if background is None:
            return {"image": img}

        algorithm = int(properties.get("algorithm", 0))

        # サイズを統一
        bg_resized = background
        if background.shape[:2] != img.shape[:2]:
            bg_resized = cv2.resize(background, (img.shape[1], img.shape[0]))

        # 毎回新しいサブトラクタを作成（状態の蓄積を防ぐ）
        subtractor = self._create_subtractor(algorithm)

        # 背景画像でモデルを初期化
        subtractor.apply(bg_resized, learningRate=1.0)

        # 前景マスクを取得
        fg_mask = subtractor.apply(img, learningRate=0)

        # マスクを3チャンネルに変換
        result = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        return {"image": result}
