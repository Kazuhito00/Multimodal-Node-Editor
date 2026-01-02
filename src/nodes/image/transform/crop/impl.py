from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class CropNodeLogic(ComputeLogic):
    """
    画像をクロップするノードロジック。
    min_x, min_y, max_x, max_yは0.0～1.0の正規化座標。
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

        # 正規化座標を取得（0.0～1.0）
        min_x = float(properties.get("min_x", 0.0))
        min_y = float(properties.get("min_y", 0.0))
        max_x = float(properties.get("max_x", 1.0))
        max_y = float(properties.get("max_y", 1.0))

        # 値を0.0～1.0にクランプ
        min_x = max(0.0, min(1.0, min_x))
        min_y = max(0.0, min(1.0, min_y))
        max_x = max(0.0, min(1.0, max_x))
        max_y = max(0.0, min(1.0, max_y))

        # min > maxの場合は入れ替え
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_y > max_y:
            min_y, max_y = max_y, min_y

        # 画像サイズを取得
        height, width = img.shape[:2]

        # ピクセル座標に変換
        x1 = int(min_x * width)
        y1 = int(min_y * height)
        x2 = int(max_x * width)
        y2 = int(max_y * height)

        # 範囲が0になる場合は元画像を返す
        if x2 <= x1 or y2 <= y1:
            return {"output": img}

        # クロップ実行
        result = img[y1:y2, x1:x2].copy()

        return {"output": result}
