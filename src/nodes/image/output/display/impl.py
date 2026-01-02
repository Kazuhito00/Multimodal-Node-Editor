from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class DisplayNodeLogic(ComputeLogic):
    """
    画像を表示するためのノードロジック。
    入力画像をそのまま出力として渡す（フロントエンドで表示用）。
    入力・出力ともにOpenCV画像（numpy配列）。Base64変換はcore.pyで自動処理。
    """
    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        # 入力のみ（プレビュー表示用、出力なし）
        return {}
