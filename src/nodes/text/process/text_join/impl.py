"""Text Joinノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic


class TextJoinLogic(ComputeLogic):
    """
    複数のテキストを結合するノードロジック。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        # 入力ポートから取得（接続されている場合）
        text1_input = inputs.get("text1")
        text2_input = inputs.get("text2")

        # プロパティから取得（入力がない場合のフォールバック）
        text1_prop = str(properties.get("text1", ""))
        text2_prop = str(properties.get("text2", ""))

        # 入力ポートが接続されていればその値を使用、なければプロパティ値
        text1 = text1_input if text1_input is not None else text1_prop
        text2 = text2_input if text2_input is not None else text2_prop

        # テキストを結合
        result = str(text1) + str(text2)

        return {"text": result}
