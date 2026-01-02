"""XOR論理演算ノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic


class XorLogic(ComputeLogic):
    """
    XOR論理演算ノード。
    どちらか一方の入力のみが0以外の場合に1を出力する。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        a = inputs.get("a") or 0
        b = inputs.get("b") or 0

        # 0以外をTrue、0をFalseとして評価し、XOR演算
        bool_a = a != 0
        bool_b = b != 0
        result = 1 if (bool_a != bool_b) else 0
        return {"result": result}
