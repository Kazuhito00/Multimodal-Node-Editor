"""OR論理演算ノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic


class OrLogic(ComputeLogic):
    """
    OR論理演算ノード。
    いずれかの入力が0以外の場合に1を出力する。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        a = inputs.get("a") or 0
        b = inputs.get("b") or 0

        # 0以外をTrue、0をFalseとして評価
        result = 1 if (a != 0 or b != 0) else 0
        return {"result": result}
