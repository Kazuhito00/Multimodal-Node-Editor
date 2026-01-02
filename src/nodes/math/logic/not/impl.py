"""NOT論理演算ノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic


class NotLogic(ComputeLogic):
    """
    NOT論理演算ノード。
    入力が0の場合に1を、0以外の場合に0を出力する。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        a = inputs.get("a") or 0

        # 0をTrue（反転して1）、0以外をFalse（反転して0）として評価
        result = 1 if a == 0 else 0
        return {"result": result}
