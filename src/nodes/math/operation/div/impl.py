from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class DivLogic(ComputeLogic):
    """二つの数値を除算するノード（ゼロ除算時は0を返す）"""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if context is None:
            context = {}

        a = float(inputs.get("a") or 0)
        b = float(inputs.get("b") or 0)
        if b == 0:
            # プレビューモードではエラー非表示
            if context.get("preview", False):
                return {"a/b": 0.0}
            return {"a/b": 0.0, "__error__": "Division by zero"}
        return {"a/b": a / b}
