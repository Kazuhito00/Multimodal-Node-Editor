from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class MulLogic(ComputeLogic):
    """二つの数値を乗算するノード"""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        a = float(inputs.get("a") or 0)
        b = float(inputs.get("b") or 0)
        return {"a*b": a * b}
