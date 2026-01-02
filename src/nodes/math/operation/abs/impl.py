from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class AbsLogic(ComputeLogic):
    """絶対値を計算するノード"""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        value = float(inputs.get("in") or 0)
        return {"out": abs(value)}
