from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class IntValueLogic(ComputeLogic):
    """整数値をfloatとして出力するノード"""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        value = float(int(properties.get("value", 0)))
        return {"value": value}
