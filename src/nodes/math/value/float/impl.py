from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class FloatValueLogic(ComputeLogic):
    """浮動小数点値を出力するノード"""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        value = float(properties.get("value", 0.0))
        return {"value": value}
