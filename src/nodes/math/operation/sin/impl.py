import math
from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class SinLogic(ComputeLogic):
    """度数法の角度からsinを計算するノード"""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        degree = float(properties.get("degree", 0.0))
        radian = math.radians(degree)
        result = math.sin(radian)
        return {"result": result}
