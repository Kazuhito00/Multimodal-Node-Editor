import math
from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class Float2IntLogic(ComputeLogic):
    """浮動小数点を整数に変換するノード（丸めモード選択可能）"""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        value = float(inputs.get("in") or 0.0)
        mode = int(properties.get("mode", 0))

        # 0: Round（四捨五入）, 1: Ceil（切り上げ）, 2: Floor（切り捨て）
        if mode == 0:
            result = round(value)
        elif mode == 1:
            result = math.ceil(value)
        else:
            result = math.floor(value)

        return {"out": result}
