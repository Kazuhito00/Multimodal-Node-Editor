"""Clampノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic


class ClampLogic(ComputeLogic):
    """
    Clampノード。
    入力値を指定された範囲内に制限する。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        value = float(inputs.get("value") or 0)
        min_val = float(properties.get("min", 0.0))
        max_val = float(properties.get("max", 1.0))

        # min > max の場合はminをmaxに合わせる
        if min_val > max_val:
            min_val = max_val

        # 値をクランプ
        result = max(min_val, min(max_val, value))
        return {"result": result}
