from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class TriggerButtonLogic(ComputeLogic):
    """ボタン押下時に1を出力し、それ以外は0を出力するトリガーノード"""

    def __init__(self):
        self._last_button_value: bool = False

    def reset(self):
        """状態をリセット"""
        self._last_button_value = False

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        button_pressed = properties.get("button", False)

        # ボタンの立ち上がりエッジを検出（押された瞬間のみ1を出力）
        if button_pressed and not self._last_button_value:
            self._last_button_value = button_pressed
            return {"out": 1.0}

        self._last_button_value = button_pressed
        return {"out": 0.0}
