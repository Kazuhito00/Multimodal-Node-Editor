from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class TextInputLogic(ComputeLogic):
    """テキスト入力ノード"""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        text = properties.get("text", "")
        return {"text_out": text}
