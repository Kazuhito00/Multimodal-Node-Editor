from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class TextDisplayLogic(ComputeLogic):
    """テキスト表示ノード"""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        text = inputs.get("text_in", "")
        # text_in: ヘッドレスモードでのテキスト出力用
        # __display_text__: フロントエンドでの表示用（後方互換性）
        return {"text_in": text, "__display_text__": text}
