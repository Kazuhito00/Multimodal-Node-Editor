"""Text Replaceノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic


class TextReplaceLogic(ComputeLogic):
    """
    テキスト内の文字列を置換するノードロジック。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        text = inputs.get("text", "")
        if text is None:
            text = ""

        before = str(properties.get("before", ""))
        after = str(properties.get("after", ""))

        # 置換前テキストが空の場合は元のテキストを返す
        if not before:
            return {"text": text}

        # テキストを置換
        result = text.replace(before, after)

        return {"text": result}
