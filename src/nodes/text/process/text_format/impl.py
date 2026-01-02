"""Text Formatノードの実装"""
from typing import Dict, Any

from node_editor.node_def import ComputeLogic


class TextFormatLogic(ComputeLogic):
    """
    テンプレート内のプレースホルダー {1}, {2}, ... を置換するノードロジック。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        format_template = str(properties.get("format", ""))

        result = format_template

        # {1} から {10} までの置換を処理
        for i in range(1, 11):
            port_name = f"Text {i}"
            placeholder = f"{{{i}}}"

            # 入力ポートから値を取得
            value = inputs.get(port_name)
            if value is not None:
                result = result.replace(placeholder, str(value))

        return {"format": result}
