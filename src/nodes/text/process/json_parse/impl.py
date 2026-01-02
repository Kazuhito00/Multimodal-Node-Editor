"""JSON Parseノードの実装"""
import json
import re
from typing import Dict, Any

from node_editor.node_def import ComputeLogic


def parse_key_path(key: str) -> list:
    """
    キーパスをパーツに分解する。
    例: "user.name" -> ["user", "name"]
    例: "items[0]" -> ["items", 0]
    例: "data.items[0].name" -> ["data", "items", 0, "name"]
    """
    if not key:
        return []

    parts = []
    # ドットで分割し、各パートを処理
    for part in key.split('.'):
        # 配列インデックスを検出: name[0] -> name, 0
        match = re.match(r'^([^\[]+)(?:\[(\d+)\])?$', part)
        if match:
            name = match.group(1)
            if name:
                parts.append(name)
            if match.group(2) is not None:
                parts.append(int(match.group(2)))
        else:
            parts.append(part)

    return parts


def get_value_by_path(data: Any, path: list) -> Any:
    """
    パスに従ってデータから値を取得する。
    """
    current = data
    for part in path:
        if isinstance(part, int):
            # 配列インデックス
            if isinstance(current, list) and 0 <= part < len(current):
                current = current[part]
            else:
                return None
        elif isinstance(current, dict):
            # 辞書キー
            if part in current:
                current = current[part]
            else:
                return None
        else:
            return None
    return current


class JSONParseLogic(ComputeLogic):
    """
    JSON文字列を解析し、指定したキー/パスの値を抽出するノードロジック。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        json_str = inputs.get("json", "")
        key = str(properties.get("key", ""))

        if not json_str:
            return {"value": ""}

        # JSONをパース
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {"value": ""}

        # キーが空の場合は全体を返す
        if not key:
            return {"value": json.dumps(data, ensure_ascii=False)}

        # パスを解析して値を取得
        path = parse_key_path(key)
        value = get_value_by_path(data, path)

        if value is None:
            return {"value": ""}

        # 値を文字列に変換
        if isinstance(value, (dict, list)):
            result = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            result = "true" if value else "false"
        else:
            result = str(value)

        return {"value": result}
