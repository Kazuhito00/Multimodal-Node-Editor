"""
JSON Array Formatノードの実装。
JSON配列から指定フィールドを抽出してテキストにフォーマットする。
"""
import json
import re
from typing import Dict, Any, List

from node_editor.node_def import ComputeLogic


def get_nested_value(obj: Any, key: str) -> Any:
    """
    ネストしたキーから値を取得する。
    例: "box[0][0]" -> obj["box"][0][0]
    """
    if not key:
        return obj

    # ドットとブラケットでパスを分解
    parts = []
    current = ""
    i = 0
    while i < len(key):
        if key[i] == '.':
            if current:
                parts.append(current)
                current = ""
        elif key[i] == '[':
            if current:
                parts.append(current)
                current = ""
            # 閉じブラケットまで読む
            j = i + 1
            while j < len(key) and key[j] != ']':
                j += 1
            index_str = key[i+1:j]
            if index_str.isdigit():
                parts.append(int(index_str))
            else:
                parts.append(index_str)
            i = j
        else:
            current += key[i]
        i += 1

    if current:
        parts.append(current)

    # パスを辿って値を取得
    value = obj
    for part in parts:
        if isinstance(part, int):
            if isinstance(value, list) and 0 <= part < len(value):
                value = value[part]
            else:
                return None
        elif isinstance(value, dict):
            if part in value:
                value = value[part]
            else:
                return None
        else:
            return None

    return value


def format_item(item: Dict[str, Any], format_str: str) -> str:
    """
    フォーマット文字列に従ってアイテムを整形する。
    {key} をアイテムの対応する値で置換する。
    """
    result = format_str

    # {key} パターンを検索
    pattern = r'\{([^}]+)\}'
    matches = re.findall(pattern, format_str)

    for key in matches:
        value = get_nested_value(item, key)
        if value is not None:
            # 値を文字列に変換
            if isinstance(value, (dict, list)):
                str_value = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, float):
                str_value = f"{value:.4f}"
            else:
                str_value = str(value)
            result = result.replace(f"{{{key}}}", str_value)
        else:
            result = result.replace(f"{{{key}}}", "")

    return result


class JSONArrayFormatLogic(ComputeLogic):
    """
    JSON配列からフィールドを抽出してフォーマットするノードロジック。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        json_str = inputs.get("json", "")
        format_str = str(properties.get("format", "{id}: {text}"))
        separator = str(properties.get("separator", "\\n"))
        array_key = str(properties.get("array_key", ""))

        # エスケープシーケンスを処理
        separator = separator.replace("\\n", "\n").replace("\\t", "\t")

        if not json_str:
            return {"text": ""}

        # JSONをパース
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {"text": ""}

        # array_keyが指定されている場合、そのキーの配列を取得
        if array_key:
            data = get_nested_value(data, array_key)

        # 配列でない場合は空を返す
        if not isinstance(data, list):
            return {"text": ""}

        # 各アイテムをフォーマット
        formatted_items = []
        for item in data:
            if isinstance(item, dict):
                formatted = format_item(item, format_str)
                formatted_items.append(formatted)

        # 区切り文字で結合
        result = separator.join(formatted_items)

        return {"text": result}
