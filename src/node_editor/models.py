import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

def generate_id(prefix: str) -> str:
    """ユニークなIDを生成します。"""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

class Port(BaseModel):
    """ノードの入出力ポートを表すモデル。"""
    id: str = Field(default_factory=lambda: generate_id("port"))
    name: str
    display_name: Optional[str] = None  # 表示名（省略時はnameを使用）
    data_type: Any
    direction: str = "in"  # "in", "out", "inout"
    preview: bool = True  # プレビュー表示（画像/ウェーブフォーム）するか

class Node(BaseModel):
    """グラフを構成するノードのインスタンスを表すモデル。"""
    id: str = Field(default_factory=lambda: generate_id("node"))
    definition_id: str
    definition_version: Optional[str] = None  # None時は最新バージョンを使用
    name: str
    inputs: List[Port] = Field(default_factory=list)
    outputs: List[Port] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    position: Dict[str, float] = {"x": 0.0, "y": 0.0}

class Connection(BaseModel):
    """ノード間の接続を表すモデル。"""
    id: str = Field(default_factory=lambda: generate_id("conn"))
    from_node_id: str
    from_port_id: str
    to_node_id: str
    to_port_id: str
