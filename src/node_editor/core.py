import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from pydantic import BaseModel, Field

from . import node_def
from .commands import AddConnectionCommand, CommandHistory, AddNodeCommand, RemoveNodeCommand
from .models import Node, Connection, generate_id
from .image_utils import ensure_base64


class Graph(BaseModel):
    """ノードエディタ全体のグラフ構造を表すモデル。"""
    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: generate_id("graph"))
    nodes: List[Node] = Field(default_factory=list)
    connections: List[Connection] = Field(default_factory=list)
    history: CommandHistory = Field(default_factory=CommandHistory, exclude=True)

    # 07_versioning_strategy.md に基づくバージョン情報
    graph_format_version: str = "0.0.1"

    def add_node(self, node: Node):
        """ノードをグラフに追加するためのコマンドを実行します。"""
        command = AddNodeCommand(self, node)
        self.history.execute(command)

    def remove_node(self, node: Node):
        """ノードをグラフから削除するためのコマンドを実行します。"""
        command = RemoveNodeCommand(self, node)
        self.history.execute(command)

    def add_connection(self, connection: Connection):
        """接続をグラフに追加するためのコマンドを実行します。"""
        command = AddConnectionCommand(self, connection)
        self.history.execute(command)

    def save(self, path: Union[str, Path]):
        """グラフをJSONファイルに保存します。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Graph":
        """JSONファイルからグラフを読み込み、バージョンチェックを行います。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = cls(**data)
        graph.validate_versions()
        return graph

    def validate_versions(self):
        """グラフとノードのバージョンを検証し、警告を出力します。"""
        # 1. Graph format version check
        app_version = Graph().graph_format_version
        file_version = self.graph_format_version
        if app_version.split('.')[0] != file_version.split('.')[0]:
            print(
                f"Warning: Graph format version mismatch. "
                f"App supports v{app_version}, but file is v{file_version}."
            )

        # 2. Node definition version check
        for node in self.nodes:
            try:
                definition = node_def.get_node_definition(node.definition_id)
                if node.definition_version != definition.version:
                    print(
                        f"Warning: Node '{node.name}' ({node.definition_id}) version mismatch. "
                        f"Graph uses v{node.definition_version}, but app has v{definition.version} registered."
                    )
            except ValueError as e:
                print(
                    f"Warning: Node '{node.name}' uses definition '{node.definition_id}', but an error occurred: {e}"
                )

    def execute(
        self, context: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any], Dict[str, str], Dict[str, Dict[str, Any]], float]:
        """グラフを実行し、出力結果、実行時間(ms)、各ノードの実行時間、エラー情報、GUI処理時間を返します。"""
        if context is None:
            context = {}
        start_time = time.perf_counter()
        node_times: Dict[str, Any] = {}
        node_errors: Dict[str, str] = {}  # ノードIDをキーにエラーメッセージを保存
        ended = False  # いずれかのノードが終了シグナルを出したか

        # 1. 依存関係グラフの構築
        adj: Dict[str, List[str]] = {node.id: [] for node in self.nodes}
        in_degree: Dict[str, int] = {node.id: 0 for node in self.nodes}
        node_map: Dict[str, Node] = {node.id: node for node in self.nodes}
        node_order: Dict[str, int] = {node.id: i for i, node in enumerate(self.nodes)}

        # 接続に関与しているノードIDを収集
        connected_node_ids: set = set()
        for conn in self.connections:
            adj[conn.from_node_id].append(conn.to_node_id)
            in_degree[conn.to_node_id] += 1
            connected_node_ids.add(conn.from_node_id)
            connected_node_ids.add(conn.to_node_id)

        # 2. トポロジカルソート (Kahn's algorithm)
        # 接続されているノードのみを先に処理（UI追加順でソート）
        initial_nodes = [
            node_id for node_id in node_order.keys()
            if in_degree[node_id] == 0 and node_id in connected_node_ids
        ]
        initial_nodes.sort(key=lambda x: node_order[x])
        queue = deque(initial_nodes)

        sorted_nodes: List[Node] = []
        while queue:
            node_id = queue.popleft()
            sorted_nodes.append(node_map[node_id])
            # 隣接ノードをUI追加順でソートして追加
            neighbors_ready = []
            for neighbor_id in adj[node_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    neighbors_ready.append(neighbor_id)
            neighbors_ready.sort(key=lambda x: node_order[x])
            queue.extend(neighbors_ready)

        # 未接続のノードを最後に追加（UI追加順を維持）
        for node in self.nodes:
            if node.id not in connected_node_ids:
                sorted_nodes.append(node)

        if len(sorted_nodes) != len(self.nodes):
            raise RuntimeError("グラフに循環があり、実行できません。")

        # 3. グラフの実行
        port_values: Dict[str, Any] = {}  # Key: port_id, Value: ポートの出力値
        node_metadata: Dict[str, Any] = {}  # Key: node_id.__key__, Value: メタデータ
        connected_properties: Dict[str, Dict[str, Any]] = {}  # 接続された入力値（プロパティ上書き用）

        is_preview = context.get("preview", False)

        for node in sorted_nodes:
            # ノードの処理時間計測開始（入力収集から）
            node_start = time.perf_counter()

            # ノード定義を取得（計測対象判定に必要）
            try:
                definition = node_def.get_node_definition(node.definition_id, node.definition_version)
            except Exception as e:
                print(f"Failed to get definition for {node.definition_id}: {e}")
                definition = None

            # STOP時にrun_when_stopped=falseのノードはキャッシュを使用
            # キャッシュはノードIDごとに保存されている
            if is_preview and definition and not definition.run_when_stopped:
                node_cache = definition.cached_outputs.get(node.id, {})
                if node_cache:
                    for out_port in node.outputs:
                        if out_port.name in node_cache:
                            port_values[out_port.id] = node_cache[out_port.name]
                continue

            try:
                # 現在のノードの入力を収集
                node_inputs: Dict[str, Any] = {}
                for in_port in node.inputs:
                    # この入力ポートへの接続を検索
                    for conn in self.connections:
                        if conn.to_port_id == in_port.id:
                            from_value = port_values.get(conn.from_port_id)
                            # int/float間の型変換
                            if from_value is not None:
                                if in_port.data_type == 'int' and isinstance(from_value, float):
                                    from_value = int(from_value)
                                elif in_port.data_type == 'float' and isinstance(from_value, int):
                                    from_value = float(from_value)
                            node_inputs[in_port.name] = from_value
                            break

                # プロパティのコピーを作成し、接続された入力値で上書き
                node_properties = dict(node.properties)
                node_connected = {}  # このノードで接続された入力値
                # プロパティ定義をマップ化
                prop_defs_map = {}
                if definition:
                    for prop_def in definition.properties:
                        prop_defs_map[prop_def.name] = prop_def
                for in_port in node.inputs:
                    input_value = node_inputs.get(in_port.name)
                    if input_value is not None and in_port.name in node_properties:
                        prop_def = prop_defs_map.get(in_port.name)
                        if prop_def:
                            # int型プロパティの場合は切り捨て
                            if prop_def.type == 'int':
                                input_value = int(input_value)
                            # min/max制限を適用
                            if prop_def.min is not None and input_value < prop_def.min:
                                input_value = prop_def.min
                            if prop_def.max is not None and input_value > prop_def.max:
                                input_value = prop_def.max
                        node_properties[in_port.name] = input_value
                        node_connected[in_port.name] = input_value
                if node_connected:
                    connected_properties[node.id] = node_connected

                # ノードのロジックを実行
                if definition is None:
                    raise ValueError(f"Node definition not found: {node.definition_id}")

                # ノードIDをコンテキストに追加
                node_context = {**context, "node_id": node.id}
                outputs = definition.compute(node_inputs, node_properties, node_context)

                # 終了シグナルをチェック
                if outputs.get("__ended__") is True:
                    ended = True

                # エラーメッセージをチェック（例外を投げずにエラーを報告）
                if "__error__" in outputs:
                    error_msg = outputs["__error__"]
                    print(f"[{node.name}] Error: {error_msg}")
                    node_errors[node.id] = error_msg
                elif node.id in node_errors:
                    # エラーがなければクリア
                    del node_errors[node.id]

                # メタデータをチェック（フレーム数など）
                if "__frame_count__" in outputs:
                    node_metadata[f"{node.id}.__frame_count__"] = outputs["__frame_count__"]

                # ダウンロード情報をチェック
                if "__download__" in outputs:
                    node_metadata[f"{node.id}.__download__"] = outputs["__download__"]

                # テキスト表示をチェック
                if "__display_text__" in outputs:
                    node_metadata[f"{node.id}.__display_text__"] = outputs["__display_text__"]

                # ビジー状態をチェック
                if "__is_busy__" in outputs:
                    node_metadata[f"{node.id}.__is_busy__"] = outputs["__is_busy__"]

                # タイマー情報をチェック
                if "__timer_interval__" in outputs:
                    node_metadata[f"{node.id}.__timer_interval__"] = outputs["__timer_interval__"]
                    node_metadata[f"{node.id}.__timer_remaining__"] = outputs.get("__timer_remaining__", 0.0)
                    node_metadata[f"{node.id}.__timer_unit__"] = outputs.get("__timer_unit__", "sec")

                # プロパティ更新をチェック（AIコード生成など）
                if "__update_property__" in outputs:
                    node_metadata[f"{node.id}.__update_property__"] = outputs["__update_property__"]

                # 出力値を保存（内部ではnumpy配列のまま保持）
                for out_port in node.outputs:
                    if out_port.name in outputs:
                        port_values[out_port.id] = outputs[out_port.name]

                # run_when_stopped=falseのノードは出力をキャッシュ（ノードIDごと）
                if definition and not definition.run_when_stopped:
                    definition.cached_outputs[node.id] = {
                        out_port.name: outputs.get(out_port.name)
                        for out_port in node.outputs
                        if out_port.name in outputs
                    }

            except Exception as e:
                # エラーメッセージを保存（エラー発生ノードは処理続行）
                error_msg = str(e)
                node_errors[node.id] = error_msg
                print(f"[ERROR] Node '{node.name}' ({node.id}): {error_msg}")

            # ノードIDをキーに、名前、時間、実行順序を保存（measure_time=Trueのノードのみ）
            should_measure = definition.measure_time if definition else True
            if should_measure:
                node_times[node.id] = {
                    "name": node.name,
                    "time": (time.perf_counter() - node_start) * 1000,
                    "order": len(node_times),
                }

        # 全ノードの出力を返す（キーはノードID.ポート名）
        # ブラウザ表示用にencode_base64=True（デフォルト）で画像をBase64に変換
        encode_base64 = context.get("encode_base64", True)
        all_outputs = {}

        # GUI用処理（Base64エンコード）の時間計測開始
        gui_start = time.perf_counter()

        # STOP中にスキップされたノードのIDを収集
        skipped_node_ids = set()
        if is_preview:
            for node in sorted_nodes:
                try:
                    definition = node_def.get_node_definition(node.definition_id, node.definition_version)
                    if definition and not definition.run_when_stopped:
                        skipped_node_ids.add(node.id)
                except Exception:
                    pass

        for node in sorted_nodes:
            # 出力ポートの値を追加
            for port in node.outputs:
                if port.id in port_values:
                    value = port_values[port.id]
                    if port.data_type == "image" and encode_base64:
                        value = ensure_base64(value, max_size=640)
                    output_key = f"{node.id}.{port.name}"
                    all_outputs[output_key] = value

            # 入力ポートの値も追加（表示用、出力がない場合）
            # - direction="in": 入力専用ポート
            # - direction="inout"で出力値がない場合: レガシー互換（compute()が空を返す場合）
            # ただし、STOP中にスキップされたノード（run_when_stopped=false）は除外
            # （入力画像をそのまま表示するのを防ぐため）
            if node.id in skipped_node_ids:
                continue
            for in_port in node.inputs:
                output_key = f"{node.id}.{in_port.name}"
                # 既に出力として追加済みならスキップ
                if output_key in all_outputs:
                    continue
                # このポートへの接続を探す
                for conn in self.connections:
                    if conn.to_port_id == in_port.id:
                        value = port_values.get(conn.from_port_id)
                        if value is not None:
                            if in_port.data_type == "image" and encode_base64:
                                value = ensure_base64(value, max_size=640)
                            all_outputs[output_key] = value
                        break

        # GUI用処理の時間計測終了
        gui_overhead_ms = (time.perf_counter() - gui_start) * 1000

        # メタデータを追加
        all_outputs.update(node_metadata)

        # 終了シグナルを追加
        if ended:
            all_outputs["__ended__"] = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return all_outputs, elapsed_ms, node_times, node_errors, connected_properties, gui_overhead_ms
