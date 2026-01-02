"""
graph.json をヘッドレスで実行するモジュール

使用方法:
    python run_graph_headless.py <graph.json>
    python run_graph_headless.py graph.json --interval 100 --count 10
"""

import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from node_editor.core import Graph
from node_editor.node_def import discover_nodes, reset_all_nodes
from node_editor.settings import init_settings

# WindowsでCOMを初期化（オーディオデバイスアクセスに必要）
if platform.system() == "Windows":
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except ImportError:
        # pythoncomがない場合はctypesで直接初期化
        try:
            import ctypes
            ctypes.windll.ole32.CoInitialize(None)
        except Exception:
            pass

# sounddeviceを事前にロードしてPortAudioを初期化（WASAPIループバックに必要）
try:
    import sounddevice as sd
    sd.query_devices()
except Exception:
    pass


# ブラウザ依存ノードをヘッドレス対応ノードに置換するマッピング
BROWSER_TO_HEADLESS_MAPPING = {
    "image.input.browser_webcam": "image.input.webcam",
    "image.input.webrtc_webcam": "image.input.webcam",
    "audio.input.browser_microphone": "audio.input.microphone",
    "audio.output.browser_speaker": "audio.output.speaker",
}


def display_images(
    results: dict,
    node_outputs: dict,
    terminal_outputs: set,
    max_size: tuple[int, int] | None = (1280, 720),
) -> None:
    """
    終端の画像出力のみをcv2.imshowで表示

    Args:
        results: 実行結果
        node_outputs: ノードIDと出力ポート情報のマッピング
        terminal_outputs: 終端出力キー（node_id.port_name）のセット
        max_size: 最大表示サイズ (width, height)。Noneの場合はリサイズしない
    """
    for key, value in results.items():
        # 終端出力かつnumpy配列のみ表示
        if key not in terminal_outputs:
            continue
        if not isinstance(value, np.ndarray):
            continue

        # キーからノードIDとポート名を取得
        node_id, port_name = key.rsplit(".", 1)
        output_info = node_outputs.get(node_id, {})
        window_name = f"{output_info.get('name', node_id)} - {port_name}"

        # 最大サイズを超える場合はリサイズ
        display_image = value
        if max_size is not None:
            h, w = value.shape[:2]
            max_w, max_h = max_size
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                display_image = cv2.resize(value, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow(window_name, display_image)


def print_text_outputs(
    results: dict,
    node_outputs: dict,
    terminal_text_outputs: set,
) -> None:
    """
    終端のテキスト出力をコンソールに表示

    Args:
        results: 実行結果
        node_outputs: ノードIDと出力ポート情報のマッピング
        terminal_text_outputs: 終端テキスト出力キー（node_id.port_name）のセット
    """
    for key in terminal_text_outputs:
        value = results.get(key)
        if value is None:
            continue

        # 文字列に変換
        if isinstance(value, str):
            text = value
        else:
            text = str(value)

        # 空文字列は表示しない
        if not text.strip():
            continue

        # ノード名とポート名を取得
        node_id, port_name = key.rsplit(".", 1)
        output_info = node_outputs.get(node_id, {})
        node_name = output_info.get("name", node_id)

        print(f"  [{node_name}.{port_name}] {text}")


def print_timer_status(results: dict, node_outputs: dict) -> None:
    """
    タイマーノードの状態をコンソールに表示

    Args:
        results: 実行結果
        node_outputs: ノードIDと出力ポート情報のマッピング
    """
    for key, value in results.items():
        if not key.endswith(".__timer_interval__"):
            continue

        node_id = key.rsplit(".", 1)[0]
        interval = value
        remaining_key = f"{node_id}.__timer_remaining__"
        unit_key = f"{node_id}.__timer_unit__"

        remaining = results.get(remaining_key, 0.0)
        unit = results.get(unit_key, "sec")

        # ノード名を取得
        output_info = node_outputs.get(node_id, {})
        node_name = output_info.get("name", "Timer")

        # 表示形式を整える
        if unit == "ms":
            interval_display = f"{interval * 1000:.0f}ms"
            remaining_display = f"{remaining * 1000:.0f}ms"
        else:
            interval_display = f"{interval:.1f}s"
            remaining_display = f"{remaining:.1f}s"

        print(f"  [{node_name}] {remaining_display} / {interval_display}")


def convert_frontend_graph_to_backend(
    frontend_data: dict,
    include_input_only_ports: bool = True,
) -> tuple[dict, dict, set, set, set, list]:
    """
    フロントエンドのgraph.json形式をバックエンドの形式に変換
    ブラウザ依存ノードはヘッドレス対応ノードに自動置換される

    Args:
        frontend_data: フロントエンドのgraph.jsonデータ
        include_input_only_ports: 入力専用ポートを終端出力に含めるか

    Returns:
        tuple: (backend_data, node_outputs, terminal_outputs, terminal_text_outputs, final_node_ids, substituted_nodes)
            - backend_data: バックエンド用のグラフデータ
            - node_outputs: ノードIDごとの出力ポート情報
            - terminal_outputs: 終端の画像出力キー（node_id.port_name）のセット
            - terminal_text_outputs: 終端のテキスト出力キー（node_id.port_name）のセット
            - final_node_ids: 下流接続がないノードIDのセット
            - substituted_nodes: 置換されたノードのリスト [(元ID, 新ID), ...]
    """
    nodes = []
    node_outputs = {}

    # 全ての画像出力ポートを収集
    all_image_outputs = set()
    # 入力専用の画像ポートを収集（終端表示用）
    input_only_image_ports = set()
    # 画像出力を持つノードを記録（入力専用ポートの終端判定用）
    nodes_with_image_output = set()
    # テキスト出力ポートを収集
    all_text_outputs = set()
    # 入力専用のテキストポートを収集（Text Display等の終端表示用）
    input_only_text_ports = set()
    # テキスト出力を持つノードを記録（入力専用ポートの終端判定用）
    nodes_with_text_output = set()

    substituted_nodes = []

    for node in frontend_data.get("nodes", []):
        data = node.get("data", {}).get("data", {})
        node_id = node["id"]
        outputs = data.get("outputs", [])
        inputs = data.get("inputs", [])

        # ブラウザ依存ノードをヘッドレス対応ノードに置換
        original_def_id = data.get("definitionId")
        definition_id = BROWSER_TO_HEADLESS_MAPPING.get(original_def_id, original_def_id)
        was_substituted = definition_id != original_def_id
        if was_substituted:
            substituted_nodes.append((original_def_id, definition_id))

        # 置換されたノードはバージョンをNoneにして最新版を使用
        definition_version = None if was_substituted else data.get("version")

        nodes.append({
            "id": node_id,
            "definition_id": definition_id,
            "definition_version": definition_version,
            "name": data.get("label"),
            "inputs": inputs,
            "outputs": outputs,
            "properties": data.get("properties", {}),
            "position": node.get("position", {"x": 0, "y": 0}),
        })

        # 出力ポート情報を保存（画像表示用）
        node_outputs[node_id] = {
            "name": data.get("label", node_id),
        }
        for output in outputs:
            port_id = output.get("id")
            port_name = output.get("name")
            if port_name:
                node_outputs[node_id][port_name] = {
                    "data_type": output.get("data_type"),
                    "port_id": port_id,
                }
                # 画像出力を収集
                if output.get("data_type") == "image":
                    all_image_outputs.add((node_id, port_id, port_name))
                    nodes_with_image_output.add(node_id)
                # テキスト出力を収集
                elif output.get("data_type") == "string":
                    all_text_outputs.add((node_id, port_id, port_name))
                    nodes_with_text_output.add(node_id)

        # 入力専用（direction="in"）の画像・テキストポートを収集
        output_names = {o.get("name") for o in outputs}
        for inp in inputs:
            port_name = inp.get("name")
            port_id = inp.get("id")
            direction = inp.get("direction", "in")
            data_type = inp.get("data_type")
            if direction == "in" and port_name not in output_names:
                if data_type == "image":
                    input_only_image_ports.add((node_id, port_id, port_name))
                    node_outputs[node_id][port_name] = {
                        "data_type": data_type,
                        "port_id": port_id,
                    }
                elif data_type == "string":
                    input_only_text_ports.add((node_id, port_id, port_name))
                    node_outputs[node_id][port_name] = {
                        "data_type": data_type,
                        "port_id": port_id,
                    }

    # 接続されている出力ポートを収集
    connected_outputs = set()
    source_nodes = set()
    connections = []
    for edge in frontend_data.get("edges", []):
        source_node = edge["source"]
        source_handle = edge.get("sourceHandle", "")
        connections.append({
            "id": edge["id"],
            "from_node_id": source_node,
            "from_port_id": source_handle,
            "to_node_id": edge["target"],
            "to_port_id": edge.get("targetHandle", ""),
        })
        connected_outputs.add((source_node, source_handle))
        source_nodes.add(source_node)

    # 全ノードIDを収集
    all_node_ids = {node["id"] for node in frontend_data.get("nodes", [])}

    # 下流接続がないノード（最終ノード）を特定
    final_node_ids = all_node_ids - source_nodes

    # 終端の出力を特定（接続されていない画像出力）
    terminal_outputs = set()
    for node_id, port_id, port_name in all_image_outputs:
        if (node_id, port_id) not in connected_outputs:
            terminal_outputs.add(f"{node_id}.{port_name}")

    # 入力専用の画像ポートを終端として扱う（オプション）
    # ただし、画像出力を持つノードは除外（出力ポートのみ表示）
    if include_input_only_ports:
        for node_id, port_id, port_name in input_only_image_ports:
            if node_id not in nodes_with_image_output:
                terminal_outputs.add(f"{node_id}.{port_name}")

    # 終端のテキスト出力を特定（接続されていないテキスト出力）
    terminal_text_outputs = set()
    for node_id, port_id, port_name in all_text_outputs:
        if (node_id, port_id) not in connected_outputs:
            terminal_text_outputs.add(f"{node_id}.{port_name}")

    # 入力専用のテキストポートを終端として扱う（Text Display等）
    # テキスト出力を持つノードの入力ポートは除外（Text Join等）
    if include_input_only_ports:
        for node_id, port_id, port_name in input_only_text_ports:
            if node_id not in nodes_with_text_output:
                terminal_text_outputs.add(f"{node_id}.{port_name}")

    backend_data = {
        "id": "graph-headless",
        "nodes": nodes,
        "connections": connections,
    }

    return backend_data, node_outputs, terminal_outputs, terminal_text_outputs, final_node_ids, substituted_nodes


def extract_definition_ids_from_graph(frontend_data: dict) -> set:
    """
    グラフJSONから使用されているノードのdefinition_idを抽出する。
    ブラウザ依存ノードはヘッドレス対応ノードに置換したIDを返す。

    Returns:
        使用されているdefinition_idの集合
    """
    definition_ids = set()
    for node in frontend_data.get("nodes", []):
        data = node.get("data", {}).get("data", {})
        original_def_id = data.get("definitionId")
        if original_def_id:
            actual_def_id = BROWSER_TO_HEADLESS_MAPPING.get(original_def_id, original_def_id)
            definition_ids.add(actual_def_id)
    return definition_ids


def run_headless(
    graph_file: Path,
    project_root: Path,
    interval_ms: int = 100,
    count: int = 0,
    show_all: bool = False,
    resize_display: bool = True,
    config_file: Path | None = None,
) -> None:
    """
    グラフをヘッドレスモードで実行する

    Args:
        graph_file: グラフJSONファイルのパス
        project_root: プロジェクトルートディレクトリ
        interval_ms: 実行間隔（ミリ秒）
        count: 実行回数（0=無限）
        show_all: 全ての終端出力を表示するか（False=最終ノードのみ）
        resize_display: 大きい画像を1280x720にリサイズして表示するか（デフォルトTrue）
        config_file: 設定ファイルのパス（Noneの場合はproject_root/config.json）
    """
    # 設定を初期化
    if config_file is None:
        config_file = project_root / "config.json"
    settings_manager = init_settings(config_file)

    # graph.jsonを先に読み込んで、使用ノードを特定
    print(f"Loading graph from: {graph_file}")
    with open(graph_file, "r", encoding="utf-8") as f:
        frontend_data = json.load(f)

    # グラフで使用されているノードIDを抽出
    required_definition_ids = extract_definition_ids_from_graph(frontend_data)
    print(f"  Required nodes: {len(required_definition_ids)}")
    for def_id in sorted(required_definition_ids):
        print(f"    - {def_id}")

    # ノード定義を検出・登録（グラフで使用されているノードのみ）
    print("\nLoading node definitions (headless mode, filtered)...")
    node_search_paths = settings_manager.get("node_search_paths", [])
    for path_str in node_search_paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = project_root / path
        if path.exists():
            discover_nodes(path, gui_type="headless", definition_ids=required_definition_ids)
            print(f"  Discovered nodes from: {path}")

    # フロントエンド形式をバックエンド形式に変換
    # 入力専用ポートも含める（最終ノードのみ表示するためフィルタは後で行う）
    backend_data, node_outputs, terminal_outputs, terminal_text_outputs, final_node_ids, substituted_nodes = (
        convert_frontend_graph_to_backend(frontend_data, include_input_only_ports=True)
    )

    # terminal_outputs, terminal_text_outputs は接続されていない出力ポートを表す
    # 各データタイプごとに独立して終端を判定（ノード単位ではなくポート単位）

    print(f"\nGraph structure:")
    print(f"  Nodes: {len(backend_data['nodes'])}")
    print(f"  Connections: {len(backend_data['connections'])}")
    print(f"  Terminal image outputs: {len(terminal_outputs)}")
    print(f"  Terminal text outputs: {len(terminal_text_outputs)}")

    # ブラウザノード置換の表示
    if substituted_nodes:
        print(f"\n  Node substitutions (browser -> headless):")
        for original, replacement in substituted_nodes:
            print(f"    {original} -> {replacement}")

    # グラフを作成
    graph = Graph(**backend_data)

    # 実行
    print("\n" + "=" * 50)
    print("Executing graph...")
    print("=" * 50)

    execution_count = 0
    interval_sec = interval_ms / 1000.0

    try:
        while True:
            execution_count += 1
            start_time = time.time()

            # コンテキストを設定（encode_base64=Falseでnumpy配列のまま取得）
            loop = count != 1
            context = {
                "loop": loop,
                "interval_ms": interval_ms,
                "encode_base64": False,
                "is_streaming": True,
            }

            # グラフを実行
            results, elapsed_ms, node_times, node_errors, _, _ = graph.execute(context)

            # 結果を表示
            print(f"\n[Execution #{execution_count}] {elapsed_ms:.1f} ms")

            # ノードごとの処理時間（実行順）
            if node_times:
                for node_id, info in sorted(node_times.items(), key=lambda x: x[1]["order"]):
                    print(f"  {info['name']}: {info['time']:.1f} ms")

            # エラーがあれば表示
            if node_errors:
                print("  Errors:")
                for node_id, error in node_errors.items():
                    print(f"    [{node_id}] {error}")

            # 終端の画像出力を表示
            max_size = (1280, 720) if resize_display else None
            display_images(results, node_outputs, terminal_outputs, max_size)

            # 終端のテキスト出力を表示
            print_text_outputs(results, node_outputs, terminal_text_outputs)

            # タイマー状態を表示
            print_timer_status(results, node_outputs)

            # cv2.waitKeyで画面更新（1ms待機）
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("\nESC pressed, exiting...")
                break
            elif key == ord('q'):
                print("\n'q' pressed, exiting...")
                break

            # 指定回数に達したら終了
            if count > 0 and execution_count >= count:
                if count == 1:
                    print("\nPress any key to exit...")
                    cv2.waitKey(0)
                else:
                    print(f"\nReached execution count limit: {count}")
                break

            # インターバル待機
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_sec - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # 終了処理：is_streaming=Falseで最後の実行を行い、VideoWriter等の保存をトリガー
    print("\nFinalizing...")
    final_context = {
        "loop": False,
        "interval_ms": interval_ms,
        "encode_base64": False,
        "is_streaming": False,
    }
    graph.execute(final_context)

    # クリーンアップ
    cv2.destroyAllWindows()
    reset_all_nodes()
    print("\nDone.")
