import importlib.util
import inspect
import platform
import sys
import threading
import tomllib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .models import Port


def is_colab() -> bool:
    """Google Colaboratory環境かどうかを判定"""
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def get_current_platform() -> str:
    """現在のプラットフォームを小文字で取得（windows, linux, darwin）"""
    return platform.system().lower()


# --- Compute Logic Base Class ---
class ComputeLogic(ABC):
    """
    ノードの計算ロジックを実装するための基底クラス。
    各ノードのimpl.pyはこのクラスを継承する必要があります。
    """

    @abstractmethod
    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        ノードの実行ロジック。
        入力ポート名と値の辞書、およびノードのプロパティを受け取り、
        出力ポート名と値の辞書を返します。
        contextにはloop等の実行時設定が含まれます。
        """
        pass

    def reset(self):
        """
        ノードの状態をリセットします。
        ストリーミング再生など状態を持つノードはこのメソッドをオーバーライドしてください。
        """
        pass

    def _get_cancel_event(self) -> threading.Event:
        """キャンセルイベントを取得（遅延初期化）"""
        if not hasattr(self, '_cancel_event'):
            self._cancel_event = threading.Event()
        return self._cancel_event

    def request_cancel(self):
        """キャンセルをリクエストします。compute()内でis_cancelled()をチェックしてください。"""
        self._get_cancel_event().set()

    def is_cancelled(self) -> bool:
        """キャンセルがリクエストされているかチェックします。"""
        return self._get_cancel_event().is_set()

    def clear_cancel(self):
        """キャンセルフラグをクリアします。compute()開始時に呼び出されます。"""
        self._get_cancel_event().clear()


# --- Visible When Condition ---
class VisibleWhen(BaseModel):
    """条件付き表示の設定。"""
    property: str  # 参照するプロパティ名
    values: List[Any]  # 表示する値のリスト


# --- Property Definition ---
class PropertyDefinition(BaseModel):
    """プロパティのメタデータを表すモデル。"""
    name: str
    display_name: str = ""
    type: str = "float"
    default: Any = None
    widget: str = "input"
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None  # counter widget step value
    options: List[Dict[str, Any]] = Field(default_factory=list)  # dropdown options
    options_source: Optional[str] = None  # dynamic options source (e.g., "cameras")
    accept: Optional[str] = None  # file_picker widget: accepted file types
    visible_when: Optional[VisibleWhen] = None  # 条件付き表示
    disabled_while_streaming: bool = False  # ストリーミング中は無効化
    requires_streaming: bool = False  # ストリーミング中のみ有効（ボタン用）
    requires_gpu: bool = False  # GPU利用可能時のみ表示
    button_label: Optional[str] = None  # ボタンのラベル
    requires_api_key: Optional[str] = None  # APIキーが必要なプロパティ
    rows: Optional[int] = None  # text_area widgetの行数


# --- Node Definition (Data Container) ---
class NodeDefinition(BaseModel):
    """
    node.tomlファイルからパースされたノードのメタデータと、
    impl.pyからロードされた計算ロジックを組み合わせたノード定義のモデル。
    """
    definition_id: str
    version: str
    display_name: str = ""
    description: str = ""
    order: int = 100  # カテゴリ内での表示順序
    gui: List[str] = Field(default_factory=list)  # 対応するGUIタイプ (reactflow, nodegraphqt, dearpygui等)
    measure_time: bool = True  # 処理時間計測の対象か否か
    run_when_stopped: bool = False  # STOP時も実行可能か（デフォルトはSTART時のみ）
    resizable: bool = False  # ノードのリサイズを許可するか（デフォルトはオフ）
    node_disable: bool = False  # 無効化フラグ（trueでサイドバーに非表示）
    no_duplicate: bool = False  # 複製不可フラグ（trueで右クリックメニューから複製不可）
    dynamic_ports: Optional[str] = None  # 動的ポートのプレフィックス（例: "Image"で"Image 1", "Image 2"...）
    inputs: List[Port] = Field(default_factory=list)
    outputs: List[Port] = Field(default_factory=list)
    properties: List[PropertyDefinition] = Field(default_factory=list)
    compute_function: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
    compute_logic: Optional[Any] = Field(default=None, exclude=True)  # ComputeLogicインスタンス（互換用）
    compute_logic_class: Optional[Any] = Field(default=None, exclude=True)  # ComputeLogicクラス
    node_instances: Dict[str, Any] = Field(default_factory=dict, exclude=True)  # ノードIDごとのインスタンス
    cached_outputs: Dict[str, Any] = Field(default_factory=dict, exclude=True)  # ノードIDごとのキャッシュ出力

    class Config:
        arbitrary_types_allowed = True

    def get_or_create_instance(self, node_id: str) -> Any:
        """ノードIDに対応するComputeLogicインスタンスを取得または作成"""
        if node_id not in self.node_instances:
            if self.compute_logic_class is not None:
                self.node_instances[node_id] = self.compute_logic_class()
            elif self.compute_logic is not None:
                # 互換性のため：compute_logic_classがない場合は既存のインスタンスを使用
                return self.compute_logic
            else:
                return None
        return self.node_instances[node_id]

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if context is None:
            context = {}

        # ノードIDがあれば、そのノード専用のインスタンスを使用
        node_id = context.get("node_id")
        if node_id and self.compute_logic_class is not None:
            instance = self.get_or_create_instance(node_id)
            if instance is not None:
                return instance.compute(inputs, properties, context)

        return self.compute_function(inputs, properties, context)

    def reset(self):
        """ノードの状態をリセット"""
        # 全てのノードインスタンスをリセット
        for instance in self.node_instances.values():
            if hasattr(instance, 'reset'):
                instance.reset()
        self.node_instances.clear()
        self.cached_outputs.clear()

        # 互換性のため：既存のcompute_logicもリセット
        if self.compute_logic is not None:
            self.compute_logic.reset()


# --- Category Definition ---
class CategoryDefinition(BaseModel):
    """カテゴリのメタデータを表すモデル。"""
    category_id: str
    display_name: str = ""
    order: int = 100
    default_open: bool = True
    requires_config: Optional[str] = None  # 必要な設定キー（設定されていない場合は非表示）


# --- Node Registry ---

# レジストリのキーは (definition_id, version) のタプルとします
_node_definition_registry: Dict[Tuple[str, str], NodeDefinition] = {}

# カテゴリレジストリ
_category_registry: Dict[str, CategoryDefinition] = {}

# requires_configでスキップされたカテゴリのセット
_skipped_categories: set = set()

def register_node(node_def_instance: NodeDefinition):
    """ノード定義のインスタンスをレジストリに登録します。"""
    key = (node_def_instance.definition_id, node_def_instance.version)
    if key in _node_definition_registry:
        print(f"Warning: Node definition '{key}' is being overwritten.")
    _node_definition_registry[key] = node_def_instance

def reset_all_nodes():
    """全てのノードの状態をリセットします。"""
    for node_def_instance in _node_definition_registry.values():
        node_def_instance.reset()
    print("All nodes reset.")


def reset_node_by_id(definition_id: str):
    """指定したdefinition_idのノードをリセットします。"""
    for (did, version), node_def_instance in _node_definition_registry.items():
        if did == definition_id:
            node_def_instance.reset()
            print(f"Node '{definition_id}' reset.")


def cancel_node_by_id(definition_id: str):
    """指定したdefinition_idのノードにキャンセルをリクエストします。"""
    for (did, version), node_def_instance in _node_definition_registry.items():
        if did == definition_id:
            if node_def_instance.compute_logic is not None:
                node_def_instance.compute_logic.request_cancel()


def cancel_all_nodes():
    """全てのノードにキャンセルをリクエストします。"""
    for node_def_instance in _node_definition_registry.values():
        if node_def_instance.compute_logic is not None:
            node_def_instance.compute_logic.request_cancel()


def get_node_definition(definition_id: str, version: Optional[str] = None) -> NodeDefinition:
    """IDとオプションのバージョンに基づいてノード定義を取得します。
    バージョンが指定されない場合、最新（最も新しいバージョン文字列）の定義を返します。
    """
    if version:
        key = (definition_id, version)
        if key not in _node_definition_registry:
            raise ValueError(f"Node definition '{key}' not found in registry.")
        return _node_definition_registry[key]
    else:
        # バージョンが指定されていない場合、最新バージョンを検索
        matching_defs = [
            (v, node_def) for (did, v), node_def in _node_definition_registry.items()
            if did == definition_id
        ]
        if not matching_defs:
            raise ValueError(f"Node definition '{definition_id}' not found in registry.")
        
        # バージョン文字列でソートして最新を取得
        # SemVerを厳密に解釈するならもっと複雑な比較が必要だが、ここでは文字列比較で簡略化
        latest_version, latest_node_def = sorted(matching_defs, key=lambda x: x[0], reverse=True)[0]
        print(f"Info: No version specified for '{definition_id}'. Using latest version '{latest_version}'.")
        return latest_node_def


def get_all_nodes_for_gui(gui_type: str) -> List[NodeDefinition]:
    """
    指定されたGUIタイプに対応するノード定義を取得します。
    guiフィールドが空のノードは全GUI非対応として除外されます。
    node_disable=trueのノードも除外されます。
    """
    result = []
    seen_ids = set()
    for (definition_id, version), node_def in _node_definition_registry.items():
        # 同じdefinition_idは最新バージョンのみ含める
        if definition_id in seen_ids:
            continue
        # 無効化されたノードはスキップ
        if node_def.node_disable:
            continue
        # guiが空の場合は全GUI非対応、指定GUIタイプが含まれている場合のみ追加
        if node_def.gui and gui_type in node_def.gui:
            result.append(node_def)
            seen_ids.add(definition_id)
    return result


def discover_categories(base_path: Path):
    """指定されたベースパス以下のcategory.tomlを走査し、カテゴリ定義をロードして登録します。"""
    from .settings import get_setting

    # スキップされたカテゴリをクリア
    _skipped_categories.clear()

    for category_toml_file in base_path.glob("**/category.toml"):
        try:
            with category_toml_file.open("rb") as f:
                config = tomllib.load(f)

            # パスからカテゴリIDを構築（base_pathからの相対パス）
            relative_path = category_toml_file.parent.relative_to(base_path)
            category_id = ".".join(relative_path.parts)

            # requires_configのチェック
            requires_config = config.get("requires_config", None)
            if requires_config:
                config_value = get_setting(requires_config, "")
                if not config_value:
                    print(f"Skipping category (requires_config={requires_config} not set): {category_id}")
                    _skipped_categories.add(category_id)
                    continue

            category_def = CategoryDefinition(
                category_id=category_id,
                display_name=config.get("display_name", category_id.split(".")[-1].capitalize()),
                order=config.get("order", 100),
                default_open=config.get("default_open", True),
                requires_config=requires_config,
            )
            _category_registry[category_id] = category_def
            print(f"Registered category: {category_id}")

        except Exception as e:
            print(f"Error processing category from {category_toml_file}: {e}")


def get_all_categories() -> List[CategoryDefinition]:
    """登録された全カテゴリをorder順でソートして返します。"""
    return sorted(_category_registry.values(), key=lambda c: c.order)


def discover_nodes(
    base_path: Path,
    gui_type: Optional[str] = None,
    definition_ids: Optional[set] = None,
):
    """指定されたベースパス以下のディレクトリを走査し、ノード定義をロードして登録します。
    構造は 'base_path/domain/node_name/impl.py' と 'base_path/domain/node_name/node.toml' を想定します。

    Args:
        base_path: ノードを検索するベースパス
        gui_type: 対応するGUIタイプでフィルタ（None=全て、"reactflow", "headless"等）
        definition_ids: 登録対象のノードID集合（None=全て登録）
    """
    # カテゴリを検出（definition_idsフィルタがない場合のみ）
    if definition_ids is None:
        discover_categories(base_path)

    found_toml_files = list(base_path.glob("**/node.toml"))

    for node_toml_file in found_toml_files:
        try:
            # 1. node.tomlをパース
            with node_toml_file.open("rb") as f:
                node_config = tomllib.load(f)

            # ノード名からカテゴリIDを取得し、カテゴリが登録されているかチェック
            definition_id = node_config.get("name", "")

            # definition_idsフィルタ（指定されている場合、対象IDのみ登録）
            if definition_ids is not None and definition_id not in definition_ids:
                continue

            # カテゴリチェック（definition_idsフィルタがない場合のみ）
            if definition_ids is None and definition_id:
                # definition_idの最後の部分を除いた部分がカテゴリID
                category_parts = definition_id.rsplit(".", 1)
                if len(category_parts) > 1:
                    category_id = category_parts[0]

                    # カテゴリまたはその祖先がスキップされていないかチェック
                    is_skipped = False
                    for skipped_cat in _skipped_categories:
                        # カテゴリIDがスキップされたカテゴリと一致、または
                        # スキップされたカテゴリの子孫である場合はスキップ
                        if category_id == skipped_cat or category_id.startswith(skipped_cat + "."):
                            print(f"Skipping node (category skipped): {definition_id}")
                            is_skipped = True
                            break
                    if is_skipped:
                        continue

                    # カテゴリが登録されていない場合（category.tomlが無い場合）
                    if category_id not in _category_registry:
                        # 親カテゴリが登録されているか確認
                        parent_registered = False
                        for registered_cat in _category_registry.keys():
                            if category_id.startswith(registered_cat + "."):
                                parent_registered = True
                                break
                        if not parent_registered:
                            print(f"Skipping node (category not registered): {definition_id}")
                            continue

            # GUIタイプフィルタ（gui_typeが指定されている場合）
            if gui_type is not None:
                node_gui_list = node_config.get("gui", [])
                # guiリストが空の場合は全GUI非対応、指定タイプが含まれていない場合もスキップ
                if not node_gui_list or gui_type not in node_gui_list:
                    continue

            # Colab環境での除外チェック（colab=falseのノードはColabで読み込まない）
            colab_enabled = node_config.get("colab", True)
            if not colab_enabled and is_colab():
                print(f"Skipping node (colab=false): {node_toml_file}")
                continue

            # ノードレベルのrequires_configチェック（設定が未設定の場合はスキップ）
            node_requires_config = node_config.get("requires_config", None)
            if node_requires_config:
                from .settings import get_setting
                config_value = get_setting(node_requires_config, "")
                if not config_value:
                    print(f"Skipping node (requires_config={node_requires_config} not set): {node_toml_file}")
                    continue

            # プラットフォームチェック（platform指定がある場合、現在のプラットフォームが含まれているか）
            platform_list = node_config.get("platform", None)
            if platform_list is not None:
                current_platform = get_current_platform()
                if current_platform not in platform_list:
                    print(f"Skipping node (platform={platform_list}, current={current_platform}): {node_toml_file}")
                    continue

            # メタデータを抽出
            definition_id = node_config["name"]
            version = node_config["version"]
            display_name = node_config.get("display_name", definition_id)
            description = node_config.get("description", "")

            # ポート情報をパース（[[ports]]形式を優先、後方互換性のため[[inputs]]/[[outputs]]もサポート）
            ports_config = node_config.get("ports", [])
            if ports_config:
                # [[ports]]形式: directionに応じてinputs/outputsに振り分け
                inputs = []
                outputs = []
                for p in ports_config:
                    port = Port(**p)
                    if port.direction in ("in", "inout"):
                        inputs.append(port)
                    if port.direction in ("out", "inout"):
                        outputs.append(port)
            else:
                # 後方互換性: [[inputs]]/[[outputs]]形式
                inputs = [Port(**p, direction="in") for p in node_config.get("inputs", [])]
                outputs = [Port(**p, direction="out") for p in node_config.get("outputs", [])]

            # プロパティ情報をパース
            properties = [PropertyDefinition(**p) for p in node_config.get("properties", [])]

            # 2. 対応するimpl.pyをロード
            impl_file = node_toml_file.parent / "impl.py"
            if not impl_file.exists():
                print(f"Warning: impl.py not found for node.toml at {node_toml_file}. Skipping.")
                continue

            # モジュールを動的にロード
            module_name = f"nodes.{'.'.join(impl_file.relative_to(base_path.parent).parent.parts)}"
            spec = importlib.util.spec_from_file_location(module_name, impl_file)
            if spec is None:
                print(f"Warning: Could not create module spec for {impl_file}. Skipping.")
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"Error loading module {impl_file}: {e}. Skipping.")
                continue
            
            # ロードされたモジュールからComputeLogicのサブクラスを検索
            compute_logic_instance = None
            compute_logic_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, ComputeLogic) and obj is not ComputeLogic:
                    try:
                        compute_logic_class = obj  # クラス自体を保存
                        compute_logic_instance = obj()
                        break
                    except Exception as e:
                        print(f"Error instantiating ComputeLogic subclass {obj.__name__} in {impl_file}: {e}")

            if compute_logic_instance is None:
                print(f"Warning: No ComputeLogic subclass found in {impl_file}. Skipping node definition.")
                continue

            # 3. NodeDefinitionインスタンスを作成し登録
            order = node_config.get("order", 100)
            gui = node_config.get("gui", [])  # 対応GUIタイプ（未設定の場合は全GUI対象）
            measure_time = node_config.get("measure_time", True)  # 処理時間計測対象（デフォルトtrue）
            run_when_stopped = node_config.get("run_when_stopped", False)  # STOP時も実行可（デフォルトfalse）
            resizable = node_config.get("resizable", False)  # リサイズ許可（デフォルトfalse）
            node_disable = node_config.get("node_disable", False)  # 無効化（デフォルトfalse）
            no_duplicate = node_config.get("no_duplicate", False)  # 複製不可（デフォルトfalse）
            dynamic_ports = node_config.get("dynamic_ports", None)  # 動的ポートのプレフィックス
            node_def_instance = NodeDefinition(
                definition_id=definition_id,
                version=version,
                display_name=display_name,
                description=description,
                order=order,
                gui=gui,
                measure_time=measure_time,
                run_when_stopped=run_when_stopped,
                resizable=resizable,
                node_disable=node_disable,
                no_duplicate=no_duplicate,
                dynamic_ports=dynamic_ports,
                inputs=inputs,
                outputs=outputs,
                properties=properties,
                compute_function=compute_logic_instance.compute,
                compute_logic=compute_logic_instance,
                compute_logic_class=compute_logic_class,
            )
            register_node(node_def_instance)
            print(f"Registered node: {node_def_instance.definition_id} v{node_def_instance.version}")

        except Exception as e:
            print(f"Error processing node definition from {node_toml_file}: {e}")
