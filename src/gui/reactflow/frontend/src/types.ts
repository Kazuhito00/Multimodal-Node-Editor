// ノード定義の型
export interface PortDef {
  id: string;
  name: string;
  display_name?: string;  // 表示名（省略時はnameを使用）
  data_type: string;
  direction: 'in' | 'out' | 'inout';
  preview?: boolean;  // プレビュー表示（画像/ウェーブフォーム）するか、デフォルトtrue
}

export interface DropdownOption {
  value: number | string;
  label: string;
}

// 条件付き表示の型
export interface VisibleWhen {
  property: string;  // 参照するプロパティ名
  values: (number | string | boolean)[];  // 表示する値のリスト
}

export interface PropertyDef {
  name: string;
  display_name: string;
  type: string;
  default: number | string | boolean | null;
  widget: string;
  min?: number;
  max?: number;
  step?: number;
  options?: DropdownOption[];
  options_source?: string;
  accept?: string;  // file_picker widget: accepted file types
  placeholder?: string;  // text_input widget: placeholder text
  visible_when?: VisibleWhen;  // 条件付き表示
  disabled_while_streaming?: boolean;  // ストリーミング中は無効化
  requires_streaming?: boolean;  // ストリーミング中のみ有効
  button_label?: string;  // button widget: ボタンのラベル
  requires_api_key?: string;  // このプロパティを表示するために必要なAPIキー（例: "openai"）
  rows?: number;  // text_area widget: 行数
}

export interface NodeDefinition {
  definition_id: string;
  version: string;
  display_name: string;
  description: string;
  order: number;
  resizable?: boolean;  // ノードのリサイズを許可するか（デフォルトfalse）
  no_duplicate?: boolean;  // 複製不可フラグ（デフォルトfalse）
  dynamic_ports?: string;  // 動的ポートのプレフィックス（例: "Image"）
  inputs: PortDef[];
  outputs: PortDef[];
  properties: PropertyDef[];
}

// カテゴリ定義の型
export interface CategoryDefinition {
  category_id: string;
  display_name: string;
  order: number;
  default_open: boolean;
}

// グラフ実行用の型
export interface GraphPort {
  id: string;
  name: string;
  display_name?: string;  // 表示名（省略時はnameを使用）
  data_type: string;
  direction: 'in' | 'out' | 'inout';
  preview?: boolean;  // プレビュー表示（画像/ウェーブフォーム）するか、デフォルトtrue
}

export interface GraphNode {
  id: string;
  definition_id: string;
  definition_version: string;
  name: string;
  inputs: GraphPort[];
  outputs: GraphPort[];
  properties: Record<string, unknown>;
  position: { x: number; y: number };
}

export interface GraphConnection {
  id: string;
  from_node_id: string;
  from_port_id: string;
  to_node_id: string;
  to_port_id: string;
}

export interface GraphData {
  id: string;
  nodes: GraphNode[];
  connections: GraphConnection[];
}

// オーディオデータ型
export interface AudioData {
  delta: number[];      // 差分サンプル（前回以降の新規データ、再生用）
  waveform: number[];   // 表示用（min/maxペア×400）
  sample_rate: number;
  duration: number;
  browser_speaker?: boolean;  // ブラウザで再生するフラグ
  label?: string;       // オーバーレイラベル（分類結果など）
}

// ReactFlow用のノードデータ型
export interface CustomNodeData {
  label: string;
  definitionId: string;
  version: string;
  resizable?: boolean;  // ノードのリサイズを許可するか
  noDuplicate?: boolean;  // 複製不可フラグ
  dynamicPorts?: string;  // 動的ポートのプレフィックス
  connectedPortIds?: string[];  // 接続されているポートのID（動的ポート用）
  comment?: string;  // ノードコメント
  inputs: GraphPort[];
  outputs: GraphPort[];
  properties: Record<string, unknown>;
  propertyDefs: PropertyDef[];
  imageData?: string;
  maskData?: string;  // マスクノード用の出力マスク
  audioData?: AudioData;
  numericOutputs?: Record<string, number>;  // 数値出力ポートの値（ポート名→値）
  stringOutputs?: Record<string, string>;  // 文字列出力ポートの値（ポート名→値）
  connectedProperties?: Record<string, number | string>;  // 接続された入力値（プロパティ名→値）
  displayText?: string;  // テキスト表示ノード用
  errorMessage?: string;
  isStreaming?: boolean;  // ストリーミング中かどうか
  isPaused?: boolean;  // 一時停止中かどうか
  intervalMs?: number;  // 実行インターバル（ミリ秒）
  isBusy?: boolean;  // ノードが処理中かどうか（API呼び出し中など）
  apiKeysStatus?: Record<string, boolean>;  // APIキー設定状況（キー名→設定有無）
}

// ノード実行時間の型
export interface NodeTimeInfo {
  name: string;
  time: number;
  order: number;
}

// WebSocketメッセージの型
export interface WSMessage {
  type: string;
  graph?: GraphData;
  interval_ms?: number;
  results?: Record<string, string | number | boolean | AudioData | Record<string, unknown>>;
  elapsed_ms?: number;
  node_times?: Record<string, NodeTimeInfo>;
  node_errors?: Record<string, string>;
  connected_properties?: Record<string, Record<string, number>>;  // ノードID→(プロパティ名→値)
  gui_overhead_ms?: number;  // GUI処理（Base64エンコード等）の時間
  message?: string;
}
