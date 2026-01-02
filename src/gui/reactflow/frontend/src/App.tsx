import { useCallback, useEffect, useMemo, useRef, useState, ChangeEvent } from 'react';
import {
  ReactFlow,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  BackgroundVariant,
  Connection,
  Edge,
  EdgeChange,
  Node,
  ReactFlowInstance,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import CustomNode from './CustomNode';
import {
  NodeDefinition,
  CategoryDefinition,
  GraphPort,
  CustomNodeData,
  WSMessage,
  NodeTimeInfo,
  AudioData,
} from './types';
import { getLayoutedElements } from './layout';
import {
  makeApiBase,
  makeWsUrl,
  generateId,
  getInitialTheme,
  buildGraphData,
  createNodeFromDefinition,
  Runtime,
} from './utils';
import { DraggingProvider, useDragging } from './contexts/DraggingContext';
import { WebRTCConnectionProvider } from './contexts/WebRTCConnectionManager';

const nodeTypes = {
  custom: CustomNode,
};

// 履歴のスナップショット型
interface HistorySnapshot {
  nodes: Node<{ data: CustomNodeData }>[];
  edges: Edge[];
}

// コンテキストメニューの状態型
interface ContextMenuState {
  nodeId: string;
  x: number;
  y: number;
}

// 履歴管理の最大サイズ
const MAX_HISTORY_SIZE = 50;

// カテゴリツリー構造の型
interface CategoryTreeNode {
  name: string;
  path: string;
  children: Map<string, CategoryTreeNode>;
  nodes: NodeDefinition[];
  order: number;
}

function AppContent() {
  const { setDragging } = useDragging();
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<{ data: CustomNodeData }>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [definitions, setDefinitions] = useState<NodeDefinition[]>([]);
  const [categories, setCategories] = useState<CategoryDefinition[]>([]);
  const [theme, setTheme] = useState<'dark' | 'light'>(getInitialTheme);
  const [playbackState, setPlaybackState] = useState<'stopped' | 'running' | 'paused'>('stopped');
  const [elapsedMs, setElapsedMs] = useState<number>(0);
  const [nodeTimes, setNodeTimes] = useState<Record<string, NodeTimeInfo>>({});
  const [guiOverheadMs, setGuiOverheadMs] = useState<number>(0);
  const [pausedElapsedMs, setPausedElapsedMs] = useState<number>(0);
  const [pausedNodeTimes, setPausedNodeTimes] = useState<Record<string, NodeTimeInfo>>({});
  const [pausedGuiOverheadMs, setPausedGuiOverheadMs] = useState<number>(0);
  const [intervalMs, setIntervalMs] = useState<number>(100);
  const [loopPlayback, setLoopPlayback] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState(true);
  const isLoadingRef = useRef(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showEditSection, setShowEditSection] = useState(true);
  const [showFileSection, setShowFileSection] = useState(true);
  const [showAutoLayoutSection, setShowAutoLayoutSection] = useState(true);
  const [nodeSearchQuery, setNodeSearchQuery] = useState('');
  const autoDownloadVideoRef = useRef(false);
  const autoDownloadWavRef = useRef(false);
  const autoDownloadCapturesRef = useRef(false);
  const autoDownloadTextRef = useRef(false);
  const apiKeysStatusRef = useRef<Record<string, boolean>>({});
  const [wsUrl, setWsUrl] = useState<string>('ws://localhost:8000/api/ws/stream');
  const wsRef = useRef<WebSocket | null>(null);
  const reactFlowInstance = useRef<ReactFlowInstance<Node<{ data: CustomNodeData }>, Edge> | null>(null);
  // ストリーミング中のset_graphスロットリング用
  const lastSetGraphTimeRef = useRef<number>(0);
  const pendingSetGraphRef = useRef<boolean>(false);
  const setGraphTimeoutRef = useRef<number | null>(null);
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  nodesRef.current = nodes;
  edgesRef.current = edges;
  // 実際の処理時間を追跡（スロットリング間隔の調整用）
  const elapsedMsRef = useRef<number>(0);

  // 履歴管理
  const historyRef = useRef<HistorySnapshot[]>([{ nodes: [], edges: [] }]);
  const historyIndexRef = useRef<number>(0);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const isUndoRedoRef = useRef(false);

  // コンテキストメニュー状態
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);

  // コメントモーダル状態
  const [commentModal, setCommentModal] = useState<{ nodeId: string; value: string } | null>(null);

  // 履歴にスナップショットを保存
  const saveToHistory = useCallback((newNodes: Node<{ data: CustomNodeData }>[], newEdges: Edge[]) => {
    // Undo/Redo操作中は保存しない
    if (isUndoRedoRef.current) return;

    const history = historyRef.current;
    const currentIndex = historyIndexRef.current;

    // 現在位置より後の履歴を削除
    history.splice(currentIndex + 1);

    // 新しいスナップショットを追加（位置情報を除いて保存）
    const snapshot: HistorySnapshot = {
      nodes: JSON.parse(JSON.stringify(newNodes)),
      edges: JSON.parse(JSON.stringify(newEdges)),
    };
    history.push(snapshot);

    // 履歴サイズ制限
    if (history.length > MAX_HISTORY_SIZE) {
      history.shift();
    } else {
      historyIndexRef.current = history.length - 1;
    }

    setCanUndo(historyIndexRef.current > 0);
    setCanRedo(false);
  }, []);

  // Undo実行
  const handleUndo = useCallback(() => {
    // START中は無効
    if (playbackState === 'running') return;

    const history = historyRef.current;
    const currentIndex = historyIndexRef.current;

    if (currentIndex <= 0) return;

    isUndoRedoRef.current = true;
    const newIndex = currentIndex - 1;
    const snapshot = history[newIndex];

    setNodes(JSON.parse(JSON.stringify(snapshot.nodes)));
    setEdges(JSON.parse(JSON.stringify(snapshot.edges)));

    historyIndexRef.current = newIndex;
    setCanUndo(newIndex > 0);
    setCanRedo(true);

    // フラグをリセット
    setTimeout(() => {
      isUndoRedoRef.current = false;
    }, 0);
  }, [setNodes, setEdges, playbackState]);

  // Redo実行
  const handleRedo = useCallback(() => {
    // START中は無効
    if (playbackState === 'running') return;

    const history = historyRef.current;
    const currentIndex = historyIndexRef.current;

    if (currentIndex >= history.length - 1) return;

    isUndoRedoRef.current = true;
    const newIndex = currentIndex + 1;
    const snapshot = history[newIndex];

    setNodes(JSON.parse(JSON.stringify(snapshot.nodes)));
    setEdges(JSON.parse(JSON.stringify(snapshot.edges)));

    historyIndexRef.current = newIndex;
    setCanUndo(true);
    setCanRedo(newIndex < history.length - 1);

    // フラグをリセット
    setTimeout(() => {
      isUndoRedoRef.current = false;
    }, 0);
  }, [setNodes, setEdges, playbackState]);

  // オートレイアウト実行（アニメーション付き）
  const handleAutoLayout = useCallback(() => {
    const targetNodes = getLayoutedElements(nodes, edges);
    const startNodes = nodes;
    const duration = 300;
    const startTime = performance.now();

    // イージング関数（ease-out）
    const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easedProgress = easeOut(progress);

      // 現在位置から目標位置へ補間
      const interpolatedNodes = startNodes.map((startNode) => {
        const targetNode = targetNodes.find((n) => n.id === startNode.id);
        if (!targetNode) return startNode;

        return {
          ...startNode,
          position: {
            x: startNode.position.x + (targetNode.position.x - startNode.position.x) * easedProgress,
            y: startNode.position.y + (targetNode.position.y - startNode.position.y) * easedProgress,
          },
        };
      });

      setNodes(interpolatedNodes);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        // アニメーション完了後に履歴保存とfitView
        saveToHistory(targetNodes, edges);
        reactFlowInstance.current?.fitView({ duration: 200, maxZoom: 1 });
      }
    };

    requestAnimationFrame(animate);
  }, [nodes, edges, setNodes, saveToHistory]);

  // キーボードショートカット用のハンドラーをrefで保持（後方で定義されるハンドラーを参照するため）
  const keyboardHandlersRef = useRef({
    handleStart: () => {},
    handleStop: () => {},
    handlePause: () => {},
    handleUndo: () => {},
    handleRedo: () => {},
    handleAutoLayout: () => {},
    handleSave: () => {},
    handleLoad: () => {},
    getPlaybackState: () => 'stopped' as 'stopped' | 'running' | 'paused',
  });

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // ローディング中は無視
      if (isLoadingRef.current) {
        return;
      }

      // 入力フィールドにフォーカスがある場合は無視
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      const handlers = keyboardHandlersRef.current;

      // Escape: STOP
      if (e.key === 'Escape') {
        e.preventDefault();
        handlers.handleStop();
        return;
      }

      if (e.ctrlKey || e.metaKey) {
        // Ctrl+Enter: START/STOP切り替え
        if (e.key === 'Enter') {
          e.preventDefault();
          if (handlers.getPlaybackState() === 'running') {
            handlers.handleStop();
          } else {
            handlers.handleStart();
          }
          return;
        }
        if (e.key === 'z' && !e.shiftKey) {
          e.preventDefault();
          handlers.handleUndo();
        } else if (e.key === 'y' || (e.key === 'z' && e.shiftKey)) {
          e.preventDefault();
          handlers.handleRedo();
        } else if (e.key === 'a') {
          e.preventDefault();
          handlers.handleAutoLayout();
        } else if (e.key === 'l') {
          e.preventDefault();
          handlers.handleLoad();
        } else if (e.key === 'p') {
          e.preventDefault();
          // Pause/Resume トグル
          if (handlers.getPlaybackState() === 'paused') {
            handlers.handleStart();
          } else if (handlers.getPlaybackState() === 'running') {
            handlers.handlePause();
          }
        } else if (e.key === 's') {
          e.preventDefault();
          handlers.handleSave();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // ノード・エッジの構造変更を検知して履歴に保存
  const prevNodesLengthRef = useRef(0);
  const prevEdgesLengthRef = useRef(0);
  const dragEndTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    // Undo/Redo操作中、または再生結果更新中は保存しない
    if (isUndoRedoRef.current) return;

    // ノード数またはエッジ数が変わった場合は即座に保存
    const nodesChanged = nodes.length !== prevNodesLengthRef.current;
    const edgesChanged = edges.length !== prevEdgesLengthRef.current;

    if (nodesChanged || edgesChanged) {
      prevNodesLengthRef.current = nodes.length;
      prevEdgesLengthRef.current = edges.length;
      saveToHistory(nodes, edges);
    }
  }, [nodes.length, edges.length, saveToHistory]);

  // ノードのドラッグ終了時に履歴保存
  const handleNodesChangeWithHistory = useCallback(
    (changes: Parameters<typeof onNodesChange>[0]) => {
      onNodesChange(changes);

      // ドラッグ終了を検知
      const hasDragEnd = changes.some(
        (change) => change.type === 'position' && change.dragging === false
      );

      if (hasDragEnd) {
        // 少し遅延してから保存（状態が確定するのを待つ）
        if (dragEndTimeoutRef.current) {
          clearTimeout(dragEndTimeoutRef.current);
        }
        dragEndTimeoutRef.current = window.setTimeout(() => {
          setNodes((currentNodes) => {
            setEdges((currentEdges) => {
              saveToHistory(currentNodes, currentEdges);
              return currentEdges;
            });
            return currentNodes;
          });
        }, 50);
      }
    },
    [onNodesChange, setNodes, setEdges, saveToHistory]
  );

  // 実行環境を判定してAPI/WS URLを設定後、各種設定を取得
  useEffect(() => {
    let retryCount = 0;
    const maxRetries = 5;
    const retryDelay = 500;

    const fetchAppData = (base: string) => {
      Promise.all([
        fetch(`${base}/nodes/definitions`).then((res) => res.json()),
        fetch(`${base}/nodes/categories`).then((res) => res.json()),
        fetch(`${base}/settings/theme`).then((res) => res.json()),
        fetch(`${base}/settings/graph`).then((res) => res.json()),
        fetch(`${base}/settings/sidebar`).then((res) => res.json()),
        fetch(`${base}/settings/auto_download`).then((res) => res.json()),
        fetch(`${base}/settings/api_keys_status`).then((res) => res.json()),
      ])
        .then(([defs, cats, themeData, graphSettings, sidebarSettings, autoDownloadSettings, apiKeysStatus]: [
          NodeDefinition[],
          CategoryDefinition[],
          { theme: string },
          { interval_ms: number },
          { show_edit: boolean; show_file: boolean; show_auto_layout: boolean },
          { video: boolean; wav: boolean; capture: boolean; text: boolean },
          Record<string, boolean>
        ]) => {
          // ノード定義が空の場合はリトライ
          if ((!defs || defs.length === 0) && retryCount < maxRetries) {
            retryCount++;
            console.log(`Definitions empty, retrying... (${retryCount}/${maxRetries})`);
            setTimeout(() => fetchAppData(base), retryDelay);
            return;
          }

          setDefinitions(defs || []);
          setCategories(cats || []);
          setTheme(themeData.theme === 'light' ? 'light' : 'dark');
          setIntervalMs(graphSettings.interval_ms);
          setShowEditSection(sidebarSettings.show_edit);
          setShowFileSection(sidebarSettings.show_file);
          setShowAutoLayoutSection(sidebarSettings.show_auto_layout);
          autoDownloadVideoRef.current = autoDownloadSettings.video;
          autoDownloadWavRef.current = autoDownloadSettings.wav;
          autoDownloadCapturesRef.current = autoDownloadSettings.capture;
          autoDownloadTextRef.current = autoDownloadSettings.text;
          apiKeysStatusRef.current = apiKeysStatus || {};
          isLoadingRef.current = false;
          setIsLoading(false);
        })
        .catch((err) => {
          console.error('Failed to fetch definitions:', err);
          if (retryCount < maxRetries) {
            retryCount++;
            console.log(`Fetch failed, retrying... (${retryCount}/${maxRetries})`);
            setTimeout(() => fetchAppData(base), retryDelay);
          } else {
            isLoadingRef.current = false;
            setIsLoading(false);
          }
        });
    };

    // まずruntimeを取得してAPI/WS URLを決定
    fetch('/api/settings/runtime')
      .then((res) => res.json())
      .then((runtime: Runtime) => {
        const base = makeApiBase(runtime.is_colab);
        setWsUrl(makeWsUrl(runtime.is_colab));
        fetchAppData(base);
      })
      .catch(() => {
        // 取得失敗時はローカル前提
        const base = makeApiBase(false);
        setWsUrl(makeWsUrl(false));
        fetchAppData(base);
      });
  }, []);

  // テーマ変更時にlocalStorageに保存
  useEffect(() => {
    try {
      localStorage.setItem('theme', theme);
    } catch {
      // localStorage使用不可
    }
  }, [theme]);

  // WebSocket接続管理（runtime判定後に接続）
  useEffect(() => {
    // ロード中はWebSocket接続しない
    if (isLoading) return;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      // 接続時にノードがあれば初期プレビュー実行
      setNodes((currentNodes) => {
        setEdges((currentEdges) => {
          if (currentNodes.length > 0) {
            const graphData = buildGraphData(currentNodes, currentEdges);
            ws.send(JSON.stringify({ type: 'execute_once', graph: graphData }));
          }
          return currentEdges;
        });
        return currentNodes;
      });
    };

    ws.onmessage = (event) => {
      const message: WSMessage = JSON.parse(event.data);

      if (message.type === 'result' && message.results) {
        // 実行時間を更新
        if (message.elapsed_ms !== undefined) {
          setElapsedMs(message.elapsed_ms);
          elapsedMsRef.current = message.elapsed_ms;
        }
        if (message.node_times) {
          setNodeTimes(message.node_times);
        }
        if (message.gui_overhead_ms !== undefined) {
          setGuiOverheadMs(message.gui_overhead_ms);
        }

        // ダウンロード情報をチェック（auto_downloadがtrueの場合のみ自動実行）
        // プロパティ更新情報を収集
        const propertyUpdates: Record<string, Record<string, unknown>> = {};
        for (const key of Object.keys(message.results)) {
          if (key.endsWith('.__download__')) {
            const downloadInfo = message.results[key] as unknown as { path: string; filename: string; type: string };
            if (downloadInfo && downloadInfo.path) {
              // タイプに応じてauto_download設定をチェック
              const isVideo = downloadInfo.type?.startsWith('video/');
              const isAudio = downloadInfo.type?.startsWith('audio/');
              const isZip = downloadInfo.type === 'application/zip';
              const isText = downloadInfo.type?.startsWith('text/');
              const shouldDownload =
                (isVideo && autoDownloadVideoRef.current) ||
                (isAudio && autoDownloadWavRef.current) ||
                (isZip && autoDownloadCapturesRef.current) ||
                (isText && autoDownloadTextRef.current);

              if (shouldDownload) {
                // ダウンロードダイアログを表示
                const downloadUrl = `/api/download?path=${encodeURIComponent(downloadInfo.path)}`;
                const a = document.createElement('a');
                a.href = downloadUrl;
                a.download = downloadInfo.filename || 'download';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
              }
            }
          }
          // プロパティ更新情報を収集
          if (key.endsWith('.__update_property__')) {
            const nodeId = key.replace('.__update_property__', '');
            const updates = message.results[key];
            if (updates && typeof updates === 'object' && !Array.isArray(updates)) {
              propertyUpdates[nodeId] = updates as unknown as Record<string, unknown>;
            }
          }
        }

        // 結果から画像/オーディオデータとエラー情報を抽出してノードを更新
        const nodeErrors = message.node_errors || {};
        setNodes((currentNodes) => {
          return currentNodes.map((node) => {
            // 出力ポートと入力専用ポートの両方を探す
            const imageOutput = node.data.data.outputs.find(
              (o) => o.data_type === 'image'
            );
            const imageInput = node.data.data.inputs.find(
              (i) => i.data_type === 'image' && i.direction === 'in'
            );
            const audioOutput = node.data.data.outputs.find(
              (o) => o.data_type === 'audio'
            );
            const audioInput = node.data.data.inputs.find(
              (i) => i.data_type === 'audio' && i.direction === 'in'
            );

            let updatedData = { ...node.data.data };
            let hasUpdate = false;

            // エラーメッセージの更新
            const errorMessage = nodeErrors[node.id];
            if (errorMessage !== updatedData.errorMessage) {
              updatedData = { ...updatedData, errorMessage };
              hasUpdate = true;
            }

            // プロパティ更新の適用
            const propUpdates = propertyUpdates[node.id];
            if (propUpdates) {
              const newProperties = { ...updatedData.properties };
              for (const [propName, propValue] of Object.entries(propUpdates)) {
                newProperties[propName] = propValue;
              }
              updatedData = { ...updatedData, properties: newProperties };
              hasUpdate = true;
            }

            // マスクノードの特別処理（入力画像と出力マスクを別々に処理）
            const isMaskNode = node.data.data.definitionId === 'image.draw.mask';
            // クロップノードの特別処理（入力画像を表示、出力はクロップ結果）
            const isCropNode = node.data.data.definitionId === 'image.transform.crop';
            // Click Perspectiveノードの特別処理（入力画像を表示）
            const isClickPerspectiveNode = node.data.data.definitionId === 'image.transform.click_perspective';
            // 描画キャンバスノードの特別処理（出力画像を表示）
            const isDrawCanvasNode = node.data.data.definitionId === 'image.draw.canvas';
            if (isMaskNode) {
              // 入力画像（背景用）
              if (imageInput) {
                const inputKey = `${node.id}.${imageInput.name}`;
                const imageData = message.results?.[inputKey];
                if (imageData && typeof imageData === 'string') {
                  updatedData = { ...updatedData, imageData };
                  hasUpdate = true;
                }
              }
              // 出力マスク
              const maskOutput = node.data.data.outputs.find(
                (o) => o.name === 'mask' && o.data_type === 'image'
              );
              if (maskOutput) {
                const maskKey = `${node.id}.${maskOutput.name}`;
                const maskData = message.results?.[maskKey];
                if (maskData && typeof maskData === 'string') {
                  updatedData = { ...updatedData, maskData };
                  hasUpdate = true;
                }
              }
            } else if (isCropNode || isClickPerspectiveNode) {
              // クロップ/Click Perspectiveノード: 入力画像を表示
              if (imageInput) {
                const inputKey = `${node.id}.${imageInput.name}`;
                const imageData = message.results?.[inputKey];
                if (imageData && typeof imageData === 'string') {
                  updatedData = { ...updatedData, imageData };
                  hasUpdate = true;
                }
              }
            } else if (isDrawCanvasNode) {
              // 描画キャンバスノード: 出力画像を表示（描画結果）
              // inoutポートはoutputsとinputsの両方に存在する可能性がある
              const drawCanvasPort = imageOutput || node.data.data.inputs.find(
                (i) => i.data_type === 'image'
              );
              if (drawCanvasPort) {
                const outputKey = `${node.id}.${drawCanvasPort.name}`;
                const imageData = message.results?.[outputKey];
                if (imageData && typeof imageData === 'string') {
                  updatedData = { ...updatedData, imageData };
                  hasUpdate = true;
                }
              }
            } else {
              // 通常の画像データの更新（出力ポートを優先、なければ入力専用ポート）
              const imagePort = imageOutput || imageInput;
              if (imagePort) {
                const outputKey = `${node.id}.${imagePort.name}`;
                const imageData = message.results?.[outputKey];
                if (imageData && typeof imageData === 'string') {
                  updatedData = { ...updatedData, imageData };
                  hasUpdate = true;
                }
              }
            }

            // オーディオデータの更新（出力ポートを優先、なければ入力専用ポート）
            const audioPort = audioOutput || audioInput;
            if (audioPort) {
              const outputKey = `${node.id}.${audioPort.name}`;
              const audioDataRaw = message.results?.[outputKey];
              if (audioDataRaw && typeof audioDataRaw === 'object') {
                updatedData = { ...updatedData, audioData: audioDataRaw as AudioData };
                hasUpdate = true;
              }
            }

            // フレーム数メタデータの更新（シークバーのmax値）
            const frameCountKey = `${node.id}.__frame_count__`;
            const frameCountRaw = message.results?.[frameCountKey];
            const frameCount = typeof frameCountRaw === 'number' ? frameCountRaw : undefined;
            if (frameCount && updatedData.propertyDefs) {
              const updatedPropertyDefs = updatedData.propertyDefs.map((prop) => {
                if (prop.widget === 'seekbar' && prop.max !== frameCount) {
                  return { ...prop, max: frameCount };
                }
                return prop;
              });
              if (JSON.stringify(updatedPropertyDefs) !== JSON.stringify(updatedData.propertyDefs)) {
                updatedData = { ...updatedData, propertyDefs: updatedPropertyDefs };
                hasUpdate = true;
              }
            }

            // 数値出力ポートの値を更新（int/float）
            const numericOutputPorts = node.data.data.outputs.filter(
              (p) => p.data_type === 'int' || p.data_type === 'float'
            );
            if (numericOutputPorts.length > 0) {
              const numericOutputs: Record<string, number> = {};
              for (const port of numericOutputPorts) {
                const outputKey = `${node.id}.${port.name}`;
                const outputValue = message.results?.[outputKey];
                if (typeof outputValue === 'number') {
                  numericOutputs[port.name] = outputValue;
                }
              }
              if (Object.keys(numericOutputs).length > 0) {
                updatedData = { ...updatedData, numericOutputs };
                hasUpdate = true;
              }
            }

            // 文字列出力ポートの値を更新（string）
            const stringOutputPorts = node.data.data.outputs.filter(
              (p) => p.data_type === 'string'
            );
            if (stringOutputPorts.length > 0) {
              const stringOutputs: Record<string, string> = {};
              for (const port of stringOutputPorts) {
                const outputKey = `${node.id}.${port.name}`;
                const outputValue = message.results?.[outputKey];
                if (typeof outputValue === 'string') {
                  stringOutputs[port.name] = outputValue;
                }
              }
              if (Object.keys(stringOutputs).length > 0) {
                updatedData = { ...updatedData, stringOutputs };
                hasUpdate = true;
              }
            }

            // テキスト表示ノード用の表示テキストを更新
            const displayTextKey = `${node.id}.__display_text__`;
            const displayTextRaw = message.results?.[displayTextKey];
            if (typeof displayTextRaw === 'string') {
              updatedData = { ...updatedData, displayText: displayTextRaw };
              hasUpdate = true;
            }

            // ビジー状態を更新
            const isBusyKey = `${node.id}.__is_busy__`;
            const isBusyRaw = message.results?.[isBusyKey];
            const isBusyValue = isBusyRaw === 1 || isBusyRaw === true;
            if (isBusyValue) {
              if (!updatedData.isBusy) {
                updatedData = { ...updatedData, isBusy: true };
                hasUpdate = true;
              }
            } else if (updatedData.isBusy) {
              updatedData = { ...updatedData, isBusy: false };
              hasUpdate = true;
            }

            // 接続されたプロパティ値を更新
            const nodeConnectedProps = message.connected_properties?.[node.id];
            if (nodeConnectedProps) {
              updatedData = { ...updatedData, connectedProperties: nodeConnectedProps };
              hasUpdate = true;
            } else if (updatedData.connectedProperties) {
              // 接続が解除された場合はクリア
              updatedData = { ...updatedData, connectedProperties: undefined };
              hasUpdate = true;
            }

            if (hasUpdate) {
              return {
                ...node,
                data: {
                  ...node.data,
                  data: updatedData,
                },
              };
            }
            return node;
          });
        });
      }

      if (message.type === 'error') {
        console.error('Execution error:', message.message);
      }

      // 動画終了メッセージを受信したらSTOP
      if (message.type === 'ended') {
        window.dispatchEvent(new CustomEvent('playback-ended'));
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    return () => {
      ws.close();
    };
  }, [isLoading, wsUrl, setNodes]);

  // グラフが変更されたらバックエンドに送信
  const playbackStateRef = useRef(playbackState);
  playbackStateRef.current = playbackState;
  const intervalMsRef = useRef(intervalMs);
  intervalMsRef.current = intervalMs;
  const previewTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    if (playbackStateRef.current === 'running') {
      // 実行中はスロットリングして送信
      // 処理時間がインターバルを超える場合は処理時間に合わせる
      const now = Date.now();
      const effectiveInterval = Math.max(intervalMsRef.current, elapsedMsRef.current);
      const minInterval = effectiveInterval / 2;
      const elapsed = now - lastSetGraphTimeRef.current;

      if (elapsed >= minInterval) {
        // 十分な時間が経過していれば即座に送信
        const graphData = buildGraphData(nodes, edges);
        ws.send(JSON.stringify({ type: 'set_graph', graph: graphData }));
        lastSetGraphTimeRef.current = now;
        pendingSetGraphRef.current = false;
        if (setGraphTimeoutRef.current) {
          clearTimeout(setGraphTimeoutRef.current);
          setGraphTimeoutRef.current = null;
        }
      } else {
        // まだ時間が経っていなければ、次のタイミングで最新データを送信
        pendingSetGraphRef.current = true;
        if (!setGraphTimeoutRef.current) {
          setGraphTimeoutRef.current = window.setTimeout(() => {
            setGraphTimeoutRef.current = null;
            if (pendingSetGraphRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
              // refから最新のデータを取得
              const graphData = buildGraphData(nodesRef.current, edgesRef.current);
              wsRef.current.send(JSON.stringify({ type: 'set_graph', graph: graphData }));
              lastSetGraphTimeRef.current = Date.now();
              pendingSetGraphRef.current = false;
            }
          }, minInterval - elapsed);
        }
      }
    } else {
      // 停止中はデバウンスしてexecute_onceでプレビュー更新
      if (previewTimeoutRef.current) {
        clearTimeout(previewTimeoutRef.current);
      }
      previewTimeoutRef.current = window.setTimeout(() => {
        const graphData = buildGraphData(nodes, edges);
        ws.send(JSON.stringify({ type: 'execute_once', graph: graphData }));
      }, 100);
    }
  }, [nodes, edges]);

  // 動的ポートノードの connectedPortIds を更新（エッジ変更時のみ）
  const prevEdgesRef = useRef<Edge[]>([]);
  useEffect(() => {
    // エッジが実際に変更されたかチェック（参照比較ではなく内容比較）
    const prevEdges = prevEdgesRef.current;
    const edgesChanged = edges.length !== prevEdges.length ||
      edges.some((e, i) => e.id !== prevEdges[i]?.id ||
        e.targetHandle !== prevEdges[i]?.targetHandle);

    if (!edgesChanged) return;
    prevEdgesRef.current = edges;

    setNodes((currentNodes) => {
      const dynamicPortNodes = currentNodes.filter((n) => n.data.data.dynamicPorts);
      if (dynamicPortNodes.length === 0) return currentNodes;

      let needsUpdate = false;
      const updatedNodes = currentNodes.map((node) => {
        if (!node.data.data.dynamicPorts) return node;

        // このノードへの接続からポートIDを収集
        const connectedIds: string[] = [];
        for (const edge of edges) {
          if (edge.target === node.id && edge.targetHandle) {
            connectedIds.push(edge.targetHandle);
          }
        }

        // 現在のconnectedPortIdsと比較して変更があるか確認
        const currentIds = node.data.data.connectedPortIds || [];
        const isSame = connectedIds.length === currentIds.length &&
          connectedIds.every((id, i) => id === currentIds[i]);

        if (isSame) return node;

        needsUpdate = true;
        return {
          ...node,
          data: {
            ...node.data,
            data: {
              ...node.data.data,
              connectedPortIds: connectedIds,
            },
          },
        };
      });

      return needsUpdate ? updatedNodes : currentNodes;
    });
  }, [edges, setNodes]);

  // 接続の型チェック
  const isValidConnection = useCallback(
    (connection: Edge | Connection) => {
      // 自己接続を禁止
      if (connection.source === connection.target) return false;

      const sourceNode = nodes.find((n) => n.id === connection.source);
      const targetNode = nodes.find((n) => n.id === connection.target);
      if (!sourceNode || !targetNode) return false;

      const sourcePort = sourceNode.data.data.outputs.find(
        (p) => p.id === connection.sourceHandle
      );
      const targetPort = targetNode.data.data.inputs.find(
        (p) => p.id === connection.targetHandle
      );
      if (!sourcePort || !targetPort) return false;

      // N:1接続を禁止（1つの入力ポートに複数の出力を接続不可）
      const existingConnection = edges.find(
        (e) => e.target === connection.target && e.targetHandle === connection.targetHandle
      );
      if (existingConnection) return false;

      // 同じデータ型、またはint/float間の接続を許可
      if (sourcePort.data_type === targetPort.data_type) {
        return true;
      }
      // int と float は相互接続可能
      const numericTypes = ['int', 'float'];
      if (numericTypes.includes(sourcePort.data_type) && numericTypes.includes(targetPort.data_type)) {
        return true;
      }
      return false;
    },
    [nodes, edges]
  );

  // 接続追加
  // 注: グラフの送信はuseEffect([nodes, edges])に任せる（行750-768）
  // これにより、ノード追加と接続追加が連続で発生しても、最新の状態で送信される
  // START中は操作を無効化
  const isRunning = playbackState === 'running';

  const onConnect = useCallback(
    (params: Connection) => {
      if (playbackState === 'running') return; // START中は接続を無効化
      if (!isValidConnection(params)) return;
      setEdges((eds) => addEdge(params, eds));
    },
    [setEdges, isValidConnection, playbackState]
  );

  // エッジ変更ハンドラ（動的ポートの詰め直し処理を含む）
  const handleEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      // まず通常のエッジ変更を適用
      onEdgesChange(changes);

      // 削除されたエッジをチェック
      const removedEdges = changes.filter((c) => c.type === 'remove');
      if (removedEdges.length === 0) return;

      // 動的ポートを持つノードへのエッジが削除されたかチェック
      setEdges((currentEdges) => {
        // 削除対象のエッジIDを収集
        const removedIds = new Set(removedEdges.map((c) => 'id' in c ? c.id : ''));

        // 各動的ポートノードについて詰め直しを行う
        const dynamicPortNodes = nodes.filter((n) => n.data.data.dynamicPorts);
        let needsUpdate = false;
        let updatedEdges = [...currentEdges];

        for (const node of dynamicPortNodes) {
          const prefix = node.data.data.dynamicPorts!;

          // このノードへのエッジで削除されたものがあるかチェック
          const removedEdgeToNode = edges.filter(
            (e) => e.target === node.id && removedIds.has(e.id)
          );
          if (removedEdgeToNode.length === 0) continue;

          // このノードの動的ポートに接続されているエッジを収集
          const nodeEdges = updatedEdges.filter((e) => e.target === node.id);

          // 動的ポート番号とエッジのマッピング
          const portEdges: { portNum: number; edge: Edge }[] = [];
          for (const edge of nodeEdges) {
            const port = node.data.data.inputs.find((p: GraphPort) => p.id === edge.targetHandle);
            if (!port) continue;
            const match = port.name.match(new RegExp(`^${prefix} (\\d+)$`));
            if (match) {
              portEdges.push({ portNum: parseInt(match[1], 10), edge });
            }
          }

          // 番号順にソート
          portEdges.sort((a, b) => a.portNum - b.portNum);

          // 詰め直し: 番号を1から振り直す
          let newNum = 1;
          for (const { portNum, edge } of portEdges) {
            if (portNum !== newNum) {
              // 新しいポートを探す
              const newPort = node.data.data.inputs.find(
                (p: GraphPort) => p.name === `${prefix} ${newNum}`
              );
              if (newPort) {
                // エッジのtargetHandleを更新
                updatedEdges = updatedEdges.map((e) =>
                  e.id === edge.id ? { ...e, targetHandle: newPort.id } : e
                );
                needsUpdate = true;
              }
            }
            newNum++;
          }
        }

        // エッジ更新後にconnectedPortIdsも即座に更新
        if (needsUpdate || removedEdges.length > 0) {
          const finalEdges = needsUpdate ? updatedEdges : currentEdges;
          setNodes((currentNodes) => {
            return currentNodes.map((node) => {
              if (!node.data.data.dynamicPorts) return node;

              // このノードへの接続からポートIDを収集
              const connectedIds: string[] = [];
              for (const edge of finalEdges) {
                if (edge.target === node.id && edge.targetHandle) {
                  connectedIds.push(edge.targetHandle);
                }
              }

              return {
                ...node,
                data: {
                  ...node.data,
                  data: {
                    ...node.data.data,
                    connectedPortIds: connectedIds,
                  },
                },
              };
            });
          });
        }

        return needsUpdate ? updatedEdges : currentEdges;
      });
    },
    [onEdgesChange, nodes, edges, setEdges, setNodes]
  );

  // ドラッグ＆ドロップでノード追加
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      // START中はノード追加を無効化
      if (playbackState === 'running') return;

      const defId = event.dataTransfer.getData('application/reactflow');
      if (!defId) return;

      const def = definitions.find((d) => d.definition_id === defId);
      if (!def) return;

      // Webcamノードかどうかを判定（バックエンド側のcv2を使うもの）
      const isWebcamNode = defId === 'image.input.webcam';

      setNodes((nds) => {
        let position: { x: number; y: number };

        // 初回ドロップ時（ノードがない場合）は固定位置を使用
        // fitViewがビューポートを変更するため
        if (nds.length === 0) {
          position = { x: 0, y: 0 };
        } else {
          // 2回目以降はスクリーン座標をフロー座標に変換
          const instance = reactFlowInstance.current;
          if (!instance) {
            position = { x: 100, y: 100 };
          } else {
            position = instance.screenToFlowPosition({
              x: event.clientX,
              y: event.clientY,
            });
          }
        }

        const newNode = createNodeFromDefinition(def, position, nds, apiKeysStatusRef.current);
        const newNodes = [...nds, newNode];

        // Webcamノードが追加された場合、プレビュー実行してカメラを事前初期化
        if (isWebcamNode) {
          setTimeout(() => {
            const ws = wsRef.current;
            if (ws && ws.readyState === WebSocket.OPEN) {
              const graphData = buildGraphData(newNodes, edges);
              ws.send(JSON.stringify({ type: 'execute_once', graph: graphData }));
            }
          }, 100);
        }

        return newNodes;
      });
    },
    [definitions, setNodes, edges]
  );

  // 再生開始
  const handleStart = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // スロットリング状態をリセット
    lastSetGraphTimeRef.current = 0;
    pendingSetGraphRef.current = false;
    if (setGraphTimeoutRef.current) {
      clearTimeout(setGraphTimeoutRef.current);
      setGraphTimeoutRef.current = null;
    }

    // RTSPノードのimageDataをクリアしてLoading表示にする
    setNodes((currentNodes) =>
      currentNodes.map((node) => {
        if (node.data?.data?.definitionId === 'image.input.rtsp') {
          return {
            ...node,
            data: {
              ...node.data,
              data: {
                ...node.data.data,
                imageData: undefined,
              },
            },
          };
        }
        return node;
      })
    );

    const graphData = buildGraphData(nodes, edges);
    ws.send(JSON.stringify({ type: 'set_graph', graph: graphData }));
    ws.send(JSON.stringify({ type: 'start', interval_ms: intervalMs, loop: loopPlayback }));
    setPlaybackState('running');
  }, [nodes, edges, intervalMs, loopPlayback]);

  // 一時停止
  const handlePause = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // スロットリング状態をクリア
    if (setGraphTimeoutRef.current) {
      clearTimeout(setGraphTimeoutRef.current);
      setGraphTimeoutRef.current = null;
    }

    // Pause時の計測結果をスナップショットとして保存
    setPausedElapsedMs(elapsedMs);
    setPausedNodeTimes({ ...nodeTimes });
    setPausedGuiOverheadMs(guiOverheadMs);

    ws.send(JSON.stringify({ type: 'stop' }));
    setPlaybackState('paused');
  }, [elapsedMs, nodeTimes, guiOverheadMs]);

  // 停止（先頭に戻る）
  const handleStop = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // スロットリング状態をクリア
    if (setGraphTimeoutRef.current) {
      clearTimeout(setGraphTimeoutRef.current);
      setGraphTimeoutRef.current = null;
    }

    ws.send(JSON.stringify({ type: 'stop' }));
    ws.send(JSON.stringify({ type: 'reset' }));
    setPlaybackState('stopped');
  }, []);

  // ノード右クリックメニュー表示
  const handleNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node<{ data: CustomNodeData }>) => {
      // START中は無効化
      if (playbackState === 'running') return;

      event.preventDefault();
      setContextMenu({
        nodeId: node.id,
        x: event.clientX,
        y: event.clientY,
      });
    },
    [playbackState]
  );

  // コンテキストメニューを閉じる
  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  // ノード複製
  const handleDuplicateNode = useCallback(() => {
    if (!contextMenu) return;

    const sourceNode = nodes.find((n) => n.id === contextMenu.nodeId);
    if (!sourceNode) {
      closeContextMenu();
      return;
    }

    // 複製不可フラグをチェック
    if (sourceNode.data.data.noDuplicate) {
      closeContextMenu();
      return;
    }

    // ノード定義を取得
    const def = definitions.find((d) => d.definition_id === sourceNode.data.data.definitionId);
    if (!def) {
      closeContextMenu();
      return;
    }

    // 新しい位置（元のノードから少しずらす）
    const newPosition = {
      x: sourceNode.position.x + 50,
      y: sourceNode.position.y + 50,
    };

    // 新しいノードを作成
    const newNode = createNodeFromDefinition(def, newPosition, nodes, apiKeysStatusRef.current);

    // プロパティをコピー
    newNode.data.data.properties = { ...sourceNode.data.data.properties };

    setNodes((nds) => [...nds, newNode]);
    closeContextMenu();
  }, [contextMenu, nodes, definitions, setNodes, closeContextMenu]);

  // ノード削除
  const handleDeleteNode = useCallback(() => {
    if (!contextMenu) return;

    // ノードを削除
    setNodes((nds) => nds.filter((n) => n.id !== contextMenu.nodeId));
    // 関連するエッジも削除
    setEdges((eds) =>
      eds.filter((e) => e.source !== contextMenu.nodeId && e.target !== contextMenu.nodeId)
    );
    closeContextMenu();
  }, [contextMenu, setNodes, setEdges, closeContextMenu]);

  // コメント追加/編集（モーダルを開く）
  const handleAddComment = useCallback(() => {
    if (!contextMenu) return;

    const targetNode = nodes.find((n) => n.id === contextMenu.nodeId);
    if (!targetNode) {
      closeContextMenu();
      return;
    }

    // モーダルを開く
    const currentComment = targetNode.data.data.comment || '';
    setCommentModal({ nodeId: contextMenu.nodeId, value: currentComment });
    closeContextMenu();
  }, [contextMenu, nodes, closeContextMenu]);

  // コメントモーダル確定
  const handleCommentSubmit = useCallback(() => {
    if (!commentModal) return;

    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === commentModal.nodeId) {
          return {
            ...n,
            data: {
              ...n.data,
              data: {
                ...n.data.data,
                comment: commentModal.value || undefined,
              },
            },
          };
        }
        return n;
      })
    );
    setCommentModal(null);
  }, [commentModal, setNodes]);

  // コメントモーダルキャンセル
  const handleCommentCancel = useCallback(() => {
    setCommentModal(null);
  }, []);

  // クリックでコンテキストメニューを閉じる
  useEffect(() => {
    const handleClick = () => {
      if (contextMenu) {
        closeContextMenu();
      }
    };

    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [contextMenu, closeContextMenu]);

  // playbackState/intervalMs変更時にノードのisStreaming/isPaused/intervalMsを更新
  useEffect(() => {
    const isStreaming = playbackState === 'running';
    const isPaused = playbackState === 'paused';
    setNodes((currentNodes) =>
      currentNodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          data: {
            ...node.data.data,
            isStreaming,
            isPaused,
            intervalMs,
          },
        },
      }))
    );
  }, [playbackState, intervalMs, setNodes]);

  // ファイルアップロード時にSTOPするイベントリスナー
  useEffect(() => {
    const handleFileUploadStop = () => {
      if (playbackStateRef.current !== 'stopped') {
        handleStop();
      }
    };

    window.addEventListener('node-file-upload', handleFileUploadStop);
    return () => {
      window.removeEventListener('node-file-upload', handleFileUploadStop);
    };
  }, [handleStop]);

  // 動画終了時にSTOPするイベントリスナー
  useEffect(() => {
    const handlePlaybackEnded = () => {
      if (playbackStateRef.current === 'running') {
        handleStop();
      }
    };

    window.addEventListener('playback-ended', handlePlaybackEnded);
    return () => {
      window.removeEventListener('playback-ended', handlePlaybackEnded);
    };
  }, [handleStop]);

  // Webcamカメラ選択変更時にプレビュー実行してカメラを再初期化
  useEffect(() => {
    const handleCameraChanged = () => {
      // 短いdelayでプロパティ更新完了を待ってから実行
      setTimeout(() => {
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          const graphData = buildGraphData(nodes, edges);
          ws.send(JSON.stringify({ type: 'execute_once', graph: graphData }));
        }
      }, 50);
    };

    window.addEventListener('webcam-camera-changed', handleCameraChanged);
    return () => {
      window.removeEventListener('webcam-camera-changed', handleCameraChanged);
    };
  }, [nodes, edges]);

  // Webcamノードの有無を追跡
  const hadWebcamRef = useRef(false);

  // 常時プレビューが必要なノードのdefinitionIdリスト
  // 注意: ImagePreview.tsx の ALWAYS_PREVIEW_NODES と同期すること
  const alwaysPreviewNodeIds = [
    'image.input.webcam',
    'image.input.browser_webcam',
    'image.input.image',
    'image.input.video',
    'image.input.video_frame',
    'image.input.url_image',
  ];

  // 常時プレビューノードがある場合、STOP中でも常時プレビュー実行
  useEffect(() => {
    // Webcamノードがあるかチェック（カメラ解放用）
    const hasWebcam = nodes.some(
      (n) => n.data?.data?.definitionId === 'image.input.webcam'
    );

    // Webcamノードがなくなった場合、カメラを解放
    if (hadWebcamRef.current && !hasWebcam) {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'release_webcams' }));
      }
    }
    hadWebcamRef.current = hasWebcam;

    // STARTしている場合は通常の実行ループで処理されるのでスキップ
    if (playbackState === 'running') return;

    // 常時プレビューが必要なノードがあるかチェック
    const hasAlwaysPreviewNode = nodes.some(
      (n) => alwaysPreviewNodeIds.includes(n.data?.data?.definitionId)
    );

    if (!hasAlwaysPreviewNode) return;

    // 定期的にexecute_onceを実行
    const intervalId = setInterval(() => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        const graphData = buildGraphData(nodes, edges);
        ws.send(JSON.stringify({ type: 'execute_once', graph: graphData }));
      }
    }, 100);

    return () => {
      clearInterval(intervalId);
    };
  }, [nodes, edges, playbackState]);

  // テキスト接続のリアルタイム更新（START不要）
  useEffect(() => {
    // string型の接続を見つけて、テキスト表示ノードを更新
    const stringConnections = edges.filter((edge) => {
      const sourceNode = nodes.find((n) => n.id === edge.source);
      const targetNode = nodes.find((n) => n.id === edge.target);
      if (!sourceNode || !targetNode) return false;

      // 出力ポートがstring型かチェック
      const sourcePort = sourceNode.data.data.outputs.find(
        (p) => p.id === edge.sourceHandle && p.data_type === 'string'
      );
      // 入力ポートがstring型かチェック
      const targetPort = targetNode.data.data.inputs.find(
        (p) => p.id === edge.targetHandle && p.data_type === 'string'
      );
      return sourcePort && targetPort;
    });

    if (stringConnections.length === 0) return;

    // 更新が必要なノードを収集
    const updates: { nodeId: string; text: string }[] = [];

    for (const edge of stringConnections) {
      const sourceNode = nodes.find((n) => n.id === edge.source);
      const targetNode = nodes.find((n) => n.id === edge.target);
      if (!sourceNode || !targetNode) continue;

      // ソースポートの名前を取得
      const sourcePort = sourceNode.data.data.outputs.find(
        (p) => p.id === edge.sourceHandle && p.data_type === 'string'
      );
      if (!sourcePort) continue;

      // ターゲットノードがテキスト表示ノードかチェック
      const hasTextInputWidget = targetNode.data.data.propertyDefs?.some(
        (p) => p.widget === 'text_area' || p.widget === 'text_input'
      );
      if (!hasTextInputWidget && targetNode.data.data.inputs.some((p) => p.data_type === 'string')) {
        // まずstringOutputsから値を取得（バックエンドからの出力値）
        const stringOutputValue = sourceNode.data.data.stringOutputs?.[sourcePort.name];
        if (stringOutputValue !== undefined) {
          updates.push({ nodeId: targetNode.id, text: stringOutputValue });
          continue;
        }

        // stringOutputsにない場合、text_area/text_inputプロパティから値を取得
        const textProp = sourceNode.data.data.propertyDefs?.find(
          (p) => (p.widget === 'text_area' || p.widget === 'text_input') &&
                 (p.name === sourcePort.name || `${p.name}_out` === sourcePort.name)
        );
        if (textProp) {
          const text = (sourceNode.data.data.properties[textProp.name] as string) ?? '';
          updates.push({ nodeId: targetNode.id, text });
        }
      }
    }

    if (updates.length === 0) return;

    // 実際に更新が必要なノードがあるかチェック
    const needsUpdate = updates.some((update) => {
      const node = nodes.find((n) => n.id === update.nodeId);
      return node && node.data.data.displayText !== update.text;
    });

    if (!needsUpdate) return;

    // displayTextを更新
    setNodes((currentNodes) =>
      currentNodes.map((node) => {
        const update = updates.find((u) => u.nodeId === node.id);
        if (update && node.data.data.displayText !== update.text) {
          return {
            ...node,
            data: {
              ...node.data,
              data: {
                ...node.data.data,
                displayText: update.text,
              },
            },
          };
        }
        return node;
      })
    );
  }, [nodes, edges, setNodes]);

  // ドラッグ開始
  const onDragStart = (event: React.DragEvent, defId: string) => {
    event.dataTransfer.setData('application/reactflow', defId);
    event.dataTransfer.effectAllowed = 'move';
  };

  // グラフを保存
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSave = useCallback(() => {
    const graphData = {
      nodes: nodes.map((node) => ({
        id: node.id,
        type: node.type,
        position: node.position,
        data: node.data,
      })),
      edges: edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle,
        targetHandle: edge.targetHandle,
      })),
    };

    const blob = new Blob([JSON.stringify(graphData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);

    // タイムスタンプ付きファイル名を生成
    const now = new Date();
    const timestamp = now.getFullYear().toString() +
      (now.getMonth() + 1).toString().padStart(2, '0') +
      now.getDate().toString().padStart(2, '0') + '_' +
      now.getHours().toString().padStart(2, '0') +
      now.getMinutes().toString().padStart(2, '0');

    const a = document.createElement('a');
    a.href = url;
    a.download = `graph_${timestamp}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [nodes, edges]);

  // キーボードショートカット用にハンドラーをrefに登録
  keyboardHandlersRef.current = {
    handleStart,
    handleStop,
    handlePause,
    handleUndo,
    handleRedo,
    handleAutoLayout,
    handleSave,
    handleLoad: () => fileInputRef.current?.click(),
    getPlaybackState: () => playbackState,
  };

  // グラフを読み込み
  const handleLoad = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      // 再生中なら停止
      handleStop();

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target?.result as string;
          const graphData = JSON.parse(content);

          if (graphData.nodes && Array.isArray(graphData.nodes)) {
            // ノード定義に基づいてポート情報をマイグレーション
            const migratedNodes = graphData.nodes.map((node: Node<{ data: CustomNodeData }>) => {
              const defId = node.data?.data?.definitionId;
              const def = definitions.find((d) => d.definition_id === defId);
              if (!def) return node;

              // 現在のノード定義からポート情報を取得
              const portMap = new Map<string, GraphPort>();
              for (const p of def.inputs) {
                if (!portMap.has(p.name)) {
                  // 既存のポートIDを保持（接続維持のため）
                  const existingPort = node.data.data.inputs?.find((i: GraphPort) => i.name === p.name);
                  portMap.set(p.name, {
                    id: existingPort?.id || generateId('port'),
                    name: p.name,
                    data_type: p.data_type,
                    direction: p.direction,
                    preview: p.preview ?? true,
                  });
                }
              }
              for (const p of def.outputs) {
                if (!portMap.has(p.name)) {
                  const existingPort = node.data.data.outputs?.find((o: GraphPort) => o.name === p.name);
                  portMap.set(p.name, {
                    id: existingPort?.id || generateId('port'),
                    name: p.name,
                    data_type: p.data_type,
                    direction: p.direction,
                    preview: p.preview ?? true,
                  });
                }
              }

              const inputs: GraphPort[] = def.inputs.map(p => portMap.get(p.name)!);
              const outputs: GraphPort[] = def.outputs.map(p => portMap.get(p.name)!);

              // WebcamノードはimageDataをクリアしてLoading表示にする（RTSPはSTART時にクリア）
              const shouldClearImage = defId === 'image.input.webcam';

              // resizable=trueのノードはスタイルを設定
              const isResizable = def.resizable === true;
              const nodeStyle = isResizable && !node.style
                ? { width: 228 }
                : node.style;

              // 複製不可フラグ
              const noDuplicate = def.no_duplicate === true;

              // 動的ポートプレフィックス
              const dynamicPorts = def.dynamic_ports || undefined;

              return {
                ...node,
                type: 'custom',
                style: nodeStyle,
                data: {
                  ...node.data,
                  data: {
                    ...node.data.data,
                    resizable: isResizable,
                    noDuplicate,
                    dynamicPorts,
                    inputs,
                    outputs,
                    propertyDefs: def.properties || node.data.data.propertyDefs,
                    apiKeysStatus: apiKeysStatusRef.current,
                    ...(shouldClearImage ? { imageData: undefined } : {}),
                  },
                },
              };
            });
            setNodes(migratedNodes);
          }
          if (graphData.edges && Array.isArray(graphData.edges)) {
            setEdges(graphData.edges);
          }
        } catch (err) {
          console.error('Failed to load graph:', err);
          alert('グラフの読み込みに失敗しました');
        }
      };
      reader.readAsText(file);

      // 同じファイルを再度選択できるようにリセット
      event.target.value = '';
    },
    [setNodes, setEdges, definitions, handleStop]
  );

  // definition_idからカテゴリツリーを構築
  const categoryTree = useMemo(() => {
    const root: CategoryTreeNode = {
      name: 'root',
      path: '',
      children: new Map(),
      nodes: [],
      order: 0,
    };

    const categoryMap = new Map(categories.map((c) => [c.category_id, c]));

    definitions.forEach((def) => {
      const parts = def.definition_id.split('.');
      // 最後の要素はノード名、それ以外はカテゴリパス
      const categoryParts = parts.slice(0, -1);

      let current = root;
      let pathSoFar = '';

      categoryParts.forEach((part) => {
        pathSoFar = pathSoFar ? `${pathSoFar}.${part}` : part;

        if (!current.children.has(part)) {
          const catDef = categoryMap.get(pathSoFar);
          current.children.set(part, {
            name: part,
            path: pathSoFar,
            children: new Map(),
            nodes: [],
            order: catDef?.order ?? 100,
          });
        }
        current = current.children.get(part)!;
      });

      current.nodes.push(def);
    });

    // 各レベルでノードをorder順にソート
    const sortNodes = (node: CategoryTreeNode) => {
      node.nodes.sort((a, b) => a.order - b.order);
      node.children.forEach(sortNodes);
    };
    sortNodes(root);

    return root;
  }, [definitions, categories]);

  // カテゴリの開閉状態（パスをキーとして使用）
  const [openCategories, setOpenCategories] = useState<Record<string, boolean>>({});

  // カテゴリ取得後にdefault_openで初期状態を設定
  useEffect(() => {
    if (categories.length === 0 && definitions.length === 0) return;

    const initialState: Record<string, boolean> = {};
    const categoryMap = new Map(categories.map((c) => [c.category_id, c]));

    // ツリーを走査して全カテゴリパスを収集
    const collectPaths = (node: CategoryTreeNode) => {
      node.children.forEach((child) => {
        const catDef = categoryMap.get(child.path);
        initialState[child.path] = catDef?.default_open ?? true;
        collectPaths(child);
      });
    };
    collectPaths(categoryTree);

    setOpenCategories(initialState);
  }, [categories, definitions, categoryTree]);

  const toggleCategory = (categoryPath: string, event: React.MouseEvent<HTMLButtonElement>) => {
    const isCurrentlyOpen = openCategories[categoryPath] ?? true;
    setOpenCategories((prev) => ({
      ...prev,
      [categoryPath]: !prev[categoryPath],
    }));

    // カテゴリを開く場合、ヘッダーを上端にスクロール
    if (!isCurrentlyOpen) {
      const target = event.currentTarget;
      setTimeout(() => {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 50);
    }
  };

  return (
    <>
      {/* ローディングオーバーレイ */}
      {isLoading && (
        <div className={`loading-overlay ${theme}`}>
          <div className="loading-spinner"></div>
          <div className="loading-text">Loading...</div>
        </div>
      )}

      <div className={`app-container ${theme}`}>
      {/* サイドバー */}
      <div className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <button
          className="sidebar-toggle"
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          title={sidebarCollapsed ? 'Open sidebar' : 'Close sidebar'}
        >
          {sidebarCollapsed ? '▶' : '◀'}
        </button>
        {!sidebarCollapsed && (
          <>
        <h2>Control</h2>
        <div className="control-buttons">
          {playbackState === 'running' ? (
            <button className="control-button" onClick={handlePause} title="Pause (Ctrl+P)">
              Pause
            </button>
          ) : (
            <button className="control-button" onClick={handleStart} title="Start (Ctrl+Enter)">
              Start
            </button>
          )}
          <button
            className="control-button stop"
            onClick={handleStop}
            disabled={playbackState === 'stopped'}
            title="Stop (Escape)"
          >
            Stop
          </button>
        </div>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={loopPlayback}
            onChange={(e) => {
              const newValue = e.target.checked;
              setLoopPlayback(newValue);
              // 再生中ならバックエンドにも通知
              const ws = wsRef.current;
              if (ws && ws.readyState === WebSocket.OPEN && playbackStateRef.current === 'running') {
                ws.send(JSON.stringify({ type: 'set_loop', loop: newValue }));
              }
            }}
          />
          Loop Playback
        </label>

        <h2 style={{ marginTop: 24 }}>Nodes</h2>
        <input
          type="text"
          className="node-search-input"
          placeholder="Search nodes..."
          value={nodeSearchQuery}
          onChange={(e) => setNodeSearchQuery(e.target.value)}
        />
        {(() => {
          // 検索クエリの正規化（小文字に変換）
          const searchQuery = nodeSearchQuery.toLowerCase().trim();

          // ノードが検索条件に一致するかチェック
          const matchesSearch = (def: NodeDefinition) => {
            if (!searchQuery) return true;
            return def.display_name.toLowerCase().includes(searchQuery);
          };

          // カテゴリ内にマッチするノードがあるかを再帰的にチェック
          const hasMatchingNodes = (node: CategoryTreeNode): boolean => {
            if (node.nodes.some(matchesSearch)) return true;
            for (const child of node.children.values()) {
              if (hasMatchingNodes(child)) return true;
            }
            return false;
          };

          // 再帰的にカテゴリツリーをレンダリング
          const renderCategoryTree = (node: typeof categoryTree, depth: number = 0) => {
            const categoryMap = new Map(categories.map((c) => [c.category_id, c]));
            // 子カテゴリをorder順にソート
            const sortedChildren = Array.from(node.children.values()).sort(
              (a, b) => a.order - b.order
            );

            return sortedChildren.map((child) => {
              // 検索中はマッチするノードがないカテゴリをスキップ
              if (searchQuery && !hasMatchingNodes(child)) return null;

              const catDef = categoryMap.get(child.path);
              const displayName = catDef?.display_name ?? child.name.charAt(0).toUpperCase() + child.name.slice(1);
              // 検索中は自動的に展開
              const isOpen = searchQuery ? true : (openCategories[child.path] ?? true);
              const hasChildren = child.children.size > 0;

              // マッチするノードをフィルタ
              const filteredNodes = child.nodes.filter(matchesSearch);
              const hasNodes = filteredNodes.length > 0;

              return (
                <div key={child.path} className="category-group" style={{ marginLeft: depth * 8 }}>
                  <button
                    className="category-header"
                    onClick={(e) => toggleCategory(child.path, e)}
                  >
                    <span className={`category-arrow ${isOpen ? 'open' : ''}`}>
                      ▶
                    </span>
                    {displayName}
                  </button>
                  {isOpen && (
                    <div className="category-items">
                      {/* 子カテゴリを先に表示 */}
                      {hasChildren && renderCategoryTree(child, depth + 1)}
                      {/* ノードを表示 */}
                      {hasNodes && filteredNodes.map((def) => (
                        <button
                          key={def.definition_id}
                          className={`node-button ${isRunning ? 'disabled' : ''}`}
                          draggable={!isRunning}
                          onDragStart={(e) => onDragStart(e, def.definition_id)}
                        >
                          {def.display_name}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            });
          };

          return renderCategoryTree(categoryTree);
        })()}

        {showEditSection && (
          <>
            <h2 style={{ marginTop: 24 }}>Edit</h2>
            <div className="control-buttons">
              <button
                className="control-button"
                onClick={handleUndo}
                disabled={!canUndo || isRunning}
                title="Undo (Ctrl+Z)"
              >
                Undo
              </button>
              <button
                className="control-button"
                onClick={handleRedo}
                disabled={!canRedo || isRunning}
                title="Redo (Ctrl+Y)"
              >
                Redo
              </button>
            </div>
          </>
        )}

        {showAutoLayoutSection && (
          <>
            <h2 style={{ marginTop: 24 }}>Layout</h2>
            <div className="control-buttons">
              <button
                className="control-button"
                onClick={handleAutoLayout}
                title="Auto Layout (Ctrl+A)"
              >
                Auto Layout
              </button>
            </div>
          </>
        )}

        {showFileSection && (
          <>
            <h2 style={{ marginTop: 24 }}>File</h2>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleLoad}
              accept=".json"
              style={{ display: 'none' }}
            />
            <div className="control-buttons">
              <button className="control-button" onClick={handleSave} title="Save (Ctrl+S)">
                Save
              </button>
              <button className="control-button" onClick={() => fileInputRef.current?.click()} title="Load (Ctrl+L)">
                Load
              </button>
            </div>
          </>
        )}
          </>
        )}
      </div>

      {/* ReactFlowキャンバス */}
      <div className={`flow-container ${isRunning ? 'running' : ''}`}>
        {/* パフォーマンス表示（Running/Pause中に表示） */}
        {(() => {
          // Pause中はスナップショット、Running中はリアルタイム値を使用
          const displayElapsedMs = playbackState === 'paused' ? pausedElapsedMs : elapsedMs;
          const displayNodeTimes = playbackState === 'paused' ? pausedNodeTimes : nodeTimes;
          const displayGuiOverheadMs = playbackState === 'paused' ? pausedGuiOverheadMs : guiOverheadMs;
          const showPerformance = (playbackState === 'running' || playbackState === 'paused') && displayElapsedMs > 0;

          return showPerformance && (
            <div className="performance-overlay">
              <div className="performance-interval">
                <span>Exec Interval: {intervalMs} ms</span>
              </div>
              <div className="performance-total">
                <span style={{ color: displayElapsedMs > intervalMs ? '#ff6b6b' : 'inherit' }}>
                  {displayElapsedMs.toFixed(1)} ms
                </span>
                <span>{Math.round(1000 / displayElapsedMs)} FPS</span>
              </div>
              <div className="performance-nodes">
                {Object.entries(displayNodeTimes)
                  .sort((a, b) => a[1].order - b[1].order)
                  .map(([id, info]) => (
                    <div key={id} className="performance-node-row">
                      <span className="node-name">{info.name}</span>
                      <span className="node-time">{info.time.toFixed(1)} ms</span>
                    </div>
                  ))}
                {displayGuiOverheadMs > 0.1 && (
                  <div className="performance-node-row gui-overhead">
                    <span className="node-name">GUI Overhead</span>
                    <span className="node-time">{displayGuiOverheadMs.toFixed(1)} ms</span>
                  </div>
                )}
              </div>
            </div>
          );
        })()}
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={handleNodesChangeWithHistory}
          onEdgesChange={handleEdgesChange}
          onConnect={onConnect}
          onDragOver={onDragOver}
          onDrop={onDrop}
          nodeTypes={nodeTypes}
          isValidConnection={isValidConnection}
          deleteKeyCode={isRunning ? null : ['Delete', 'Backspace']}
          nodesConnectable={!isRunning}
          edgesReconnectable={!isRunning}
          fitView
          fitViewOptions={{ maxZoom: 1 }}
          minZoom={0.1}
          maxZoom={4}
          onInit={(instance) => { reactFlowInstance.current = instance; }}
          onNodeDragStart={() => setDragging(true)}
          onNodeDragStop={() => setDragging(false)}
          onNodeContextMenu={handleNodeContextMenu}
          proOptions={{ hideAttribution: true }}
        >
          <Controls showZoom={false} />
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
        </ReactFlow>

        {/* ノードコンテキストメニュー */}
        {contextMenu && (() => {
          const targetNode = nodes.find((n) => n.id === contextMenu.nodeId);
          const canDuplicate = targetNode && !targetNode.data.data.noDuplicate;
          const hasComment = targetNode && targetNode.data.data.comment;
          return (
            <div
              className="node-context-menu"
              style={{
                position: 'fixed',
                left: contextMenu.x,
                top: contextMenu.y,
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <button onClick={handleAddComment}>
                {hasComment ? 'Edit Comment' : 'Add Comment'}
              </button>
              {canDuplicate && (
                <button onClick={handleDuplicateNode}>Duplicate</button>
              )}
              <button onClick={handleDeleteNode}>Delete</button>
            </div>
          );
        })()}

        {/* コメント入力モーダル */}
        {commentModal && (
          <div className="comment-modal-overlay" onClick={handleCommentCancel}>
            <div className="comment-modal" onClick={(e) => e.stopPropagation()}>
              <div className="comment-modal-header">Comment</div>
              <textarea
                className="comment-modal-input"
                value={commentModal.value}
                onChange={(e) => setCommentModal({ ...commentModal, value: e.target.value })}
                placeholder="Enter comment..."
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleCommentSubmit();
                  } else if (e.key === 'Escape') {
                    handleCommentCancel();
                  }
                }}
              />
              <div className="comment-modal-buttons">
                <button className="comment-modal-btn cancel" onClick={handleCommentCancel}>
                  Cancel
                </button>
                <button className="comment-modal-btn submit" onClick={handleCommentSubmit}>
                  OK
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
    </>
  );
}

// ドラッグ状態コンテキストでラップしたAppコンポーネント
export default function App() {
  return (
    <DraggingProvider>
      <WebRTCConnectionProvider>
        <AppContent />
      </WebRTCConnectionProvider>
    </DraggingProvider>
  );
}