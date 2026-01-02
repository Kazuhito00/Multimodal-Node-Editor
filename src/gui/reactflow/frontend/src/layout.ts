import dagre from 'dagre';
import { Node, Edge } from '@xyflow/react';

// ノードのデフォルトサイズ（実際のサイズに合わせて調整）
const DEFAULT_NODE_WIDTH = 220;
const DEFAULT_NODE_HEIGHT = 200;

export interface LayoutOptions {
  direction?: 'TB' | 'BT' | 'LR' | 'RL';  // Top-Bottom, Left-Right など
  nodeSpacing?: number;  // ノード間の垂直間隔
  rankSpacing?: number;  // 階層間の水平間隔
}

/**
 * dagreを使用してノードを自動レイアウトする
 */
export function getLayoutedElements<T extends Record<string, unknown>>(
  nodes: Node<T>[],
  edges: Edge[],
  options: LayoutOptions = {}
): Node<T>[] {
  // ノードがない場合はそのまま返す
  if (nodes.length === 0) {
    return nodes;
  }

  const {
    direction = 'LR',
    nodeSpacing = 40,
    rankSpacing = 60,
  } = options;

  // dagreグラフを作成
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));

  // グラフの方向とスペーシングを設定
  // ranker: network-simplex はエッジ交差を最小化するアルゴリズム
  // edgesep: エッジ間の最小間隔
  dagreGraph.setGraph({
    rankdir: direction,
    nodesep: nodeSpacing,
    ranksep: rankSpacing,
    edgesep: 20,
    ranker: 'network-simplex',
  });

  // ノードをdagreグラフに追加（実際のサイズがあれば使用）
  nodes.forEach((node) => {
    const width = node.measured?.width ?? DEFAULT_NODE_WIDTH;
    const height = node.measured?.height ?? DEFAULT_NODE_HEIGHT;
    dagreGraph.setNode(node.id, { width, height });
  });

  // エッジをdagreグラフに追加
  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  // レイアウトを計算
  dagre.layout(dagreGraph);

  // 計算された位置をノードに適用
  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);

    // dagreから位置が取得できない場合は元の位置を維持
    if (!nodeWithPosition) {
      return node;
    }

    // 実際のサイズを使用して左上座標に変換
    const width = node.measured?.width ?? DEFAULT_NODE_WIDTH;
    const height = node.measured?.height ?? DEFAULT_NODE_HEIGHT;
    return {
      ...node,
      position: {
        // dagreは中心座標を返すので、左上座標に変換
        x: nodeWithPosition.x - width / 2,
        y: nodeWithPosition.y - height / 2,
      },
    };
  });

  return layoutedNodes;
}
