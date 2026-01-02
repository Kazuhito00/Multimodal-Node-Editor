import { Node, Edge } from '@xyflow/react';
import {
  NodeDefinition,
  GraphData,
  GraphNode,
  GraphConnection,
  GraphPort,
  CustomNodeData,
} from '../types';
import { generateId } from './api';

// ReactFlowノードからグラフデータを構築
export function buildGraphData(
  nodes: Node<{ data: CustomNodeData }>[],
  edges: Edge[]
): GraphData {
  const graphNodes: GraphNode[] = nodes.map((node) => ({
    id: node.id,
    definition_id: node.data.data.definitionId,
    definition_version: node.data.data.version,
    name: node.data.data.label,
    inputs: node.data.data.inputs,
    outputs: node.data.data.outputs,
    properties: node.data.data.properties,
    position: { x: node.position.x, y: node.position.y },
  }));

  const graphConnections: GraphConnection[] = edges.map((edge) => ({
    id: edge.id,
    from_node_id: edge.source,
    from_port_id: edge.sourceHandle || '',
    to_node_id: edge.target,
    to_port_id: edge.targetHandle || '',
  }));

  return {
    id: 'graph-main',
    nodes: graphNodes,
    connections: graphConnections,
  };
}

// ノード定義からReactFlowノードを作成
export function createNodeFromDefinition(
  def: NodeDefinition,
  position: { x: number; y: number },
  existingNodes: Node<{ data: CustomNodeData }>[],
  apiKeysStatus?: Record<string, boolean>
): Node<{ data: CustomNodeData }> {
  const nodeId = generateId('node');

  // inoutポートを考慮したポート作成（同じ名前のポートは同じIDを使用）
  const portMap = new Map<string, GraphPort>();

  // 入力ポートを処理
  for (const p of def.inputs) {
    if (!portMap.has(p.name)) {
      portMap.set(p.name, {
        id: generateId('port'),
        name: p.name,
        data_type: p.data_type,
        direction: p.direction,
        preview: p.preview ?? true,
      });
    }
  }

  // 出力ポートを処理（inoutの場合は同じポートを再利用）
  for (const p of def.outputs) {
    if (!portMap.has(p.name)) {
      portMap.set(p.name, {
        id: generateId('port'),
        name: p.name,
        data_type: p.data_type,
        direction: p.direction,
        preview: p.preview ?? true,
      });
    }
  }

  // inputs/outputsに振り分け
  const inputs: GraphPort[] = def.inputs.map(p => portMap.get(p.name)!);
  const outputs: GraphPort[] = def.outputs.map(p => portMap.get(p.name)!);

  // デフォルトプロパティ値を設定
  const defaultProperties: Record<string, unknown> = {};
  for (const prop of def.properties || []) {
    defaultProperties[prop.name] = prop.default;
  }

  // 同じdefinition_idを持つ既存ノードのラベルから使用中の番号を抽出
  const sameTypeNodes = existingNodes.filter(
    (n) => n.data.data.definitionId === def.definition_id
  );
  const baseName = def.display_name;

  // 使用中の番号を収集（番号なし=1として扱う）
  const usedNumbers = new Set<number>();
  for (const n of sameTypeNodes) {
    const nodeLabel = n.data.data.label;
    if (nodeLabel === baseName) {
      usedNumbers.add(1);
    } else {
      // "Canny 2", "Canny 3" などから番号を抽出
      const match = nodeLabel.match(new RegExp(`^${baseName} (\\d+)$`));
      if (match) {
        usedNumbers.add(parseInt(match[1], 10));
      }
    }
  }

  // 使われていない最小の番号を見つける
  let newNumber = 1;
  while (usedNumbers.has(newNumber)) {
    newNumber++;
  }

  const label = newNumber === 1 ? baseName : `${baseName} ${newNumber}`;

  // resizable=trueのノードは初期サイズを設定
  const isResizable = def.resizable === true;
  const style = isResizable ? { width: 228 } : undefined;

  // 複製不可フラグ
  const noDuplicate = def.no_duplicate === true;

  // 動的ポートプレフィックス
  const dynamicPorts = def.dynamic_ports || undefined;

  return {
    id: nodeId,
    type: 'custom',
    position,
    style,
    data: {
      data: {
        label,
        definitionId: def.definition_id,
        version: def.version,
        resizable: isResizable,
        noDuplicate,
        dynamicPorts,
        inputs,
        outputs,
        properties: defaultProperties,
        propertyDefs: def.properties || [],
        apiKeysStatus,
      },
    },
  };
}
