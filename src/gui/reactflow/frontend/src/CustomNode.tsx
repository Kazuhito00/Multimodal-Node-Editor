import { memo, useCallback, useMemo, useState } from 'react';
import { Handle, Position, useReactFlow, Node, NodeResizer } from '@xyflow/react';
import { CustomNodeData, PropertyDef } from './types';

// プロパティの表示条件をチェック
function isPropertyVisible(
  prop: PropertyDef,
  properties: Record<string, unknown>,
  apiKeysStatus?: Record<string, boolean>
): boolean {
  // APIキーが必要なプロパティで、そのキーが設定されていない場合は非表示
  if (prop.requires_api_key) {
    const keyName = prop.requires_api_key;
    if (!apiKeysStatus || !apiKeysStatus[keyName]) {
      return false;
    }
  }

  if (!prop.visible_when) {
    return true;
  }
  const refValue = properties[prop.visible_when.property];
  return prop.visible_when.values.includes(refValue as number | string | boolean);
}

// 共通ウィジェット
import {
  NumberInputWidget,
  TextInputWidget,
  TextAreaWidget,
  TextDisplayWidget,
  FilePickerWidget,
  SeekbarWidget,
  XYInputWidget,
  ColorPickerWidget,
  CheckboxWidget,
  Matrix3x3Widget,
  ButtonWidget,
} from './widgets';

// ノード固有コンポーネント
import {
  WaveformCanvas,
  TimerRing,
  BrowserSpeaker,
  BrowserMicrophone,
  BrowserWebcam,
  MaskCanvas,
  DrawCanvas,
  ImagePreview,
  CropCanvas,
  OmnidirectionalCanvas,
  PIPCanvas,
  PerspectiveCanvas,
} from './components';
import { WebRTCWebcam } from './components/WebRTCWebcam';
import { WebRTCMicrophone } from './components/WebRTCMicrophone';

// ノードデータの型
type NodeData = { data: CustomNodeData };

// カスタムノードコンポーネント
function CustomNode({ id, data, selected }: { id: string; data: NodeData; selected?: boolean }) {
  const nodeData = data.data;
  const { setNodes } = useReactFlow<Node<NodeData>>();

  // connectedPortIds は App.tsx から渡される（動的ポートノードのみ）
  const connectedPortIds = useMemo(() => {
    return new Set<string>(nodeData.connectedPortIds || []);
  }, [nodeData.connectedPortIds]);

  // プロパティ変更ハンドラ
  const handlePropertyChange = useCallback(
    (propName: string, value: number | string | boolean) => {
      setNodes((nodes) =>
        nodes.map((node) => {
          if (node.id === id) {
            return {
              ...node,
              data: {
                ...node.data,
                data: {
                  ...node.data.data,
                  properties: {
                    ...node.data.data.properties,
                    [propName]: value,
                  },
                },
              },
            };
          }
          return node;
        })
      );
    },
    [id, setNodes]
  );

  // リサイズハンドラ（resizable=trueのノード用）
  const handleResize = useCallback(
    (_event: unknown, params: { width: number; height: number }) => {
      setNodes((nodes) =>
        nodes.map((node) => {
          if (node.id === id) {
            return {
              ...node,
              style: { ...node.style, width: params.width, height: params.height },
            };
          }
          return node;
        })
      );
    },
    [id, setNodes]
  );

  // 画像プレビュー更新ハンドラ（ファイルアップロード時の最初のフレーム用）
  const handleImageUpdate = useCallback(
    (imageData: string) => {
      setNodes((nodes) =>
        nodes.map((node) => {
          if (node.id === id) {
            return {
              ...node,
              data: {
                ...node.data,
                data: {
                  ...node.data.data,
                  imageData,
                  // ウェーブフォームもリセット
                  audioData: undefined,
                },
              },
            };
          }
          return node;
        })
      );
    },
    [id, setNodes]
  );

  // フレーム数更新ハンドラ（シークバーのmax値を更新）
  const handleFrameCountUpdate = useCallback(
    (frameCount: number) => {
      setNodes((nodes) =>
        nodes.map((node) => {
          if (node.id === id) {
            const updatedPropertyDefs = node.data.data.propertyDefs?.map((prop) => {
              if (prop.widget === 'seekbar') {
                return { ...prop, max: frameCount };
              }
              return prop;
            });
            return {
              ...node,
              data: {
                ...node.data,
                data: {
                  ...node.data.data,
                  propertyDefs: updatedPropertyDefs,
                },
              },
            };
          }
          return node;
        })
      );
    },
    [id, setNodes]
  );

  // シークバーがあるかチェック
  const hasSeekbar = nodeData.propertyDefs?.some((p) => p.widget === 'seekbar');

  // 画像ポートを探す
  // preview=falseでないものをメイン入力とする（preview=falseの入力は追加ポートとして表示）
  // ただしinoutポートは常にメインポートとして扱う
  const imageInputPort = nodeData.inputs.find((i) => i.data_type === 'image' && (i.preview !== false || i.direction === 'inout'));
  const imageOutputPort = nodeData.outputs.find((o) => o.data_type === 'image');
  const imagePort = imageInputPort || imageOutputPort;
  const hasImagePort = !!imagePort;
  // プレビュー表示判定：出力ポートがある場合はその設定を優先（デフォルトはtrue）
  const showImagePreview = imageOutputPort
    ? imageOutputPort.preview !== false
    : imageInputPort?.preview !== false;
  // 入力ハンドル: 入力ポートがある、またはinoutポート
  const showImageInputHandle = imageInputPort && (imageInputPort.direction === 'in' || imageInputPort.direction === 'inout');
  // 出力ハンドル: 出力ポートがある、またはinoutポート
  const showImageOutputHandle = imageOutputPort && (imageOutputPort.direction === 'out' || imageOutputPort.direction === 'inout');
  // 追加の画像入力ポート（メイン画像以外のすべて - preview=falseのものも含む）
  // inoutポートはメインとして使用されるので除外
  const allAdditionalImageInputPorts = nodeData.inputs.filter(
    (i) => i.data_type === 'image' && i !== imageInputPort && i.direction !== 'inout'
  );

  // 動的ポートの処理: dynamicPortsが設定されている場合、接続済みポート+1個の空ポートのみ表示
  const additionalImageInputPorts = useMemo(() => {
    if (!nodeData.dynamicPorts) {
      return allAdditionalImageInputPorts;
    }

    // 動的ポートのプレフィックスでフィルタ（例: "Image"）
    const prefix = nodeData.dynamicPorts;
    const dynamicPorts = allAdditionalImageInputPorts.filter(
      (p) => p.name.startsWith(prefix)
    );
    const nonDynamicPorts = allAdditionalImageInputPorts.filter(
      (p) => !p.name.startsWith(prefix)
    );

    // 接続済みの動的ポートを収集
    const connectedDynamicPorts = dynamicPorts.filter((p) => connectedPortIds.has(p.id));

    // 最後の接続済みポートの番号を取得
    let maxConnectedIndex = 0;  // 0始まりで、最低でもImage 1を表示
    for (const port of connectedDynamicPorts) {
      const match = port.name.match(new RegExp(`^${prefix} (\\d+)$`));
      if (match) {
        const index = parseInt(match[1], 10);
        if (index > maxConnectedIndex) {
          maxConnectedIndex = index;
        }
      }
    }

    // 表示するポート: 接続済み + 次の1つ (最低でもImage 1を表示)
    const nextIndex = maxConnectedIndex + 1;
    const visibleDynamicPorts = dynamicPorts.filter((p) => {
      const match = p.name.match(new RegExp(`^${prefix} (\\d+)$`));
      if (!match) return false;
      const index = parseInt(match[1], 10);
      return index <= nextIndex;
    });

    return [...visibleDynamicPorts, ...nonDynamicPorts];
  }, [allAdditionalImageInputPorts, nodeData.dynamicPorts, connectedPortIds]);

  // オーディオポートを探す
  // 入出力ポートを分離して処理（Mixerのような複数ポートノード対応）
  const audioInputPorts = nodeData.inputs.filter((i) => i.data_type === 'audio');
  const audioOutputPort = nodeData.outputs.find((o) => o.data_type === 'audio');

  // メインオーディオポート（ウェーブフォーム表示用）: 出力があれば出力、なければpreview=trueの入力
  // preview=falseの入力は追加ポートとして別表示
  const audioInputPortsWithPreview = audioInputPorts.filter((p) => p.preview !== false);
  const mainAudioPort = audioOutputPort || audioInputPortsWithPreview[0];
  const hasAudioPort = !!mainAudioPort;
  const showAudioPreview = mainAudioPort?.preview !== false;  // デフォルトはtrue

  // メインオーディオポートのハンドル表示判定
  const showAudioInputHandle = mainAudioPort && (mainAudioPort.direction === 'in' || mainAudioPort.direction === 'inout');
  const showAudioOutputHandle = mainAudioPort && (mainAudioPort.direction === 'out' || mainAudioPort.direction === 'inout');

  // 追加のオーディオ入力ポート（メインポート以外、ウェーブフォームなし）
  // inoutポートはメインとして使用されるので除外
  const additionalAudioInputPorts = audioInputPorts.filter(
    (p) => p !== mainAudioPort && p.direction !== 'inout'
  );

  // オーディオ入力→画像出力の流れの場合、オーディオを先に表示
  // （オーディオ入力のみ + 画像出力のみ の場合）
  // ただし、複数のオーディオ入力がある場合やpreview=falseの場合は除外
  const isAudioToImageFlow = showAudioInputHandle && !showAudioOutputHandle
                           && showImageOutputHandle && !showImageInputHandle
                           && audioInputPorts.length === 1
                           && showAudioPreview;

  // int/floatポートを収集（image/audio以外の数値型）
  const numericInputPorts = nodeData.inputs.filter(
    (p) => p.data_type === 'int' || p.data_type === 'float'
  );
  const numericOutputPorts = nodeData.outputs.filter(
    (p) => p.data_type === 'int' || p.data_type === 'float'
  );

  // stringポートを収集
  const stringInputPorts = nodeData.inputs.filter((p) => p.data_type === 'string');
  const stringOutputPorts = nodeData.outputs.filter((p) => p.data_type === 'string');

  // string inoutポートを収集（入力と出力の両方に存在し、同じ名前のもの）
  const stringInoutPorts = stringInputPorts.filter((inputPort) =>
    stringOutputPorts.some((outputPort) => outputPort.name === inputPort.name)
  );
  const stringInoutPortNames = new Set(stringInoutPorts.map((p) => p.name));

  // inoutでない純粋な入力/出力ポート
  const allPureStringInputPorts = stringInputPorts.filter((p) => !stringInoutPortNames.has(p.name));
  const pureStringOutputPorts = stringOutputPorts.filter((p) => !stringInoutPortNames.has(p.name));

  // 動的stringポートの処理: dynamicPortsが設定されている場合、接続済みポート+1個の空ポートのみ表示
  const pureStringInputPorts = useMemo(() => {
    if (!nodeData.dynamicPorts) {
      return allPureStringInputPorts;
    }

    // 動的ポートのプレフィックスでフィルタ（例: "Text"）
    const prefix = nodeData.dynamicPorts;
    const dynamicPorts = allPureStringInputPorts.filter(
      (p) => p.name.startsWith(prefix)
    );
    const nonDynamicPorts = allPureStringInputPorts.filter(
      (p) => !p.name.startsWith(prefix)
    );

    // 接続済みの動的ポートを収集
    const connectedDynamicPorts = dynamicPorts.filter((p) => connectedPortIds.has(p.id));

    // 最後の接続済みポートの番号を取得
    let maxConnectedIndex = 0;
    for (const port of connectedDynamicPorts) {
      const match = port.name.match(new RegExp(`^${prefix} (\\d+)$`));
      if (match) {
        const index = parseInt(match[1], 10);
        if (index > maxConnectedIndex) {
          maxConnectedIndex = index;
        }
      }
    }

    // 表示するポート: 接続済み + 次の1つ (最低でもText 1を表示)
    const nextIndex = maxConnectedIndex + 1;
    const visibleDynamicPorts = dynamicPorts.filter((p) => {
      const match = p.name.match(new RegExp(`^${prefix} (\\d+)$`));
      if (!match) return false;
      const index = parseInt(match[1], 10);
      return index <= nextIndex;
    });

    return [...nonDynamicPorts, ...visibleDynamicPorts];
  }, [allPureStringInputPorts, nodeData.dynamicPorts, connectedPortIds]);

  // triggerポートを収集
  const triggerInputPorts = nodeData.inputs.filter((p) => p.data_type === 'trigger');
  const triggerOutputPorts = nodeData.outputs.filter((p) => p.data_type === 'trigger');

  // テキスト表示ノードかどうか判定
  // （string入力があり、text_area/text_inputウィジェットがなく、画像ポートもなく、string出力もなく、ボタンもない）
  const hasTextWidgets = nodeData.propertyDefs?.some(
    (p) => p.widget === 'text_area' || p.widget === 'text_input'
  );
  const hasButtonWidget = nodeData.propertyDefs?.some(
    (p) => p.widget === 'button'
  );
  const isTextDisplayNode =
    stringInputPorts.length > 0 && !hasTextWidgets && !hasButtonWidget && !hasImagePort && stringOutputPorts.length === 0;

  // プロパティ名と入力ポートのマッピング（同名のポートをプロパティ横に表示するため）
  const propertyInputPorts = new Map(
    numericInputPorts
      .filter((p) => nodeData.propertyDefs?.some((prop) => prop.name === p.name))
      .map((p) => [p.name, p])
  );

  // プロパティに対応しない入力ポート（独立して表示）
  const standaloneInputPorts = numericInputPorts.filter(
    (p) => !propertyInputPorts.has(p.name)
  );

  // マスクノードかどうか判定
  const isMaskNode = nodeData.definitionId === 'image.draw.mask';
  const penSize = (nodeData.properties.pen_size as number) ?? 10;

  // 描画キャンバスノードかどうか判定
  const isDrawCanvasNode = nodeData.definitionId === 'image.draw.canvas';
  const drawPenSize = (nodeData.properties.pen_size as number) ?? 5;
  const drawPenColor = (nodeData.properties.pen_color as string) ?? '#000000';

  // クロップノードかどうか判定
  const isCropNode = nodeData.definitionId === 'image.transform.crop';
  const cropMinX = (nodeData.properties.min_x as number) ?? 0.0;
  const cropMinY = (nodeData.properties.min_y as number) ?? 0.0;
  const cropMaxX = (nodeData.properties.max_x as number) ?? 1.0;
  const cropMaxY = (nodeData.properties.max_y as number) ?? 1.0;

  // クロップ領域変更ハンドラ
  const handleCropChange = useCallback(
    (minX: number, minY: number, maxX: number, maxY: number) => {
      handlePropertyChange('min_x', minX);
      handlePropertyChange('min_y', minY);
      handlePropertyChange('max_x', maxX);
      handlePropertyChange('max_y', maxY);
    },
    [handlePropertyChange]
  );

  // クロッププロパティ変更ハンドラ（min/max制約付き）
  const handleCropPropertyChange = useCallback(
    (propName: string, value: number) => {
      const step = 0.01;
      let newValue = value;

      // min_xがmax_xを超えた場合、max_x - stepに制限
      if (propName === 'min_x' && value >= cropMaxX) {
        newValue = Math.max(0, cropMaxX - step);
      }
      // max_xがmin_xを下回った場合、min_x + stepに制限
      if (propName === 'max_x' && value <= cropMinX) {
        newValue = Math.min(1, cropMinX + step);
      }
      // min_yがmax_yを超えた場合、max_y - stepに制限
      if (propName === 'min_y' && value >= cropMaxY) {
        newValue = Math.max(0, cropMaxY - step);
      }
      // max_yがmin_yを下回った場合、min_y + stepに制限
      if (propName === 'max_y' && value <= cropMinY) {
        newValue = Math.min(1, cropMinY + step);
      }

      handlePropertyChange(propName, Math.round(newValue * 100) / 100);
    },
    [handlePropertyChange, cropMinX, cropMinY, cropMaxX, cropMaxY]
  );

  // Draw Textノードかどうか判定（画像クリックで座標指定）
  const isDrawTextNode = nodeData.definitionId === 'image.draw.text';

  // Omnidirectional Viewerノードかどうか判定
  const isOmnidirectionalNode = nodeData.definitionId === 'image.filter.omnidirectional_viewer';
  const omniPitch = (nodeData.properties.pitch as number) ?? 0;
  const omniYaw = (nodeData.properties.yaw as number) ?? 0;
  const omniRoll = (nodeData.properties.roll as number) ?? 0;

  // タイマーノードかどうか判定
  const isTimerNode = nodeData.definitionId === 'utility.timer';

  // Comparison Sliderノードかどうか判定
  const isComparisonSliderNode = nodeData.definitionId === 'image.draw.image_comparison_slider';

  // Waveformスタイル画像ノードかどうか判定（VAD, Waveform to Image）
  const isWaveformStyleImageNode =
    nodeData.definitionId === 'audio.analysis.voice_activity_detection' ||
    nodeData.definitionId === 'audio.utility.waveform_to_image';

  // Picture In Pictureノードかどうか判定
  const isPIPNode = nodeData.definitionId === 'image.draw.picture_in_picture';

  // Click Perspectiveノードかどうか判定
  const isPerspectiveNode = nodeData.definitionId === 'image.transform.click_perspective';
  const perspectiveX1 = (nodeData.properties.x1 as number) ?? 0.0;
  const perspectiveY1 = (nodeData.properties.y1 as number) ?? 0.0;
  const perspectiveX2 = (nodeData.properties.x2 as number) ?? 1.0;
  const perspectiveY2 = (nodeData.properties.y2 as number) ?? 0.0;
  const perspectiveX3 = (nodeData.properties.x3 as number) ?? 1.0;
  const perspectiveY3 = (nodeData.properties.y3 as number) ?? 1.0;
  const perspectiveX4 = (nodeData.properties.x4 as number) ?? 0.0;
  const perspectiveY4 = (nodeData.properties.y4 as number) ?? 1.0;
  const perspectiveCurrentPoint = (nodeData.properties.current_point as number) ?? 1;

  // Clampノードかどうか判定
  const isClampNode = nodeData.definitionId === 'math.value.clamp';
  const clampMin = (nodeData.properties.min as number) ?? 0.0;
  const clampMax = (nodeData.properties.max as number) ?? 1.0;

  // Execute Pythonノードかどうか判定
  const isExecutePythonNode = nodeData.definitionId === 'image.other.execute_python';
  const pipMinX = (nodeData.properties.min_x as number) ?? 0.7;
  const pipMinY = (nodeData.properties.min_y as number) ?? 0.7;
  const pipMaxX = (nodeData.properties.max_x as number) ?? 0.9;
  const pipMaxY = (nodeData.properties.max_y as number) ?? 0.9;

  // PIP領域変更ハンドラ
  const handlePIPChange = useCallback(
    (minX: number, minY: number, maxX: number, maxY: number) => {
      handlePropertyChange('min_x', minX);
      handlePropertyChange('min_y', minY);
      handlePropertyChange('max_x', maxX);
      handlePropertyChange('max_y', maxY);
    },
    [handlePropertyChange]
  );

  // PIPプロパティ変更ハンドラ（min/max制約付き）
  const handlePIPPropertyChange = useCallback(
    (propName: string, value: number) => {
      const step = 0.01;
      let newValue = value;

      // min_xがmax_xを超えた場合、max_x - stepに制限
      if (propName === 'min_x' && value >= pipMaxX) {
        newValue = Math.max(0, pipMaxX - step);
      }
      // max_xがmin_xを下回った場合、min_x + stepに制限
      if (propName === 'max_x' && value <= pipMinX) {
        newValue = Math.min(1, pipMinX + step);
      }
      // min_yがmax_yを超えた場合、max_y - stepに制限
      if (propName === 'min_y' && value >= pipMaxY) {
        newValue = Math.max(0, pipMaxY - step);
      }
      // max_yがmin_yを下回った場合、min_y + stepに制限
      if (propName === 'max_y' && value <= pipMinY) {
        newValue = Math.min(1, pipMinY + step);
      }

      handlePropertyChange(propName, Math.round(newValue * 100) / 100);
    },
    [handlePropertyChange, pipMinX, pipMinY, pipMaxX, pipMaxY]
  );

  // Perspectiveポイント変更ハンドラ
  const handlePerspectivePointChange = useCallback(
    (pointNum: number, x: number, y: number) => {
      const xProp = `x${pointNum}`;
      const yProp = `y${pointNum}`;
      handlePropertyChange(xProp, Math.round(x * 100) / 100);
      handlePropertyChange(yProp, Math.round(y * 100) / 100);
    },
    [handlePropertyChange]
  );

  // Perspective現在ポイント変更ハンドラ
  const handlePerspectiveCurrentPointChange = useCallback(
    (pointNum: number) => {
      handlePropertyChange('current_point', pointNum);
    },
    [handlePropertyChange]
  );

  // Clampプロパティ変更ハンドラ（min/max逆転防止）
  const handleClampPropertyChange = useCallback(
    (propName: string, value: number) => {
      if (propName === 'max') {
        // Maxが変更された場合、MinがMaxを超えていたらMinも更新
        if (clampMin > value) {
          handlePropertyChange('min', value);
        }
        handlePropertyChange('max', value);
      } else if (propName === 'min') {
        // Minが変更された場合、MaxがMinを下回っていたらMaxも更新
        if (clampMax < value) {
          handlePropertyChange('max', value);
        }
        handlePropertyChange('min', value);
      } else {
        handlePropertyChange(propName, value);
      }
    },
    [handlePropertyChange, clampMin, clampMax]
  );

  // 画像クリックハンドラ（Draw Text用）
  const handleImageClick = useCallback(
    (e: React.MouseEvent<HTMLImageElement>) => {
      if (!isDrawTextNode) return;

      const img = e.currentTarget;
      const rect = img.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      // Draw Text: 画像座標に変換
      const scaleX = img.naturalWidth / rect.width;
      const scaleY = img.naturalHeight / rect.height;
      const x = Math.round(clickX * scaleX);
      const y = Math.round(clickY * scaleY);
      handlePropertyChange('x', x);
      handlePropertyChange('y', y);
    },
    [isDrawTextNode, handlePropertyChange]
  );

  // 画像ドラッグハンドラ（Comparison Slider用）
  const handleImageDrag = useCallback(
    (normalizedX: number) => {
      if (!isComparisonSliderNode) return;
      handlePropertyChange('position', normalizedX);
    },
    [isComparisonSliderNode, handlePropertyChange]
  );

  // ブラウザWebカメラノードかどうか判定
  const hasBrowserWebcam = nodeData.propertyDefs?.some((p) => p.widget === 'browser_webcam');
  const browserWebcamProp = nodeData.propertyDefs?.find((p) => p.widget === 'browser_webcam');
  const resolutionProp = nodeData.propertyDefs?.find((p) => p.name === 'resolution');
  const browserWebcamResolution = (nodeData.properties['resolution'] as string) ?? resolutionProp?.default ?? '1280x720';

  // WebRTC Webcamノードかどうか判定
  const hasWebRTCWebcam = nodeData.propertyDefs?.some((p) => p.widget === 'webrtc_webcam');
  const webrtcWebcamConnectionId = `webrtc_webcam_${id}`;

  // ブラウザマイクノードかどうか判定
  const hasBrowserMicrophone = nodeData.propertyDefs?.some((p) => p.widget === 'browser_microphone');
  const browserMicrophoneProp = nodeData.propertyDefs?.find((p) => p.widget === 'browser_microphone');
  const echoCancellation = (nodeData.properties['echo_cancellation'] as boolean) ?? false;
  const noiseSuppression = (nodeData.properties['noise_suppression'] as boolean) ?? false;
  const autoGainControl = (nodeData.properties['auto_gain_control'] as boolean) ?? false;

  // WebRTC Microphoneノードかどうか判定
  const hasWebRTCMicrophone = nodeData.propertyDefs?.some((p) => p.widget === 'webrtc_microphone');
  const webrtcMicConnectionId = `webrtc_microphone_${id}`;
  const [isWebRTCMicReady, setIsWebRTCMicReady] = useState(false);

  // ブラウザスピーカーノードかどうか判定
  const hasBrowserSpeaker = nodeData.propertyDefs?.some((p) => p.widget === 'browser_speaker');

  // 描画コマンドハンドラ
  const handleDrawCommand = useCallback(
    (command: object) => {
      // 現在のコマンドを取得して新しいコマンドを追加
      const currentCommands = nodeData.properties.draw_commands as string || '[]';
      let commands: object[] = [];
      try {
        commands = JSON.parse(currentCommands);
      } catch {
        commands = [];
      }

      // clearコマンドの場合はリストをクリアしてclearのみにする
      if ((command as { type: string }).type === 'clear') {
        commands = [command];
      } else {
        commands.push(command);
      }

      // コマンド数を制限（古いものを削除）
      if (commands.length > 100) {
        commands = commands.slice(-100);
      }

      handlePropertyChange('draw_commands', JSON.stringify(commands));
    },
    [nodeData.properties.draw_commands, handlePropertyChange]
  );

  // リサイズ可能なノードかどうか
  const isResizable = nodeData.resizable === true;
  const resizableClass = isResizable ? 'resizable-node' : '';

  // コメント
  const comment = nodeData.comment;

  // ヘッダーフォントサイズの自動調整（長い名前は縮小）
  const headerFontSize = useMemo(() => {
    const label = nodeData.label || '';
    const baseSize = 18;
    const minSize = 11;
    const maxWidth = 200;
    const avgCharWidth = 0.55;

    // 推定テキスト幅 = 文字数 × フォントサイズ × 平均文字幅係数
    const estimatedWidth = label.length * baseSize * avgCharWidth;

    if (estimatedWidth <= maxWidth) {
      return baseSize;
    }

    // 幅に収まるようにフォントサイズを縮小
    const scale = maxWidth / estimatedWidth;
    const newSize = Math.max(minSize, Math.floor(baseSize * scale));
    return newSize;
  }, [nodeData.label]);

  // ノード固有のクラス
  const nodeTypeClass = isExecutePythonNode ? 'execute-python-node' : '';

  return (
    <div className={`custom-node ${resizableClass} ${nodeTypeClass} ${selected ? 'selected' : ''} ${nodeData.errorMessage ? 'has-error' : ''}`}>
      {/* コメント表示（ノード右上角から線を引く） */}
      {comment && (
        <div className="node-comment-wrapper">
          <svg className="node-comment-diagonal" viewBox="0 0 20 30" width="20" height="30">
            <path d="M 0 30 L 20 0" stroke="currentColor" strokeWidth="1.5" fill="none" />
          </svg>
          <div className="node-comment-content">
            <span className="node-comment-text">{comment}</span>
            <div className="node-comment-line"></div>
          </div>
        </div>
      )}
      {/* リサイズハンドル（resizable=trueのノードのみ） */}
      {isResizable && (
        <NodeResizer
          minWidth={200}
          minHeight={100}
          isVisible={selected}
          keepAspectRatio={true}
          lineClassName="node-resizer-line"
          handleClassName="node-resizer-handle"
          onResize={handleResize}
        />
      )}
      {/* エラーメッセージ表示 */}
      {nodeData.errorMessage && (
        <div className="node-error" title={nodeData.errorMessage}>
          {nodeData.errorMessage}
        </div>
      )}
      <div className="node-header" style={{ fontSize: headerFontSize }}>{nodeData.label}</div>
      <div className="node-content">
        {/* 画像ウィジェット（directionに基づいてハンドル表示） */}
        {hasImagePort && imagePort && (
          <div className="media-widget-row">
            {/* オーディオ入力→画像出力フローの場合、オーディオ入力ハンドルを左側に配置 */}
            {isAudioToImageFlow && hasAudioPort && mainAudioPort && showAudioInputHandle && (
              <Handle
                type="target"
                position={Position.Left}
                id={mainAudioPort.id}
                className="media-handle audio left"
              />
            )}
            {showImageInputHandle && imageInputPort && (
              <Handle
                type="target"
                position={Position.Left}
                id={imageInputPort.id}
                className="media-handle image left"
              />
            )}
            {isMaskNode ? (
              <MaskCanvas
                backgroundImage={nodeData.imageData}
                maskImage={nodeData.maskData}
                penSize={penSize}
                onDrawCommand={handleDrawCommand}
              />
            ) : isDrawCanvasNode ? (
              <DrawCanvas
                backgroundImage={nodeData.imageData}
                penSize={drawPenSize}
                penColor={drawPenColor}
                onDrawCommand={handleDrawCommand}
              />
            ) : isCropNode ? (
              <CropCanvas
                imageData={nodeData.imageData}
                minX={cropMinX}
                minY={cropMinY}
                maxX={cropMaxX}
                maxY={cropMaxY}
                isStreaming={nodeData.isStreaming ?? false}
                isPaused={nodeData.isPaused ?? false}
                onCropChange={handleCropChange}
              />
            ) : isPIPNode ? (
              <PIPCanvas
                imageData={nodeData.imageData}
                minX={pipMinX}
                minY={pipMinY}
                maxX={pipMaxX}
                maxY={pipMaxY}
                isStreaming={nodeData.isStreaming ?? false}
                isPaused={nodeData.isPaused ?? false}
                onRegionChange={handlePIPChange}
              />
            ) : isOmnidirectionalNode ? (
              <OmnidirectionalCanvas
                imageData={nodeData.imageData}
                pitch={omniPitch}
                yaw={omniYaw}
                roll={omniRoll}
                isStreaming={nodeData.isStreaming ?? false}
                isPaused={nodeData.isPaused ?? false}
                onPitchChange={(v) => handlePropertyChange('pitch', v)}
                onYawChange={(v) => handlePropertyChange('yaw', v)}
                onRollChange={(v) => handlePropertyChange('roll', v)}
              />
            ) : isPerspectiveNode ? (
              <PerspectiveCanvas
                imageData={nodeData.imageData}
                x1={perspectiveX1}
                y1={perspectiveY1}
                x2={perspectiveX2}
                y2={perspectiveY2}
                x3={perspectiveX3}
                y3={perspectiveY3}
                x4={perspectiveX4}
                y4={perspectiveY4}
                currentPoint={perspectiveCurrentPoint}
                isStreaming={nodeData.isStreaming ?? false}
                isPaused={nodeData.isPaused ?? false}
                onPointChange={handlePerspectivePointChange}
                onCurrentPointChange={handlePerspectiveCurrentPointChange}
              />
            ) : hasWebRTCWebcam ? (
              <WebRTCWebcam
                connectionId={webrtcWebcamConnectionId}
                isStreaming={nodeData.isStreaming ?? false}
                onConnectionReady={() => handlePropertyChange('webrtc_webcam', webrtcWebcamConnectionId)}
              />
            ) : hasBrowserWebcam && browserWebcamProp ? (
              <BrowserWebcam
                resolution={browserWebcamResolution}
                onFrame={(base64) => handlePropertyChange(browserWebcamProp.name, base64)}
                isStreaming={nodeData.isStreaming ?? false}
                intervalMs={nodeData.intervalMs ?? 100}
              />
            ) : showImagePreview ? (
              <ImagePreview
                definitionId={nodeData.definitionId}
                imageData={nodeData.imageData}
                isStreaming={nodeData.isStreaming ?? false}
                isPaused={nodeData.isPaused ?? false}
                isClickable={isDrawTextNode}
                isDraggable={isComparisonSliderNode}
                waveformStyle={isWaveformStyleImageNode}
                onClick={isDrawTextNode ? handleImageClick : undefined}
                onDrag={isComparisonSliderNode ? handleImageDrag : undefined}
              />
            ) : (
              <div className="port-label">Image</div>
            )}
            {showImageOutputHandle && imageOutputPort && (
              <Handle
                type="source"
                position={Position.Right}
                id={imageOutputPort.id}
                className="media-handle image right"
              />
            )}
          </div>
        )}

        {/* オーディオウィジェット（通常フロー - オーディオ入力→画像出力でない場合） */}
        {!isAudioToImageFlow && hasAudioPort && mainAudioPort && (
          <div className="media-widget-row">
            {showAudioInputHandle && (
              <Handle
                type="target"
                position={Position.Left}
                id={mainAudioPort.id}
                className="media-handle audio left"
              />
            )}
            <div className="audio-widget-content">
              {/* WebRTCマイクが準備中の場合はLoadingを表示 */}
              {hasWebRTCMicrophone && !isWebRTCMicReady && (
                <div className="webrtc-mic-loading">Loading...</div>
              )}
              {/* WebRTCマイクが準備完了、またはWebRTCマイクでない場合はWaveform表示 */}
              {showAudioPreview && (!hasWebRTCMicrophone || isWebRTCMicReady) && (
                <WaveformCanvas audioData={nodeData.audioData} />
              )}
              {!showAudioPreview && (
                <div className="port-label">Audio</div>
              )}
              {hasWebRTCMicrophone && (
                <WebRTCMicrophone
                  echoCancellation={echoCancellation}
                  noiseSuppression={noiseSuppression}
                  autoGainControl={autoGainControl}
                  connectionId={webrtcMicConnectionId}
                  isStreaming={nodeData.isStreaming ?? false}
                  onConnectionReady={() => handlePropertyChange('webrtc_microphone', webrtcMicConnectionId)}
                  onReadyStateChange={setIsWebRTCMicReady}
                />
              )}
              {hasBrowserMicrophone && browserMicrophoneProp && (
                <BrowserMicrophone
                  echoCancellation={echoCancellation}
                  noiseSuppression={noiseSuppression}
                  autoGainControl={autoGainControl}
                  onAudio={(json) => handlePropertyChange(browserMicrophoneProp.name, json)}
                  isStreaming={nodeData.isStreaming ?? false}
                />
              )}
              {hasBrowserSpeaker && (
                <BrowserSpeaker
                  audioData={nodeData.audioData}
                  isStreaming={nodeData.isStreaming ?? false}
                />
              )}
            </div>
            {showAudioOutputHandle && (
              <Handle
                type="source"
                position={Position.Right}
                id={mainAudioPort.id}
                className="media-handle audio right"
              />
            )}
          </div>
        )}

        {/* 追加のオーディオ入力ポート（メインオーディオ以外、ウェーブフォームなし） */}
        {additionalAudioInputPorts.map((port) => (
          <div key={port.id} className="audio-port-row">
            <Handle
              type="target"
              position={Position.Left}
              id={port.id}
              className="media-handle audio left"
            />
            <span className="port-label">{port.display_name || port.name}</span>
          </div>
        ))}

        {/* 追加の画像入力ポート（メイン画像以外、例：マスク） */}
        {additionalImageInputPorts.map((port) => (
          <div key={port.id} className="image-port-row">
            <Handle
              type="target"
              position={Position.Left}
              id={port.id}
              className="media-handle image left"
            />
            <span className="port-label">{port.name}</span>
          </div>
        ))}

        {/* トリガー入力ポート（ボタンに統合されていないもののみ） */}
        {triggerInputPorts
          .filter((port) => !nodeData.propertyDefs?.some((p) => p.widget === 'button' && p.name === port.name))
          .map((port) => (
            <div key={port.id} className="trigger-port-row">
              <Handle
                type="target"
                position={Position.Left}
                id={port.id}
                className="trigger-handle left"
              />
              <span className="port-label">{port.name}</span>
            </div>
          ))}

        {/* 数値入力ポート (int/float) - プロパティに対応しないもののみ */}
        {standaloneInputPorts.map((port) => (
          <div key={port.id} className="numeric-port-row">
            <Handle
              type="target"
              position={Position.Left}
              id={port.id}
              className={`numeric-handle left ${port.data_type}`}
            />
            <span className="port-label">{port.name}</span>
          </div>
        ))}

        {/* 数値出力ポート (int/float) */}
        {numericOutputPorts
          .filter((port) => !(isTimerNode && port.name === 'trigger'))
          .map((port) => {
            const outputValue = nodeData.numericOutputs?.[port.name];
            const displayValue = outputValue !== undefined
              ? `=${Number.isInteger(outputValue) ? outputValue : outputValue.toFixed(2)}`
              : '';
            return (
              <div key={port.id} className="numeric-port-row output">
                <span className="port-label">{port.name}{displayValue}</span>
                <Handle
                  type="source"
                  position={Position.Right}
                  id={port.id}
                  className={`numeric-handle right ${port.data_type}`}
                />
              </div>
            );
          })}

        {/* 文字列入出力ポート (string inout) - 両端にハンドル */}
        {stringInoutPorts.map((inputPort) => {
          const outputPort = stringOutputPorts.find((p) => p.name === inputPort.name);
          return (
            <div key={inputPort.id} className="string-port-row inout">
              <Handle
                type="target"
                position={Position.Left}
                id={inputPort.id}
                className="string-handle left"
              />
              <span className="port-label">{inputPort.display_name || inputPort.name}</span>
              {outputPort && (
                <Handle
                  type="source"
                  position={Position.Right}
                  id={outputPort.id}
                  className="string-handle right"
                />
              )}
            </div>
          );
        })}

        {/* テキストエリア（プロパティ定義順で表示、文字列入力ポートより先に表示） */}
        {nodeData.propertyDefs?.filter((p) =>
          p.widget === 'text_area' &&
          isPropertyVisible(p, nodeData.properties, nodeData.apiKeysStatus)
        ).map((prop) => {
          const inputPort = stringInputPorts.find((p) => p.name === prop.name);
          const outputPort = stringOutputPorts.find((p) => p.name === `${prop.name}_out` || p.name === prop.name);
          const connectedValue = nodeData.connectedProperties?.[prop.name];
          const isConnected = inputPort !== undefined && connectedValue !== undefined;
          const displayValue = isConnected
            ? String(connectedValue)
            : ((nodeData.properties[prop.name] as string) ?? prop.default ?? '');
          return (
            <div key={prop.name} className="text-area-widget-row">
              {inputPort && (
                <Handle
                  type="target"
                  position={Position.Left}
                  id={inputPort.id}
                  className={`string-handle left ${isConnected ? 'connected' : ''}`}
                />
              )}
              <TextAreaWidget
                prop={prop}
                value={displayValue}
                onChange={(value) => handlePropertyChange(prop.name, value)}
                readOnly={isConnected}
              />
              {outputPort && (
                <Handle
                  type="source"
                  position={Position.Right}
                  id={outputPort.id}
                  className="string-handle right"
                />
              )}
            </div>
          );
        })}

        {/* 文字列入力ポート (string) - テキスト表示/テキストエリアに統合されていないもの、かつinoutでないもののみ */}
        {pureStringInputPorts
          .filter((port) => {
            if (isTextDisplayNode) return false;
            // text_areaウィジェットに統合されたポートは除外
            const textAreaProps = nodeData.propertyDefs?.filter((p) => p.widget === 'text_area') || [];
            return !textAreaProps.some((prop) => prop.name === port.name);
          })
          .map((port) => (
            <div key={port.id} className="string-port-row">
              <Handle
                type="target"
                position={Position.Left}
                id={port.id}
                className="string-handle left"
              />
              <span className="port-label">{port.name}</span>
            </div>
          ))}

        {/* XY座標入力（横並び） */}
        {(() => {
          const xyProps = nodeData.propertyDefs?.filter((p) => p.widget === 'xy_input') || [];
          const xProp = xyProps.find((p) => p.name === 'x');
          const yProp = xyProps.find((p) => p.name === 'y');
          if (!xProp || !yProp) return null;

          const xValue = (nodeData.properties['x'] as number) ?? xProp.default ?? 0;
          const yValue = (nodeData.properties['y'] as number) ?? yProp.default ?? 0;

          return (
            <XYInputWidget
              xValue={xValue}
              yValue={yValue}
              onXChange={(value) => handlePropertyChange('x', value)}
              onYChange={(value) => handlePropertyChange('y', value)}
            />
          );
        })()}

        {/* シークバー（ラベルなしスライダー） */}
        {nodeData.propertyDefs?.filter((p) => p.widget === 'seekbar' && isPropertyVisible(p, nodeData.properties, nodeData.apiKeysStatus)).map((prop) => (
          <SeekbarWidget
            key={prop.name}
            prop={prop}
            value={(nodeData.properties[prop.name] as number) ?? prop.default ?? 1}
            onChange={(value) => handlePropertyChange(prop.name, value)}
          />
        ))}

        {/* 文字列出力ポート (string) - テキストウィジェットに統合されていないもの */}
        {pureStringOutputPorts
          .filter((port) => {
            const textProps = nodeData.propertyDefs?.filter((p) => p.widget === 'text_input' || p.widget === 'text_area') || [];
            return !textProps.some((prop) => port.name === `${prop.name}_out` || port.name === prop.name);
          })
          .map((port) => (
            <div key={port.id} className="string-port-row output">
              <span className="port-label">{port.name}</span>
              <Handle
                type="source"
                position={Position.Right}
                id={port.id}
                className="string-handle right"
              />
            </div>
          ))}

        {/* スライダー、ドロップダウン、ファイルピッカー、チェックボックス、ボタン、カラーピッカー（TOML定義順で表示） */}
        {nodeData.propertyDefs?.filter((p) =>
          (p.widget === 'slider' || p.widget === 'dropdown' || p.widget === 'file_picker' || p.widget === 'checkbox' || p.widget === 'button' || p.widget === 'color_picker') &&
          isPropertyVisible(p, nodeData.properties, nodeData.apiKeysStatus)
        ).map((prop) => {
          if (prop.widget === 'slider') {
            const inputPort = propertyInputPorts.get(prop.name);
            const connectedValue = nodeData.connectedProperties?.[prop.name];
            const isConnected = connectedValue !== undefined && typeof connectedValue === 'number';
            const rawValue = isConnected ? connectedValue : ((nodeData.properties[prop.name] as number) ?? prop.default ?? 0);
            const displayValue = (isConnected && prop.type === 'int') ? Math.floor(rawValue) : rawValue;
            return (
              <div key={prop.name} className="property-with-handle">
                {inputPort && (
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={inputPort.id}
                    className={`numeric-handle property-handle ${inputPort.data_type}`}
                  />
                )}
                <div className={`property-row nodrag ${isConnected ? 'connected' : ''}`}>
                  {prop.display_name && (
                    <label className="property-label">{prop.display_name}</label>
                  )}
                  <input
                    type="range"
                    className="property-slider"
                    min={prop.min ?? 0}
                    max={prop.max ?? 100}
                    step={prop.step ?? 1}
                    value={displayValue}
                    onChange={(e) => {
                      const value = Number(e.target.value);
                      if (isCropNode && ['min_x', 'min_y', 'max_x', 'max_y'].includes(prop.name)) {
                        handleCropPropertyChange(prop.name, value);
                      } else if (isPIPNode && ['min_x', 'min_y', 'max_x', 'max_y'].includes(prop.name)) {
                        handlePIPPropertyChange(prop.name, value);
                      } else {
                        handlePropertyChange(prop.name, value);
                      }
                    }}
                    disabled={isConnected || (prop.disabled_while_streaming && (nodeData.isStreaming || nodeData.isPaused))}
                  />
                  {prop.display_name && (
                    <span className="property-value">
                      {Number.isInteger(displayValue) ? displayValue : displayValue.toFixed(1)}
                    </span>
                  )}
                </div>
              </div>
            );
          } else if (prop.widget === 'dropdown') {
            return (
              <div key={prop.name} className="property-row nodrag">
                <label className="property-label">{prop.display_name || prop.name}</label>
                <select
                  className="property-dropdown"
                  value={(nodeData.properties[prop.name] as number | string) ?? prop.default ?? ''}
                  onChange={(e) => {
                    const val = prop.type === 'int' ? Number(e.target.value) : e.target.value;
                    handlePropertyChange(prop.name, val);

                    // Webcamノードのcamera_id変更時にプレビュー実行をトリガー
                    if (nodeData.definitionId === 'image.input.webcam' && prop.name === 'camera_id') {
                      window.dispatchEvent(new CustomEvent('webcam-camera-changed'));
                    }
                  }}
                  disabled={prop.disabled_while_streaming && (nodeData.isStreaming || nodeData.isPaused)}
                >
                  {prop.options?.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            );
          } else if (prop.widget === 'file_picker') {
            return (
              <FilePickerWidget
                key={prop.name}
                prop={prop}
                value={(nodeData.properties[prop.name] as string) ?? ''}
                onChange={(value) => handlePropertyChange(prop.name, value)}
                onFirstFrame={hasImagePort ? handleImageUpdate : undefined}
                onFrameCount={hasSeekbar ? handleFrameCountUpdate : undefined}
                disabled={prop.disabled_while_streaming && (nodeData.isStreaming || nodeData.isPaused)}
              />
            );
          } else if (prop.widget === 'checkbox') {
            return (
              <CheckboxWidget
                key={prop.name}
                prop={prop}
                checked={(nodeData.properties[prop.name] as boolean) ?? prop.default ?? false}
                onChange={(value) => handlePropertyChange(prop.name, value)}
                disabled={prop.disabled_while_streaming && (nodeData.isStreaming || nodeData.isPaused)}
              />
            );
          } else if (prop.widget === 'button') {
            const triggerInPort = triggerInputPorts.find((p2) => p2.name === prop.name);
            const triggerOutPort = triggerOutputPorts.find((p2) => p2.name === 'out');
            const requiresStreaming = prop.requires_streaming === true;
            const isDisabled = (requiresStreaming && !nodeData.isStreaming) || (nodeData.isBusy === true);
            return (
              <div key={prop.name} className="button-with-trigger">
                {triggerInPort && (
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={triggerInPort.id}
                    className="trigger-handle left"
                  />
                )}
                <ButtonWidget
                  prop={prop}
                  disabled={isDisabled}
                  onClick={() => {
                    if (isDrawCanvasNode && prop.name === 'reset') {
                      handlePropertyChange('draw_commands', '[]');
                    }
                    handlePropertyChange(prop.name, true);
                    setTimeout(() => handlePropertyChange(prop.name, false), 100);
                  }}
                />
                {triggerOutPort && (
                  <Handle
                    type="source"
                    position={Position.Right}
                    id={triggerOutPort.id}
                    className="trigger-handle right"
                  />
                )}
              </div>
            );
          } else {
            // color_picker
            return (
              <ColorPickerWidget
                key={prop.name}
                prop={prop}
                value={(nodeData.properties[prop.name] as string) ?? prop.default ?? '#ff0000'}
                onChange={(value) => handlePropertyChange(prop.name, value)}
              />
            );
          }
        })}

        {/* 3x3マトリックス入力 */}
        {nodeData.propertyDefs?.filter((p) => p.widget === 'matrix3x3' && isPropertyVisible(p, nodeData.properties, nodeData.apiKeysStatus)).map((prop) => (
          <Matrix3x3Widget
            key={prop.name}
            prop={prop}
            value={(nodeData.properties[prop.name] as string) ?? prop.default ?? '0,0,0,0,1,0,0,0,0'}
            onChange={(value) => handlePropertyChange(prop.name, value)}
          />
        ))}

        {/* 数値入力 */}
        {nodeData.propertyDefs?.filter((p) => p.widget === 'number_input' && isPropertyVisible(p, nodeData.properties, nodeData.apiKeysStatus)).map((prop) => {
          const inputPort = propertyInputPorts.get(prop.name);
          const connectedValue = nodeData.connectedProperties?.[prop.name];
          const isConnected = connectedValue !== undefined && typeof connectedValue === 'number';
          // intプロパティに接続された場合は切り捨て
          const rawValue = isConnected ? connectedValue : ((nodeData.properties[prop.name] as number) ?? prop.default ?? 0);
          const displayValue = (isConnected && prop.type === 'int') ? Math.floor(rawValue) : rawValue;
          return (
            <div key={prop.name} className="property-with-handle">
              {inputPort && (
                <Handle
                  type="target"
                  position={Position.Left}
                  id={inputPort.id}
                  className={`numeric-handle property-handle ${inputPort.data_type}`}
                />
              )}
              {isConnected ? (
                <div className="property-row nodrag connected">
                  <label className="property-label">{prop.display_name || prop.name}</label>
                  <span className="connected-value">{Number.isInteger(displayValue) ? displayValue : displayValue.toFixed(1)}</span>
                </div>
              ) : (
                <NumberInputWidget
                  prop={prop}
                  value={displayValue}
                  onChange={(value) => {
                    // Clampノードの場合はmin/max逆転防止ハンドラを使用
                    if (isClampNode && (prop.name === 'min' || prop.name === 'max')) {
                      handleClampPropertyChange(prop.name, value);
                    } else {
                      handlePropertyChange(prop.name, value);
                    }
                  }}
                />
              )}
            </div>
          );
        })}

        {/* テキスト入力 */}
        {nodeData.propertyDefs?.filter((p) => p.widget === 'text_input' && isPropertyVisible(p, nodeData.properties, nodeData.apiKeysStatus)).map((prop) => {
          // 同名のstring出力ポートを探す（text -> text_out のマッピング）
          const outputPort = stringOutputPorts.find((p) => p.name === `${prop.name}_out` || p.name === prop.name);
          return (
            <div key={prop.name} className="text-input-with-handle">
              <TextInputWidget
                prop={prop}
                value={(nodeData.properties[prop.name] as string) ?? prop.default ?? ''}
                onChange={(value) => handlePropertyChange(prop.name, value)}
              />
              {outputPort && (
                <Handle
                  type="source"
                  position={Position.Right}
                  id={outputPort.id}
                  className="string-handle right"
                />
              )}
            </div>
          );
        })}

        {/* テキスト表示（読み取り専用、string入力ポートの内容を表示） */}
        {isTextDisplayNode && (
          <div className="text-display-widget-row">
            {stringInputPorts[0] && (
              <Handle
                type="target"
                position={Position.Left}
                id={stringInputPorts[0].id}
                className="string-handle left"
              />
            )}
            <TextDisplayWidget text={nodeData.displayText ?? ''} />
          </div>
        )}

        {/* タイマーリング */}
        {isTimerNode && (() => {
          const interval = (nodeData.properties.interval as number) ?? 1;
          const unit = (nodeData.properties.unit as string) ?? 'sec';
          const triggerPort = triggerOutputPorts.find((p) => p.name === 'trigger');
          return (
            <div className="timer-ring-row">
              <TimerRing
                interval={interval}
                unit={unit}
                isStreaming={nodeData.isStreaming ?? false}
              />
              {triggerPort && (
                <Handle
                  type="source"
                  position={Position.Right}
                  id={triggerPort.id}
                  className="trigger-handle right"
                />
              )}
            </div>
          );
        })()}

      </div>
    </div>
  );
}

export default memo(CustomNode);
