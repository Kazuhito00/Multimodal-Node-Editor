import { useCallback, useRef, useState } from 'react';

interface PerspectiveCanvasProps {
  imageData?: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  x3: number;
  y3: number;
  x4: number;
  y4: number;
  currentPoint: number;
  isStreaming: boolean;
  isPaused: boolean;
  onPointChange: (pointNum: number, x: number, y: number) => void;
  onCurrentPointChange: (pointNum: number) => void;
}

const POINT_COLOR = '#00ff00';

interface ImageBounds {
  offsetX: number;
  offsetY: number;
  width: number;
  height: number;
}

export function PerspectiveCanvas({
  imageData,
  x1, y1, x2, y2, x3, y3, x4, y4,
  currentPoint,
  isStreaming,
  isPaused,
  onPointChange,
  onCurrentPointChange,
}: PerspectiveCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [imageBounds, setImageBounds] = useState<ImageBounds | null>(null);

  const points = [
    { x: x1, y: y1 },
    { x: x2, y: y2 },
    { x: x3, y: y3 },
    { x: x4, y: y4 },
  ];

  // CSSピクセルでのboundsを計算
  const calculateBounds = useCallback((): ImageBounds | null => {
    const container = containerRef.current;
    const img = imgRef.current;
    if (!container || !img) return null;
    if (img.naturalWidth === 0 || img.naturalHeight === 0) return null;

    // CSSピクセルでのコンテナサイズ
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;

    const containerRatio = containerWidth / containerHeight;
    const imageRatio = naturalWidth / naturalHeight;

    let displayWidth: number;
    let displayHeight: number;

    if (imageRatio > containerRatio) {
      // 画像が横長：幅に合わせる（上下にレターボックス）
      displayWidth = containerWidth;
      displayHeight = containerWidth / imageRatio;
    } else {
      // 画像が縦長：高さに合わせる（左右にピラーボックス）
      displayHeight = containerHeight;
      displayWidth = containerHeight * imageRatio;
    }

    const offsetX = (containerWidth - displayWidth) / 2;
    const offsetY = (containerHeight - displayHeight) / 2;

    return { offsetX, offsetY, width: displayWidth, height: displayHeight };
  }, []);

  // 画像読み込み時にboundsを更新
  const handleImageLoad = useCallback(() => {
    const bounds = calculateBounds();
    if (bounds) {
      setImageBounds(bounds);
    }
  }, [calculateBounds]);

  // クリックハンドラ - ズーム/パンを考慮
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const container = containerRef.current;
      const img = imgRef.current;
      if (!container || !img) return;
      if (img.naturalWidth === 0 || img.naturalHeight === 0) return;

      // boundsを再計算（常に最新の値を使用）
      const bounds = calculateBounds();
      if (!bounds) return;
      setImageBounds(bounds);

      // スクリーンピクセルでの矩形
      const containerRect = container.getBoundingClientRect();

      // ズーム係数（スクリーンピクセル / CSSピクセル）
      const zoomX = containerRect.width / container.clientWidth;
      const zoomY = containerRect.height / container.clientHeight;

      // クリック位置（スクリーンピクセル、コンテナ基準）
      const clickXScreen = e.clientX - containerRect.left;
      const clickYScreen = e.clientY - containerRect.top;

      // CSSピクセルに変換
      const clickXInContainer = clickXScreen / zoomX;
      const clickYInContainer = clickYScreen / zoomY;

      // 画像コンテンツ領域内での位置
      const clickXInImage = clickXInContainer - bounds.offsetX;
      const clickYInImage = clickYInContainer - bounds.offsetY;

      // 正規化座標（0-1）
      const normalizedX = clickXInImage / bounds.width;
      const normalizedY = clickYInImage / bounds.height;

      // 範囲を0-1にクランプ
      const clampedX = Math.max(0, Math.min(1, normalizedX));
      const clampedY = Math.max(0, Math.min(1, normalizedY));

      onPointChange(currentPoint, Math.round(clampedX * 100) / 100, Math.round(clampedY * 100) / 100);

      const nextPoint = currentPoint >= 4 ? 1 : currentPoint + 1;
      onCurrentPointChange(nextPoint);
    },
    [currentPoint, onPointChange, onCurrentPointChange, calculateBounds]
  );

  const isActiveOrPaused = isStreaming || isPaused;
  const shouldShowPreview = isActiveOrPaused && !!imageData;

  return (
    <div
      ref={containerRef}
      className="perspective-canvas-container nodrag"
      onClick={handleClick}
      style={{ position: 'relative', cursor: 'crosshair' }}
    >
      {shouldShowPreview ? (
        <img
          ref={imgRef}
          src={`data:image/jpeg;base64,${imageData}`}
          alt="preview"
          className="node-image"
          draggable={false}
          onLoad={handleImageLoad}
        />
      ) : (
        <div className="node-image-placeholder" />
      )}

      {/* ポイントと線のオーバーレイ（画像コンテンツ領域に配置） */}
      {shouldShowPreview && imageBounds && (
        <svg
          style={{
            position: 'absolute',
            top: imageBounds.offsetY,
            left: imageBounds.offsetX,
            width: imageBounds.width,
            height: imageBounds.height,
            pointerEvents: 'none',
          }}
        >
          {/* 点と点を結ぶ線 */}
          {points.map((point, index) => {
            const nextIndex = (index + 1) % 4;
            const nextPoint = points[nextIndex];
            return (
              <line
                key={`line-${index}`}
                x1={`${point.x * 100}%`}
                y1={`${point.y * 100}%`}
                x2={`${nextPoint.x * 100}%`}
                y2={`${nextPoint.y * 100}%`}
                stroke={POINT_COLOR}
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
            );
          })}

          {/* 各ポイントの小さい丸と右上のラベル */}
          {points.map((point, index) => {
            const isCurrentPoint = index + 1 === currentPoint;
            return (
              <g key={index}>
                <circle
                  cx={`${point.x * 100}%`}
                  cy={`${point.y * 100}%`}
                  r={isCurrentPoint ? 6 : 4}
                  fill={POINT_COLOR}
                  stroke={isCurrentPoint ? '#ffffff' : 'rgba(0, 0, 0, 0.5)'}
                  strokeWidth={isCurrentPoint ? 2 : 1}
                  vectorEffect="non-scaling-stroke"
                />
                <text
                  x={`${point.x * 100 + 1.5}%`}
                  y={`${point.y * 100 - 1.5}%`}
                  fill={POINT_COLOR}
                  fontSize="14"
                  fontWeight="bold"
                  textAnchor="start"
                  dominantBaseline="auto"
                  stroke="#000"
                  strokeWidth="0.5"
                  paintOrder="stroke"
                >
                  {index + 1}
                </text>
              </g>
            );
          })}

          {/* 現在選択中のポイント表示 */}
          <rect x="5" y="5" width="120" height="20" fill="rgba(0, 0, 0, 0.7)" rx="3" />
          <text x="10" y="18" fill={POINT_COLOR} fontSize="12" fontWeight="bold">
            Click Point {currentPoint}
          </text>
        </svg>
      )}
    </div>
  );
}
