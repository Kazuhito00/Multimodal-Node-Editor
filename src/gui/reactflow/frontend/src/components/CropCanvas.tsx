import { useState, useCallback, useRef } from 'react';

interface CropCanvasProps {
  imageData?: string;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  isStreaming: boolean;
  isPaused: boolean;
  onCropChange: (minX: number, minY: number, maxX: number, maxY: number) => void;
}

interface ImageBounds {
  offsetX: number;
  offsetY: number;
  width: number;
  height: number;
}

/**
 * 画像上でドラッグしてクロップ領域を指定するキャンバス。
 * 左ドラッグで矩形を描画し、離すとmin_x/min_y/max_x/max_yを更新する。
 */
export default function CropCanvas({
  imageData,
  minX,
  minY,
  maxX,
  maxY,
  isStreaming,
  isPaused,
  onCropChange,
}: CropCanvasProps) {
  const isActiveOrPaused = isStreaming || isPaused;
  const shouldShowPreview = isActiveOrPaused && !!imageData;
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragEnd, setDragEnd] = useState<{ x: number; y: number } | null>(null);
  const [imageBounds, setImageBounds] = useState<ImageBounds | null>(null);

  // CSSピクセルでのboundsを計算（object-fit: contain対応）
  const calculateBounds = useCallback((): ImageBounds | null => {
    const container = containerRef.current;
    const img = imgRef.current;
    if (!container || !img) return null;
    if (img.naturalWidth === 0 || img.naturalHeight === 0) return null;

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;

    const containerRatio = containerWidth / containerHeight;
    const imageRatio = naturalWidth / naturalHeight;

    let displayWidth: number;
    let displayHeight: number;

    if (imageRatio > containerRatio) {
      displayWidth = containerWidth;
      displayHeight = containerWidth / imageRatio;
    } else {
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

  // マウス座標を正規化座標（0～1）に変換（ズーム/object-fit対応）
  const getNormalizedCoords = useCallback(
    (e: React.MouseEvent<HTMLDivElement>): { x: number; y: number } => {
      const container = containerRef.current;
      if (!container) return { x: 0, y: 0 };

      // boundsを再計算（常に最新の値を使用）
      const bounds = calculateBounds();
      if (!bounds) return { x: 0, y: 0 };

      const rect = container.getBoundingClientRect();

      // ズーム係数（スクリーンピクセル / CSSピクセル）
      const zoomX = rect.width / container.clientWidth;
      const zoomY = rect.height / container.clientHeight;

      // クリック位置（スクリーンピクセル、コンテナ基準）
      const clickXScreen = e.clientX - rect.left;
      const clickYScreen = e.clientY - rect.top;

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
      const x = Math.max(0, Math.min(1, normalizedX));
      const y = Math.max(0, Math.min(1, normalizedY));

      return { x, y };
    },
    [calculateBounds]
  );

  // ドラッグ開始
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();

      const coords = getNormalizedCoords(e);
      setIsDragging(true);
      setDragStart(coords);
      setDragEnd(coords);
    },
    [getNormalizedCoords]
  );

  // ドラッグ中
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!isDragging) return;
      e.preventDefault();
      e.stopPropagation();

      const coords = getNormalizedCoords(e);
      setDragEnd(coords);
    },
    [isDragging, getNormalizedCoords]
  );

  // ドラッグ終了
  const handleMouseUp = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!isDragging || !dragStart || !dragEnd) return;
      e.preventDefault();
      e.stopPropagation();

      setIsDragging(false);

      const newMinX = Math.min(dragStart.x, dragEnd.x);
      const newMaxX = Math.max(dragStart.x, dragEnd.x);
      const newMinY = Math.min(dragStart.y, dragEnd.y);
      const newMaxY = Math.max(dragStart.y, dragEnd.y);

      onCropChange(
        Math.round(newMinX * 100) / 100,
        Math.round(newMinY * 100) / 100,
        Math.round(newMaxX * 100) / 100,
        Math.round(newMaxY * 100) / 100
      );

      setDragStart(null);
      setDragEnd(null);
    },
    [isDragging, dragStart, dragEnd, onCropChange]
  );

  // マウスがコンテナ外に出た場合
  const handleMouseLeave = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (isDragging && dragStart && dragEnd) {
        handleMouseUp(e);
      }
    },
    [isDragging, dragStart, dragEnd, handleMouseUp]
  );

  // 選択領域の表示位置（画像コンテンツ領域基準）
  const getSelectionStyle = useCallback(() => {
    const bounds = imageBounds;
    if (!bounds) return {};

    const currentMinX = isDragging && dragStart && dragEnd ? Math.min(dragStart.x, dragEnd.x) : minX;
    const currentMinY = isDragging && dragStart && dragEnd ? Math.min(dragStart.y, dragEnd.y) : minY;
    const currentMaxX = isDragging && dragStart && dragEnd ? Math.max(dragStart.x, dragEnd.x) : maxX;
    const currentMaxY = isDragging && dragStart && dragEnd ? Math.max(dragStart.y, dragEnd.y) : maxY;

    return {
      left: bounds.offsetX + currentMinX * bounds.width,
      top: bounds.offsetY + currentMinY * bounds.height,
      width: (currentMaxX - currentMinX) * bounds.width,
      height: (currentMaxY - currentMinY) * bounds.height,
    };
  }, [imageBounds, isDragging, dragStart, dragEnd, minX, minY, maxX, maxY]);

  const isFullSelection = minX === 0 && minY === 0 && maxX === 1 && maxY === 1;
  const showSelection = shouldShowPreview && imageBounds && (isDragging || !isFullSelection);

  return (
    <div
      ref={containerRef}
      className="crop-canvas nodrag"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
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

      {/* 選択領域のオーバーレイ（画像コンテンツ領域に配置） */}
      {showSelection && (
        <div
          className="crop-selection"
          style={{
            position: 'absolute',
            ...getSelectionStyle(),
            border: '2px dashed #00ff00',
            backgroundColor: 'rgba(0, 255, 0, 0.1)',
            pointerEvents: 'none',
            boxSizing: 'border-box',
          }}
        />
      )}
    </div>
  );
}
