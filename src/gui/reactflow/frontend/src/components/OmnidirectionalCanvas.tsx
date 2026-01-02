import { useState, useCallback, useRef } from 'react';

interface OmnidirectionalCanvasProps {
  imageData?: string;
  pitch: number;
  yaw: number;
  roll: number;
  isStreaming: boolean;
  isPaused: boolean;
  onPitchChange: (value: number) => void;
  onYawChange: (value: number) => void;
  onRollChange: (value: number) => void;
}

/**
 * 360度画像ビューアのキャンバス。
 * 左ドラッグ: Pitch/Yaw変更
 * 右ドラッグ: Roll変更
 */
export default function OmnidirectionalCanvas({
  imageData,
  pitch,
  yaw,
  roll,
  isStreaming,
  isPaused,
  onPitchChange,
  onYawChange,
  onRollChange,
}: OmnidirectionalCanvasProps) {
  const isActiveOrPaused = isStreaming || isPaused;
  const shouldShowPreview = isActiveOrPaused && !!imageData;
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isRightDrag, setIsRightDrag] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [startValues, setStartValues] = useState<{ pitch: number; yaw: number; roll: number }>({ pitch: 0, yaw: 0, roll: 0 });

  // ドラッグ開始
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();

      const isRight = e.button === 2;
      setIsDragging(true);
      setIsRightDrag(isRight);
      setDragStart({ x: e.clientX, y: e.clientY });
      setStartValues({ pitch, yaw, roll });
    },
    [pitch, yaw, roll]
  );

  // ドラッグ中
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!isDragging || !dragStart) return;
      e.preventDefault();
      e.stopPropagation();

      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;

      // 感度調整（ピクセルあたりの角度変化）
      const sensitivity = 0.5;

      if (isRightDrag) {
        // 右ドラッグ: Roll変更
        let newRoll = startValues.roll + Math.round(dx * sensitivity);
        // 0-359の範囲にラップ
        newRoll = ((newRoll % 360) + 360) % 360;
        onRollChange(newRoll);
      } else {
        // 左ドラッグ: Yaw/Pitch変更
        let newYaw = startValues.yaw + Math.round(dx * sensitivity);
        let newPitch = startValues.pitch - Math.round(dy * sensitivity);

        // 0-359の範囲にラップ
        newYaw = ((newYaw % 360) + 360) % 360;
        newPitch = ((newPitch % 360) + 360) % 360;

        onYawChange(newYaw);
        onPitchChange(newPitch);
      }
    },
    [isDragging, dragStart, isRightDrag, startValues, onPitchChange, onYawChange, onRollChange]
  );

  // ドラッグ終了
  const handleMouseUp = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      setDragStart(null);
    },
    []
  );

  // マウスがコンテナ外に出た場合
  const handleMouseLeave = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (isDragging) {
        handleMouseUp(e);
      }
    },
    [isDragging, handleMouseUp]
  );

  // 右クリックメニューを無効化
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
  }, []);

  return (
    <div
      ref={containerRef}
      className="omnidirectional-canvas nodrag"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      onContextMenu={handleContextMenu}
      style={{ position: 'relative', cursor: isDragging ? 'grabbing' : 'grab' }}
    >
      {shouldShowPreview ? (
        <img
          src={`data:image/jpeg;base64,${imageData}`}
          alt="preview"
          className="node-image"
          draggable={false}
        />
      ) : (
        <div className="node-image-placeholder" />
      )}
    </div>
  );
}
