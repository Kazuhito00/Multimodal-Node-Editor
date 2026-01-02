import { useCallback, useEffect, useRef } from 'react';

// 描画キャンバスコンポーネント（2層構造：背景 + 描画レイヤー）
export function DrawCanvas({
  backgroundImage,
  penSize,
  penColor,
  onDrawCommand,
}: {
  backgroundImage?: string;  // 入力画像（背景）
  penSize: number;
  penColor: string;  // HEX形式の色
  onDrawCommand: (command: object) => void;
}) {
  const bgCanvasRef = useRef<HTMLCanvasElement>(null);
  const drawCanvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawingRef = useRef(false);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);
  const currentStrokeRef = useRef<{ x: number; y: number }[]>([]);
  const currentColorRef = useRef<string>(penColor);

  // 画像の表示領域（contain表示時のオフセットとサイズ）
  const drawBoundsRef = useRef<{ offsetX: number; offsetY: number; width: number; height: number }>({
    offsetX: 0, offsetY: 0, width: 640, height: 360
  });

  // 現在の色を更新
  useEffect(() => {
    currentColorRef.current = penColor;
  }, [penColor]);

  // 背景画像を描画（backgroundImageが変わったときのみ）
  // 描画中でなければ描画レイヤーもクリア（Resetボタン対応）
  useEffect(() => {
    const canvas = bgCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (backgroundImage) {
      const img = new Image();
      img.onload = () => {
        // 背景をクリアして描画
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // アスペクト比を維持して収まるように描画（object-fit: contain）
        const imgAspect = img.width / img.height;
        const canvasAspect = canvas.width / canvas.height;
        let drawWidth: number;
        let drawHeight: number;
        let offsetX: number;
        let offsetY: number;

        if (imgAspect > canvasAspect) {
          // 画像が横長：幅に合わせる
          drawWidth = canvas.width;
          drawHeight = canvas.width / imgAspect;
          offsetX = 0;
          offsetY = (canvas.height - drawHeight) / 2;
        } else {
          // 画像が縦長：高さに合わせる
          drawHeight = canvas.height;
          drawWidth = canvas.height * imgAspect;
          offsetX = (canvas.width - drawWidth) / 2;
          offsetY = 0;
        }

        // 描画領域を保存（マウス座標変換用）
        drawBoundsRef.current = { offsetX, offsetY, width: drawWidth, height: drawHeight };

        ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);

        // 背景描画後に描画レイヤーをクリア（ちらつき防止）
        if (!isDrawingRef.current) {
          const drawCanvas = drawCanvasRef.current;
          const drawCtx = drawCanvas?.getContext('2d');
          if (drawCtx && drawCanvas) {
            drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
          }
        }
      };
      img.src = `data:image/jpeg;base64,${backgroundImage}`;
    } else {
      // 画像がない場合は白背景、全体を描画領域とする
      // 描画レイヤーもクリア
      if (!isDrawingRef.current) {
        const drawCanvas = drawCanvasRef.current;
        const drawCtx = drawCanvas?.getContext('2d');
        if (drawCtx && drawCanvas) {
          drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        }
      }
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      drawBoundsRef.current = { offsetX: 0, offsetY: 0, width: canvas.width, height: canvas.height };
    }
  }, [backgroundImage]);

  // マウス座標を画像領域の座標に変換（0-640, 0-360の範囲）
  const getCanvasPos = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = drawCanvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // キャンバス上のピクセル座標
    const canvasX = (e.clientX - rect.left) * scaleX;
    const canvasY = (e.clientY - rect.top) * scaleY;

    // 画像領域の座標に変換（オフセットを引いて、0-640/0-360にスケール）
    const bounds = drawBoundsRef.current;
    const relX = canvasX - bounds.offsetX;
    const relY = canvasY - bounds.offsetY;

    // 画像領域外はクランプ
    const clampedX = Math.max(0, Math.min(bounds.width, relX));
    const clampedY = Math.max(0, Math.min(bounds.height, relY));

    // 0-640, 0-360の座標にスケール
    return {
      x: (clampedX / bounds.width) * 640,
      y: (clampedY / bounds.height) * 360,
    };
  }, []);

  // 座標を画像領域からキャンバス座標に変換（プレビュー描画用）
  const toCanvasCoord = useCallback((pos: { x: number; y: number }) => {
    const bounds = drawBoundsRef.current;
    return {
      x: bounds.offsetX + (pos.x / 640) * bounds.width,
      y: bounds.offsetY + (pos.y / 360) * bounds.height,
    };
  }, []);

  // ペンサイズをキャンバススケールに変換
  const getScaledPenSize = useCallback(() => {
    const bounds = drawBoundsRef.current;
    const scale = bounds.width / 640;
    return penSize * scale;
  }, [penSize]);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    // 左クリックのみ
    if (e.button !== 0) return;

    e.preventDefault();
    e.stopPropagation();

    isDrawingRef.current = true;

    const pos = getCanvasPos(e);
    lastPosRef.current = pos;
    currentStrokeRef.current = [pos];

    // 即座に点を描画（プレビュー用、描画キャンバスに）
    const canvas = drawCanvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (ctx) {
      const canvasPos = toCanvasCoord(pos);
      const scaledPenSize = getScaledPenSize();
      ctx.fillStyle = currentColorRef.current;
      ctx.beginPath();
      ctx.arc(canvasPos.x, canvasPos.y, scaledPenSize / 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [getCanvasPos, toCanvasCoord, getScaledPenSize]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) return;

    e.preventDefault();
    e.stopPropagation();

    const pos = getCanvasPos(e);
    const lastPos = lastPosRef.current;

    if (lastPos) {
      // プレビュー用に描画キャンバスに直接描画
      const canvas = drawCanvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (ctx) {
        const canvasPos = toCanvasCoord(pos);
        const canvasLastPos = toCanvasCoord(lastPos);
        const scaledPenSize = getScaledPenSize();
        ctx.strokeStyle = currentColorRef.current;
        ctx.lineWidth = scaledPenSize;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(canvasLastPos.x, canvasLastPos.y);
        ctx.lineTo(canvasPos.x, canvasPos.y);
        ctx.stroke();
      }
    }

    currentStrokeRef.current.push(pos);
    lastPosRef.current = pos;
  }, [getCanvasPos, toCanvasCoord, getScaledPenSize]);

  const handleMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) return;

    e.preventDefault();
    e.stopPropagation();

    isDrawingRef.current = false;

    // ストロークを送信
    if (currentStrokeRef.current.length > 0) {
      onDrawCommand({
        type: 'stroke',
        points: currentStrokeRef.current,
        size: penSize,
        color: currentColorRef.current,
      });
    }

    // 描画レイヤーのクリアは背景画像更新時に行う（ちらつき防止）

    currentStrokeRef.current = [];
    lastPosRef.current = null;
  }, [onDrawCommand, penSize]);

  const handleMouseLeave = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDrawingRef.current) {
      handleMouseUp(e);
    }
  }, [handleMouseUp]);

  // 右クリックメニューを無効化
  const handleContextMenu = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  return (
    <div className="draw-canvas-container">
      {/* 背景キャンバス（下層） */}
      <canvas
        ref={bgCanvasRef}
        width={640}
        height={360}
        className="draw-canvas-bg"
      />
      {/* 描画キャンバス（上層、描画操作を受け付ける） */}
      <canvas
        ref={drawCanvasRef}
        width={640}
        height={360}
        className="draw-canvas nodrag"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onContextMenu={handleContextMenu}
        title="Left drag to draw"
      />
    </div>
  );
}
