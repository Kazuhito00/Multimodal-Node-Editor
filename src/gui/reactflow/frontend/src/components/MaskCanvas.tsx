import { useCallback, useEffect, useRef } from 'react';

// マスク描画キャンバスコンポーネント（2層構造：背景 + マスク）
export function MaskCanvas({
  backgroundImage,
  maskImage,
  penSize,
  onDrawCommand,
}: {
  backgroundImage?: string;  // 入力画像（背景）
  maskImage?: string;        // マスク画像（オーバーレイ）
  penSize: number;
  onDrawCommand: (command: object) => void;
}) {
  const bgCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawingRef = useRef(false);
  const isErasingRef = useRef(false);  // 右ドラッグで消しゴムモード
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);
  const currentStrokeRef = useRef<{ x: number; y: number }[]>([]);
  const bgImageRef = useRef<HTMLImageElement | null>(null);
  // 画像の表示領域（contain表示時のオフセットとサイズ）
  const drawBoundsRef = useRef<{ offsetX: number; offsetY: number; width: number; height: number }>({
    offsetX: 0, offsetY: 0, width: 640, height: 360
  });

  // 背景画像を描画（backgroundImageが変わったときのみ）
  useEffect(() => {
    const canvas = bgCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (backgroundImage) {
      const img = new Image();
      img.onload = () => {
        // 画像読み込み完了後にクリアして描画（ちらつき防止）
        ctx.fillStyle = '#000';
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
        bgImageRef.current = img;
      };
      img.src = `data:image/jpeg;base64,${backgroundImage}`;
    } else {
      // 画像がない場合は黒背景、全体を描画領域とする
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      drawBoundsRef.current = { offsetX: 0, offsetY: 0, width: canvas.width, height: canvas.height };
      bgImageRef.current = null;
    }
  }, [backgroundImage]);

  // マスク画像を描画（maskImageが変わったときのみ）
  useEffect(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (maskImage) {
      const mask = new Image();
      mask.onload = () => {
        // 画像読み込み完了後にクリアして描画（ちらつき防止）
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // アスペクト比を維持して収まるように描画（背景と同じ位置に）
        const imgAspect = mask.width / mask.height;
        const canvasAspect = canvas.width / canvas.height;
        let drawWidth: number;
        let drawHeight: number;
        let offsetX: number;
        let offsetY: number;

        if (imgAspect > canvasAspect) {
          drawWidth = canvas.width;
          drawHeight = canvas.width / imgAspect;
          offsetX = 0;
          offsetY = (canvas.height - drawHeight) / 2;
        } else {
          drawHeight = canvas.height;
          drawWidth = canvas.height * imgAspect;
          offsetX = (canvas.width - drawWidth) / 2;
          offsetY = 0;
        }

        ctx.globalAlpha = 0.5;
        ctx.drawImage(mask, offsetX, offsetY, drawWidth, drawHeight);
        ctx.globalAlpha = 1.0;
      };
      mask.src = `data:image/jpeg;base64,${maskImage}`;
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [maskImage]);

  // マウス座標を画像領域の座標に変換（0-640, 0-360の範囲）
  const getCanvasPos = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = maskCanvasRef.current;
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
    e.preventDefault();
    e.stopPropagation();

    // 左クリック(0)で描画、右クリック(2)で消しゴム
    const isErasing = e.button === 2;
    isDrawingRef.current = !isErasing;
    isErasingRef.current = isErasing;

    const pos = getCanvasPos(e);
    lastPosRef.current = pos;
    currentStrokeRef.current = [pos];

    // 即座に点を描画（プレビュー用、マスクキャンバスに）
    const canvas = maskCanvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (ctx) {
      const canvasPos = toCanvasCoord(pos);
      const scaledPenSize = getScaledPenSize();
      ctx.fillStyle = isErasing ? '#000' : '#fff';
      ctx.beginPath();
      ctx.arc(canvasPos.x, canvasPos.y, scaledPenSize / 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [getCanvasPos, toCanvasCoord, getScaledPenSize]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current && !isErasingRef.current) return;

    e.preventDefault();
    e.stopPropagation();

    const pos = getCanvasPos(e);
    const lastPos = lastPosRef.current;

    if (lastPos) {
      // プレビュー用にマスクキャンバスに直接描画
      const canvas = maskCanvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (ctx) {
        const canvasPos = toCanvasCoord(pos);
        const canvasLastPos = toCanvasCoord(lastPos);
        const scaledPenSize = getScaledPenSize();
        ctx.strokeStyle = isErasingRef.current ? '#000' : '#fff';
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
    if (!isDrawingRef.current && !isErasingRef.current) return;

    e.preventDefault();
    e.stopPropagation();

    const wasErasing = isErasingRef.current;
    isDrawingRef.current = false;
    isErasingRef.current = false;

    // ストロークを送信（消しゴムの場合はeraseタイプ）
    if (currentStrokeRef.current.length > 0) {
      onDrawCommand({
        type: wasErasing ? 'erase' : 'stroke',
        points: currentStrokeRef.current,
        size: penSize,
      });
    }

    currentStrokeRef.current = [];
    lastPosRef.current = null;
  }, [onDrawCommand, penSize]);

  const handleMouseLeave = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDrawingRef.current || isErasingRef.current) {
      handleMouseUp(e);
    }
  }, [handleMouseUp]);

  // 右クリックメニューを無効化（右ドラッグで消しゴムを使うため）
  const handleContextMenu = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  return (
    <div className="mask-canvas-container">
      {/* 背景キャンバス（下層） */}
      <canvas
        ref={bgCanvasRef}
        width={640}
        height={360}
        className="mask-canvas-bg"
      />
      {/* マスクキャンバス（上層、描画操作を受け付ける） */}
      <canvas
        ref={maskCanvasRef}
        width={640}
        height={360}
        className="mask-canvas nodrag"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onContextMenu={handleContextMenu}
        title="Left drag: draw, Right drag: erase"
      />
    </div>
  );
}
