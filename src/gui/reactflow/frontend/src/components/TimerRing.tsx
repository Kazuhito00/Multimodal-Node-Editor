import { useEffect, useRef } from 'react';

// タイマーリングコンポーネント
export function TimerRing({
  interval,
  unit,
  isStreaming,
}: {
  interval: number;
  unit: string;
  isStreaming: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // インターバルをミリ秒に変換
    const intervalMs = unit === 'ms' ? interval : interval * 1000;

    // ライトモード判定
    const isLightMode = document.querySelector('.app-container.light') !== null;
    const bgColor = isLightMode ? '#ddd' : '#222';
    const trackColor = isLightMode ? '#bbb' : '#444';
    const progressColor = '#4caf50';

    const size = canvas.width;
    const centerX = size / 2;
    const centerY = size / 2;
    const radius = size / 2 - 8;
    const lineWidth = 6;

    const drawRing = (progress: number) => {
      ctx.clearRect(0, 0, size, size);

      // 背景円
      ctx.fillStyle = bgColor;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius + lineWidth / 2, 0, Math.PI * 2);
      ctx.fill();

      // トラック（グレーのリング）
      ctx.strokeStyle = trackColor;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.stroke();

      // プログレス（アニメーションリング）
      if (progress > 0) {
        ctx.strokeStyle = progressColor;
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.beginPath();
        // 上（-90度）から時計回りに描画
        const startAngle = -Math.PI / 2;
        const endAngle = startAngle + (Math.PI * 2 * progress);
        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
        ctx.stroke();
      }

      // 中央にインターバル表示
      ctx.fillStyle = isLightMode ? '#333' : '#fff';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const displayText = unit === 'ms' ? `${interval}ms` : `${interval}s`;
      ctx.fillText(displayText, centerX, centerY);
    };

    if (!isStreaming) {
      // 停止中は0%で描画
      drawRing(0);
      startTimeRef.current = 0;
      return;
    }

    // アニメーション開始
    startTimeRef.current = performance.now();

    const animate = () => {
      const now = performance.now();
      const elapsed = now - startTimeRef.current;
      const progress = (elapsed % intervalMs) / intervalMs;

      drawRing(progress);

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [interval, unit, isStreaming]);

  return (
    <canvas
      ref={canvasRef}
      width={80}
      height={80}
      className="timer-ring"
    />
  );
}
