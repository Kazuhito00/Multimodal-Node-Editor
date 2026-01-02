import { useEffect, useRef } from 'react';
import { AudioData } from '../types';

// Waveform描画コンポーネント
export function WaveformCanvas({ audioData }: { audioData?: AudioData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // ライトモード判定
    const isLightMode = document.querySelector('.app-container.light') !== null;

    const width = canvas.width;
    const height = canvas.height;
    const centerY = height / 2;

    // 背景をクリア
    ctx.fillStyle = isLightMode ? '#ddd' : '#000';
    ctx.fillRect(0, 0, width, height);

    if (!audioData || !audioData.waveform || audioData.waveform.length === 0) {
      // データがない場合は中央線だけ描画
      ctx.strokeStyle = isLightMode ? '#bbb' : '#333';
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();
      return;
    }

    const waveform = audioData.waveform;

    // 中央線
    ctx.strokeStyle = isLightMode ? '#bbb' : '#333';
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();

    // 波形を描画（min/maxペアが事前計算済み）
    ctx.strokeStyle = isLightMode ? '#00aa00' : '#00ff00';
    ctx.lineWidth = 1;

    // waveformは[min0, max0, min1, max1, ...]の形式
    const numPixels = Math.floor(waveform.length / 2);
    for (let x = 0; x < numPixels && x < width; x++) {
      const minVal = waveform[x * 2] || 0;
      const maxVal = waveform[x * 2 + 1] || 0;

      // -1 to 1 の範囲を height にマッピング
      const yMin = centerY - minVal * centerY * 0.9;
      const yMax = centerY - maxVal * centerY * 0.9;

      ctx.beginPath();
      ctx.moveTo(x, yMin);
      ctx.lineTo(x, yMax);
      ctx.stroke();
    }

    // ラベルを左上に描画
    if (audioData.label) {
      ctx.font = '11px sans-serif';
      ctx.fillStyle = isLightMode ? '#000' : '#fff';
      ctx.textBaseline = 'top';
      ctx.fillText(audioData.label, 4, 4);
    }
  }, [audioData]);

  return (
    <canvas
      ref={canvasRef}
      width={200}
      height={80}
      className="node-waveform"
    />
  );
}
