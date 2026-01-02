import { useEffect, useRef, useState } from 'react';
import { useDragging } from '../contexts/DraggingContext';

// ブラウザWebカメラキャプチャコンポーネント
// ノード配置時にカメラ許可を要求し、STARTに同期してキャプチャを開始/停止
// プレビューはvideo要素を直接表示（パフォーマンス向上のため）
// ノードドラッグ中はフレームキャプチャを一時停止してパフォーマンスを維持
export function BrowserWebcam({
  resolution,
  onFrame,
  isStreaming,
  intervalMs = 100,
}: {
  resolution: string;
  onFrame: (base64: string) => void;
  isStreaming: boolean;
  intervalMs?: number;
}) {
  const { isDragging } = useDragging();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const isSendingRef = useRef<boolean>(false);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 解像度をパース
  const parseResolution = (res: string): { width: number; height: number } => {
    const parts = res.split('x');
    if (parts.length === 2) {
      return { width: parseInt(parts[0], 10), height: parseInt(parts[1], 10) };
    }
    return { width: 1280, height: 720 };
  };

  const { width, height } = parseResolution(resolution);

  // ノード配置時または解像度変更時にカメラを初期化
  useEffect(() => {
    let isCancelled = false;

    const initCamera = async () => {
      // 既存のストリームがあれば先に停止
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        setIsCameraReady(false);
      }

      // キャンセルされていたら中止
      if (isCancelled) return;

      try {
        setError(null);

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: width }, height: { ideal: height } },
          audio: false,
        });

        // キャンセルされていたらストリームを停止
        if (isCancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          if (!isCancelled) {
            setIsCameraReady(true);
          }
        }
      } catch (err) {
        if (isCancelled) return;
        // エラーの種類に応じたメッセージを表示
        if (err instanceof Error) {
          if (err.name === 'NotAllowedError') {
            setError('Camera access denied');
          } else if (err.name === 'NotFoundError') {
            setError('Camera not found');
          } else if (err.name === 'NotReadableError' || err.name === 'AbortError') {
            setError('Camera is in use by another app');
          } else if (err.name === 'OverconstrainedError') {
            setError('Resolution not supported');
          } else {
            setError(`Camera error: ${err.name}`);
          }
        } else {
          setError('Failed to initialize camera');
        }
        console.error('getUserMedia error:', err);
      }
    };

    initCamera();

    // ページリロード・クローズ時にカメラを解放（F5、Ctrl+F5、Ctrl+Shift+R対応）
    const handleBeforeUnload = () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);

    // アンマウント時にカメラを解放
    return () => {
      isCancelled = true;
      window.removeEventListener('beforeunload', handleBeforeUnload);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, [width, height]);

  // STARTに同期してフレームをバックエンドに送信（フレームレート制限・スキップ機能付き）
  useEffect(() => {
    if (!isStreaming || !isCameraReady || isDragging) {
      // キャプチャ停止（ドラッグ中も一時停止）
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      isSendingRef.current = false;
      return;
    }

    // 実行インターバルに合わせたフレームレート制限
    const frameInterval = intervalMs;
    let lastFrameTime = 0;

    // フレームキャプチャ開始
    const captureFrame = (currentTime: number) => {
      // フレームレート制限
      if (currentTime - lastFrameTime >= frameInterval) {
        // 前のフレームがまだ処理中ならスキップ（バックプレッシャー対策）
        if (!isSendingRef.current) {
          lastFrameTime = currentTime;

          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (video && canvas && video.readyState === video.HAVE_ENOUGH_DATA) {
            const ctx = canvas.getContext('2d');
            if (ctx) {
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
              ctx.drawImage(video, 0, 0);
              const base64 = canvas.toDataURL('image/jpeg', 0.8);
              isSendingRef.current = true;
              onFrame(base64);
              // 次のフレームを許可（非同期処理の完了を待たずに一定時間後に解除）
              setTimeout(() => {
                isSendingRef.current = false;
              }, frameInterval * 0.8);
            }
          }
        }
      }
      animationRef.current = requestAnimationFrame(captureFrame);
    };

    animationRef.current = requestAnimationFrame(captureFrame);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      isSendingRef.current = false;
    };
  }, [isStreaming, isCameraReady, isDragging, intervalMs, onFrame]);

  return (
    <div className="browser-webcam-container nodrag">
      <video
        ref={videoRef}
        playsInline
        muted
        className="node-image"
        style={{ display: isCameraReady ? 'block' : 'none' }}
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      {!isCameraReady && !error && <div className="node-image-placeholder" />}
      {error && <div className="browser-webcam-error">{error}</div>}
    </div>
  );
}
