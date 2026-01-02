import { useEffect, useRef, useState } from 'react';
import { useWebRTCConnection } from '../contexts/WebRTCConnectionManager';

// WebRTC Webcamコンポーネント
// WebRTCConnectionManagerを使用して接続を一元管理
// 複数のWebRTC接続（カメラ+マイク）を1つのPeerConnectionで処理
export function WebRTCWebcam({
  connectionId,
  isStreaming: _isStreaming,
  onConnectionReady,
}: {
  connectionId: string;
  isStreaming: boolean;
  onConnectionReady?: () => void;
}) {
  void _isStreaming;
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // connectionIdからnodeIdを抽出
  const nodeId = connectionId.replace('webrtc_webcam_', '');

  // WebRTC接続マネージャー
  const { registerTrack, unregisterTrack, getConnectionState } = useWebRTCConnection();
  const connectionState = getConnectionState('video', nodeId);

  // 固定解像度（1280x720）
  const width = 1280;
  const height = 720;

  // カメラ初期化
  useEffect(() => {
    let isCancelled = false;

    const initCamera = async () => {
      try {
        setError(null);

        const newStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: width }, height: { ideal: height } },
          audio: false,
        });

        if (isCancelled) {
          newStream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = newStream;

        if (videoRef.current) {
          videoRef.current.srcObject = newStream;
          await videoRef.current.play();
          if (!isCancelled) {
            setIsCameraReady(true);
          }
        }
      } catch (err) {
        if (isCancelled) return;
        console.error('[WebRTCWebcam] getUserMedia error:', err);
        if (err instanceof Error) {
          if (err.name === 'NotAllowedError') {
            setError('Camera access denied');
          } else if (err.name === 'NotFoundError') {
            setError('Camera not found');
          } else if (err.name === 'NotReadableError' || err.name === 'AbortError') {
            setError('Camera is in use');
          } else {
            setError(`Camera error: ${err.name}`);
          }
        } else {
          setError('Camera init failed');
        }
      }
    };

    initCamera();

    // ページアンロード時にカメラ解放
    const handleBeforeUnload = () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      isCancelled = true;
      window.removeEventListener('beforeunload', handleBeforeUnload);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  // カメラ準備完了時にマネージャーにトラックを登録
  const isRegisteredRef = useRef(false);

  useEffect(() => {
    if (isCameraReady && streamRef.current) {
      console.log(`[WebRTCWebcam] Registering video track for ${nodeId}`);
      registerTrack('video', nodeId, streamRef.current);
      isRegisteredRef.current = true;
    }

    return () => {
      // 登録済みの場合のみ解除
      if (isRegisteredRef.current) {
        console.log(`[WebRTCWebcam] Unregistering video track for ${nodeId}`);
        unregisterTrack('video', nodeId);
        isRegisteredRef.current = false;
      }
    };
  }, [isCameraReady, nodeId, registerTrack, unregisterTrack]);

  // マウント時にconnection_idをプロパティとして通知
  useEffect(() => {
    if (onConnectionReady) {
      onConnectionReady();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // カメラ準備完了かつWebRTC接続完了で表示
  const isReady = isCameraReady && connectionState.isConnected;

  return (
    <div className="browser-webcam-container nodrag">
      <video
        ref={videoRef}
        playsInline
        muted
        className="node-image"
        style={{ display: isReady ? 'block' : 'none' }}
      />
      {!isReady && !error && !connectionState.error && (
        <div className="node-image-placeholder node-image-loading">Loading...</div>
      )}
      {error && <div className="browser-webcam-error">{error}</div>}
      {connectionState.error && <div className="browser-webcam-error">{connectionState.error}</div>}
    </div>
  );
}
