import { useEffect, useRef, useState } from 'react';
import { useWebRTCConnection } from '../contexts/WebRTCConnectionManager';

// WebRTC Microphoneコンポーネント
// WebRTCConnectionManagerを使用して接続を一元管理
// 複数のWebRTC接続（カメラ+マイク）を1つのPeerConnectionで処理
export function WebRTCMicrophone({
  echoCancellation,
  noiseSuppression,
  autoGainControl,
  connectionId,
  isStreaming: _isStreaming,
  onConnectionReady,
  onReadyStateChange,
}: {
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
  connectionId: string;
  isStreaming: boolean;
  onConnectionReady?: () => void;
  onReadyStateChange?: (isReady: boolean) => void;
}) {
  void _isStreaming;
  const streamRef = useRef<MediaStream | null>(null);
  const [isMicReady, setIsMicReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const optionsRef = useRef({ echoCancellation, noiseSuppression, autoGainControl });

  // connectionIdからnodeIdを抽出
  const nodeId = connectionId.replace('webrtc_microphone_', '');

  // WebRTC接続マネージャー
  const { registerTrack, unregisterTrack, getConnectionState } = useWebRTCConnection();
  const connectionState = getConnectionState('audio', nodeId);

  // マイク初期化
  useEffect(() => {
    console.log(`[WebRTCMicrophone] Mount/init useEffect: connectionId=${connectionId}`);
    let isCancelled = false;

    const initMicrophone = async () => {
      console.log('[WebRTCMicrophone] initMicrophone called');
      // 既存のストリームがあれば停止
      if (streamRef.current) {
        console.log('[WebRTCMicrophone] Stopping existing stream');
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        setIsMicReady(false);
      }

      // マイク解放が完了するまで少し待機
      await new Promise((resolve) => setTimeout(resolve, 100));

      if (isCancelled) return;

      try {
        setError(null);

        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: optionsRef.current.echoCancellation,
            noiseSuppression: optionsRef.current.noiseSuppression,
            autoGainControl: optionsRef.current.autoGainControl,
          },
          video: false,
        });

        if (isCancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        if (!isCancelled) {
          setIsMicReady(true);
        }
      } catch (err) {
        if (isCancelled) return;
        if (err instanceof Error) {
          if (err.name === 'NotAllowedError') {
            setError('Microphone access denied');
          } else if (err.name === 'NotFoundError') {
            setError('Microphone not found');
          } else if (err.name === 'NotReadableError' || err.name === 'AbortError') {
            setError('Microphone is in use');
          } else {
            setError(`Microphone error: ${err.name}`);
          }
        } else {
          setError('Microphone init failed');
        }
        console.error('[WebRTCMicrophone] getUserMedia error:', err);
      }
    };

    initMicrophone();

    // ページアンロード時にマイク解放
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
  }, [connectionId]);

  // オプション変更時にマイクを再初期化
  const prevOptionsRef = useRef({ echoCancellation, noiseSuppression, autoGainControl });

  useEffect(() => {
    // 接続完了していない場合は何もしない
    if (!isMicReady || !connectionState.isConnected) return;

    // オプションが実際に変更されたかチェック
    const optionsChanged =
      prevOptionsRef.current.echoCancellation !== echoCancellation ||
      prevOptionsRef.current.noiseSuppression !== noiseSuppression ||
      prevOptionsRef.current.autoGainControl !== autoGainControl;

    if (!optionsChanged) return;

    prevOptionsRef.current = { echoCancellation, noiseSuppression, autoGainControl };
    optionsRef.current = { echoCancellation, noiseSuppression, autoGainControl };

    const reinitMicrophone = async () => {
      // トラックを解除
      unregisterTrack('audio', nodeId);

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      setIsMicReady(false);

      await new Promise((resolve) => setTimeout(resolve, 100));

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation, noiseSuppression, autoGainControl },
          video: false,
        });
        streamRef.current = stream;
        setIsMicReady(true);
      } catch (err) {
        console.error('[WebRTCMicrophone] Reinit error:', err);
        setError('Failed to reinitialize microphone');
      }
    };

    reinitMicrophone();
  }, [echoCancellation, noiseSuppression, autoGainControl, isMicReady, connectionState.isConnected, nodeId, unregisterTrack]);

  // マイク準備完了時にマネージャーにトラックを登録
  const isRegisteredRef = useRef(false);

  useEffect(() => {
    if (isMicReady && streamRef.current) {
      console.log(`[WebRTCMicrophone] Registering audio track for ${nodeId}`);
      registerTrack('audio', nodeId, streamRef.current);
      isRegisteredRef.current = true;
    }

    return () => {
      // 登録済みの場合のみ解除
      if (isRegisteredRef.current) {
        console.log(`[WebRTCMicrophone] Unregistering audio track for ${nodeId}`);
        unregisterTrack('audio', nodeId);
        isRegisteredRef.current = false;
      }
    };
  }, [isMicReady, nodeId, registerTrack, unregisterTrack]);

  // マウント時にconnection_idをプロパティとして通知
  useEffect(() => {
    if (onConnectionReady) {
      onConnectionReady();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 接続状態が変化したら親に通知
  const prevReadyRef = useRef<boolean | null>(null);
  useEffect(() => {
    const isReady = isMicReady && connectionState.isConnected && !error;
    if (prevReadyRef.current !== isReady) {
      prevReadyRef.current = isReady;
      if (onReadyStateChange) {
        onReadyStateChange(isReady);
      }
    }
  }, [isMicReady, connectionState.isConnected, error, onReadyStateChange]);

  // エラー時のみエラー表示を返す
  if (error) {
    return <div className="browser-microphone-error">{error}</div>;
  }

  if (connectionState.error) {
    return <div className="browser-microphone-error">{connectionState.error}</div>;
  }

  // WebRTCMicrophoneはコネクション管理のみ担当し、UIは返さない
  return null;
}
