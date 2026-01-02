import { useEffect, useRef, useState } from 'react';

// リサンプリング関数
function resampleAudio(inputSamples: Float32Array, inputSampleRate: number, targetRate: number): number[] {
  if (inputSampleRate === targetRate) {
    return Array.from(inputSamples);
  }

  const ratio = inputSampleRate / targetRate;
  const outputLength = Math.floor(inputSamples.length / ratio);
  const output: number[] = [];

  for (let i = 0; i < outputLength; i++) {
    const srcIndex = i * ratio;
    const srcIndexFloor = Math.floor(srcIndex);
    const srcIndexCeil = Math.min(srcIndexFloor + 1, inputSamples.length - 1);
    const t = srcIndex - srcIndexFloor;
    // 線形補間
    const value = inputSamples[srcIndexFloor] * (1 - t) + inputSamples[srcIndexCeil] * t;
    output.push(value);
  }

  return output;
}

// ブラウザマイクキャプチャコンポーネント
// ノード配置時にマイク許可を要求し、STARTに同期してキャプチャを開始/停止
export function BrowserMicrophone({
  echoCancellation,
  noiseSuppression,
  autoGainControl,
  onAudio,
  isStreaming,
}: {
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
  onAudio: (json: string) => void;
  isStreaming: boolean;
}) {
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sequenceRef = useRef<number>(0);
  const targetSampleRateRef = useRef<number>(16000);
  const isSendingRef = useRef<boolean>(false);
  const skipCountRef = useRef<number>(0);
  const [isMicReady, setIsMicReady] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // オプション変更検知用
  const optionsRef = useRef({ echoCancellation, noiseSuppression, autoGainControl });

  // 起動時にサンプルレート設定を取得
  useEffect(() => {
    fetch('/api/settings/audio')
      .then((res) => res.json())
      .then((data) => {
        if (data.sample_rate) {
          targetSampleRateRef.current = data.sample_rate;
        }
      })
      .catch(() => {
        // エラー時はデフォルト値を使用
      });
  }, []);

  // ノード配置時にマイク許可を要求してプレビュー開始
  useEffect(() => {
    let isCancelled = false;

    const initMicrophone = async () => {
      // 既存のストリームがあれば先に停止（React Strict Mode対策）
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
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

        // AudioContextを作成
        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;

        const source = audioContext.createMediaStreamSource(stream);
        sourceRef.current = source;

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
            setError('Microphone is in use by another app');
          } else {
            setError(`Microphone error: ${err.name}`);
          }
        } else {
          setError('Failed to initialize microphone');
        }
        console.error('getUserMedia error:', err);
      }
    };

    initMicrophone();

    // ページリロード・クローズ時にマイクを解放（F5、Ctrl+F5、Ctrl+Shift+R対応）
    const handleBeforeUnload = () => {
      if (processorRef.current) {
        processorRef.current.disconnect();
      }
      if (sourceRef.current) {
        sourceRef.current.disconnect();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      isCancelled = true;
      window.removeEventListener('beforeunload', handleBeforeUnload);
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      if (sourceRef.current) {
        sourceRef.current.disconnect();
        sourceRef.current = null;
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  // STARTに同期してキャプチャを開始/停止
  useEffect(() => {
    if (!isMicReady || !audioContextRef.current || !sourceRef.current) return;

    if (isStreaming && !isCapturing) {
      // キャプチャ開始
      const audioContext = audioContextRef.current;
      const source = sourceRef.current;

      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      sequenceRef.current = 0;
      skipCountRef.current = 0;
      isSendingRef.current = false;

      const targetRate = targetSampleRateRef.current;
      processor.onaudioprocess = (e) => {
        // 前の送信がまだ処理中ならスキップ（バックプレッシャー対策）
        if (isSendingRef.current) {
          skipCountRef.current++;
          // 連続スキップが多すぎる場合はリセット（詰まり防止）
          if (skipCountRef.current > 10) {
            isSendingRef.current = false;
            skipCountRef.current = 0;
          }
          return;
        }
        skipCountRef.current = 0;
        isSendingRef.current = true;

        const inputData = e.inputBuffer.getChannelData(0);
        const resampled = resampleAudio(inputData, audioContext.sampleRate, targetRate);
        const seq = sequenceRef.current++;
        onAudio(JSON.stringify({ samples: resampled, seq }));

        // 次の送信を許可（短い遅延後）
        setTimeout(() => {
          isSendingRef.current = false;
        }, 50);
      };

      source.connect(processor);
      // 無音で出力（processorを動作させるため必要）
      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      processor.connect(silentGain);
      silentGain.connect(audioContext.destination);

      setIsCapturing(true);
      optionsRef.current = { echoCancellation, noiseSuppression, autoGainControl };
    } else if (!isStreaming && isCapturing) {
      // キャプチャ停止
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      sequenceRef.current = 0;
      skipCountRef.current = 0;
      isSendingRef.current = false;
      setIsCapturing(false);
    }
  }, [isStreaming, isMicReady, isCapturing, echoCancellation, noiseSuppression, autoGainControl, onAudio]);

  // オプション変更時にマイクを再初期化
  useEffect(() => {
    if (!isMicReady) return;

    const optionsChanged =
      optionsRef.current.echoCancellation !== echoCancellation ||
      optionsRef.current.noiseSuppression !== noiseSuppression ||
      optionsRef.current.autoGainControl !== autoGainControl;

    if (optionsChanged) {
      optionsRef.current = { echoCancellation, noiseSuppression, autoGainControl };

      // マイクを再取得する必要がある場合は一度停止して再初期化
      const reinitMicrophone = async () => {
        if (processorRef.current) {
          processorRef.current.disconnect();
          processorRef.current = null;
        }
        if (sourceRef.current) {
          sourceRef.current.disconnect();
          sourceRef.current = null;
        }
        if (audioContextRef.current) {
          audioContextRef.current.close();
          audioContextRef.current = null;
        }
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
        }

        setIsMicReady(false);
        setIsCapturing(false);

        await new Promise((resolve) => setTimeout(resolve, 100));

        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation, noiseSuppression, autoGainControl },
            video: false,
          });
          streamRef.current = stream;

          const audioContext = new AudioContext();
          audioContextRef.current = audioContext;

          const source = audioContext.createMediaStreamSource(stream);
          sourceRef.current = source;

          setIsMicReady(true);
        } catch (err) {
          console.error('Microphone reinit error:', err);
          setError('Failed to reinitialize microphone');
        }
      };

      reinitMicrophone();
    }
  }, [echoCancellation, noiseSuppression, autoGainControl, isMicReady]);

  if (error) {
    return <div className="browser-microphone-error">{error}</div>;
  }
  return null;
}
