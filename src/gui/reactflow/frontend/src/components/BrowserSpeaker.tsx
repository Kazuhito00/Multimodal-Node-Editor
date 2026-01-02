import { useEffect, useRef, useState } from 'react';
import { AudioData } from '../types';

// ブラウザスピーカーコンポーネント
// ノード配置時にAudioContextを初期化し、STARTに同期して再生を開始/停止
export function BrowserSpeaker({
  audioData,
  isStreaming,
}: {
  audioData?: AudioData;
  isStreaming: boolean;
}) {
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef<number>(0);
  const isPlayingRef = useRef<boolean>(false);
  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  // ノード配置時にAudioContextを初期化
  useEffect(() => {
    const initAudioContext = () => {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
        nextPlayTimeRef.current = 0;
        setIsReady(true);
      }
    };

    initAudioContext();

    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
    };
  }, []);

  // STARTに同期して再生を開始/停止
  useEffect(() => {
    if (!isReady || !audioContextRef.current) return;

    if (isStreaming && !isPlaying) {
      // 再生開始
      const ctx = audioContextRef.current;
      if (ctx.state === 'suspended') {
        ctx.resume();
      }
      isPlayingRef.current = true;
      nextPlayTimeRef.current = 0;
      setIsPlaying(true);
    } else if (!isStreaming && isPlaying) {
      // 再生停止
      isPlayingRef.current = false;
      nextPlayTimeRef.current = 0;
      setIsPlaying(false);
    }
  }, [isStreaming, isReady, isPlaying]);

  // audioDataが更新されたらバッファに追加して再生
  useEffect(() => {
    if (!isPlaying || !isPlayingRef.current) return;
    if (!audioData || !audioData.delta || audioData.delta.length === 0) return;

    const ctx = audioContextRef.current;
    if (!ctx) return;

    const sampleRate = audioData.sample_rate || 16000;
    const samples = audioData.delta;

    // AudioBufferを作成
    const buffer = ctx.createBuffer(1, samples.length, sampleRate);
    const channelData = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) {
      channelData[i] = samples[i];
    }

    // BufferSourceNodeで再生
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    // 再生時刻を計算（連続再生のため）
    const currentTime = ctx.currentTime;
    if (nextPlayTimeRef.current < currentTime) {
      nextPlayTimeRef.current = currentTime;
    }

    source.start(nextPlayTimeRef.current);
    nextPlayTimeRef.current += buffer.duration;
  }, [audioData, isPlaying]);

  return <div className="browser-speaker-container nodrag" />;
}
