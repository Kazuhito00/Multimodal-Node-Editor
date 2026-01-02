import { createContext, useContext, useRef, useCallback, useState, useEffect, ReactNode } from 'react';

// トラック情報
interface TrackInfo {
  kind: 'video' | 'audio';
  nodeId: string;
  track: MediaStreamTrack;
  stream: MediaStream;
}

// 接続状態
interface ConnectionState {
  isConnecting: boolean;
  isConnected: boolean;
  error: string | null;
}

// コンテキストの型
interface WebRTCConnectionContextType {
  // トラックを登録（カメラ/マイクコンポーネントから呼び出し）
  registerTrack: (kind: 'video' | 'audio', nodeId: string, stream: MediaStream) => void;
  // トラックを解除
  unregisterTrack: (kind: 'video' | 'audio', nodeId: string) => void;
  // 接続状態を取得
  getConnectionState: (kind: 'video' | 'audio', nodeId: string) => ConnectionState;
  // 特定ノードの接続が完了しているか
  isNodeConnected: (kind: 'video' | 'audio', nodeId: string) => boolean;
}

const WebRTCConnectionContext = createContext<WebRTCConnectionContextType | null>(null);

// WebRTC接続を一元管理するプロバイダー
export function WebRTCConnectionProvider({ children }: { children: ReactNode }) {
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const tracksRef = useRef<Map<string, TrackInfo>>(new Map());
  const [connectionStates, setConnectionStates] = useState<Map<string, ConnectionState>>(new Map());
  const isConnectingRef = useRef(false);
  const needsReconnectRef = useRef(false);  // 接続中にトラックが変更された場合のフラグ
  const lastTrackCountRef = useRef(0);  // 接続開始時のトラック数
  const sessionIdRef = useRef<string>(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const connectionTimeoutRef = useRef<number | null>(null);  // 接続遅延タイマー
  const isFirstConnectionRef = useRef(true);  // 初回接続かどうか

  // キーを生成
  const getTrackKey = (kind: 'video' | 'audio', nodeId: string) => `${kind}_${nodeId}`;

  // 接続状態を更新
  const updateConnectionState = useCallback((kind: 'video' | 'audio', nodeId: string, state: Partial<ConnectionState>) => {
    const key = getTrackKey(kind, nodeId);
    setConnectionStates(prev => {
      const newMap = new Map(prev);
      const current = newMap.get(key) || { isConnecting: false, isConnected: false, error: null };
      newMap.set(key, { ...current, ...state });
      return newMap;
    });
  }, []);

  // PeerConnectionを作成または取得
  const getOrCreatePeerConnection = useCallback(() => {
    if (pcRef.current && pcRef.current.connectionState !== 'closed') {
      return pcRef.current;
    }

    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    });

    pc.onconnectionstatechange = () => {
      console.log(`[WebRTCManager] Connection state: ${pc.connectionState}`);
      if (pc.connectionState === 'connected') {
        // 全ての登録済みトラックを接続完了にする
        tracksRef.current.forEach((info) => {
          updateConnectionState(info.kind, info.nodeId, { isConnecting: false, isConnected: true });
        });
      } else if (pc.connectionState === 'failed' || pc.connectionState === 'closed') {
        // 全てのトラックを切断状態にする
        tracksRef.current.forEach((info) => {
          updateConnectionState(info.kind, info.nodeId, { isConnecting: false, isConnected: false });
        });
      }
    };

    pcRef.current = pc;
    return pc;
  }, [updateConnectionState]);

  // 既存のPeerConnectionに追加済みのトラックを追跡
  const addedTracksRef = useRef<Set<string>>(new Set());

  // 接続を確立または再ネゴシエーション
  const establishConnection = useCallback(async () => {
    if (isConnectingRef.current) {
      // 接続中に呼ばれた場合、再接続が必要とマーク
      console.log('[WebRTCManager] Already connecting, marking for reconnect');
      needsReconnectRef.current = true;
      return;
    }

    if (tracksRef.current.size === 0) {
      console.log('[WebRTCManager] No tracks registered, skipping');
      return;
    }

    isConnectingRef.current = true;
    needsReconnectRef.current = false;
    lastTrackCountRef.current = tracksRef.current.size;

    try {
      const isNewConnection = !pcRef.current || pcRef.current.connectionState === 'closed';

      // 既存の接続がある場合は閉じて新しく作り直す（Colab対応）
      if (pcRef.current && pcRef.current.connectionState !== 'closed') {
        console.log('[WebRTCManager] Closing existing connection for renegotiation');
        pcRef.current.close();
        pcRef.current = null;
        addedTracksRef.current.clear();
        // 新しいセッションIDを生成
        sessionIdRef.current = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        // バックエンドのクリーンアップを待つ
        await new Promise(resolve => setTimeout(resolve, 300));
      }

      const pc = getOrCreatePeerConnection();

      // 全トラックを追加
      const trackInfos: { kind: string; node_id: string }[] = [];
      tracksRef.current.forEach((info, key) => {
        pc.addTrack(info.track, info.stream);
        addedTracksRef.current.add(key);
        trackInfos.push({ kind: info.kind, node_id: info.nodeId });
        updateConnectionState(info.kind, info.nodeId, { isConnecting: true, error: null });
      });

      console.log(`[WebRTCManager] Creating offer with ${trackInfos.length} tracks (new=${isNewConnection})`);

      // Offerを作成
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // ICE候補の収集を待つ
      await new Promise<void>((resolve) => {
        if (pc.iceGatheringState === 'complete') {
          resolve();
        } else {
          const checkState = () => {
            if (pc.iceGatheringState === 'complete') {
              pc.removeEventListener('icegatheringstatechange', checkState);
              resolve();
            }
          };
          pc.addEventListener('icegatheringstatechange', checkState);
          setTimeout(resolve, 3000);
        }
      });

      // バックエンドにOfferを送信（トラック情報を含む）
      const response = await fetch('/api/webrtc/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription?.sdp,
          type: pc.localDescription?.type,
          connection_id: `webrtc_combined_${sessionIdRef.current}`,
          track_type: 'combined',
          tracks: trackInfos,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const answer = await response.json();
      await pc.setRemoteDescription(new RTCSessionDescription(answer));

      console.log('[WebRTCManager] WebRTC connection established');
      isFirstConnectionRef.current = false;  // 初回接続完了
    } catch (err) {
      console.error('[WebRTCManager] Failed to establish connection:', err);
      tracksRef.current.forEach((info) => {
        updateConnectionState(info.kind, info.nodeId, {
          isConnecting: false,
          isConnected: false,
          error: 'Connection failed',
        });
      });
    } finally {
      isConnectingRef.current = false;

      // 接続中にトラックが変更された場合、再ネゴシエーション
      if (needsReconnectRef.current || tracksRef.current.size !== lastTrackCountRef.current) {
        console.log('[WebRTCManager] Track count changed during connection, renegotiating...');
        needsReconnectRef.current = false;
        setTimeout(() => {
          establishConnection();
        }, 100);
      }
    }
  }, [getOrCreatePeerConnection, updateConnectionState]);

  // 遅延接続をスケジュール
  const scheduleConnection = useCallback((delay: number) => {
    // 既存のタイマーをクリア
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
    }

    connectionTimeoutRef.current = window.setTimeout(() => {
      connectionTimeoutRef.current = null;
      establishConnection();
    }, delay);
  }, [establishConnection]);

  // トラックを登録
  const registerTrack = useCallback((kind: 'video' | 'audio', nodeId: string, stream: MediaStream) => {
    const key = getTrackKey(kind, nodeId);
    const track = kind === 'video' ? stream.getVideoTracks()[0] : stream.getAudioTracks()[0];

    if (!track) {
      console.error(`[WebRTCManager] No ${kind} track in stream`);
      return;
    }

    console.log(`[WebRTCManager] Registering ${kind} track for node ${nodeId}`);

    const info: TrackInfo = { kind, nodeId, track, stream };
    tracksRef.current.set(key, info);

    // 初回接続は長めに待機して全トラックが揃うのを待つ
    // 2回目以降（トラック追加/変更）は短い遅延
    if (isFirstConnectionRef.current) {
      // 初回: 500ms待機して複数トラックをまとめる
      scheduleConnection(500);
    } else {
      // 2回目以降: 接続中でなければ即座に再接続をスケジュール
      if (!isConnectingRef.current) {
        scheduleConnection(100);
      } else {
        // 接続中の場合は再接続フラグを立てる
        needsReconnectRef.current = true;
      }
    }
  }, [scheduleConnection]);

  // トラックを解除
  const unregisterTrack = useCallback((kind: 'video' | 'audio', nodeId: string) => {
    const key = getTrackKey(kind, nodeId);
    console.log(`[WebRTCManager] Unregistering ${kind} track for node ${nodeId}`);

    tracksRef.current.delete(key);
    addedTracksRef.current.delete(key);
    setConnectionStates(prev => {
      const newMap = new Map(prev);
      newMap.delete(key);
      return newMap;
    });

    // 残りのトラックがあれば再ネゴシエーション、なければ閉じる
    if (tracksRef.current.size === 0) {
      // タイマーをクリア
      if (connectionTimeoutRef.current) {
        clearTimeout(connectionTimeoutRef.current);
        connectionTimeoutRef.current = null;
      }
      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
      }
      // 追加済みトラックをクリア
      addedTracksRef.current.clear();
      // 次回は初回接続として扱う
      isFirstConnectionRef.current = true;
    } else {
      // トラックが削除された場合、PeerConnectionを再作成する必要がある
      // （WebRTCはトラック削除の再ネゴシエーションが複雑なため）
      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
        addedTracksRef.current.clear();
      }
      scheduleConnection(200);
    }
  }, [scheduleConnection]);

  // 接続状態を取得
  const getConnectionState = useCallback((kind: 'video' | 'audio', nodeId: string): ConnectionState => {
    const key = getTrackKey(kind, nodeId);
    return connectionStates.get(key) || { isConnecting: false, isConnected: false, error: null };
  }, [connectionStates]);

  // 特定ノードの接続が完了しているか
  const isNodeConnected = useCallback((kind: 'video' | 'audio', nodeId: string): boolean => {
    const state = getConnectionState(kind, nodeId);
    return state.isConnected;
  }, [getConnectionState]);

  // クリーンアップ
  useEffect(() => {
    return () => {
      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
      }
    };
  }, []);

  const value: WebRTCConnectionContextType = {
    registerTrack,
    unregisterTrack,
    getConnectionState,
    isNodeConnected,
  };

  return (
    <WebRTCConnectionContext.Provider value={value}>
      {children}
    </WebRTCConnectionContext.Provider>
  );
}

// カスタムフック
export function useWebRTCConnection() {
  const context = useContext(WebRTCConnectionContext);
  if (!context) {
    throw new Error('useWebRTCConnection must be used within WebRTCConnectionProvider');
  }
  return context;
}
