"""
WebRTCを使用してブラウザからマイク音声を取得するノード。
Base64/JSONエンコードを経由しないため低遅延。
"""
from typing import Any, Dict, List

import numpy as np

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting


BUFFER_SECONDS = 5.0


def get_sample_rate() -> int:
    """設定からサンプリングレートを取得"""
    return get_setting("audio.sample_rate", 16000)


class WebRTCMicrophoneNodeLogic(ComputeLogic):
    """
    WebRTC経由でブラウザからのマイク音声を受け取るノードロジック。
    バックエンドのWebRTCFrameStoreから最新サンプルを取得する。
    """

    def __init__(self):
        self.waveform_buffer = np.zeros(0, dtype=np.float32)
        self.sample_rate = get_sample_rate()
        self.max_samples = int(self.sample_rate * BUFFER_SECONDS)
        self._was_streaming = False

    def reset(self):
        """ノードの状態をリセット（STOP時に呼ばれる）"""
        self.waveform_buffer = np.zeros(0, dtype=np.float32)
        self._was_streaming = False

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        if context is None:
            context = {}

        # webrtc_microphoneプロパティからconnection_idを取得
        # フロントエンドが "webrtc_microphone_node-xxx" 形式で設定
        connection_id = properties.get("webrtc_microphone", "")

        if not connection_id:
            return self._build_output([])

        is_streaming = context.get("is_streaming", False)

        # START直後の最初の実行を検出
        # この場合、START前に蓄積された古いオーディオをクリア
        is_first_streaming_execution = is_streaming and not self._was_streaming
        self._was_streaming = is_streaming

        # WebRTCFrameStoreからオーディオサンプルを取得
        # バックエンドで既にリサンプリング済み（設定のsample_rateに変換済み）
        delta = []
        try:
            from src.gui.reactflow.backend.main import webrtc_frame_store

            # START直後は古いバッファをクリア
            if is_first_streaming_execution:
                webrtc_frame_store.clear_audio_buffer(connection_id)
                return self._build_output([])

            buffer_data = webrtc_frame_store.get_audio_buffer(connection_id)
            if buffer_data is not None:
                delta = buffer_data["samples"].tolist()
                # サンプル数が異常に多い場合（1秒以上）は警告
                if len(delta) > self.sample_rate:
                    print(f"[WebRTC Mic] WARNING: Large buffer! {len(delta)} samples = {len(delta)/self.sample_rate:.2f}s")
                # サンプルレートの不一致は警告
                buffer_rate = buffer_data.get("sample_rate", self.sample_rate)
                if buffer_rate != self.sample_rate:
                    print(f"[WebRTC Mic] WARNING: sample_rate mismatch! buffer={buffer_rate}, node={self.sample_rate}")
        except ImportError as e:
            print(f"[WebRTC Mic] ImportError: {e}")
        except Exception as e:
            print(f"[WebRTC Mic] Error getting audio buffer: {e}")

        if not delta:
            return self._build_output([])

        # サンプルをバッファに追加
        delta_array = np.array(delta, dtype=np.float32)
        self.waveform_buffer = np.concatenate([self.waveform_buffer, delta_array])

        # バッファサイズを制限
        if len(self.waveform_buffer) > self.max_samples:
            self.waveform_buffer = self.waveform_buffer[-self.max_samples:]

        return self._build_output(delta)

    def _build_output(self, delta: List[float]) -> Dict[str, Any]:
        """出力データを構築"""
        # 表示用にmin/max計算（200ピクセル分）
        display_width = 200
        samples_per_pixel = self.max_samples // display_width

        waveform_display = []
        data_pixels = len(self.waveform_buffer) // samples_per_pixel if samples_per_pixel > 0 else 0
        empty_pixels = display_width - data_pixels

        # 左側の空白部分
        for _ in range(empty_pixels):
            waveform_display.append(0.0)
            waveform_display.append(0.0)

        # 右側の実データ部分
        for i in range(data_pixels):
            start = i * samples_per_pixel
            end = start + samples_per_pixel
            segment = self.waveform_buffer[start:end]
            if len(segment) > 0:
                waveform_display.append(float(np.min(segment)))
                waveform_display.append(float(np.max(segment)))
            else:
                waveform_display.append(0.0)
                waveform_display.append(0.0)

        return {
            "audio_out": {
                "delta": delta,
                "waveform": waveform_display,
                "sample_rate": self.sample_rate,
                "duration": BUFFER_SECONDS,
            }
        }
