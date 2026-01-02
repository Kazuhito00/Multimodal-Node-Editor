"""
ブラウザのgetUserMedia() APIを使用してマイク音声を取得するノード。
フロントエンドでキャプチャした音声データを受け取り、出力する。
"""
import json
from typing import Any, Dict

import numpy as np

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting


BUFFER_SECONDS = 5.0


def get_sample_rate() -> int:
    """設定からサンプリングレートを取得"""
    return get_setting("audio.sample_rate", 16000)


class BrowserMicrophoneNodeLogic(ComputeLogic):
    """
    ブラウザからのマイク音声を受け取るノードロジック。
    フロントエンドがgetUserMedia()でキャプチャした音声を
    browser_audioプロパティ経由で受け取る。
    """

    def __init__(self):
        self.waveform_buffer = np.zeros(0, dtype=np.float32)
        self.sample_rate = get_sample_rate()
        self.max_samples = int(self.sample_rate * BUFFER_SECONDS)
        self.last_seq = -1  # 最後に処理したシーケンス番号

    def reset(self):
        """バッファをリセット"""
        self.waveform_buffer = np.zeros(0, dtype=np.float32)
        self.last_seq = -1

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        browser_audio = properties.get("browser_audio", "")

        if not browser_audio:
            return {"audio_out": None}

        try:
            # JSON形式でサンプルデータを受け取る
            audio_data = json.loads(browser_audio)
            delta = audio_data.get("samples", [])
            seq = audio_data.get("seq", -1)

            if not delta:
                return {"audio_out": None}

            # シーケンス番号で重複チェック
            if seq != -1:
                # seqが小さくなった場合はマイク再起動と判断してリセット
                if seq < self.last_seq:
                    self.last_seq = -1
                    self.waveform_buffer = np.zeros(0, dtype=np.float32)
                elif seq == self.last_seq:
                    # 同じseqは重複なのでスキップ
                    return self._build_output([])

            self.last_seq = seq

            # 差分サンプルをバッファに追加
            delta_array = np.array(delta, dtype=np.float32)
            self.waveform_buffer = np.concatenate([self.waveform_buffer, delta_array])

            # バッファサイズを制限
            if len(self.waveform_buffer) > self.max_samples:
                self.waveform_buffer = self.waveform_buffer[-self.max_samples:]

            return self._build_output(delta)

        except Exception as e:
            print(f"BrowserMicrophone decode error: {e}")
            return {"audio_out": None}

    def _build_output(self, delta: list) -> Dict[str, Any]:
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
