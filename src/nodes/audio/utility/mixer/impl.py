"""
Mixer ノードの実装。
2つの音声信号をミックスして出力する。
"""
from typing import Any, Dict, List

import numpy as np
import threading

from node_editor.node_def import ComputeLogic


class WaveformBuffer:
    """処理済みサンプルを蓄積する循環バッファ"""

    def __init__(self, sample_rate: int, buffer_seconds: float = 5.0):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_seconds)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.valid_samples = 0
        self.lock = threading.Lock()

    def write(self, data: List[float]):
        """データをバッファに書き込む"""
        if not data:
            return
        with self.lock:
            arr = np.array(data, dtype=np.float32)
            n = len(arr)
            if n >= self.buffer_size:
                self.buffer[:] = arr[-self.buffer_size:]
                self.write_pos = 0
                self.valid_samples = self.buffer_size
            else:
                end_pos = self.write_pos + n
                if end_pos <= self.buffer_size:
                    self.buffer[self.write_pos:end_pos] = arr
                else:
                    first_part = self.buffer_size - self.write_pos
                    self.buffer[self.write_pos:] = arr[:first_part]
                    self.buffer[:n - first_part] = arr[first_part:]
                self.write_pos = end_pos % self.buffer_size
                self.valid_samples = min(self.valid_samples + n, self.buffer_size)

    def get_waveform_display(self, display_width: int = 200) -> List[float]:
        """表示用のmin/maxペアを生成"""
        with self.lock:
            if self.valid_samples == 0:
                return [0.0] * (display_width * 2)

            # バッファからデータを時系列順に取得
            if self.valid_samples >= self.buffer_size:
                waveform = np.concatenate([
                    self.buffer[self.write_pos:],
                    self.buffer[:self.write_pos]
                ])
            else:
                start_pos = (self.write_pos - self.valid_samples) % self.buffer_size
                if start_pos < self.write_pos:
                    waveform = self.buffer[start_pos:self.write_pos].copy()
                else:
                    waveform = np.concatenate([
                        self.buffer[start_pos:],
                        self.buffer[:self.write_pos]
                    ])

            # min/max計算
            samples_per_pixel = self.buffer_size // display_width
            data_pixels = len(waveform) // samples_per_pixel if samples_per_pixel > 0 else 0
            empty_pixels = display_width - data_pixels

            result = []
            for _ in range(empty_pixels):
                result.extend([0.0, 0.0])

            for i in range(data_pixels):
                start = i * samples_per_pixel
                end = start + samples_per_pixel
                segment = waveform[start:end]
                if len(segment) > 0:
                    result.append(float(np.min(segment)))
                    result.append(float(np.max(segment)))
                else:
                    result.extend([0.0, 0.0])

            return result


class MixerNodeLogic(ComputeLogic):
    """
    Mixerノードのロジック。
    2つの音声信号をミックスして出力する。
    """

    def __init__(self):
        self._waveform_buffer: WaveformBuffer | None = None
        self._last_sample_rate: int = -1

    def reset(self):
        """バッファをリセット"""
        self._waveform_buffer = None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        audio_1 = inputs.get("audio_1")
        audio_2 = inputs.get("audio_2")

        # 両方の入力がない場合
        if audio_1 is None and audio_2 is None:
            return {"audio_out": None}

        # サンプルレートとdurationを取得
        sample_rate = 16000
        duration = 5.0

        if audio_1 is not None:
            sample_rate = audio_1.get("sample_rate", sample_rate)
            duration = audio_1.get("duration", duration)
        elif audio_2 is not None:
            sample_rate = audio_2.get("sample_rate", sample_rate)
            duration = audio_2.get("duration", duration)

        # サンプルレート変更時はバッファを再初期化
        if self._last_sample_rate != sample_rate:
            self._waveform_buffer = None
            self._last_sample_rate = sample_rate

        # バッファを初期化（初回のみ）
        if self._waveform_buffer is None:
            self._waveform_buffer = WaveformBuffer(sample_rate, duration)

        # deltaを取得
        delta_1 = audio_1.get("delta", []) if audio_1 else []
        delta_2 = audio_2.get("delta", []) if audio_2 else []

        # ミックス処理
        mixed_delta = self._mix_audio(delta_1, delta_2)

        # 処理済みdeltaをバッファに蓄積
        self._waveform_buffer.write(mixed_delta)

        # バッファから表示用waveformを生成
        waveform_display = self._waveform_buffer.get_waveform_display(200)

        return {
            "audio_out": {
                "delta": mixed_delta,
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": duration,
            }
        }

    def _mix_audio(self, delta_1: List[float], delta_2: List[float]) -> List[float]:
        """2つのオーディオデルタをミックスする"""
        # どちらかが空の場合はもう一方をそのまま返す
        if len(delta_1) == 0 and len(delta_2) == 0:
            return []

        if len(delta_1) == 0:
            return delta_2

        if len(delta_2) == 0:
            return delta_1

        # 長さを揃える
        arr_1 = np.array(delta_1, dtype=np.float32)
        arr_2 = np.array(delta_2, dtype=np.float32)

        max_len = max(len(arr_1), len(arr_2))

        if len(arr_1) < max_len:
            arr_1 = np.pad(arr_1, (0, max_len - len(arr_1)), mode='constant')

        if len(arr_2) < max_len:
            arr_2 = np.pad(arr_2, (0, max_len - len(arr_2)), mode='constant')

        # ミックス（単純加算）
        mixed = arr_1 + arr_2

        return mixed.tolist()
