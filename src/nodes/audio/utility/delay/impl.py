"""
Delay ノードの実装。
指定したミリ秒だけ音声信号を遅延させる。
"""
from typing import Any, Dict, List

import numpy as np
import threading

from node_editor.node_def import ComputeLogic


class DelayBuffer:
    """遅延用リングバッファ"""

    def __init__(self, sample_rate: int, max_delay_ms: int = 5000):
        self.sample_rate = sample_rate
        # 最大遅延分のバッファを確保
        self.max_delay_samples = int(sample_rate * max_delay_ms / 1000)
        self.buffer = np.zeros(self.max_delay_samples, dtype=np.float32)
        self.write_pos = 0

    def process(self, input_data: np.ndarray, delay_samples: int) -> np.ndarray:
        """入力データを遅延させて出力"""
        if len(input_data) == 0:
            return np.array([], dtype=np.float32)

        # 遅延サンプル数をクランプ
        delay_samples = max(0, min(delay_samples, self.max_delay_samples - 1))

        output = np.zeros(len(input_data), dtype=np.float32)

        for i, sample in enumerate(input_data):
            # バッファに書き込み
            self.buffer[self.write_pos] = sample

            # 遅延分前の位置から読み出し
            read_pos = (self.write_pos - delay_samples) % self.max_delay_samples
            output[i] = self.buffer[read_pos]

            # 書き込み位置を進める
            self.write_pos = (self.write_pos + 1) % self.max_delay_samples

        return output


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


class DelayNodeLogic(ComputeLogic):
    """
    Delayノードのロジック。
    指定したミリ秒だけ音声信号を遅延させる。
    """

    def __init__(self):
        self._waveform_buffer: WaveformBuffer | None = None
        self._delay_buffer: DelayBuffer | None = None
        self._last_sample_rate: int = -1

    def reset(self):
        """バッファをリセット"""
        self._waveform_buffer = None
        self._delay_buffer = None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        audio_in = inputs.get("audio")

        if audio_in is None:
            return {"audio": None}

        # パラメータ取得（ポート入力を優先）
        delay_ms = inputs.get("delay_ms")
        if delay_ms is None:
            delay_ms = int(properties.get("delay_ms", 0))
        else:
            delay_ms = int(delay_ms)

        # 値をクランプ
        delay_ms = max(0, min(5000, delay_ms))

        sample_rate = audio_in.get("sample_rate", 16000)
        duration = audio_in.get("duration", 5.0)

        # サンプルレート変更時はバッファを再初期化
        if self._last_sample_rate != sample_rate:
            self._waveform_buffer = None
            self._delay_buffer = None
            self._last_sample_rate = sample_rate

        # バッファを初期化（初回のみ）
        if self._waveform_buffer is None:
            self._waveform_buffer = WaveformBuffer(sample_rate, duration)

        if self._delay_buffer is None:
            self._delay_buffer = DelayBuffer(sample_rate, max_delay_ms=5000)

        # 遅延サンプル数を計算
        delay_samples = int(sample_rate * delay_ms / 1000)

        # deltaを取得して遅延処理
        delta = audio_in.get("delta", [])
        if len(delta) > 0:
            delta_array = np.array(delta, dtype=np.float32)
            delayed = self._delay_buffer.process(delta_array, delay_samples)
            delayed_delta = delayed.tolist()
        else:
            delayed_delta = []

        # 処理済みdeltaをバッファに蓄積
        self._waveform_buffer.write(delayed_delta)

        # バッファから表示用waveformを生成
        waveform_display = self._waveform_buffer.get_waveform_display(200)

        return {
            "audio": {
                "delta": delayed_delta,
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": duration,
            }
        }
