import time
import numpy as np
from typing import Dict, Any, Optional
from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting


def get_sample_rate() -> int:
    return get_setting("audio.sample_rate", 16000)


BUFFER_SECONDS = 5.0


class ZeroAudioLogic(ComputeLogic):
    """無音（ゼロデータ）を出力するノード"""

    def __init__(self):
        self._sample_rate: int = 0
        self._buffer: Optional[np.ndarray] = None
        self._write_pos: int = 0
        self._valid_samples: int = 0
        self._last_compute_time: Optional[float] = None

    def reset(self):
        """状態をリセット"""
        self._buffer = None
        self._write_pos = 0
        self._valid_samples = 0
        self._last_compute_time = None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        sample_rate = get_sample_rate()

        # バッファ初期化
        buffer_size = int(sample_rate * BUFFER_SECONDS)
        if self._buffer is None or self._sample_rate != sample_rate:
            self._buffer = np.zeros(buffer_size, dtype=np.float32)
            self._sample_rate = sample_rate
            self._write_pos = 0
            self._valid_samples = 0

        # 実時間ベースでサンプル数を計算
        current_time = time.perf_counter()
        if self._last_compute_time is None:
            num_samples = int(sample_rate * 0.05)
        else:
            elapsed = current_time - self._last_compute_time
            num_samples = int(sample_rate * elapsed)
            num_samples = max(1, min(num_samples, int(sample_rate * 0.2)))
        self._last_compute_time = current_time

        # ゼロデータ生成
        zero_data = np.zeros(num_samples, dtype=np.float32)

        # バッファに書き込み（常に0なので実際は不要だが、waveform表示のために更新）
        n = len(zero_data)
        if n >= buffer_size:
            self._write_pos = 0
            self._valid_samples = buffer_size
        else:
            end_pos = self._write_pos + n
            self._write_pos = end_pos % buffer_size
            self._valid_samples = min(self._valid_samples + n, buffer_size)

        # 表示用waveform生成（全て0）
        display_width = 200
        waveform_display = [0.0] * (display_width * 2)

        return {
            "audio_out": {
                "delta": zero_data.tolist(),
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": BUFFER_SECONDS,
            }
        }
