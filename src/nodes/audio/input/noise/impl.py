import time
import numpy as np
from typing import Dict, Any, Optional
from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting


def get_sample_rate() -> int:
    return get_setting("audio.sample_rate", 16000)


BUFFER_SECONDS = 5.0


class NoiseGeneratorLogic(ComputeLogic):
    """各種ノイズを生成するノード"""

    def __init__(self):
        self._sample_rate: int = 0
        self._buffer: Optional[np.ndarray] = None
        self._write_pos: int = 0
        self._valid_samples: int = 0
        self._last_compute_time: Optional[float] = None

        # ピンクノイズ用フィルタ状態
        self._pink_state = np.zeros(7, dtype=np.float32)
        # ハムノイズ用位相（サンプル単位）
        self._hum_sample_idx: int = 0
        # パルスノイズ用（次のパルスまでのサンプル数）
        self._pulse_countdown: int = 0
        # ヒスノイズ用（前回の値）
        self._hiss_prev: float = 0.0

    def reset(self):
        """状態をリセット"""
        self._buffer = None
        self._write_pos = 0
        self._valid_samples = 0
        self._last_compute_time = None
        self._pink_state = np.zeros(7, dtype=np.float32)
        self._hum_sample_idx = 0
        self._pulse_countdown = 0
        self._hiss_prev = 0.0

    def _generate_white_noise(self, num_samples: int) -> np.ndarray:
        """ホワイトノイズ: 全周波数が均等（振幅 -1 〜 1）"""
        return np.random.uniform(-1.0, 1.0, num_samples).astype(np.float32)

    def _generate_pink_noise(self, num_samples: int) -> np.ndarray:
        """ピンクノイズ: 1/f特性（低周波が強い、振幅 -1 〜 1）"""
        output = np.zeros(num_samples, dtype=np.float32)
        b = self._pink_state

        for i in range(num_samples):
            white = np.random.uniform(-1, 1)
            b[0] = 0.99886 * b[0] + white * 0.0555179
            b[1] = 0.99332 * b[1] + white * 0.0750759
            b[2] = 0.96900 * b[2] + white * 0.1538520
            b[3] = 0.86650 * b[3] + white * 0.3104856
            b[4] = 0.55000 * b[4] + white * 0.5329522
            b[5] = -0.7616 * b[5] - white * 0.0168980
            output[i] = (b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * 0.5362) * 0.11
            b[6] = white * 0.115926

        return np.clip(output, -1.0, 1.0)

    def _generate_hiss_noise(self, num_samples: int) -> np.ndarray:
        """ヒスノイズ: 高周波強調のホワイトノイズ"""
        white = np.random.uniform(-0.5, 0.5, num_samples + 1).astype(np.float32)
        white[0] = self._hiss_prev
        hiss = np.diff(white) * 2.0
        self._hiss_prev = white[-1]
        return np.clip(hiss, -1.0, 1.0).astype(np.float32)

    def _generate_hum_noise(self, num_samples: int, sample_rate: int) -> np.ndarray:
        """ハムノイズ: 電源ハム（50Hz + 高調波）"""
        indices = np.arange(num_samples) + self._hum_sample_idx
        t = indices / sample_rate
        hum = (
            0.5 * np.sin(2 * np.pi * 50 * t) +
            0.25 * np.sin(2 * np.pi * 100 * t) +
            0.125 * np.sin(2 * np.pi * 150 * t)
        )
        self._hum_sample_idx += num_samples
        # オーバーフロー防止（1秒周期でリセット）
        if self._hum_sample_idx >= sample_rate:
            self._hum_sample_idx %= sample_rate
        return (hum * 0.5).astype(np.float32)

    def _generate_pulse_noise(self, num_samples: int, sample_rate: int) -> np.ndarray:
        """パルスノイズ: ランダム間隔のクリック音（正負ランダム）"""
        output = np.zeros(num_samples, dtype=np.float32)
        pulse_width = max(1, int(sample_rate * 0.001))  # 1msのパルス幅
        i = 0

        while i < num_samples:
            if self._pulse_countdown > 0:
                skip = min(self._pulse_countdown, num_samples - i)
                i += skip
                self._pulse_countdown -= skip
            else:
                # 正負をランダムに決定
                sign = 1.0 if np.random.random() > 0.5 else -1.0
                # パルス生成
                for j in range(pulse_width):
                    if i + j < num_samples:
                        decay = 1.0 - j / pulse_width
                        output[i + j] = sign * 0.8 * decay
                i += pulse_width
                # 次のパルスまでのランダム間隔（50ms〜500ms）
                self._pulse_countdown = int(sample_rate * np.random.uniform(0.05, 0.5))

        return output

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        sample_rate = get_sample_rate()
        noise_type = properties.get("noise_type", "white")

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
            num_samples = int(sample_rate * 0.05)  # 初回は50ms分
        else:
            elapsed = current_time - self._last_compute_time
            num_samples = int(sample_rate * elapsed)
            # 異常値を制限
            num_samples = max(1, min(num_samples, int(sample_rate * 0.2)))
        self._last_compute_time = current_time

        # ノイズ生成
        if noise_type == "white":
            noise = self._generate_white_noise(num_samples)
        elif noise_type == "pink":
            noise = self._generate_pink_noise(num_samples)
        elif noise_type == "hiss":
            noise = self._generate_hiss_noise(num_samples)
        elif noise_type == "hum":
            noise = self._generate_hum_noise(num_samples, sample_rate)
        elif noise_type == "pulse":
            noise = self._generate_pulse_noise(num_samples, sample_rate)
        else:
            noise = self._generate_white_noise(num_samples)

        # バッファに書き込み
        n = len(noise)
        if n >= buffer_size:
            self._buffer[:] = noise[-buffer_size:]
            self._write_pos = 0
            self._valid_samples = buffer_size
        else:
            end_pos = self._write_pos + n
            if end_pos <= buffer_size:
                self._buffer[self._write_pos:end_pos] = noise
            else:
                first_part = buffer_size - self._write_pos
                self._buffer[self._write_pos:] = noise[:first_part]
                self._buffer[:n - first_part] = noise[first_part:]
            self._write_pos = end_pos % buffer_size
            self._valid_samples = min(self._valid_samples + n, buffer_size)

        # 表示用waveform生成
        display_width = 200
        samples_per_pixel = buffer_size // display_width

        waveform_display = []
        if self._valid_samples >= buffer_size:
            waveform = np.concatenate([
                self._buffer[self._write_pos:],
                self._buffer[:self._write_pos]
            ])
        else:
            start_pos = (self._write_pos - self._valid_samples) % buffer_size
            if start_pos < self._write_pos:
                waveform = self._buffer[start_pos:self._write_pos]
            else:
                waveform = np.concatenate([
                    self._buffer[start_pos:],
                    self._buffer[:self._write_pos]
                ])

        data_pixels = len(waveform) // samples_per_pixel if samples_per_pixel > 0 else 0
        empty_pixels = display_width - data_pixels

        for _ in range(empty_pixels):
            waveform_display.extend([0.0, 0.0])

        for i in range(data_pixels):
            start = i * samples_per_pixel
            end = start + samples_per_pixel
            segment = waveform[start:end]
            if len(segment) > 0:
                waveform_display.append(float(np.min(segment)))
                waveform_display.append(float(np.max(segment)))
            else:
                waveform_display.extend([0.0, 0.0])

        return {
            "audio_out": {
                "delta": noise.tolist(),
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": BUFFER_SECONDS,
            }
        }
