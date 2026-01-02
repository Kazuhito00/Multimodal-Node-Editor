"""Tone Generatorノードの実装"""
import time
from typing import Any, Dict, Optional

import numpy as np

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting


def get_sample_rate() -> int:
    return get_setting("audio.sample_rate", 16000)


BUFFER_SECONDS = 5.0


class ToneGeneratorLogic(ComputeLogic):
    """各種波形を生成するノード"""

    def __init__(self):
        self._sample_rate: int = 0
        self._buffer: Optional[np.ndarray] = None
        self._write_pos: int = 0
        self._valid_samples: int = 0
        self._last_compute_time: Optional[float] = None
        # 波形生成用の位相（0.0〜1.0の範囲で連続）
        self._phase: float = 0.0
        # 端数サンプルの累積（切り捨て誤差防止）
        self._fractional_samples: float = 0.0
        # 起動からの累積サンプル数（フェードイン用）
        self._total_samples_generated: int = 0
        # 安定するまでのウォームアップ回数
        self._warmup_count: int = 0

    def reset(self):
        """状態をリセット"""
        self._buffer = None
        self._write_pos = 0
        self._valid_samples = 0
        self._last_compute_time = None
        self._phase = 0.0
        self._fractional_samples = 0.0
        self._total_samples_generated = 0
        self._warmup_count = 0

    def _generate_wave(
        self, num_samples: int, frequency: int, sample_rate: int, waveform_type: str
    ) -> np.ndarray:
        """波形を生成（位相連続性を保証）"""
        if num_samples <= 0:
            return np.array([], dtype=np.float32)

        # 1サンプルあたりの位相増分
        phase_increment = frequency / sample_rate

        # 各サンプルの位相を計算（連続的に増加）
        phases = self._phase + np.arange(num_samples, dtype=np.float64) * phase_increment

        # 次回の開始位相を保存（最後のサンプルの次の位相）
        self._phase = phases[-1] + phase_increment

        # 位相が大きくなりすぎたら正規化（精度維持のため）
        if self._phase > 1000.0:
            self._phase = self._phase % 1.0

        # 波形生成用に0〜1の範囲に正規化
        wave_phases = phases % 1.0

        # 波形生成
        if waveform_type == "sine":
            wave = np.sin(2 * np.pi * wave_phases)
        elif waveform_type == "square":
            wave = np.where(wave_phases < 0.5, 1.0, -1.0)
        elif waveform_type == "triangle":
            wave = 2 * np.abs(2 * wave_phases - 1) - 1
        elif waveform_type == "sawtooth":
            wave = 2 * wave_phases - 1
        else:
            wave = np.sin(2 * np.pi * wave_phases)

        # 起動時のフェードイン（最初の20msで0から1へ）
        fade_in_samples = int(sample_rate * 0.02)  # 20ms
        if self._total_samples_generated < fade_in_samples:
            fade_start = self._total_samples_generated
            fade_end = min(fade_start + num_samples, fade_in_samples)
            fade_length = fade_end - fade_start

            if fade_length > 0:
                # フェードカーブを計算
                fade_indices = np.arange(fade_length)
                fade_curve = (fade_start + fade_indices) / fade_in_samples
                wave[:fade_length] = wave[:fade_length] * fade_curve

        self._total_samples_generated += num_samples

        return wave.astype(np.float32)

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        sample_rate = get_sample_rate()
        frequency = int(properties.get("frequency", 440))
        waveform_type = properties.get("waveform", "sine")

        # バッファ初期化
        buffer_size = int(sample_rate * BUFFER_SECONDS)
        if self._buffer is None or self._sample_rate != sample_rate:
            self._buffer = np.zeros(buffer_size, dtype=np.float32)
            self._sample_rate = sample_rate
            self._write_pos = 0
            self._valid_samples = 0

        # 実時間ベースでサンプル数を計算（端数を累積して切り捨て誤差を防止）
        current_time = time.perf_counter()

        # ウォームアップ期間（最初の5回）は固定チャンクサイズで安定させる
        warmup_frames = 5
        chunk_size = int(sample_rate * 0.05)  # 50ms

        if self._warmup_count < warmup_frames:
            num_samples = chunk_size
            self._warmup_count += 1
            # ウォームアップ終了時に時間ベース計算を開始
            if self._warmup_count == warmup_frames:
                self._fractional_samples = 0.0
        else:
            if self._last_compute_time is None:
                num_samples = chunk_size
            else:
                elapsed = current_time - self._last_compute_time
                # 端数を累積
                self._fractional_samples += sample_rate * elapsed
                num_samples = int(self._fractional_samples)
                # 異常値を制限（クランプ前に保存）
                max_samples = int(sample_rate * 0.2)
                num_samples = max(1, min(num_samples, max_samples))
                # 実際に生成するサンプル数を引く（クランプ後の値）
                self._fractional_samples -= num_samples

        self._last_compute_time = current_time

        # 波形生成（位相連続性を保証）
        wave = self._generate_wave(num_samples, frequency, sample_rate, waveform_type)

        # バッファに書き込み
        n = len(wave)
        if n >= buffer_size:
            self._buffer[:] = wave[-buffer_size:]
            self._write_pos = 0
            self._valid_samples = buffer_size
        else:
            end_pos = self._write_pos + n
            if end_pos <= buffer_size:
                self._buffer[self._write_pos:end_pos] = wave
            else:
                first_part = buffer_size - self._write_pos
                self._buffer[self._write_pos:] = wave[:first_part]
                self._buffer[:n - first_part] = wave[first_part:]
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
                "delta": wave.tolist(),
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": BUFFER_SECONDS,
            }
        }
