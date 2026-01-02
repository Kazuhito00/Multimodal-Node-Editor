from typing import Dict, Any, Optional
from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting
import numpy as np
import threading
import time
from math import gcd

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    signal = None


def get_sample_rate() -> int:
    """設定からサンプリングレートを取得"""
    return get_setting("audio.sample_rate", 16000)


def get_interval_ms() -> int:
    """設定からグラフ実行間隔を取得"""
    return get_setting("graph.interval_ms", 50)


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """オーディオをリサンプリング"""
    if orig_sr == target_sr:
        return data.astype(np.float32)

    if SCIPY_AVAILABLE:
        # scipy.signal.resample_polyを使用（高品質）
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        resampled = signal.resample_poly(data, up, down)
        return resampled.astype(np.float32)
    else:
        # フォールバック: 線形補間
        duration = len(data) / orig_sr
        new_length = int(duration * target_sr)
        old_indices = np.linspace(0, len(data) - 1, new_length)
        return np.interp(old_indices, np.arange(len(data)), data).astype(np.float32)


class AudioFileBuffer:
    """オーディオファイルのストリーミングバッファ"""

    def __init__(self, sample_rate: int, buffer_seconds: float = 5.0):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.buffer_size = int(sample_rate * buffer_seconds)

        # 表示用循環バッファ
        self.display_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.valid_samples = 0

        # ファイルデータ
        self.file_data: Optional[np.ndarray] = None
        self.file_path: str = ""
        self.play_pos = 0
        self.is_playing = False

        self.lock = threading.Lock()

    def load_file(self, file_path: str) -> bool:
        """オーディオファイルを読み込む"""
        if not SOUNDFILE_AVAILABLE or not file_path:
            return False

        try:
            # soundfileでファイルを読み込み（自動的に-1.0~1.0に正規化される）
            data, orig_sr = sf.read(file_path, dtype='float32')

            # ステレオの場合はモノラルに変換
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # サンプルレートを変換
            samples = resample_audio(data, orig_sr, self.sample_rate)

            with self.lock:
                self.file_data = samples
                self.file_path = file_path
                self.play_pos = 0
                self.is_playing = True
                # 表示バッファをリセット
                self.display_buffer = np.zeros(self.buffer_size, dtype=np.float32)
                self.write_pos = 0
                self.valid_samples = 0

            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False

    def get_next_samples(self, num_samples: int, loop: bool = True) -> tuple:
        """
        次のサンプルを取得
        戻り値: (samples, ended)
        ended: ループがオフでファイルが終了した場合True
        """
        with self.lock:
            if self.file_data is None or len(self.file_data) == 0:
                return np.zeros(num_samples, dtype=np.float32), False

            result = np.zeros(num_samples, dtype=np.float32)
            remaining = num_samples
            write_idx = 0
            ended = False

            while remaining > 0:
                available = len(self.file_data) - self.play_pos
                to_copy = min(remaining, available)

                result[write_idx:write_idx + to_copy] = self.file_data[self.play_pos:self.play_pos + to_copy]

                self.play_pos += to_copy
                write_idx += to_copy
                remaining -= to_copy

                # ファイル終端に到達
                if self.play_pos >= len(self.file_data):
                    if loop:
                        # ループ再生
                        self.play_pos = 0
                    else:
                        # ループオフの場合は終了シグナル
                        ended = True
                        break

            # 表示バッファに書き込み
            self._write_to_display_buffer(result)

            return result, ended

    def _write_to_display_buffer(self, data: np.ndarray):
        """表示用バッファに書き込む"""
        n = len(data)
        if n >= self.buffer_size:
            self.display_buffer[:] = data[-self.buffer_size:]
            self.write_pos = 0
            self.valid_samples = self.buffer_size
        else:
            end_pos = self.write_pos + n
            if end_pos <= self.buffer_size:
                self.display_buffer[self.write_pos:end_pos] = data
            else:
                first_part = self.buffer_size - self.write_pos
                self.display_buffer[self.write_pos:] = data[:first_part]
                self.display_buffer[:n - first_part] = data[first_part:]
            self.write_pos = end_pos % self.buffer_size
            self.valid_samples = min(self.valid_samples + n, self.buffer_size)

    def get_waveform_display(self, display_width: int = 200) -> list:
        """表示用のmin/maxペアを生成"""
        with self.lock:
            if self.valid_samples == 0:
                return [0.0] * (display_width * 2)

            # バッファからデータを時系列順に取得
            if self.valid_samples >= self.buffer_size:
                waveform = np.concatenate([
                    self.display_buffer[self.write_pos:],
                    self.display_buffer[:self.write_pos]
                ])
            else:
                start_pos = (self.write_pos - self.valid_samples) % self.buffer_size
                if start_pos < self.write_pos:
                    waveform = self.display_buffer[start_pos:self.write_pos].copy()
                else:
                    waveform = np.concatenate([
                        self.display_buffer[start_pos:],
                        self.display_buffer[:self.write_pos]
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


class AudioFilePlayerLogic(ComputeLogic):
    """オーディオファイル再生ノードロジック"""

    def __init__(self):
        self._buffer: Optional[AudioFileBuffer] = None
        self._current_file: str = ""
        self._last_compute_time: Optional[float] = None

    def reset(self):
        """再生位置を先頭にリセット"""
        if self._buffer is not None:
            with self._buffer.lock:
                self._buffer.play_pos = 0
                self._buffer.display_buffer = np.zeros(self._buffer.buffer_size, dtype=np.float32)
                self._buffer.write_pos = 0
                self._buffer.valid_samples = 0
        self._last_compute_time = None
        print("AudioFilePlayer reset")

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if context is None:
            context = {}

        if not SOUNDFILE_AVAILABLE:
            return {
                "audio_out": None,
                "__error__": "soundfile is not installed",
            }

        sample_rate = get_sample_rate()
        interval_ms = context.get("interval_ms", get_interval_ms())
        loop = context.get("loop", True)
        file_path = properties.get("file_path", "")

        # バッファ初期化
        if self._buffer is None:
            self._buffer = AudioFileBuffer(sample_rate, buffer_seconds=5.0)

        # ファイルが変更されたら読み込み
        if file_path and file_path != self._current_file:
            if self._buffer.load_file(file_path):
                self._current_file = file_path
                self._last_compute_time = None  # 時間計測をリセット
            else:
                self._current_file = ""

        # ファイルが読み込まれていない場合
        if not self._current_file:
            self._last_compute_time = None
            return {"audio_out": None}

        # 実際の経過時間からサンプル数を計算
        current_time = time.perf_counter()
        if self._last_compute_time is None:
            # 初回は interval_ms を使用
            elapsed_ms = interval_ms
        else:
            elapsed_ms = (current_time - self._last_compute_time) * 1000
            # 異常値を制限（最小: interval_msの半分、最大: interval_msの3倍）
            elapsed_ms = max(interval_ms * 0.5, min(elapsed_ms, interval_ms * 3))
        self._last_compute_time = current_time

        num_samples = int(sample_rate * elapsed_ms / 1000)
        delta_samples, ended = self._buffer.get_next_samples(num_samples, loop=loop)

        # 表示用waveformを生成
        waveform_display = self._buffer.get_waveform_display(200)

        result = {
            "audio_out": {
                "delta": delta_samples.tolist(),
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": 5.0,
            }
        }

        # 終了シグナルを追加（ループがオフでファイルが終了した場合）
        if ended:
            result["__ended__"] = True

        return result
