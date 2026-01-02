"""
Lowpass Filter ノードの実装。
指定したカットオフ周波数より高い周波数成分を除去する。
"""
from typing import Dict, Any, List
from node_editor.node_def import ComputeLogic
import numpy as np
import threading

try:
    from scipy.signal import butter, sosfilt, sosfilt_zi
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class LowpassFilter:
    """Butterworthローパスフィルター"""

    def __init__(
        self,
        cutoff_freq: int = 8000,
        filter_order: int = 5,
        sample_rate: int = 16000,
    ):
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        self.sample_rate = sample_rate

        # ナイキスト周波数
        nyquist = sample_rate / 2

        # カットオフ周波数がナイキスト周波数を超えないようにクランプ
        normalized_cutoff = min(cutoff_freq / nyquist, 0.99)

        # Butterworthフィルター係数を計算（SOS形式）
        self.sos = butter(filter_order, normalized_cutoff, btype='low', output='sos')

        # フィルター状態を初期化
        self.zi = sosfilt_zi(self.sos)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """チャンクをフィルタリング"""
        if len(chunk) == 0:
            return chunk

        # フィルター適用（状態を保持してストリーミング処理）
        filtered, self.zi = sosfilt(self.sos, chunk, zi=self.zi)
        return filtered.astype(np.float32)


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


class LowpassFilterLogic(ComputeLogic):
    """
    ローパスフィルターノードのロジック。
    指定したカットオフ周波数より高い周波数成分を除去する。
    """

    def __init__(self):
        self._buffer: WaveformBuffer | None = None
        self._filter: LowpassFilter | None = None
        self._last_cutoff_freq: int = -1
        self._last_filter_order: int = -1
        self._last_sample_rate: int = -1

    def reset(self):
        """バッファとフィルターをリセット"""
        self._buffer = None
        self._filter = None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not SCIPY_AVAILABLE:
            return {
                "audio": None,
                "__error__": "scipy is not installed"
            }

        audio_in = inputs.get("audio")

        if audio_in is None:
            return {"audio": None}

        # パラメータ取得
        cutoff_freq = int(properties.get("cutoff_freq", 8000))
        filter_order = int(properties.get("filter_order", 5))

        # 値をクランプ
        cutoff_freq = max(100, min(20000, cutoff_freq))
        filter_order = max(1, min(10, filter_order))

        sample_rate = audio_in.get("sample_rate", 16000)
        duration = audio_in.get("duration", 5.0)

        # カットオフ周波数がナイキスト周波数を超えないようにする
        max_cutoff = int(sample_rate / 2 - 1)
        cutoff_freq = min(cutoff_freq, max_cutoff)

        # バッファを初期化（初回のみ）
        if self._buffer is None:
            self._buffer = WaveformBuffer(sample_rate, duration)

        # パラメータが変更された場合、フィルターを再作成
        params_changed = (
            self._last_cutoff_freq != cutoff_freq or
            self._last_filter_order != filter_order or
            self._last_sample_rate != sample_rate
        )

        if self._filter is None or params_changed:
            self._filter = LowpassFilter(
                cutoff_freq=cutoff_freq,
                filter_order=filter_order,
                sample_rate=sample_rate,
            )
            self._last_cutoff_freq = cutoff_freq
            self._last_filter_order = filter_order
            self._last_sample_rate = sample_rate

        # deltaを取得してフィルタリング
        delta = audio_in.get("delta", [])
        if len(delta) > 0:
            delta_array = np.array(delta, dtype=np.float32)
            filtered = self._filter.process(delta_array)
            filtered_delta = filtered.tolist()
        else:
            filtered_delta = []

        # 処理済みdeltaをバッファに蓄積
        self._buffer.write(filtered_delta)

        # バッファから表示用waveformを生成
        waveform_display = self._buffer.get_waveform_display(200)

        return {
            "audio": {
                "delta": filtered_delta,
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": duration,
            }
        }
