"""
Equalizer ノードの実装。
指定した周波数帯域のゲインを調整する。
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


class EqualizerFilter:
    """パラメトリックイコライザー（バンドゲイン調整）"""

    def __init__(
        self,
        low_freq: int = 2000,
        high_freq: int = 4000,
        gain_db: float = 0.0,
        filter_order: int = 4,
        sample_rate: int = 16000,
    ):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.gain_db = gain_db
        self.filter_order = filter_order
        self.sample_rate = sample_rate

        # dBからリニアゲインに変換
        self.linear_gain = 10 ** (gain_db / 20)

        # ナイキスト周波数
        nyquist = sample_rate / 2

        # カットオフ周波数を正規化
        low_normalized = max(low_freq / nyquist, 0.01)
        low_normalized = min(low_normalized, 0.98)

        high_normalized = max(high_freq / nyquist, 0.02)
        high_normalized = min(high_normalized, 0.99)

        # low < high を保証
        if low_normalized >= high_normalized:
            high_normalized = min(low_normalized + 0.01, 0.99)

        # バンドパスフィルター（対象帯域を抽出）
        self.sos_band = butter(
            filter_order,
            [low_normalized, high_normalized],
            btype='band',
            output='sos'
        )
        self.zi_band = sosfilt_zi(self.sos_band)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """チャンクを処理"""
        if len(chunk) == 0:
            return chunk

        # 対象帯域を抽出
        band_signal, self.zi_band = sosfilt(self.sos_band, chunk, zi=self.zi_band)

        # 元の信号 + 帯域信号 × (ゲイン - 1)
        # ゲイン=1(0dB)の場合は変化なし
        # ゲイン>1の場合はブースト、ゲイン<1の場合はカット
        output = chunk + band_signal * (self.linear_gain - 1)

        # クリッピング防止
        output = np.clip(output, -1.0, 1.0)

        return output.astype(np.float32)


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


class EqualizerLogic(ComputeLogic):
    """
    イコライザーノードのロジック。
    指定した周波数帯域のゲインを調整する。
    """

    def __init__(self):
        self._buffer: WaveformBuffer | None = None
        self._filter: EqualizerFilter | None = None
        self._last_low_freq: int = -1
        self._last_high_freq: int = -1
        self._last_gain_db: float = -999.0
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

        # パラメータ取得（ポート入力を優先）
        high_freq = inputs.get("high_freq")
        if high_freq is None:
            high_freq = int(properties.get("high_freq", 4000))
        else:
            high_freq = int(high_freq)

        low_freq = inputs.get("low_freq")
        if low_freq is None:
            low_freq = int(properties.get("low_freq", 2000))
        else:
            low_freq = int(low_freq)

        gain_db = inputs.get("gain_db")
        if gain_db is None:
            gain_db = float(properties.get("gain_db", 0.0))
        else:
            gain_db = float(gain_db)

        # 値をクランプ
        low_freq = max(20, min(20000, low_freq))
        high_freq = max(100, min(20000, high_freq))
        gain_db = max(-24.0, min(24.0, gain_db))

        # low < high を保証
        if low_freq >= high_freq:
            high_freq = low_freq + 100

        sample_rate = audio_in.get("sample_rate", 16000)
        duration = audio_in.get("duration", 5.0)

        # カットオフ周波数がナイキスト周波数を超えないようにする
        max_freq = int(sample_rate / 2 - 1)
        low_freq = min(low_freq, max_freq - 100)
        high_freq = min(high_freq, max_freq)

        # バッファを初期化（初回のみ）
        if self._buffer is None:
            self._buffer = WaveformBuffer(sample_rate, duration)

        # パラメータが変更された場合、フィルターを再作成
        params_changed = (
            self._last_low_freq != low_freq or
            self._last_high_freq != high_freq or
            self._last_gain_db != gain_db or
            self._last_sample_rate != sample_rate
        )

        if self._filter is None or params_changed:
            self._filter = EqualizerFilter(
                low_freq=low_freq,
                high_freq=high_freq,
                gain_db=gain_db,
                sample_rate=sample_rate,
            )
            self._last_low_freq = low_freq
            self._last_high_freq = high_freq
            self._last_gain_db = gain_db
            self._last_sample_rate = sample_rate

        # deltaを取得して処理
        delta = audio_in.get("delta", [])
        if len(delta) > 0:
            delta_array = np.array(delta, dtype=np.float32)
            processed = self._filter.process(delta_array)
            processed_delta = processed.tolist()
        else:
            processed_delta = []

        # 処理済みdeltaをバッファに蓄積
        self._buffer.write(processed_delta)

        # バッファから表示用waveformを生成
        waveform_display = self._buffer.get_waveform_display(200)

        return {
            "audio": {
                "delta": processed_delta,
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": duration,
            }
        }
