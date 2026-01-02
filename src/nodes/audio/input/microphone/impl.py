from typing import Dict, Any, Optional
from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting
import numpy as np
import atexit
import threading
import time

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None


def get_sample_rate() -> int:
    """設定からサンプリングレートを取得"""
    return get_setting("audio.sample_rate", 16000)


class AudioBuffer:
    """スレッドセーフな循環オーディオバッファ"""

    def __init__(self, sample_rate: int, buffer_seconds: float = 5.0):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.buffer_size = int(sample_rate * buffer_seconds)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0  # 累計書き込みサンプル数
        self.valid_samples = 0  # バッファ内の有効サンプル数（最大buffer_size）
        self.lock = threading.Lock()

    def write(self, data: np.ndarray):
        """データをバッファに書き込む"""
        with self.lock:
            data = data.flatten().astype(np.float32)
            n = len(data)
            if n >= self.buffer_size:
                # バッファより大きい場合は最後の部分だけ使用
                self.buffer[:] = data[-self.buffer_size:]
                self.write_pos = 0
                self.valid_samples = self.buffer_size
            else:
                # 循環書き込み
                end_pos = self.write_pos + n
                if end_pos <= self.buffer_size:
                    self.buffer[self.write_pos:end_pos] = data
                else:
                    first_part = self.buffer_size - self.write_pos
                    self.buffer[self.write_pos:] = data[:first_part]
                    self.buffer[:n - first_part] = data[first_part:]
                self.write_pos = end_pos % self.buffer_size
                self.valid_samples = min(self.valid_samples + n, self.buffer_size)
            self.total_written += n

    def read(self) -> np.ndarray:
        """有効なデータのみを時系列順で読み出す"""
        with self.lock:
            if self.valid_samples == 0:
                return np.array([], dtype=np.float32)

            if self.valid_samples >= self.buffer_size:
                # バッファが満杯の場合は全体を返す
                return np.concatenate(
                    [self.buffer[self.write_pos:], self.buffer[:self.write_pos]]
                )
            else:
                # バッファが満杯でない場合は有効なデータのみ返す
                start_pos = (self.write_pos - self.valid_samples) % self.buffer_size
                if start_pos < self.write_pos:
                    return self.buffer[start_pos:self.write_pos].copy()
                else:
                    return np.concatenate([
                        self.buffer[start_pos:],
                        self.buffer[:self.write_pos]
                    ])

    def read_new_samples(self, last_total: int) -> tuple[np.ndarray, int]:
        """前回読み取り以降の新しいサンプルを取得"""
        with self.lock:
            current_total = self.total_written
            new_count = current_total - last_total

            if new_count <= 0:
                return np.array([], dtype=np.float32), current_total

            if new_count >= self.buffer_size:
                # バッファサイズ以上の新規データがある場合は全バッファを返す
                result = np.concatenate(
                    [self.buffer[self.write_pos :], self.buffer[: self.write_pos]]
                )
                return result, current_total

            # 新しいサンプルだけを抽出
            start_pos = (self.write_pos - new_count) % self.buffer_size
            if start_pos < self.write_pos:
                result = self.buffer[start_pos : self.write_pos].copy()
            else:
                result = np.concatenate(
                    [self.buffer[start_pos:], self.buffer[: self.write_pos]]
                )
            return result, current_total


class MicrophoneStream:
    """マイク入力ストリーム管理"""

    def __init__(
        self, device_id: Optional[int], sample_rate: int, buffer_seconds: float = 5.0
    ):
        self.device_id = device_id if device_id != -1 else None
        self.sample_rate = sample_rate
        self.audio_buffer = AudioBuffer(sample_rate, buffer_seconds)
        self.stream: Optional[sd.InputStream] = None
        self.running = False
        self.last_read_total = 0  # 前回読み取り時の累計サンプル数

    def _audio_callback(self, indata, frames, time_info, status):
        """オーディオコールバック"""
        if status:
            print(f"Audio status: {status}")
        # モノラルに変換して保存
        if indata.ndim > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata.flatten()
        self.audio_buffer.write(mono)

    def start(self):
        """ストリーム開始"""
        if self.running:
            return
        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                blocksize=256,  # 低遅延のため小さいブロックサイズ
                latency='low',
            )
            self.stream.start()
            self.running = True
        except Exception as e:
            print(f"Failed to start microphone: {e}")
            self.running = False

    def stop(self):
        """ストリーム停止"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False

    def get_waveform(self) -> np.ndarray:
        """現在のバッファからWaveformデータを取得"""
        return self.audio_buffer.read()

    def get_new_samples(self) -> np.ndarray:
        """前回取得以降の新しいサンプルを取得"""
        new_samples, self.last_read_total = self.audio_buffer.read_new_samples(
            self.last_read_total
        )
        return new_samples


# デバイスIDごとのストリームを管理
_streams: Dict[int, MicrophoneStream] = {}
_streams_lock = threading.Lock()


BUFFER_SECONDS = 5.0  # 固定5秒バッファ


class MicrophoneNodeLogic(ComputeLogic):
    """
    マイクからオーディオを取得するノードロジック。
    出力はWaveformデータ（numpy配列）。
    """

    def reset(self):
        """全ストリームのバッファをクリア"""
        global _streams
        with _streams_lock:
            for stream in _streams.values():
                with stream.audio_buffer.lock:
                    stream.audio_buffer.buffer.fill(0)
                    stream.audio_buffer.write_pos = 0
                    stream.audio_buffer.total_written = 0

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not SOUNDDEVICE_AVAILABLE:
            return {
                "audio_out": None,
                "__error__": "sounddevice is not installed",
            }

        global _streams
        device_id = int(properties.get("device_id", -1))
        sample_rate = get_sample_rate()

        with _streams_lock:
            # ストリームを取得または作成
            if device_id not in _streams:
                stream = MicrophoneStream(
                    device_id, sample_rate=sample_rate, buffer_seconds=BUFFER_SECONDS
                )
                stream.start()
                _streams[device_id] = stream

            stream = _streams[device_id]
            if not stream.running:
                stream.start()

            # Waveformデータを取得
            waveform = stream.get_waveform()
            # 差分サンプルを取得（前回取得以降の新しいサンプル）
            delta_samples = stream.get_new_samples()

        # 表示用にmin/max計算（200ピクセル分、フロントエンドのcanvas幅に合わせる）
        display_width = 200
        full_buffer_size = int(sample_rate * BUFFER_SECONDS)
        samples_per_pixel = full_buffer_size // display_width

        # 5秒バッファ全体を基準にした表示（データがない部分は0）
        waveform_display = []
        # 現在のデータは右端（最新）に表示、左側はまだデータがない部分
        data_pixels = len(waveform) // samples_per_pixel if samples_per_pixel > 0 else 0
        empty_pixels = display_width - data_pixels

        # 左側の空白部分（まだデータがない）
        for _ in range(empty_pixels):
            waveform_display.append(0.0)
            waveform_display.append(0.0)

        # 右側の実データ部分
        for i in range(data_pixels):
            start = i * samples_per_pixel
            end = start + samples_per_pixel
            segment = waveform[start:end]
            if len(segment) > 0:
                waveform_display.append(float(np.min(segment)))
                waveform_display.append(float(np.max(segment)))
            else:
                waveform_display.append(0.0)
                waveform_display.append(0.0)

        return {
            "audio_out": {
                "delta": delta_samples.tolist(),  # 差分サンプル（前回以降の新規データ）
                "waveform": waveform_display,  # 表示用（min/maxペア×400）
                "sample_rate": sample_rate,
                "duration": BUFFER_SECONDS,
            }
        }


def cleanup_streams():
    """全ストリームを解放"""
    global _streams
    with _streams_lock:
        for stream in _streams.values():
            stream.stop()
        _streams.clear()


atexit.register(cleanup_streams)
