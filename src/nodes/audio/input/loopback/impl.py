import platform
from typing import Dict, Any, Optional, List
from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting
import numpy as np
import atexit
import threading

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None


def get_sample_rate() -> int:
    """設定からサンプリングレートを取得"""
    return get_setting("audio.sample_rate", 16000)


def find_loopback_devices() -> List[int]:
    """
    Windowsのループバック入力デバイスを検索し、候補リストを返す。
    優先度順にソートされたデバイスIDのリスト。
    """
    if platform.system() != "Windows":
        return []

    if not SOUNDDEVICE_AVAILABLE:
        return []

    candidate_devices = []

    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[Loopback] Failed to query audio devices: {e}")
        return []

    # 優先度別に分類
    priority_levels: Dict[int, List[int]] = {1: [], 2: [], 3: []}

    for device_id, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            device_name = device.get("name", "").lower()

            # 最高優先度: ステレオミキサー系
            if any(k in device_name for k in ["stereo mix", "ステレオ ミキサー", "what u hear"]):
                priority_levels[1].append(device_id)
            # 中優先度: PCスピーカー系
            elif "pc スピーカー" in device_name or "pc speaker" in device_name:
                priority_levels[2].append(device_id)
            # 低優先度: Steam系
            elif "steam streaming speakers" in device_name and "input" in device_name:
                priority_levels[3].append(device_id)

    # 優先度順に結合
    for level in sorted(priority_levels.keys()):
        candidate_devices.extend(priority_levels[level])

    # 重複を除去
    candidate_devices = list(dict.fromkeys(candidate_devices))

    return candidate_devices


class AudioBuffer:
    """スレッドセーフな循環オーディオバッファ"""

    def __init__(self, sample_rate: int, buffer_seconds: float = 5.0):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.buffer_size = int(sample_rate * buffer_seconds)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0
        self.valid_samples = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray):
        """データをバッファに書き込む"""
        with self.lock:
            data = data.flatten().astype(np.float32)
            n = len(data)
            if n >= self.buffer_size:
                self.buffer[:] = data[-self.buffer_size:]
                self.write_pos = 0
                self.valid_samples = self.buffer_size
            else:
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
        """有効なデータを時系列順で読み出す"""
        with self.lock:
            if self.valid_samples == 0:
                return np.array([], dtype=np.float32)

            if self.valid_samples >= self.buffer_size:
                return np.concatenate([
                    self.buffer[self.write_pos:],
                    self.buffer[:self.write_pos]
                ])
            else:
                start_pos = (self.write_pos - self.valid_samples) % self.buffer_size
                if start_pos < self.write_pos:
                    return self.buffer[start_pos:self.write_pos].copy()
                else:
                    return np.concatenate([
                        self.buffer[start_pos:],
                        self.buffer[:self.write_pos]
                    ])

    def read_new_samples(self, last_total: int) -> tuple:
        """前回読み取り以降の新しいサンプルを取得"""
        with self.lock:
            current_total = self.total_written
            new_count = current_total - last_total

            # total_writtenがリセットされた場合（last_total > current_total）
            if new_count < 0:
                new_count = current_total
                last_total = 0

            if new_count <= 0:
                return np.array([], dtype=np.float32), current_total

            if new_count >= self.buffer_size:
                result = np.concatenate([
                    self.buffer[self.write_pos:],
                    self.buffer[:self.write_pos]
                ])
                return result, current_total

            start_pos = (self.write_pos - new_count) % self.buffer_size
            if start_pos < self.write_pos:
                result = self.buffer[start_pos:self.write_pos].copy()
            else:
                result = np.concatenate([
                    self.buffer[start_pos:],
                    self.buffer[:self.write_pos]
                ])
            return result, current_total


class LoopbackStream:
    """Windowsループバック入力ストリーム管理"""

    def __init__(self, target_sample_rate: int, buffer_seconds: float = 5.0):
        self.target_sample_rate = target_sample_rate
        self.audio_buffer = AudioBuffer(target_sample_rate, buffer_seconds)
        self.stream: Optional[Any] = None
        self.running = False
        self.last_read_total = 0
        self.device_id: Optional[int] = None
        self.device_sample_rate: int = target_sample_rate
        self.candidate_devices = find_loopback_devices()
        self.current_device_index = 0

        if self.candidate_devices:
            self.device_id = self.candidate_devices[0]
            print(f"[Loopback] Found {len(self.candidate_devices)} candidate devices")
            for i, dev_id in enumerate(self.candidate_devices):
                try:
                    name = sd.query_devices(dev_id)['name']
                    print(f"[Loopback]   {i}: ID={dev_id}, Name={name}")
                except Exception:
                    pass
        else:
            print("[Loopback] No loopback device found. Enable 'Stereo Mix' in Windows sound settings.")

    def _audio_callback(self, indata, frames, time_info, status):
        """オーディオコールバック"""
        if status:
            print(f"[Loopback] Stream status: {status}")

        # モノラルに変換
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()

        # リサンプリング（デバイスのサンプルレート → ターゲットサンプルレート）
        if self.device_sample_rate != self.target_sample_rate:
            ratio = self.target_sample_rate / self.device_sample_rate
            new_length = int(len(mono) * ratio)
            if new_length > 0:
                indices = np.linspace(0, len(mono) - 1, new_length)
                mono = np.interp(indices, np.arange(len(mono)), mono).astype(np.float32)

        self.audio_buffer.write(mono)

    def start(self) -> bool:
        """ストリーム開始"""
        if self.running:
            return True

        if platform.system() != "Windows":
            print("[Loopback] This node only works on Windows")
            return False

        if self.device_id is None:
            print("[Loopback] No loopback device available")
            return False

        try:
            device_info = sd.query_devices(self.device_id)
            self.device_sample_rate = int(device_info["default_samplerate"])
            channels = min(2, device_info["max_input_channels"])

            print(f"[Loopback] Starting stream: device={self.device_id}, rate={self.device_sample_rate}, ch={channels}")

            self.stream = sd.InputStream(
                device=self.device_id,
                channels=channels,
                samplerate=self.device_sample_rate,
                callback=self._audio_callback,
                blocksize=512,
                latency='low',
                dtype='float32',
            )
            self.stream.start()
            self.running = True
            return True

        except Exception as e:
            print(f"[Loopback] Failed to start stream on device {self.device_id}: {e}")
            self.running = False
            return self._try_next_device()

    def _try_next_device(self) -> bool:
        """次の候補デバイスを試す"""
        self.current_device_index += 1
        if self.current_device_index < len(self.candidate_devices):
            self.device_id = self.candidate_devices[self.current_device_index]
            print(f"[Loopback] Trying next device: ID={self.device_id}")
            return self.start()
        else:
            print("[Loopback] All candidate devices failed")
            self.device_id = None
            return False

    def stop(self):
        """ストリーム停止"""
        if self.stream is not None:
            try:
                if not self.stream.closed:
                    self.stream.stop()
                    self.stream.close()
            except Exception as e:
                print(f"[Loopback] Error closing stream: {e}")
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


# グローバルストリーム管理
_loopback_stream: Optional[LoopbackStream] = None
_stream_lock = threading.Lock()


BUFFER_SECONDS = 5.0


class LoopbackNodeLogic(ComputeLogic):
    """
    Windowsのループバックデバイスからシステム音声をキャプチャするノード。
    """

    def reset(self):
        """ストリームのバッファをクリア"""
        global _loopback_stream
        with _stream_lock:
            if _loopback_stream is not None:
                with _loopback_stream.audio_buffer.lock:
                    _loopback_stream.audio_buffer.buffer.fill(0)
                    _loopback_stream.audio_buffer.write_pos = 0
                    _loopback_stream.audio_buffer.total_written = 0
                    _loopback_stream.audio_buffer.valid_samples = 0
                _loopback_stream.last_read_total = 0

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        # Windows以外では動作しない
        if platform.system() != "Windows":
            return {
                "audio_out": None,
                "__error__": "Loopback node only works on Windows",
            }

        if not SOUNDDEVICE_AVAILABLE:
            return {
                "audio_out": None,
                "__error__": "sounddevice is not installed",
            }

        global _loopback_stream
        sample_rate = get_sample_rate()

        with _stream_lock:
            # ストリームを作成または取得
            if _loopback_stream is None:
                _loopback_stream = LoopbackStream(
                    target_sample_rate=sample_rate,
                    buffer_seconds=BUFFER_SECONDS
                )

            stream = _loopback_stream

            # ストリーム開始
            if not stream.running:
                if not stream.start():
                    return {
                        "audio_out": {
                            "delta": [],
                            "waveform": [0.0] * 400,
                            "sample_rate": sample_rate,
                            "duration": BUFFER_SECONDS,
                        },
                        "__error__": "No loopback device available. Enable 'Stereo Mix' in Windows sound settings.",
                    }

            # Waveformデータを取得
            waveform = stream.get_waveform()
            delta_samples = stream.get_new_samples()

        # 表示用にmin/max計算
        display_width = 200
        full_buffer_size = int(sample_rate * BUFFER_SECONDS)
        samples_per_pixel = full_buffer_size // display_width

        waveform_display = []
        data_pixels = len(waveform) // samples_per_pixel if samples_per_pixel > 0 else 0
        empty_pixels = display_width - data_pixels

        # 左側の空白部分
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
                "delta": delta_samples.tolist(),
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": BUFFER_SECONDS,
            }
        }


def cleanup_loopback():
    """ストリームを解放"""
    global _loopback_stream
    with _stream_lock:
        if _loopback_stream is not None:
            _loopback_stream.stop()
            _loopback_stream = None


atexit.register(cleanup_loopback)
