from typing import Dict, Any, Optional
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


def get_interval_ms() -> int:
    """設定からグラフ実行間隔を取得"""
    return get_setting("graph.interval_ms", 50)


class SpeakerBuffer:
    """スレッドセーフな循環オーディオバッファ（出力用）"""

    def __init__(self, sample_rate: int, buffer_seconds: float = 0.5):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_seconds)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.available = 0  # 読み取り可能なサンプル数
        self.lock = threading.Lock()

    def write(self, data: np.ndarray):
        """データをバッファに書き込む"""
        with self.lock:
            data = data.flatten().astype(np.float32)
            n = len(data)

            # バッファオーバーフロー防止（古いデータを捨てる）
            if n > self.buffer_size:
                data = data[-self.buffer_size:]
                n = self.buffer_size

            # 空き容量チェック
            free_space = self.buffer_size - self.available
            if n > free_space:
                # 古いデータを捨てて空きを作る
                skip = n - free_space
                self.read_pos = (self.read_pos + skip) % self.buffer_size
                self.available -= skip

            # 循環書き込み
            end_pos = self.write_pos + n
            if end_pos <= self.buffer_size:
                self.buffer[self.write_pos:end_pos] = data
            else:
                first_part = self.buffer_size - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:n - first_part] = data[first_part:]

            self.write_pos = end_pos % self.buffer_size
            self.available += n

    def read(self, frames: int) -> np.ndarray:
        """指定フレーム数を読み取る（不足分は0で埋める）"""
        with self.lock:
            output = np.zeros(frames, dtype=np.float32)

            if self.available == 0:
                return output

            # 読み取り可能な量
            to_read = min(frames, self.available)

            # 循環読み取り
            end_pos = self.read_pos + to_read
            if end_pos <= self.buffer_size:
                output[:to_read] = self.buffer[self.read_pos:end_pos]
            else:
                first_part = self.buffer_size - self.read_pos
                output[:first_part] = self.buffer[self.read_pos:]
                output[first_part:to_read] = self.buffer[:to_read - first_part]

            self.read_pos = end_pos % self.buffer_size
            self.available -= to_read

            return output


class SpeakerStream:
    """スピーカー出力ストリーム管理"""
    def __init__(self, device_id: Optional[int], sample_rate: int, interval_ms: int):
        self.device_id = device_id if device_id != -1 else None
        self.sample_rate = sample_rate
        self.interval_ms = interval_ms
        self.stream: Optional[sd.OutputStream] = None
        self.running = False
        # バッファサイズ: interval_msの4倍分を確保（遅延とのバランス）
        buffer_seconds = (interval_ms * 4) / 1000.0
        self.audio_buffer = SpeakerBuffer(sample_rate, buffer_seconds=buffer_seconds)
        # ブロックサイズ: interval_msに相当するサンプル数
        self.blocksize = int(sample_rate * interval_ms / 1000)

    def _audio_callback(self, outdata, frames, time_info, status):
        """オーディオ出力コールバック"""
        if status:
            print(f"Output status: {status}")

        # バッファからデータを読み取り
        data = self.audio_buffer.read(frames)
        outdata[:, 0] = data

    def start(self):
        """ストリーム開始"""
        if self.running:
            return
        try:
            self.stream = sd.OutputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                blocksize=self.blocksize,
                latency=0.1,
            )
            self.stream.start()
            self.running = True
        except Exception as e:
            print(f"Failed to start speaker: {e}")
            self.running = False

    def stop(self):
        """ストリーム停止"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False

    def play(self, audio_data: np.ndarray):
        """オーディオデータをバッファに追加"""
        self.audio_buffer.write(audio_data)


# デバイスIDごとのストリームを管理
_speaker_streams: Dict[int, SpeakerStream] = {}
_speaker_lock = threading.Lock()


class SpeakerNodeLogic(ComputeLogic):
    """
    スピーカーにオーディオを出力するノードロジック。
    入力されたオーディオを再生し、Waveformをパススルーで出力する。
    """
    def reset(self):
        """スピーカーバッファをクリア"""
        global _speaker_streams
        with _speaker_lock:
            for stream in _speaker_streams.values():
                # バッファをクリア
                with stream.audio_buffer.lock:
                    stream.audio_buffer.buffer.fill(0)
                    stream.audio_buffer.write_pos = 0
                    stream.audio_buffer.read_pos = 0
                    stream.audio_buffer.available = 0

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        audio_in = inputs.get("audio")

        if audio_in is None:
            return {}

        if not SOUNDDEVICE_AVAILABLE:
            return {"__error__": "sounddevice is not installed"}

        global _speaker_streams
        device_id = int(properties.get("device_id", -1))
        sample_rate = get_sample_rate()
        interval_ms = get_interval_ms()

        # 入力から差分サンプルデータを抽出（前回以降の新規データ）
        delta = audio_in.get("delta", [])
        if not delta:
            return {}

        with _speaker_lock:
            # ストリームを取得または作成
            if device_id not in _speaker_streams:
                stream = SpeakerStream(device_id, sample_rate=sample_rate, interval_ms=interval_ms)
                stream.start()
                _speaker_streams[device_id] = stream

            stream = _speaker_streams[device_id]
            if not stream.running:
                stream.start()

            # 差分サンプルを再生
            audio_data = np.array(delta, dtype=np.float32)
            if len(audio_data) > 0:
                stream.play(audio_data)

        # 出力なし（入力のみ）
        return {}


def cleanup_speaker_streams():
    """全スピーカーストリームを解放"""
    global _speaker_streams
    with _speaker_lock:
        for stream in _speaker_streams.values():
            stream.stop()
        _speaker_streams.clear()


atexit.register(cleanup_speaker_streams)
