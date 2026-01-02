"""
Noise Gate ノードの実装。
閾値を下回った音声信号を完全にカット（ミュート）する。
"""
from typing import Dict, Any, List
from node_editor.node_def import ComputeLogic
import numpy as np
import threading


class NoiseGate:
    """ノイズゲート処理"""

    def __init__(
        self,
        threshold: float = 0.02,
        attack_ms: int = 0,
        release_ms: int = 0,
        hold_ms: int = 0,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate

        # アタック/リリース係数を計算
        self.alpha_attack = (
            0.0
            if attack_ms == 0
            else np.exp(-1.0 / (sample_rate * (attack_ms / 1000.0)))
        )
        self.alpha_release = (
            0.0
            if release_ms == 0
            else np.exp(-1.0 / (sample_rate * (release_ms / 1000.0)))
        )

        # ホールドタイム（サンプル数）
        self.hold_samples = int(sample_rate * (hold_ms / 1000.0))

        # エンベロープ状態（ゲート開=1.0、閉=0.0）
        self.env = 1.0
        # ホールドカウンター（閾値を下回ってからの経過サンプル数）
        self.hold_counter = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """チャンクをサンプル単位で処理してゲート"""
        if len(chunk) == 0:
            return chunk

        output = np.zeros_like(chunk)

        for i, sample in enumerate(chunk):
            # サンプルごとにレベル検出
            level = abs(sample)

            # ゲイン計算
            if level >= self.threshold:
                # 閾値以上：ゲート開（gain = 1.0）
                gain = 1.0
                # ホールドカウンターをリセット
                self.hold_counter = 0
            else:
                # 閾値未満：ホールドタイムをチェック
                if self.hold_counter < self.hold_samples:
                    # ホールド期間中：まだゲートを閉じない
                    gain = 1.0
                    self.hold_counter += 1
                else:
                    # ホールド期間終了：ゲートを閉じる
                    gain = 0.0

            # エンベロープ追従（サンプル単位）
            if gain < self.env:
                # ゲインを下げる（リリース：ゲートを閉じる）
                self.env = (
                    self.alpha_release * self.env + (1 - self.alpha_release) * gain
                )
            else:
                # ゲインを上げる（アタック：ゲートを開く）
                self.env = (
                    self.alpha_attack * self.env + (1 - self.alpha_attack) * gain
                )

            output[i] = sample * self.env

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


class NoiseGateLogic(ComputeLogic):
    """
    ノイズゲートノードのロジック。
    閾値を下回った音声信号を完全にカットする。
    """

    def __init__(self):
        self._buffer: WaveformBuffer | None = None
        self._gate: NoiseGate | None = None
        self._last_threshold: float = -1
        self._last_attack_ms: int = -1
        self._last_release_ms: int = -1
        self._last_hold_ms: int = -1
        self._last_sample_rate: int = -1

    def reset(self):
        """バッファとゲートをリセット"""
        self._buffer = None
        self._gate = None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        audio_in = inputs.get("audio")

        if audio_in is None:
            return {"audio": None}

        # パラメータ取得
        threshold = float(properties.get("threshold", 0.02))
        attack_ms = int(properties.get("attack_ms", 0))
        release_ms = int(properties.get("release_ms", 0))
        hold_ms = int(properties.get("hold_ms", 0))

        # 値をクランプ
        threshold = max(0.0, min(1.0, threshold))
        attack_ms = max(0, attack_ms)
        release_ms = max(0, release_ms)
        hold_ms = max(0, hold_ms)

        sample_rate = audio_in.get("sample_rate", 16000)
        duration = audio_in.get("duration", 5.0)

        # バッファを初期化（初回のみ）
        if self._buffer is None:
            self._buffer = WaveformBuffer(sample_rate, duration)

        # パラメータが変更された場合、ゲートを再作成
        params_changed = (
            self._last_threshold != threshold or
            self._last_attack_ms != attack_ms or
            self._last_release_ms != release_ms or
            self._last_hold_ms != hold_ms or
            self._last_sample_rate != sample_rate
        )

        if self._gate is None or params_changed:
            self._gate = NoiseGate(
                threshold=threshold,
                attack_ms=attack_ms,
                release_ms=release_ms,
                hold_ms=hold_ms,
                sample_rate=sample_rate,
            )
            self._last_threshold = threshold
            self._last_attack_ms = attack_ms
            self._last_release_ms = release_ms
            self._last_hold_ms = hold_ms
            self._last_sample_rate = sample_rate

        # deltaを取得してゲート処理
        delta = audio_in.get("delta", [])
        if len(delta) > 0:
            delta_array = np.array(delta, dtype=np.float32)
            gated = self._gate.process(delta_array)
            gated_delta = gated.tolist()
        else:
            gated_delta = []

        # 処理済みdeltaをバッファに蓄積
        self._buffer.write(gated_delta)

        # バッファから表示用waveformを生成
        waveform_display = self._buffer.get_waveform_display(200)

        return {
            "audio": {
                "delta": gated_delta,
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": duration,
            }
        }
