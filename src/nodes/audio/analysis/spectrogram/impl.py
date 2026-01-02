"""
Spectrogram ノードの実装。
音声信号のスペクトログラムを表示する。
"""
from typing import Any, Dict, Optional

import numpy as np

from node_editor.node_def import ComputeLogic

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# 窓関数マッピング
WINDOW_FUNCTIONS = {
    0: "hamming",
    1: "hanning",
}


class SpectrogramBuffer:
    """スペクトログラム用バッファ"""

    def __init__(
        self,
        sample_rate: int,
        fft_size: int = 1024,
        shift_size: int = 512,
        duration: float = 5.0,
    ):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.duration = duration

        # 表示時間に対応するフレーム数を計算
        total_samples = int(sample_rate * duration)
        self.num_frames = max(1, total_samples // shift_size)

        # FFT結果の周波数ビン数（0〜ナイキスト周波数）
        self.num_bins = fft_size // 2 + 1

        # 入力バッファ
        self.input_buffer = np.zeros(0, dtype=np.float32)

        # スペクトログラムデータ（周波数×時間）
        self.spectrogram = np.zeros((self.num_bins, self.num_frames), dtype=np.float32)
        self.spectrogram.fill(-80)  # 初期値は-80dB

        # 平滑化用の履歴
        self.smoothing_history: list = []

    def set_params(self, fft_size: int, shift_size: int, duration: float):
        """パラメータ変更時にバッファをリセット"""
        needs_reset = (
            fft_size != self.fft_size or
            shift_size != self.shift_size or
            duration != self.duration
        )
        if needs_reset:
            self.fft_size = fft_size
            self.shift_size = shift_size
            self.duration = duration
            self.num_bins = fft_size // 2 + 1

            total_samples = int(self.sample_rate * duration)
            self.num_frames = max(1, total_samples // shift_size)

            self.input_buffer = np.zeros(0, dtype=np.float32)
            self.spectrogram = np.zeros((self.num_bins, self.num_frames), dtype=np.float32)
            self.spectrogram.fill(-80)
            self.smoothing_history = []

    def process(
        self,
        audio_data: np.ndarray,
        window_type: str = "hamming",
        smoothing: int = 1,
    ):
        """音声データを処理してスペクトログラムを更新"""
        if len(audio_data) == 0:
            return

        # 入力バッファに追加
        self.input_buffer = np.concatenate([self.input_buffer, audio_data])

        # 窓関数を生成
        if window_type == "hamming":
            window = np.hamming(self.fft_size)
        else:  # hanning
            window = np.hanning(self.fft_size)

        # シフトサイズごとにFFTを実行
        frames_processed = 0
        while len(self.input_buffer) >= self.fft_size:
            frame = self.input_buffer[:self.fft_size]
            self.input_buffer = self.input_buffer[self.shift_size:]

            # 窓関数適用
            windowed = frame * window

            # FFT
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum)

            # dBスケールに変換（最小値をクリップ）
            magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))

            # 平滑化
            if smoothing > 1:
                self.smoothing_history.append(magnitude_db)
                if len(self.smoothing_history) > smoothing:
                    self.smoothing_history.pop(0)
                magnitude_db = np.mean(self.smoothing_history, axis=0)

            # スペクトログラムを左にシフトして新しいフレームを追加
            self.spectrogram = np.roll(self.spectrogram, -1, axis=1)
            self.spectrogram[:, -1] = magnitude_db
            frames_processed += 1

    def get_image(self, width: int = 400, height: int = 200, colormap: int = 2) -> np.ndarray:
        """スペクトログラムを画像として取得"""
        # 正規化（-80dB〜0dBの範囲にクリップ）
        spec_normalized = np.clip(self.spectrogram, -80, 0)
        spec_normalized = (spec_normalized + 80) / 80  # 0〜1に正規化

        # 縦軸を反転（低周波が下になるように）
        spec_flipped = np.flipud(spec_normalized)

        # 0〜255の範囲に変換
        spec_uint8 = (spec_flipped * 255).astype(np.uint8)

        # リサイズ
        spec_resized = cv2.resize(spec_uint8, (width, height), interpolation=cv2.INTER_LINEAR)

        # カラーマップ適用（OpenCVのカラーマップ定数と一致）
        spec_color = cv2.applyColorMap(spec_resized, colormap)

        return spec_color


class SpectrogramLogic(ComputeLogic):
    """
    Spectrogramノードのロジック。
    音声信号のスペクトログラムを表示する。
    """

    def __init__(self):
        self._buffer: Optional[SpectrogramBuffer] = None
        self._last_sample_rate: int = -1
        self._last_duration: float = -1
        self._image_width: int = 400
        self._image_height: int = 200

    def reset(self):
        """バッファをリセット"""
        self._buffer = None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {
                "image": None,
                "__error__": "opencv-python is not installed"
            }

        audio_in = inputs.get("audio")

        # カラーマップは常に取得（音声入力がない場合も使用）
        colormap = int(properties.get("colormap", 2))  # デフォルトはJet

        if audio_in is None:
            # 音声入力がない場合でも既存のスペクトログラムを表示
            if self._buffer is not None:
                image = self._buffer.get_image(self._image_width, self._image_height, colormap)
                return {"image": image}
            return {"image": None}

        # パラメータ取得
        shift_size = int(properties.get("shift_size", 512))
        window_func_idx = int(properties.get("window_function", 0))
        smoothing = int(properties.get("smoothing", 1))

        window_type = WINDOW_FUNCTIONS.get(window_func_idx, "hamming")

        # FFTサイズはシフトサイズの2倍
        fft_size = shift_size * 2

        sample_rate = audio_in.get("sample_rate", 16000)
        duration = audio_in.get("duration", 5.0)

        # サンプルレートまたは表示時間の変更時はバッファを再初期化
        if self._last_sample_rate != sample_rate or self._last_duration != duration:
            self._buffer = None
            self._last_sample_rate = sample_rate
            self._last_duration = duration

        # バッファを初期化（初回のみ）
        if self._buffer is None:
            self._buffer = SpectrogramBuffer(
                sample_rate=sample_rate,
                fft_size=fft_size,
                shift_size=shift_size,
                duration=duration,
            )
        else:
            # パラメータ変更時はバッファを更新
            self._buffer.set_params(fft_size, shift_size, duration)

        # 音声データを処理
        delta = audio_in.get("delta", [])
        if len(delta) > 0:
            audio_array = np.array(delta, dtype=np.float32)
            self._buffer.process(audio_array, window_type, smoothing)

        # スペクトログラム画像を生成
        image = self._buffer.get_image(self._image_width, self._image_height, colormap)

        return {"image": image}
