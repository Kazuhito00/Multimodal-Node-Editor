"""
Power Spectrum ノードの実装。
音声信号のパワースペクトルを表示する。
"""
from typing import Any, Dict, Optional, List

import numpy as np

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# 画像サイズ
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 200

# 窓関数マッピング
WINDOW_FUNCTIONS = {
    0: "hamming",
    1: "hanning",
}


class PowerSpectrumBuffer:
    """パワースペクトル用バッファ"""

    def __init__(self, sample_rate: int, fft_size: int = 1024):
        self.sample_rate = sample_rate
        self.fft_size = fft_size

        # FFT結果の周波数ビン数（0〜ナイキスト周波数）
        self.num_bins = fft_size // 2 + 1

        # 入力バッファ
        self.input_buffer = np.zeros(0, dtype=np.float32)

        # 現在のパワースペクトル
        self.power_spectrum = np.zeros(self.num_bins, dtype=np.float32)
        self.power_spectrum.fill(-80)  # 初期値は-80dB

        # 平滑化用の履歴
        self.smoothing_history: List[np.ndarray] = []

    def set_params(self, fft_size: int):
        """パラメータ変更時にバッファをリセット"""
        if fft_size != self.fft_size:
            self.fft_size = fft_size
            self.num_bins = fft_size // 2 + 1
            self.input_buffer = np.zeros(0, dtype=np.float32)
            self.power_spectrum = np.zeros(self.num_bins, dtype=np.float32)
            self.power_spectrum.fill(-80)
            self.smoothing_history = []

    def process(
        self,
        audio_data: np.ndarray,
        window_type: str = "hamming",
        smoothing: int = 1,
    ):
        """音声データを処理してパワースペクトルを更新"""
        if len(audio_data) == 0:
            return

        # 入力バッファに追加
        self.input_buffer = np.concatenate([self.input_buffer, audio_data])

        # 窓関数を生成
        if window_type == "hamming":
            window = np.hamming(self.fft_size)
        else:  # hanning
            window = np.hanning(self.fft_size)

        # FFTサイズ分のデータがあれば処理
        while len(self.input_buffer) >= self.fft_size:
            frame = self.input_buffer[:self.fft_size]
            self.input_buffer = self.input_buffer[self.fft_size // 2:]  # 50%オーバーラップ

            # 窓関数適用
            windowed = frame * window

            # FFT
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum)

            # dBスケールに変換（最小値をクリップ）
            magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))

            # 平滑化
            self.smoothing_history.append(magnitude_db)
            if len(self.smoothing_history) > smoothing:
                self.smoothing_history.pop(0)

            self.power_spectrum = np.mean(self.smoothing_history, axis=0)

    def get_image(self, width: int = 400, height: int = 200, is_light_mode: bool = True) -> np.ndarray:
        """パワースペクトルを画像として取得"""
        # 色設定（BGR形式）
        if is_light_mode:
            bg_color = [221, 221, 221]       # #ddd
            grid_color = [187, 187, 187]     # #bbb
            spectrum_color = [0, 170, 0]     # #00aa00
        else:
            bg_color = [0, 0, 0]             # #000
            grid_color = [51, 51, 51]        # #333
            spectrum_color = [0, 255, 0]     # #00ff00

        # 背景色で画像を作成
        image = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # 正規化（-80dB〜+40dBの範囲にクリップ）
        db_min = -80
        db_max = 40
        db_range = db_max - db_min  # 120dB
        spec_normalized = np.clip(self.power_spectrum, db_min, db_max)
        spec_normalized = (spec_normalized - db_min) / db_range  # 0〜1に正規化

        # 周波数ビンを画像幅にリサンプリング
        x_indices = np.linspace(0, len(spec_normalized) - 1, width).astype(int)
        resampled = spec_normalized[x_indices]

        # グリッド線を描画（横線）
        for i in range(1, 4):
            y = int(height * i / 4)
            image[y, :] = grid_color

        # Y座標を計算
        y_coords = (height - 1) - (resampled * (height - 1))
        y_coords = y_coords.astype(int)
        y_coords = np.clip(y_coords, 0, height - 1)

        # パワースペクトルを描画（線と同じ色で塗りつぶし）
        for x in range(width):
            y_top = y_coords[x]
            # 上端から下端まで塗りつぶし
            image[y_top:height, x] = spectrum_color

        return image


class PowerSpectrumLogic(ComputeLogic):
    """
    Power Spectrumノードのロジック。
    音声信号のパワースペクトルを表示する。
    """

    def __init__(self):
        self._buffer: Optional[PowerSpectrumBuffer] = None
        self._last_sample_rate: int = -1
        self._image_width: int = IMAGE_WIDTH
        self._image_height: int = IMAGE_HEIGHT

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

        # UIテーマ設定に従ってモードを決定
        theme = get_setting("ui.theme", "dark")
        is_light_mode = (theme == "light")

        if audio_in is None:
            # 音声入力がない場合でも既存のスペクトルを表示
            if self._buffer is not None:
                image = self._buffer.get_image(self._image_width, self._image_height, is_light_mode)
                return {"image": image}
            return {"image": None}

        # パラメータ取得
        fft_size = int(properties.get("fft_size", 1024))
        window_func_idx = int(properties.get("window_function", 0))
        smoothing = int(properties.get("smoothing", 3))

        window_type = WINDOW_FUNCTIONS.get(window_func_idx, "hamming")

        sample_rate = audio_in.get("sample_rate", 16000)

        # サンプルレート変更時はバッファを再初期化
        if self._last_sample_rate != sample_rate:
            self._buffer = None
            self._last_sample_rate = sample_rate

        # バッファを初期化（初回のみ）
        if self._buffer is None:
            self._buffer = PowerSpectrumBuffer(
                sample_rate=sample_rate,
                fft_size=fft_size,
            )
        else:
            # パラメータ変更時はバッファを更新
            self._buffer.set_params(fft_size)

        # 音声データを処理
        delta = audio_in.get("delta", [])
        if len(delta) > 0:
            audio_array = np.array(delta, dtype=np.float32)
            self._buffer.process(audio_array, window_type, smoothing)

        # パワースペクトル画像を生成
        image = self._buffer.get_image(self._image_width, self._image_height, is_light_mode)

        return {"image": image}
