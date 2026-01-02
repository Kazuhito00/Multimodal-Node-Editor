"""
MSC (Magnitude Squared Coherence) ノードの実装。
2つの音声信号間の周波数毎のコヒーレンス（類似度）を計算・表示する。
"""
from typing import Any, Dict, Optional, List

import numpy as np

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting


# 画像サイズ
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 200

# 窓関数マッピング
WINDOW_FUNCTIONS = {
    0: "hamming",
    1: "hanning",
}


class MSCBuffer:
    """MSC計算用バッファ"""

    def __init__(self, sample_rate: int, fft_size: int = 1024):
        self.sample_rate = sample_rate
        self.fft_size = fft_size

        # FFT結果の周波数ビン数（0〜ナイキスト周波数）
        self.num_bins = fft_size // 2 + 1

        # 入力バッファ（2チャンネル分）
        self.input_buffer_1 = np.zeros(0, dtype=np.float32)
        self.input_buffer_2 = np.zeros(0, dtype=np.float32)

        # MSC結果
        self.msc = np.zeros(self.num_bins, dtype=np.float32)

        # 平滑化用の履歴（Pxx, Pyy, Pxy を保持）
        self.pxx_history: List[np.ndarray] = []
        self.pyy_history: List[np.ndarray] = []
        self.pxy_history: List[np.ndarray] = []

    def set_params(self, fft_size: int):
        """パラメータ変更時にバッファをリセット"""
        if fft_size != self.fft_size:
            self.fft_size = fft_size
            self.num_bins = fft_size // 2 + 1
            self.input_buffer_1 = np.zeros(0, dtype=np.float32)
            self.input_buffer_2 = np.zeros(0, dtype=np.float32)
            self.msc = np.zeros(self.num_bins, dtype=np.float32)
            self.pxx_history = []
            self.pyy_history = []
            self.pxy_history = []

    def process(
        self,
        audio_data_1: np.ndarray,
        audio_data_2: np.ndarray,
        window_type: str = "hamming",
        smoothing: int = 5,
    ):
        """2つの音声データを処理してMSCを更新"""
        if len(audio_data_1) == 0 or len(audio_data_2) == 0:
            return

        # 長さを揃える
        min_len = min(len(audio_data_1), len(audio_data_2))
        audio_data_1 = audio_data_1[:min_len]
        audio_data_2 = audio_data_2[:min_len]

        # 入力バッファに追加
        self.input_buffer_1 = np.concatenate([self.input_buffer_1, audio_data_1])
        self.input_buffer_2 = np.concatenate([self.input_buffer_2, audio_data_2])

        # 窓関数を生成
        if window_type == "hamming":
            window = np.hamming(self.fft_size)
        else:  # hanning
            window = np.hanning(self.fft_size)

        # FFTサイズ分のデータがあれば処理
        while (len(self.input_buffer_1) >= self.fft_size and
               len(self.input_buffer_2) >= self.fft_size):
            frame_1 = self.input_buffer_1[:self.fft_size]
            frame_2 = self.input_buffer_2[:self.fft_size]

            # 50%オーバーラップ
            self.input_buffer_1 = self.input_buffer_1[self.fft_size // 2:]
            self.input_buffer_2 = self.input_buffer_2[self.fft_size // 2:]

            # 窓関数適用
            windowed_1 = frame_1 * window
            windowed_2 = frame_2 * window

            # FFT
            spectrum_1 = np.fft.rfft(windowed_1)
            spectrum_2 = np.fft.rfft(windowed_2)

            # パワースペクトル密度とクロススペクトル密度
            pxx = np.abs(spectrum_1) ** 2
            pyy = np.abs(spectrum_2) ** 2
            pxy = spectrum_1 * np.conj(spectrum_2)

            # 履歴に追加
            self.pxx_history.append(pxx)
            self.pyy_history.append(pyy)
            self.pxy_history.append(pxy)

            # smoothing数を超えたら古いものを削除
            if len(self.pxx_history) > smoothing:
                self.pxx_history.pop(0)
                self.pyy_history.pop(0)
                self.pxy_history.pop(0)

            # 平均を計算
            pxx_avg = np.mean(self.pxx_history, axis=0)
            pyy_avg = np.mean(self.pyy_history, axis=0)
            pxy_avg = np.mean(self.pxy_history, axis=0)

            # MSC = |Pxy|^2 / (Pxx * Pyy)
            denominator = pxx_avg * pyy_avg
            # ゼロ除算を防ぐ
            denominator = np.maximum(denominator, 1e-10)
            self.msc = np.abs(pxy_avg) ** 2 / denominator

            # 0〜1の範囲にクリップ
            self.msc = np.clip(self.msc, 0, 1)

    def get_image(self, width: int = 400, height: int = 200, is_light_mode: bool = True) -> np.ndarray:
        """MSCを画像として取得"""
        # 色設定（BGR形式）
        if is_light_mode:
            bg_color = [221, 221, 221]       # #ddd
            grid_color = [187, 187, 187]     # #bbb
            spectrum_color = [170, 0, 0]     # #0000aa (BGR) -> blue
        else:
            bg_color = [0, 0, 0]             # #000
            grid_color = [51, 51, 51]        # #333
            spectrum_color = [255, 100, 100] # light blue

        # 背景色で画像を作成
        image = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # MSCは0〜1の範囲
        msc_normalized = self.msc

        # 周波数ビンを画像幅にリサンプリング
        x_indices = np.linspace(0, len(msc_normalized) - 1, width).astype(int)
        resampled = msc_normalized[x_indices]

        # グリッド線を描画（横線：25%, 50%, 75%）
        for i in range(1, 4):
            y = int(height * i / 4)
            image[y, :] = grid_color

        # Y座標を計算（MSCが1のとき上端、0のとき下端）
        y_coords = (height - 1) - (resampled * (height - 1))
        y_coords = y_coords.astype(int)
        y_coords = np.clip(y_coords, 0, height - 1)

        # MSCを描画（塗りつぶし）
        for x in range(width):
            y_top = y_coords[x]
            image[y_top:height, x] = spectrum_color

        return image


class MSCLogic(ComputeLogic):
    """
    MSCノードのロジック。
    2つの音声信号間のMagnitude Squared Coherenceを計算・表示する。
    """

    def __init__(self):
        self._buffer: Optional[MSCBuffer] = None
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
        audio_1 = inputs.get("audio_1")
        audio_2 = inputs.get("audio_2")

        # UIテーマ設定に従ってモードを決定
        theme = get_setting("ui.theme", "dark")
        is_light_mode = (theme == "light")

        # 両方の音声入力がない場合は既存のMSCを表示
        if audio_1 is None or audio_2 is None:
            if self._buffer is not None:
                image = self._buffer.get_image(self._image_width, self._image_height, is_light_mode)
                return {"image": image}
            return {"image": None}

        # パラメータ取得
        fft_size = int(properties.get("fft_size", 1024))
        window_func_idx = int(properties.get("window_function", 0))
        smoothing = int(properties.get("smoothing", 5))

        window_type = WINDOW_FUNCTIONS.get(window_func_idx, "hamming")

        # サンプルレート取得（両方同じと仮定）
        sample_rate = audio_1.get("sample_rate", 16000)

        # サンプルレート変更時はバッファを再初期化
        if self._last_sample_rate != sample_rate:
            self._buffer = None
            self._last_sample_rate = sample_rate

        # バッファを初期化（初回のみ）
        if self._buffer is None:
            self._buffer = MSCBuffer(
                sample_rate=sample_rate,
                fft_size=fft_size,
            )
        else:
            # パラメータ変更時はバッファを更新
            self._buffer.set_params(fft_size)

        # 音声データを処理
        delta_1 = audio_1.get("delta", [])
        delta_2 = audio_2.get("delta", [])

        if len(delta_1) > 0 and len(delta_2) > 0:
            audio_array_1 = np.array(delta_1, dtype=np.float32)
            audio_array_2 = np.array(delta_2, dtype=np.float32)
            self._buffer.process(audio_array_1, audio_array_2, window_type, smoothing)

        # MSC画像を生成
        image = self._buffer.get_image(self._image_width, self._image_height, is_light_mode)

        return {"image": image}
