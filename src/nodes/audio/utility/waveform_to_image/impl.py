"""
Waveform to Image ノードの実装。
オーディオ波形を画像に変換する。
"""
from typing import Any, Dict, List

import numpy as np
import threading

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting


# 画像サイズ（UI表示は200x80、内部は2倍解像度）
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 160


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

    def get_waveform_minmax(self, display_width: int) -> List[tuple]:
        """表示用のmin/maxペアを生成"""
        with self.lock:
            if self.valid_samples == 0:
                return [(0.0, 0.0)] * display_width

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

            # 空のピクセル
            for _ in range(empty_pixels):
                result.append((0.0, 0.0))

            # データがあるピクセル
            for i in range(data_pixels):
                start = i * samples_per_pixel
                end = start + samples_per_pixel
                segment = waveform[start:end]
                if len(segment) > 0:
                    result.append((float(np.min(segment)), float(np.max(segment))))
                else:
                    result.append((0.0, 0.0))

            return result

    def get_waveform_display(self, display_width: int = 200) -> List[float]:
        """表示用のmin/maxペアを生成（フラットリスト形式）"""
        minmax = self.get_waveform_minmax(display_width)
        result = []
        for min_val, max_val in minmax:
            result.extend([min_val, max_val])
        return result


class WaveformToImageLogic(ComputeLogic):
    """
    Waveform to Imageノードのロジック。
    オーディオ波形を画像に変換する。
    """

    def __init__(self):
        self._waveform_buffer: WaveformBuffer | None = None
        self._last_sample_rate: int = -1

    def reset(self):
        """バッファをリセット"""
        self._waveform_buffer = None

    def _render_waveform_image(self, minmax_data: List[tuple], is_light_mode: bool) -> np.ndarray:
        """波形を画像にレンダリング"""
        # 色設定（BGR形式）
        if is_light_mode:
            bg_color = [221, 221, 221]       # #ddd
            line_color = [187, 187, 187]     # #bbb
            waveform_color = [0, 170, 0]     # #00aa00
        else:
            bg_color = [0, 0, 0]             # #000
            line_color = [51, 51, 51]        # #333
            waveform_color = [0, 255, 0]     # #00ff00

        # 背景色で画像を作成
        image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), bg_color, dtype=np.uint8)

        center_y = IMAGE_HEIGHT // 2

        # 中央線
        image[center_y, :] = line_color

        # 波形を描画
        for x, (min_val, max_val) in enumerate(minmax_data):
            if x >= IMAGE_WIDTH:
                break

            # -1 to 1 の範囲を height にマッピング
            y_min = int(center_y - min_val * center_y * 0.9)
            y_max = int(center_y - max_val * center_y * 0.9)

            # 範囲をクランプ
            y_min = max(0, min(IMAGE_HEIGHT - 1, y_min))
            y_max = max(0, min(IMAGE_HEIGHT - 1, y_max))

            # 縦線を描画（y_minが上、y_maxが下になるように）
            if y_min > y_max:
                y_min, y_max = y_max, y_min

            image[y_min:y_max + 1, x] = waveform_color

        return image

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        audio_in = inputs.get("audio")

        if audio_in is None:
            return {"image": None}

        # サンプルレートとdurationを取得
        sample_rate = audio_in.get("sample_rate", 16000)
        duration = audio_in.get("duration", 5.0)

        # サンプルレート変更時はバッファを再初期化
        if self._last_sample_rate != sample_rate:
            self._waveform_buffer = None
            self._last_sample_rate = sample_rate

        # バッファを初期化（初回のみ）
        if self._waveform_buffer is None:
            self._waveform_buffer = WaveformBuffer(sample_rate, duration)

        # deltaを取得してバッファに追加
        delta = audio_in.get("delta", [])
        if delta:
            self._waveform_buffer.write(delta)

        # UIテーマ設定に従ってモードを決定
        theme = get_setting("ui.theme", "dark")
        is_light_mode = (theme == "light")

        # 波形データを取得
        minmax_data = self._waveform_buffer.get_waveform_minmax(IMAGE_WIDTH)

        # 画像にレンダリング
        image = self._render_waveform_image(minmax_data, is_light_mode)

        return {"image": image}
