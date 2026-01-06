"""
Audio Classification ノードの実装。
MediaPipe YamNet を使用して音声を分類する。
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from node_editor.node_def import ComputeLogic
import numpy as np
import threading

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python.components import containers
    from mediapipe.tasks.python import audio
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


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


class AudioClassificationBuffer:
    """分類用のオーディオバッファ（YamNetは約0.975秒のウィンドウを使用）"""

    def __init__(self, sample_rate: int, window_seconds: float = 1.0):
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_seconds)
        self.buffer = np.zeros(0, dtype=np.float32)

    def add_samples(self, samples: np.ndarray):
        """サンプルをバッファに追加"""
        self.buffer = np.concatenate([self.buffer, samples])

    def get_window(self) -> Optional[np.ndarray]:
        """分類用のウィンドウを取得（十分なデータがあれば）"""
        if len(self.buffer) >= self.window_size:
            window = self.buffer[:self.window_size]
            # バッファから処理済み分を削除（50%オーバーラップ）
            self.buffer = self.buffer[self.window_size // 2:]
            return window
        return None


class AudioClassificationLogic(ComputeLogic):
    """
    Audio Classificationノードのロジック。
    MediaPipe YamNet を使用して音声を分類する。
    """

    def __init__(self):
        self._waveform_buffer: WaveformBuffer | None = None
        self._classification_buffer: AudioClassificationBuffer | None = None
        self._classifier: Any = None
        self._last_model_index: int = -1
        self._last_sample_rate: int = -1
        self._last_result: Dict[str, Any] = {}
        self._max_results: int = 5

        # モデルパスを設定
        current_dir = Path(__file__).parent
        self._yamnet_model_path = str(current_dir / "yamnet" / "model" / "yamnet_float32.tflite")

    def reset(self):
        """バッファをリセット"""
        self._waveform_buffer = None
        self._classification_buffer = None
        self._last_result = {}

    def _create_classifier(self, model_index: int):
        """分類器を作成"""
        if model_index == 0:  # MediaPipe YamNet
            if not Path(self._yamnet_model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self._yamnet_model_path}")

            base_options = python.BaseOptions(model_asset_path=self._yamnet_model_path)
            options = audio.AudioClassifierOptions(
                base_options=base_options,
                max_results=self._max_results,
            )
            return audio.AudioClassifier.create_from_options(options)
        return None

    def _classify_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """オーディオを分類"""
        if self._classifier is None:
            return {}

        # MediaPipe用にAudioDataを作成
        audio_clip = containers.AudioData.create_from_array(
            audio_data.astype(float),
            sample_rate
        )

        # 分類実行
        results = self._classifier.classify(audio_clip)

        if not results:
            return {}

        # 最初の結果を使用
        result = results[0]
        if not result.classifications:
            return {}

        # カテゴリ情報を抽出
        categories = []
        for category in result.classifications[0].categories:
            categories.append({
                "name": category.category_name,
                "score": round(float(category.score), 4),
            })

        return {
            "categories": categories,
            "top_category": categories[0]["name"] if categories else "",
            "top_score": categories[0]["score"] if categories else 0.0,
        }

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not MEDIAPIPE_AVAILABLE:
            return {
                "audio": None,
                "result_json": "",
                "__error__": "mediapipe is not installed"
            }

        audio_in = inputs.get("audio")

        if audio_in is None:
            return {"audio": None, "result_json": ""}

        # パラメータ取得
        model_index = int(properties.get("model", 0))
        sample_rate = audio_in.get("sample_rate", 16000)
        duration = audio_in.get("duration", 5.0)

        # ウェーブフォームバッファを初期化（初回のみ）
        if self._waveform_buffer is None:
            self._waveform_buffer = WaveformBuffer(sample_rate, duration)

        # 分類バッファを初期化（初回のみ）
        if self._classification_buffer is None:
            self._classification_buffer = AudioClassificationBuffer(sample_rate, 1.0)

        # モデルが変更された場合、分類器を再作成
        if self._classifier is None or self._last_model_index != model_index:
            try:
                self._classifier = self._create_classifier(model_index)
                self._last_model_index = model_index
            except FileNotFoundError as e:
                return {
                    "audio": None,
                    "result_json": "",
                    "__error__": str(e)
                }

        # サンプルレートが変更された場合、バッファをリセット
        if self._last_sample_rate != sample_rate:
            self._classification_buffer = AudioClassificationBuffer(sample_rate, 1.0)
            self._last_sample_rate = sample_rate

        # deltaを取得
        delta = audio_in.get("delta", [])
        delta_list = delta if isinstance(delta, list) else []

        if len(delta_list) > 0:
            delta_array = np.array(delta_list, dtype=np.float32)

            # ウェーブフォームバッファに蓄積
            self._waveform_buffer.write(delta_list)

            # 分類バッファに蓄積
            self._classification_buffer.add_samples(delta_array)

            # 十分なデータがあれば分類を実行
            window = self._classification_buffer.get_window()
            if window is not None:
                result = self._classify_audio(window, sample_rate)
                if result:
                    self._last_result = result

        # バッファから表示用waveformを生成
        waveform_display = self._waveform_buffer.get_waveform_display(200)

        # 結果をJSON文字列に変換
        result_json = json.dumps(self._last_result, ensure_ascii=False) if self._last_result else ""

        # Top1結果を抽出
        top_label = ""
        if self._last_result:
            top_category = self._last_result.get("top_category", "")
            top_score = self._last_result.get("top_score", 0.0)
            if top_category:
                top_label = f"{top_category} ({top_score:.0%})"

        return {
            "audio": {
                "delta": delta_list,
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": duration,
                "label": top_label,
            },
            "result_json": result_json,
        }
