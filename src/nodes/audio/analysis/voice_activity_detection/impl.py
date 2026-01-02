"""
Voice Activity Detection (VAD) ノードの実装。
音声区間を検出する。
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False


# 画像サイズ（UI表示は200x80、内部は2倍解像度）
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 160


class SileroVadWrapper:
    """Silero VAD ONNXモデルのラッパー"""

    def __init__(self, model_path: str, sample_rate: int = 16000):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts
        )
        self.sample_rate = sample_rate
        self.reset_states()

    def reset_states(self, batch_size: int = 1):
        """状態をリセット"""
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = None

    def __call__(self, x: np.ndarray) -> float:
        """音声チャンクに対してVAD推論を実行"""
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        num_samples = 512 if self.sample_rate == 16000 else 256
        context_size = 64 if self.sample_rate == 16000 else 32

        # 入力サイズを調整
        if x.shape[-1] != num_samples:
            # パディングまたはトリミング
            if x.shape[-1] < num_samples:
                x = np.pad(x, ((0, 0), (0, num_samples - x.shape[-1])), mode='constant')
            else:
                x = x[:, :num_samples]

        batch_size = x.shape[0]

        if self._context is None:
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x_with_context = np.concatenate([self._context, x], axis=1)

        ort_inputs = {
            "input": x_with_context.astype(np.float32),
            "state": self._state.astype(np.float32),
            "sr": np.array(self.sample_rate, dtype=np.int64),
        }

        out, new_state = self.session.run(None, ort_inputs)
        self._state = new_state
        self._context = x_with_context[:, -context_size:]

        return float(out[0])


class WebRTCVadWrapper:
    """WebRTC VADのラッパー"""

    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 1):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(aggressiveness)
        self.sample_rate = sample_rate
        # WebRTC VADは10, 20, 30msのフレームのみサポート
        self.frame_ms = 30
        self.frame_size = int(sample_rate * self.frame_ms / 1000)

    def __call__(self, x: np.ndarray) -> float:
        """音声チャンクに対してVAD推論を実行"""
        if len(x) < self.frame_size:
            x = np.pad(x, (0, self.frame_size - len(x)), mode='constant')

        # float32 -> int16に変換
        frame_int16 = (x[:self.frame_size] * 32767).astype(np.int16)

        try:
            is_speech = self.vad.is_speech(frame_int16.tobytes(), self.sample_rate)
            return 1.0 if is_speech else 0.0
        except Exception:
            return 0.0

    def reset_states(self):
        """状態をリセット（WebRTC VADは状態を持たない）"""
        pass


class VADDisplayBuffer:
    """VAD結果表示用のバッファ"""

    def __init__(self, sample_rate: int, frame_size: int, duration: float = 5.0):
        self.duration = duration
        self.sample_rate = sample_rate
        self.frame_size = frame_size

        # durationの間に発生するVADフレーム数をバッファサイズとする
        self.buffer_size = int(sample_rate * duration / frame_size)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)

    def add_result(self, value: float):
        """VAD結果を追加"""
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = value

    def get_image(self, width: int, height: int, is_light_mode: bool = True) -> np.ndarray:
        """VAD結果を画像として取得"""
        # 色設定（BGR形式）
        if is_light_mode:
            bg_color = [221, 221, 221]       # #ddd
            grid_color = [187, 187, 187]     # #bbb
            vad_color = [0, 170, 0]          # #00aa00
        else:
            bg_color = [0, 0, 0]             # #000
            grid_color = [51, 51, 51]        # #333
            vad_color = [0, 255, 0]          # #00ff00

        # 背景色で画像を作成
        image = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # 中央線
        center_y = height // 2
        image[center_y, :] = grid_color

        # バッファを画像幅にリサンプリング
        x_indices = np.linspace(0, len(self.buffer) - 1, width).astype(int)
        resampled = self.buffer[x_indices]

        # VAD結果を描画（0は下、1は上）
        for x in range(width):
            value = resampled[x]
            if value > 0.5:
                # 音声検出時: 上半分を塗りつぶし
                image[0:center_y, x] = vad_color
            else:
                # 非音声時: 下半分に小さいマーカー
                image[center_y + 1:height, x] = grid_color

        return image


class VADLogic(ComputeLogic):
    """
    Voice Activity Detectionノードのロジック。
    音声区間を検出する。
    """

    def __init__(self):
        self._vad_model: Any = None
        self._display_buffer: Optional[VADDisplayBuffer] = None
        self._audio_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._last_model_index: int = -1
        self._last_sample_rate: int = -1
        self._last_frame_size: int = -1
        self._last_aggressiveness: int = -1
        self._last_result: int = 0

        # モデルパス
        current_dir = Path(__file__).parent
        self._silero_v5_path = str(current_dir / "model" / "silero_vad_v5.onnx")
        self._silero_v6_path = str(current_dir / "model" / "silero_vad_v6.onnx")

    def reset(self):
        """状態をリセット"""
        self._display_buffer = None
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._last_result = 0
        if self._vad_model is not None and hasattr(self._vad_model, 'reset_states'):
            self._vad_model.reset_states()

    def _create_model(self, model_index: int, sample_rate: int, aggressiveness: int = 0):
        """VADモデルを作成"""
        if model_index == 0:  # WebRTC VAD
            if not WEBRTCVAD_AVAILABLE:
                raise ImportError("webrtcvad is not installed")
            return WebRTCVadWrapper(sample_rate, aggressiveness)
        elif model_index == 1:  # Silero VAD v5
            if not ONNX_AVAILABLE:
                raise ImportError("onnxruntime is not installed")
            return SileroVadWrapper(self._silero_v5_path, sample_rate)
        elif model_index == 2:  # Silero VAD v6
            if not ONNX_AVAILABLE:
                raise ImportError("onnxruntime is not installed")
            return SileroVadWrapper(self._silero_v6_path, sample_rate)
        return None

    def _get_frame_size(self, model_index: int, sample_rate: int) -> int:
        """VADフレームサイズを取得"""
        if model_index == 0:  # WebRTC
            return int(sample_rate * 30 / 1000)  # 30ms
        else:  # Silero
            return 512 if sample_rate == 16000 else 256

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        audio_in = inputs.get("audio")

        # プロパティ取得
        model_index = int(properties.get("model", 2))
        aggressiveness = int(properties.get("aggressiveness", 0))

        # UIテーマ設定に従ってモードを決定
        theme = get_setting("ui.theme", "dark")
        is_light_mode = (theme == "light")

        # 音声入力がない場合
        if audio_in is None:
            if self._display_buffer is None:
                # デフォルト値で初期化
                self._display_buffer = VADDisplayBuffer(16000, 512, duration=5.0)
            image = self._display_buffer.get_image(IMAGE_WIDTH, IMAGE_HEIGHT, is_light_mode)
            return {"image": image, "result": self._last_result}

        sample_rate = audio_in.get("sample_rate", 16000)
        frame_size = self._get_frame_size(model_index, sample_rate)

        # モデルまたはフレームサイズが変更された場合、バッファを再作成
        if (self._display_buffer is None or
            self._last_frame_size != frame_size or
            self._last_sample_rate != sample_rate):
            self._display_buffer = VADDisplayBuffer(sample_rate, frame_size, duration=5.0)
            self._last_frame_size = frame_size

        # モデルが変更された場合、再作成（WebRTC VADの場合はaggressivenessも確認）
        need_recreate = (
            self._vad_model is None or
            self._last_model_index != model_index or
            self._last_sample_rate != sample_rate or
            (model_index == 0 and self._last_aggressiveness != aggressiveness)
        )
        if need_recreate:
            try:
                self._vad_model = self._create_model(model_index, sample_rate, aggressiveness)
                self._last_model_index = model_index
                self._last_sample_rate = sample_rate
                self._last_aggressiveness = aggressiveness
                self._audio_buffer = np.zeros(0, dtype=np.float32)
            except ImportError as e:
                image = self._display_buffer.get_image(IMAGE_WIDTH, IMAGE_HEIGHT, is_light_mode)
                return {"image": image, "result": 0, "__error__": str(e)}

        # 音声データを処理
        delta = audio_in.get("delta", [])
        if len(delta) > 0:
            delta_array = np.array(delta, dtype=np.float32)
            self._audio_buffer = np.concatenate([self._audio_buffer, delta_array])

            while len(self._audio_buffer) >= frame_size:
                frame = self._audio_buffer[:frame_size]
                self._audio_buffer = self._audio_buffer[frame_size:]

                # VAD推論
                prob = self._vad_model(frame)
                is_speech = prob > 0.5
                vad_result = 1 if is_speech else 0

                # 表示バッファに追加
                self._display_buffer.add_result(float(vad_result))

                # result出力を保存
                self._last_result = vad_result

        # 画像を生成
        image = self._display_buffer.get_image(IMAGE_WIDTH, IMAGE_HEIGHT, is_light_mode)

        return {"image": image, "result": self._last_result}
