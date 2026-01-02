"""
Speech Enhancement ノードの実装。
Deep Learningモデルを使用して音声を強調する。
"""
from pathlib import Path
from typing import Dict, Any, List
from node_editor.node_def import ComputeLogic
import numpy as np
import threading

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class GTCRNModel:
    """GTCRNモデルによる音声強調"""

    def __init__(self, model_path: str, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_size = 512
        self.hop_size = 256

        # モデル読み込み
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # キャッシュ初期化
        self.conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        self.tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        self.inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)

        # 入力バッファ
        self.input_buffer = np.zeros(0, dtype=np.float32)

        # オーバーラップ加算用バッファ
        self.temp_enhanced_buffer = np.zeros(self.frame_size + self.hop_size, dtype=np.float32)
        self.temp_norm_buffer = np.zeros(self.frame_size + self.hop_size, dtype=np.float32)
        self.enhanced_buffer = np.array([], dtype=np.float32)

        # 初期化フラグ
        self.init_flag = False

        # √ハン窓
        self.sqrt_hanning = np.sqrt(np.hanning(self.frame_size)).astype(np.float32)

    def reset(self):
        """状態をリセット"""
        self.conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        self.tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        self.inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.temp_enhanced_buffer = np.zeros(self.frame_size + self.hop_size, dtype=np.float32)
        self.temp_norm_buffer = np.zeros(self.frame_size + self.hop_size, dtype=np.float32)
        self.enhanced_buffer = np.array([], dtype=np.float32)
        self.init_flag = False

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """チャンクを処理して強調音声を返す"""
        if len(chunk) == 0:
            return np.array([], dtype=np.float32)

        # 入力バッファに追加
        self.input_buffer = np.concatenate([self.input_buffer, chunk.astype(np.float32)])

        # フレーム単位で処理
        while len(self.input_buffer) >= self.frame_size:
            temp_frame = self.input_buffer[:self.frame_size]

            # STFT (1 frame)
            frame_windowed = temp_frame * self.sqrt_hanning
            frame_spec = np.fft.rfft(frame_windowed, n=self.frame_size)
            real = np.real(frame_spec).astype(np.float32)
            imag = np.imag(frame_spec).astype(np.float32)
            frame_input = np.stack([real, imag], axis=-1)[None, :, None, :]  # (1, 257, 1, 2)

            # ONNX inference
            out_i, conv_cache, tra_cache, inter_cache = self.session.run(
                None,
                {
                    "mix": frame_input,
                    "conv_cache": self.conv_cache,
                    "tra_cache": self.tra_cache,
                    "inter_cache": self.inter_cache,
                },
            )
            self.conv_cache = conv_cache
            self.tra_cache = tra_cache
            self.inter_cache = inter_cache

            # IRFFT
            out_real = out_i[0][:, 0, 0]
            out_imag = out_i[0][:, 0, 1]
            spec_enh = out_real + 1j * out_imag
            time_frame = np.fft.irfft(spec_enh, n=self.frame_size)[:self.frame_size]
            time_frame = (time_frame * self.sqrt_hanning).astype(np.float32)

            # オーバーラップ加算
            if not self.init_flag:
                self.temp_enhanced_buffer[:self.frame_size] += time_frame
                self.temp_norm_buffer[:self.frame_size] += self.sqrt_hanning ** 2
                self.init_flag = True
            else:
                self.temp_enhanced_buffer[self.hop_size:self.frame_size + self.hop_size] += time_frame
                self.temp_norm_buffer[self.hop_size:self.frame_size + self.hop_size] += self.sqrt_hanning ** 2

                # 正規化
                normalized_frame = np.zeros(self.frame_size, dtype=np.float32)
                denominator = self.temp_norm_buffer[:self.frame_size]
                nonzero = denominator > 1e-8
                normalized_frame[nonzero] = (
                    self.temp_enhanced_buffer[:self.frame_size][nonzero] /
                    denominator[nonzero]
                )

                # 出力バッファに追加
                self.enhanced_buffer = np.concatenate([
                    self.enhanced_buffer,
                    normalized_frame[:self.hop_size]
                ])

                # バッファをシフト
                self.temp_enhanced_buffer[:-self.hop_size] = self.temp_enhanced_buffer[self.hop_size:]
                self.temp_norm_buffer[:-self.hop_size] = self.temp_norm_buffer[self.hop_size:]
                self.temp_enhanced_buffer[-self.hop_size:] = 0.0
                self.temp_norm_buffer[-self.hop_size:] = 0.0

            # 入力バッファから処理済み部分を削除
            self.input_buffer = self.input_buffer[self.hop_size:]

        # 出力チャンクを取り出す
        output_size = len(chunk)
        if len(self.enhanced_buffer) >= output_size:
            output = self.enhanced_buffer[:output_size]
            self.enhanced_buffer = self.enhanced_buffer[output_size:]
            return output
        else:
            # まだ十分なデータがない場合は0埋め
            output = np.zeros(output_size, dtype=np.float32)
            if len(self.enhanced_buffer) > 0:
                output[:len(self.enhanced_buffer)] = self.enhanced_buffer
                self.enhanced_buffer = np.array([], dtype=np.float32)
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


class SpeechEnhancementLogic(ComputeLogic):
    """
    Speech Enhancementノードのロジック。
    Deep Learningモデルを使用して音声を強調する。
    """

    def __init__(self):
        self._buffer: WaveformBuffer | None = None
        self._model: GTCRNModel | None = None
        self._last_model_index: int = -1
        self._last_sample_rate: int = -1

        # モデルパスを設定
        current_dir = Path(__file__).parent
        self._gtcrn_model_path = str(current_dir / "model" / "gtcrn_simple.onnx")

    def reset(self):
        """バッファとモデルをリセット"""
        self._buffer = None
        if self._model is not None:
            self._model.reset()

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not ONNX_AVAILABLE:
            return {
                "audio": None,
                "__error__": "onnxruntime is not installed"
            }

        audio_in = inputs.get("audio")

        if audio_in is None:
            return {"audio": None}

        # パラメータ取得（モデルは整数インデックス: 0=GTCRN）
        model_index = int(properties.get("model", 0))

        sample_rate = audio_in.get("sample_rate", 16000)
        duration = audio_in.get("duration", 5.0)

        # バッファを初期化（初回のみ）
        if self._buffer is None:
            self._buffer = WaveformBuffer(sample_rate, duration)

        # モデルが変更された場合、モデルを再作成
        if self._model is None or self._last_model_index != model_index or self._last_sample_rate != sample_rate:
            if model_index == 0:  # GTCRN
                if not Path(self._gtcrn_model_path).exists():
                    return {
                        "audio": None,
                        "__error__": f"Model file not found: {self._gtcrn_model_path}"
                    }
                self._model = GTCRNModel(self._gtcrn_model_path, sample_rate)

            self._last_model_index = model_index
            self._last_sample_rate = sample_rate

        # deltaを取得して処理
        delta = audio_in.get("delta", [])
        if len(delta) > 0:
            delta_array = np.array(delta, dtype=np.float32)
            enhanced = self._model.process(delta_array)
            enhanced_delta = enhanced.tolist()
        else:
            enhanced_delta = []

        # 処理済みdeltaをバッファに蓄積
        self._buffer.write(enhanced_delta)

        # バッファから表示用waveformを生成
        waveform_display = self._buffer.get_waveform_display(200)

        return {
            "audio": {
                "delta": enhanced_delta,
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": duration,
            }
        }
