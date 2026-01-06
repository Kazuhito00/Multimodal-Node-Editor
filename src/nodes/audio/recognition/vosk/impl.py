"""
Vosk Speech-to-Text ノードの実装。
Voskを使用してオフライン音声認識を行う。
"""
import json
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from node_editor.node_def import ComputeLogic


try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


# モデル設定（インデックス: モデルディレクトリ名）
MODEL_DIRS = {
    0: "vosk-model-small-ja-0.22",
    1: "vosk-model-ja-0.22",
    2: "vosk-model-small-en-us-0.15",
    3: "vosk-model-en-us-0.22",
}


class VoskSession:
    """Voskストリーミングセッション"""

    def __init__(self, model_path: str, sample_rate: int = 16000):
        self.model_path = model_path
        self.sample_rate = sample_rate

        # 音声キュー
        self.audio_queue: queue.Queue = queue.Queue()

        # 認識結果
        self.current_transcript = ""

        # 制御フラグ
        self.running = False
        self.stt_thread: Optional[threading.Thread] = None
        self.recognizer = None

    def start(self):
        """STTセッションを開始"""
        if self.running:
            return

        self.running = True
        self.current_transcript = ""

        self.stt_thread = threading.Thread(target=self._run_stt_loop, daemon=True)
        self.stt_thread.start()

    def stop(self):
        """STTセッションを停止"""
        self.running = False
        self.audio_queue.put(None)

        if self.stt_thread is not None:
            self.stt_thread.join(timeout=2.0)
            self.stt_thread = None

        # キューをクリア
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.recognizer = None

    def feed_audio(self, audio_data: np.ndarray):
        """音声データをフィード"""
        if not self.running:
            return

        if len(audio_data) == 0:
            return

        # float32 [-1, 1] → int16 に変換してbytesへ
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        self.audio_queue.put(audio_bytes)

    def get_transcript(self) -> str:
        """現在の認識結果を取得"""
        return self.current_transcript

    def _run_stt_loop(self):
        """STTループを実行"""
        # Voskモデルの初期化
        try:
            vosk.SetLogLevel(-1)
            model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(model, self.sample_rate)
            self.recognizer.SetPartialWords(True)
        except Exception as e:
            print(f"[Vosk] Failed to initialize model: {e}")
            return

        # 認識ループ
        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if data is None:
                break

            # Voskで認識
            if self.recognizer.AcceptWaveform(data):
                # 最終結果
                result = json.loads(self.recognizer.Result())
                if "text" in result and result["text"]:
                    self.current_transcript = result["text"]
            else:
                # 部分結果
                partial_result = json.loads(self.recognizer.PartialResult())
                if "partial" in partial_result and partial_result["partial"]:
                    self.current_transcript = partial_result["partial"]


class VoskSTTLogic(ComputeLogic):
    """
    Vosk Speech-to-Text ノードのロジック。
    Voskを使用してオフライン音声認識を行う。
    """

    def __init__(self):
        self._session: Optional[VoskSession] = None
        self._last_model_index: int = -1
        self._last_sample_rate: int = -1

        # モデルパスを設定
        current_dir = Path(__file__).parent
        self._model_dir = current_dir / "model"

    def reset(self):
        """セッションをリセット"""
        if self._session is not None:
            self._session.stop()
            self._session = None

    def _get_model_path(self, model_index: int) -> tuple:
        """モデルパスを取得。(path, error_message)を返す"""
        model_dirname = MODEL_DIRS.get(model_index)
        if model_dirname is None:
            return None, f"Unknown model index: {model_index}"

        model_path = self._model_dir / model_dirname

        if not model_path.exists():
            return None, f"Model not found: {model_path}"

        return str(model_path), None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not VOSK_AVAILABLE:
            return {
                "text": "",
                "__error__": "vosk is not installed. Run: pip install vosk"
            }

        audio_in = inputs.get("audio")
        if audio_in is None:
            # 音声入力がない場合、セッションが実行中なら結果を返す
            if self._session is not None:
                return {"text": self._session.get_transcript()}
            return {"text": ""}

        # パラメータ取得
        model_index = int(properties.get("model", 0))
        sample_rate = audio_in.get("sample_rate", 16000)

        # 音声データがあるか確認
        delta = audio_in.get("delta", [])
        has_audio = len(delta) > 0

        # パラメータ変更時はセッション再作成
        params_changed = (
            self._last_model_index != model_index or
            self._last_sample_rate != sample_rate
        )

        if params_changed and self._session is not None:
            self._session.stop()
            self._session = None

        # セッションが未起動で音声がある場合、セッションを開始
        if has_audio and self._session is None:
            model_path, model_error = self._get_model_path(model_index)
            if model_error:
                return {"text": "", "__error__": model_error}

            self._session = VoskSession(model_path, sample_rate)
            self._session.start()

        self._last_model_index = model_index
        self._last_sample_rate = sample_rate

        # 音声データを処理
        if self._session is not None and has_audio:
            audio_array = np.array(delta, dtype=np.float32)
            self._session.feed_audio(audio_array)

        # 認識結果を返す
        if self._session is not None:
            return {"text": self._session.get_transcript()}

        return {"text": ""}
