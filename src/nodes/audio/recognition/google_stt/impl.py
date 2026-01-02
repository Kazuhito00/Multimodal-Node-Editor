"""
Google Speech-to-Text ノードの実装。
Google Cloud Speech-to-Text APIを使用して音声認識を行う。
"""
import datetime
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting

try:
    from google.cloud import speech
    GOOGLE_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False


# 言語コードマッピング
LANGUAGE_CODES = {
    0: "ja-JP",  # 日本語
    1: "en-US",  # English
}


class GoogleSTTSession:
    """Google STTストリーミングセッション"""

    def __init__(self, sample_rate: int = 16000, language_code: str = "ja-JP"):
        self.sample_rate = sample_rate
        self.language_code = language_code
        self.chunk_size = 1024

        # 認証情報パスを設定から取得
        credentials_path = get_setting("api_keys.google_stt", "")
        if credentials_path and os.path.isfile(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        # 音声キュー
        self.audio_queue: queue.Queue = queue.Queue()

        # 認識結果
        self.current_transcript = ""
        self.final_transcript = ""

        # 制御フラグ
        self.running = False
        self.stt_thread: Optional[threading.Thread] = None
        self.speech_client = None

    def start(self):
        """STTセッションを開始"""
        if self.running:
            return

        self.running = True
        self.current_transcript = ""
        self.final_transcript = ""

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

    def feed_audio(self, audio_data: np.ndarray):
        """音声データをフィード"""
        if not self.running:
            return

        if len(audio_data) == 0:
            return

        # float32 [-1, 1] → int16 に変換
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        self.audio_queue.put(audio_bytes)

    def get_transcript(self) -> str:
        """現在の認識結果を取得"""
        return self.current_transcript

    def _audio_generator(self):
        """音声データジェネレーター"""
        timeout_time = self.chunk_size / self.sample_rate * 2
        silent_chunk = np.zeros(self.chunk_size, dtype=np.int16).tobytes()

        while self.running:
            try:
                data = self.audio_queue.get(timeout=timeout_time)
            except queue.Empty:
                # タイムアウト時は無音を送信
                yield speech.StreamingRecognizeRequest(audio_content=silent_chunk)
                continue

            if data is None:
                break

            yield speech.StreamingRecognizeRequest(audio_content=data)

    def _run_stt_loop(self):
        """STTループを実行（自動再接続付き）"""
        while self.running:
            try:
                self._run_stt_session()
            except Exception as e:
                print(f"[Google STT] Session error: {e}")

            if self.running:
                time.sleep(1.0)  # クールダウン後に再接続

    def _run_stt_session(self):
        """単一のSTTセッションを実行"""
        try:
            self.speech_client = speech.SpeechClient()
        except Exception:
            return

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            use_enhanced=True,
            model="default",
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
        )

        try:
            responses = self.speech_client.streaming_recognize(
                streaming_config, self._audio_generator()
            )

            start_time = datetime.datetime.now()

            for response in responses:
                if not self.running:
                    break

                # 最大ストリーム時間（Googleの仕様: 305秒）を超えたら終了
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                if elapsed > 300:
                    break

                for result in response.results:
                    if result.alternatives:
                        transcript = result.alternatives[0].transcript
                        self.current_transcript = transcript

                        if result.is_final:
                            self.final_transcript = transcript
        except Exception:
            pass


class GoogleSTTLogic(ComputeLogic):
    """
    Google Speech-to-Text ノードのロジック。
    Google Cloud Speech-to-Text APIを使用して音声認識を行う。
    """

    def __init__(self):
        self._session: Optional[GoogleSTTSession] = None
        self._last_language: int = -1
        self._last_sample_rate: int = -1
        self._is_streaming = False

    def reset(self):
        """セッションをリセット"""
        if self._session is not None:
            self._session.stop()
            self._session = None
        self._is_streaming = False

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not GOOGLE_SPEECH_AVAILABLE:
            return {
                "text": None,
                "__error__": "google-cloud-speech is not installed"
            }

        # 認証情報の確認
        credentials_path = get_setting("api_keys.google_stt", "")
        if not credentials_path or not os.path.isfile(credentials_path):
            return {
                "text": None,
                "__error__": "Google credentials file not configured in api_keys.google_stt"
            }

        audio_in = inputs.get("audio")
        if audio_in is None:
            # 音声入力がない場合、セッションが実行中なら結果を返す
            if self._session is not None:
                return {"text": self._session.get_transcript()}
            return {"text": ""}

        # パラメータ取得
        language_index = int(properties.get("language", 0))
        language_code = LANGUAGE_CODES.get(language_index, "ja-JP")
        sample_rate = audio_in.get("sample_rate", 16000)

        # ストリーミング状態の確認（contextから、または音声データの有無で判定）
        is_streaming = context.get("is_streaming", False) if context else False

        # 音声データがある場合はストリーミング中と判断
        delta = audio_in.get("delta", [])
        has_audio = len(delta) > 0

        # セッションが未起動で音声がある場合、セッションを開始
        if has_audio and self._session is None:
            self._start_session(sample_rate, language_code)
            self._is_streaming = True

        # パラメータ変更時はセッション再作成
        params_changed = (
            self._last_language != language_index or
            self._last_sample_rate != sample_rate
        )
        if params_changed and self._session is not None:
            self._start_session(sample_rate, language_code)

        self._last_language = language_index
        self._last_sample_rate = sample_rate

        # 音声データを処理
        if self._session is not None and has_audio:
            audio_array = np.array(delta, dtype=np.float32)
            self._session.feed_audio(audio_array)

        # 認識結果を返す
        if self._session is not None:
            transcript = self._session.get_transcript()
            return {"text": transcript}

        return {"text": ""}

    def _start_session(self, sample_rate: int, language_code: str):
        """新しいセッションを開始"""
        if self._session is not None:
            self._session.stop()

        self._session = GoogleSTTSession(
            sample_rate=sample_rate,
            language_code=language_code
        )
        self._session.start()
