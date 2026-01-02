import base64
import json
import threading
from typing import Any, Dict

import numpy as np

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting


class RealtimeSTTLogic(ComputeLogic):
    """OpenAI Realtime APIを使用した音声認識ノード"""

    def __init__(self):
        self._ws = None
        self._ws_thread = None
        self._is_connected = False
        self._is_connecting = False
        self._lock = threading.Lock()
        self._result_text = ""
        self._partial_text = ""
        self._error: str | None = None
        self._was_streaming = False

    def reset(self):
        """状態をリセット"""
        self._disconnect()
        self._result_text = ""
        self._partial_text = ""
        self._error = None
        self._was_streaming = False

    def _connect(self, api_key: str, model: str):
        """WebSocket接続を開始（別スレッド）"""
        try:
            import websocket

            url = f"wss://api.openai.com/v1/realtime?model={model}"
            headers = [
                f"Authorization: Bearer {api_key}",
                "OpenAI-Beta: realtime=v1",
            ]

            self._ws = websocket.WebSocketApp(
                url,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            self._ws.run_forever()

        except Exception as e:
            with self._lock:
                self._error = str(e)
                self._is_connected = False
                self._is_connecting = False

    def _on_open(self, ws):
        """WebSocket接続時のコールバック"""
        with self._lock:
            self._is_connected = True
            self._is_connecting = False
            self._error = None

        # セッション設定を送信（音声入力モード）
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                }
            }
        }
        ws.send(json.dumps(session_config))

    def _on_message(self, ws, message):
        """WebSocketメッセージ受信時のコールバック"""
        try:
            data = json.loads(message)
            event_type = data.get("type", "")

            # 音声入力の文字起こし結果（確定）
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = data.get("transcript", "")
                if transcript:
                    with self._lock:
                        if self._result_text:
                            self._result_text += " " + transcript
                        else:
                            self._result_text = transcript
                        # 確定したのでpartialをクリア
                        self._partial_text = ""

            # 部分的な文字起こし
            elif event_type == "conversation.item.input_audio_transcription.delta":
                delta = data.get("delta", "")
                if delta:
                    with self._lock:
                        self._partial_text += delta

            # エラー
            elif event_type == "error":
                error_info = data.get("error", {})
                error_msg = error_info.get("message", "Unknown error")
                with self._lock:
                    self._error = error_msg

        except json.JSONDecodeError:
            pass

    def _on_error(self, ws, error):
        """WebSocketエラー時のコールバック"""
        with self._lock:
            self._error = str(error)

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket切断時のコールバック"""
        with self._lock:
            self._is_connected = False
            self._is_connecting = False

    def _disconnect(self):
        """WebSocket接続を切断"""
        with self._lock:
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None
            self._is_connected = False
            self._is_connecting = False

    def _send_audio(self, audio_data: np.ndarray):
        """音声データを送信"""
        if not self._is_connected or self._ws is None:
            return

        try:
            # float32 [-1, 1] を int16 に変換
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            # 音声データを送信
            event = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }
            self._ws.send(json.dumps(event))

        except Exception as e:
            with self._lock:
                self._error = str(e)

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if context is None:
            context = {}

        is_streaming = not context.get("preview", False)
        model = properties.get("model", "gpt-4o-realtime-preview")

        # ストリーミング開始時に接続
        if is_streaming and not self._was_streaming:
            api_key = get_setting("api_keys.openai", "")
            if api_key:
                with self._lock:
                    if not self._is_connected and not self._is_connecting:
                        self._is_connecting = True
                        self._result_text = ""
                        self._partial_text = ""
                        self._error = None
                        self._ws_thread = threading.Thread(
                            target=self._connect,
                            args=(api_key, model),
                            daemon=True
                        )
                        self._ws_thread.start()
            else:
                self._error = "OpenAI API key not configured"

        # ストリーミング終了時に切断
        if not is_streaming and self._was_streaming:
            self._disconnect()

        self._was_streaming = is_streaming

        # 音声データを送信
        audio_in = inputs.get("audio")
        if is_streaming and self._is_connected and audio_in:
            delta = audio_in.get("delta", [])
            if delta:
                audio_array = np.array(delta, dtype=np.float32)
                self._send_audio(audio_array)

        # 結果を返す
        with self._lock:
            display_text = self._result_text
            if self._partial_text:
                display_text += " " + self._partial_text

        result = {"result_text": display_text}
        if self._error:
            result["__error__"] = self._error

        return result
