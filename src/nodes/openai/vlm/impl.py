import base64
import json
import threading
from typing import Any, Dict

import cv2
import numpy as np

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting


class VLMLogic(ComputeLogic):
    """OpenAI Vision Language Model APIを呼び出すノード（ストリーミング対応）"""

    def __init__(self):
        self._last_button_value: bool = False
        self._last_trigger_value: float = 0.0
        self._is_executing: bool = False
        self._lock = threading.Lock()
        self._result: str = ""
        self._content: str = ""  # ストリーミング中のコンテンツ
        self._error: str | None = None

    def reset(self):
        """状態をリセット"""
        self._last_button_value = False
        self._last_trigger_value = 0.0
        self._is_executing = False
        self._result = ""
        self._content = ""
        self._error = None

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """画像をBase64エンコードする"""
        # OpenCV形式（BGR）からJPEGにエンコード
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def _call_openai_api_stream(
        self,
        api_key: str,
        model: str,
        user_prompt: str,
        image_base64: str,
        temperature: float
    ):
        """OpenAI Vision APIをストリーミングで呼び出す（別スレッドで実行）"""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)

            # Vision API用のメッセージ形式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt if user_prompt else "What's in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]

            # リクエストパラメータを構築
            request_params = {
                "model": model,
                "messages": messages,
                "stream": True
            }
            # gpt-5系以外はtemperatureを指定
            if not model.startswith("gpt-5"):
                request_params["temperature"] = temperature

            # ストリーミングリクエスト
            stream = client.chat.completions.create(**request_params)

            # ストリーミング中のコンテンツを蓄積
            self._content = ""
            self._error = None

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    self._content += chunk.choices[0].delta.content
                    # ストリーミング中も完了後もcontentのみ出力
                    self._result = self._content

        except Exception as e:
            error_msg = str(e)
            self._result = json.dumps({"error": error_msg}, ensure_ascii=False)
            self._error = error_msg

        finally:
            with self._lock:
                self._is_executing = False

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        # ボタンまたはトリガー入力を検出
        button_pressed = properties.get("execute", False)
        trigger_value = inputs.get("execute")
        if trigger_value is None:
            trigger_value = 0.0

        # ボタンの立ち上がりエッジを検出
        button_triggered = button_pressed and not self._last_button_value
        self._last_button_value = button_pressed

        # トリガー入力の立ち上がりエッジを検出
        trigger_triggered = trigger_value > 0.5 and self._last_trigger_value <= 0.5
        self._last_trigger_value = trigger_value

        should_execute = button_triggered or trigger_triggered

        if should_execute:
            with self._lock:
                if not self._is_executing:
                    self._is_executing = True
                    self._content = ""
                    self._result = ""
                    self._error = None

                    # APIキーを取得
                    api_key = get_setting("api_keys.openai", "")
                    if not api_key:
                        self._error = "OpenAI API key not configured"
                        self._result = json.dumps(
                            {"error": self._error},
                            ensure_ascii=False
                        )
                        self._is_executing = False
                        return {"result_text": self._result, "__error__": self._error}

                    # 画像を取得
                    image = inputs.get("image")
                    if image is None:
                        self._error = "No image input"
                        self._result = json.dumps(
                            {"error": self._error},
                            ensure_ascii=False
                        )
                        self._is_executing = False
                        return {"result_text": self._result, "__error__": self._error}

                    # 画像をBase64エンコード
                    image_base64 = self._encode_image_to_base64(image)

                    # プロパティを取得（入力ポートがあれば上書き）
                    user_prompt = inputs.get("user_prompt") or properties.get("user_prompt", "")
                    model = properties.get("model", "gpt-4.1-mini")
                    temperature = properties.get("temperature", 0.7)

                    # 別スレッドでストリーミングAPI呼び出し
                    thread = threading.Thread(
                        target=self._call_openai_api_stream,
                        args=(api_key, model, user_prompt, image_base64, temperature),
                        daemon=True
                    )
                    thread.start()

        # 結果を返す
        result = {"result_text": self._result}
        if self._error:
            result["__error__"] = self._error
        if self._is_executing:
            result["__is_busy__"] = True
        return result
