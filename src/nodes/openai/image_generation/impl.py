import base64
import threading
from typing import Any, Dict

import cv2
import numpy as np

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting


class ImageGenerationLogic(ComputeLogic):
    """OpenAI Image Generation APIを呼び出すノード"""

    def __init__(self):
        self._last_button_value: bool = False
        self._last_trigger_value: float = 0.0
        self._is_executing: bool = False
        self._lock = threading.Lock()
        self._image: np.ndarray | None = None
        self._error: str | None = None

    def reset(self):
        """状態をリセット"""
        self._last_button_value = False
        self._last_trigger_value = 0.0
        self._is_executing = False
        self._image = None
        self._error = None

    def _call_openai_api(
        self,
        api_key: str,
        model: str,
        prompt: str
    ):
        """OpenAI Image Generation APIを呼び出す（別スレッドで実行）"""
        try:
            import openai
            import urllib.request
            client = openai.OpenAI(api_key=api_key)

            # 画像生成リクエスト（gpt-image-1はresponse_format不要）
            response = client.images.generate(
                model=model,
                prompt=prompt,
                n=1
            )

            # レスポンスから画像を取得
            image_data = response.data[0]

            # b64_jsonがある場合はBase64デコード、なければURLからダウンロード
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                image_bytes = base64.b64decode(image_data.b64_json)
            elif hasattr(image_data, 'url') and image_data.url:
                with urllib.request.urlopen(image_data.url) as resp:
                    image_bytes = resp.read()
            else:
                raise ValueError("No image data in response")

            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

            # RGBA画像の場合はBGRに変換
            if decoded is not None and len(decoded.shape) == 3 and decoded.shape[2] == 4:
                self._image = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
            else:
                self._image = decoded

            self._error = None

        except Exception as e:
            error_msg = str(e)
            self._error = error_msg
            self._image = None

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
        trigger_value = inputs.get("execute", 0.0)

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
                    self._error = None

                    # APIキーを取得
                    api_key = get_setting("api_keys.openai", "")
                    if not api_key:
                        self._error = "OpenAI API key not configured"
                        self._is_executing = False
                        return {"image": self._image, "__error__": self._error}

                    # プロパティを取得（入力ポートがあれば上書き）
                    prompt = inputs.get("prompt") or properties.get("prompt", "")
                    if not prompt:
                        self._error = "Prompt is required"
                        self._is_executing = False
                        return {"image": self._image, "__error__": self._error}

                    model = properties.get("model", "dall-e-3")

                    # 別スレッドでAPI呼び出し
                    thread = threading.Thread(
                        target=self._call_openai_api,
                        args=(api_key, model, prompt),
                        daemon=True
                    )
                    thread.start()

        # 結果を返す
        result = {"image": self._image}
        if self._error:
            result["__error__"] = self._error
        if self._is_executing:
            result["__is_busy__"] = True
        return result
