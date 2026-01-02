"""
Execute Python ノードの実装。
テキストエリアに入力されたPythonコードを実行し、画像を処理する。
AIコード生成機能付き。
"""
import re
import threading
from typing import Dict, Any

import numpy as np
import cv2

from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting


# コード生成用システムプロンプト
CODEGEN_SYSTEM_PROMPT = """You are a Python code generator for image processing.
Output only valid Python code without any explanations or markdown formatting.

Available variables:
- input_image: numpy.ndarray (BGR format, from OpenCV)
- output_image: numpy.ndarray (must be set by your code)

Available libraries (already imported):
- cv2 (OpenCV)
- np (NumPy)

Requirements:
- Your code must set output_image to the processed result
- Use OpenCV (cv2) and NumPy (np) for image processing
- Do not include import statements
- Do not include markdown code blocks (no ```python or ```)
- Output raw Python code only"""


def overlay_error_text(image: np.ndarray, error_message: str) -> np.ndarray:
    """画像にエラーメッセージを赤文字で重畳表示"""
    if image is None:
        # 画像がない場合は黒背景を作成
        result = np.zeros((200, 400, 3), dtype=np.uint8)
    else:
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    h, w = result.shape[:2]

    # 半透明の黒背景を追加
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.5, result, 0.5, 0)

    # エラーメッセージを複数行に分割
    lines = error_message.split('\n')
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    color = (0, 0, 255)  # BGR: 赤

    # 各行を描画
    y_offset = 40
    line_height = 36
    for line in lines:
        # 長い行は折り返す（フォントサイズに応じて調整）
        max_chars = max(10, w // 16)
        while len(line) > max_chars:
            cv2.putText(result, line[:max_chars], (5, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
            line = line[max_chars:]
            if y_offset > h - 10:
                break
        if y_offset > h - 10:
            break
        cv2.putText(result, line, (5, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        if y_offset > h - 10:
            break

    return result


def clean_generated_code(code: str) -> str:
    """生成されたコードからマークダウン記法を除去"""
    # マークダウンコードブロックを除去
    code = re.sub(r'^```python\s*\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'^```\s*\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n?```$', '', code, flags=re.MULTILINE)
    return code.strip()


class ExecutePythonLogic(ComputeLogic):
    """
    Execute Pythonノードのロジック。
    テキストエリアのPythonコードを実行し、input_imageを処理してoutput_imageを返す。
    AIコード生成機能付き。
    """

    def __init__(self):
        self._last_error: str = ""
        self._last_button_value: bool = False
        self._is_generating: bool = False
        self._generated_code: str | None = None
        self._generation_error: str | None = None
        self._lock = threading.Lock()

    def reset(self):
        """ノードの状態をリセット"""
        self._last_error = ""
        self._last_button_value = False
        self._is_generating = False
        self._generated_code = None
        self._generation_error = None

    def _print_error_once(self, error_msg: str):
        """エラーメッセージが変わった場合のみprint"""
        if error_msg != self._last_error:
            self._last_error = error_msg
            print(f"[Execute Python] {error_msg}")

    def _generate_code_async(self, api_key: str, model: str, user_prompt: str):
        """OpenAI APIでコードを生成（別スレッド）"""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CODEGEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
            )

            generated = response.choices[0].message.content
            cleaned_code = clean_generated_code(generated)

            with self._lock:
                self._generated_code = cleaned_code
                self._generation_error = None

        except Exception as e:
            with self._lock:
                self._generation_error = str(e)
                self._generated_code = None

        finally:
            with self._lock:
                self._is_generating = False

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        input_image = inputs.get("image")
        code = properties.get("code", "output_image = input_image")

        result: Dict[str, Any] = {}

        # コード生成ボタンの検出
        generate_pressed = properties.get("generate_code", False)
        button_triggered = generate_pressed and not self._last_button_value
        self._last_button_value = generate_pressed

        # コード生成を開始
        if button_triggered and not self._is_generating:
            api_key = get_setting("api_keys.openai", "")
            if not api_key:
                result["__error__"] = "OpenAI API key not configured"
            else:
                user_prompt = properties.get("codegen_prompt", "")
                if not user_prompt.strip():
                    result["__error__"] = "Please enter a prompt for code generation"
                else:
                    model = properties.get("codegen_model", "gpt-4.1-mini")
                    with self._lock:
                        self._is_generating = True
                        self._generated_code = None
                        self._generation_error = None

                    thread = threading.Thread(
                        target=self._generate_code_async,
                        args=(api_key, model, user_prompt),
                        daemon=True
                    )
                    thread.start()

        # 生成中フラグ
        if self._is_generating:
            result["__is_busy__"] = True

        # 生成完了時にcodeプロパティを更新
        with self._lock:
            if self._generated_code is not None:
                result["__update_property__"] = {
                    "code": self._generated_code
                }
                code = self._generated_code
                self._generated_code = None

            if self._generation_error is not None:
                result["__error__"] = f"Code generation failed: {self._generation_error}"
                self._generation_error = None

        # コードが空の場合は入力をそのまま返す
        if not code or not code.strip():
            result["image"] = input_image
            return result

        # 実行用の名前空間を準備
        namespace = {
            "input_image": input_image,
            "output_image": None,
            "np": np,
            "cv2": cv2,
        }

        try:
            # コードをコンパイル（構文チェック）
            compiled_code = compile(code, "<user_code>", "exec")

            # コードを実行
            exec(compiled_code, namespace)

            # 結果を取得
            output_image = namespace.get("output_image")

            # output_imageが設定されていない場合
            if output_image is None:
                error_msg = "Error: output_image is not defined"
                self._print_error_once(error_msg)
                error_img = overlay_error_text(input_image, error_msg)
                result["image"] = error_img
                return result

            # 成功時はエラー状態をクリア
            self._last_error = ""
            result["image"] = output_image
            return result

        except SyntaxError as e:
            # 構文エラー
            error_msg = f"SyntaxError at line {e.lineno}:\n{e.msg}"
            if e.text:
                error_msg += f"\n{e.text.strip()}"
            self._print_error_once(error_msg)
            error_img = overlay_error_text(input_image, error_msg)
            result["image"] = error_img
            return result

        except Exception as e:
            # 実行時エラー
            import traceback
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            # ユーザーコード部分のみ抽出
            error_lines = []
            for line in tb_lines:
                if "<user_code>" in line or not line.startswith("  File"):
                    error_lines.append(line.strip())
            error_msg = "\n".join(error_lines[-5:])  # 最後の5行のみ
            if not error_msg:
                error_msg = f"{type(e).__name__}: {str(e)}"
            self._print_error_once(error_msg)
            error_img = overlay_error_text(input_image, error_msg)
            result["image"] = error_img
            return result
