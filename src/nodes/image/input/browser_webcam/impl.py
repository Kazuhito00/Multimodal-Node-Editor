"""
ブラウザのgetUserMedia() APIを使用してカメラ画像を取得するノード。
フロントエンドでキャプチャした画像をBase64形式で受け取り、出力する。
"""
import base64
from typing import Any, Dict

import cv2
import numpy as np

from node_editor.node_def import ComputeLogic


class BrowserWebcamNodeLogic(ComputeLogic):
    """
    ブラウザからのカメラ画像を受け取るノードロジック。
    フロントエンドがgetUserMedia()でキャプチャした画像を
    browser_frameプロパティ経由で受け取る。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        browser_frame = properties.get("browser_frame", "")

        if not browser_frame:
            return {"image_out": None}

        try:
            # Base64データからヘッダーを除去（data:image/jpeg;base64, など）
            if "," in browser_frame:
                browser_frame = browser_frame.split(",", 1)[1]

            # Base64デコードしてnumpy配列に変換
            image_bytes = base64.b64decode(browser_frame)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if frame is None:
                return {"image_out": None}

            return {"image_out": frame}

        except Exception as e:
            print(f"BrowserWebcam decode error: {e}")
            return {"image_out": None}
