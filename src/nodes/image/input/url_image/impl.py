"""URL Imageノードの実装"""
import threading
from typing import Any, Dict, Optional

import numpy as np

from node_editor.node_def import ComputeLogic

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import urllib.request
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False


class URLImageLogic(ComputeLogic):
    """
    URLから画像を読み込むノード。
    非同期でダウンロードし、完了まではLoading状態を表示。
    """

    def __init__(self):
        self._current_url: str = ""
        self._cached_image: Optional[np.ndarray] = None
        self._is_loading: bool = False
        self._error_message: Optional[str] = None
        self._load_thread: Optional[threading.Thread] = None

    def reset(self):
        """キャッシュをクリア"""
        self._current_url = ""
        self._cached_image = None
        self._is_loading = False
        self._error_message = None

    def _load_image_from_url(self, url: str):
        """URLから画像を非同期で読み込む"""
        try:
            # URLから画像データを取得
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                image_data = response.read()

            # バイトデータをnumpy配列に変換
            image_array = np.frombuffer(image_data, dtype=np.uint8)

            # OpenCVで画像をデコード
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                self._error_message = "Failed to decode image"
                self._cached_image = None
            else:
                self._cached_image = image
                self._error_message = None

        except Exception as e:
            self._error_message = str(e)
            self._cached_image = None
        finally:
            self._is_loading = False

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {
                "image": None,
                "__error__": "opencv-python is not installed",
            }

        if not URLLIB_AVAILABLE:
            return {
                "image": None,
                "__error__": "urllib is not available",
            }

        url = properties.get("url", "").strip()

        # URLが空の場合
        if not url:
            self._current_url = ""
            self._cached_image = None
            self._is_loading = False
            return {"image": None}

        # URLが変更された場合、非同期で読み込み開始
        if url != self._current_url:
            self._current_url = url
            self._cached_image = None
            self._is_loading = True
            self._error_message = None

            # 既存のスレッドがあれば完了を待たずに新しいスレッドを開始
            self._load_thread = threading.Thread(
                target=self._load_image_from_url,
                args=(url,),
                daemon=True
            )
            self._load_thread.start()

        # 読み込み中の場合
        if self._is_loading:
            return {
                "image": None,
                "__is_busy__": True,
                "__display_text__": "Loading...",
            }

        # エラーがある場合
        if self._error_message:
            return {
                "image": None,
                "__error__": self._error_message,
            }

        return {"image": self._cached_image}
