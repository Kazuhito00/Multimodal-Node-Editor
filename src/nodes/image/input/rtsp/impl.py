from typing import Dict, Any, Optional
from node_editor.node_def import ComputeLogic
import numpy as np
import threading
import time

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


class RTSPLogic(ComputeLogic):
    """RTSPストリームからフレームを取得するノードロジック（別スレッドで常時取得）"""

    def __init__(self):
        self._current_url: str = ""
        self._video_capture: Optional[Any] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

    def _capture_loop(self):
        """別スレッドで常にフレームを取得し続ける"""
        while self._running:
            if self._video_capture is None or not self._video_capture.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self._video_capture.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._error = None
            else:
                with self._frame_lock:
                    self._error = "Failed to read frame"
                time.sleep(0.1)

    def _start_capture(self, url: str) -> bool:
        """キャプチャスレッドを開始"""
        self._stop_capture()

        self._video_capture = cv2.VideoCapture(url)
        if not self._video_capture.isOpened():
            # 失敗時もリソースを解放
            self._video_capture.release()
            self._video_capture = None
            self._current_url = ""  # リセットして次回再試行可能に
            return False

        self._current_url = url
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _stop_capture(self):
        """キャプチャスレッドを停止"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
        with self._frame_lock:
            self._latest_frame = None
            self._error = None

    def reset(self):
        """リソースを解放"""
        self._stop_capture()
        self._current_url = ""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {
                "image_out": None,
                "__error__": "opencv-python is not installed",
            }

        # STOP時はVideoCaptureを解放
        is_streaming = context.get("is_streaming", False) if context else False
        if not is_streaming:
            self._stop_capture()
            return {"image_out": None}

        url = properties.get("url", "").strip()

        # URLが空の場合
        if not url:
            self._stop_capture()
            return {"image_out": None}

        # URLが変更された、または接続が切れている場合は再接続
        need_reconnect = (url != self._current_url) or (self._video_capture is None)
        if need_reconnect:
            if not self._start_capture(url):
                return {
                    "image_out": None,
                    "__error__": f"Failed to connect: {url}",
                }

        # 最新フレームを取得
        with self._frame_lock:
            frame = self._latest_frame
            error = self._error

        if error:
            return {
                "image_out": frame,
                "__error__": error,
            }

        return {"image_out": frame}
