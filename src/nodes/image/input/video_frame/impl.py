"""
Video Frame ノードの実装。
動画ファイルから指定フレームを出力する。
"""
from typing import Dict, Any, Optional
from node_editor.node_def import ComputeLogic
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


class VideoFrameLogic(ComputeLogic):
    """動画から指定フレームを出力するノードロジック"""

    def __init__(self):
        self.current_file: str = ""
        self.video_capture: Optional[Any] = None
        self.total_frames: int = 1
        self.cached_frame: Optional[np.ndarray] = None
        self.cached_position: int = -1

    def reset(self):
        """リソースを解放"""
        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except Exception:
                pass
            self.video_capture = None
        self.current_file = ""
        self.total_frames = 1
        self.cached_frame = None
        self.cached_position = -1

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

        # キャンセルチェック
        if self.is_cancelled():
            self.clear_cancel()
            return {
                "image_out": self.cached_frame,
                "__frame_count__": self.total_frames,
            }

        file_path = properties.get("file_path", "")
        frame_position = int(properties.get("frame_position", 1))

        # ファイルパスが空の場合
        if not file_path:
            return {
                "image_out": None,
                "__frame_count__": 1,
            }

        # ファイルが変更された場合は再読み込み
        if file_path != self.current_file:
            self.reset()

            self.video_capture = cv2.VideoCapture(file_path)
            if not self.video_capture.isOpened():
                self.video_capture = None
                return {
                    "image_out": None,
                    "__error__": f"Failed to open: {file_path}",
                    "__frame_count__": 1,
                }

            self.current_file = file_path
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.total_frames < 1:
                self.total_frames = 1

        if self.video_capture is None:
            return {
                "image_out": None,
                "__frame_count__": 1,
            }

        # フレーム位置をクランプ
        frame_position = max(1, min(self.total_frames, frame_position))

        # キャッシュされたフレームと同じ位置なら再読み込み不要
        if frame_position == self.cached_position and self.cached_frame is not None:
            return {
                "image_out": self.cached_frame,
                "__frame_count__": self.total_frames,
            }

        try:
            # フレームをシーク（0始まりなので-1する）
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position - 1)
            ret, frame = self.video_capture.read()
        except Exception as e:
            return {
                "image_out": self.cached_frame,
                "__error__": f"Video read error: {e}",
                "__frame_count__": self.total_frames,
            }

        if not ret or frame is None:
            return {
                "image_out": None,
                "__error__": f"Failed to read frame {frame_position}",
                "__frame_count__": self.total_frames,
            }

        self.cached_frame = frame
        self.cached_position = frame_position

        return {
            "image_out": frame,
            "__frame_count__": self.total_frames,
        }
