from typing import Dict, Any, Optional
from node_editor.node_def import ComputeLogic
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


class ImageLoaderLogic(ComputeLogic):
    """静止画像を読み込むノードロジック"""

    def __init__(self):
        self._current_file: str = ""
        self._cached_image: Optional[np.ndarray] = None

    def reset(self):
        """キャッシュをクリア"""
        self._current_file = ""
        self._cached_image = None

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

        file_path = properties.get("file_path", "")

        # ファイルパスが空の場合
        if not file_path:
            return {"image_out": None}

        # ファイルが変更された場合のみ再読み込み
        if file_path != self._current_file:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    return {
                        "image_out": None,
                        "__error__": f"Failed to load: {file_path}",
                    }
                self._cached_image = image
                self._current_file = file_path
            except Exception as e:
                return {
                    "image_out": None,
                    "__error__": str(e),
                }

        return {"image_out": self._cached_image}
