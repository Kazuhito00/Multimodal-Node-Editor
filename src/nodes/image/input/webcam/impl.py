from typing import Dict, Any, Tuple
from node_editor.node_def import ComputeLogic
import cv2
import atexit

# カメラID・解像度ごとのキャプチャオブジェクトを管理
# キー: (camera_id, width, height)
_caps: Dict[Tuple[int, int, int], cv2.VideoCapture] = {}


def release_all_webcams():
    """全てのWebカメラを解放"""
    global _caps
    for cap in _caps.values():
        cap.release()
    _caps.clear()


def parse_resolution(resolution_str: str) -> Tuple[int, int]:
    """解像度文字列をwidth, heightのタプルに変換"""
    try:
        width, height = resolution_str.split("x")
        return int(width), int(height)
    except (ValueError, AttributeError):
        return 1280, 720  # デフォルト


class WebcamNodeLogic(ComputeLogic):
    """
    Webカメラから画像フレームを取得するノードロジック。
    出力はOpenCV画像（numpy配列）。Base64変換はcore.pyで自動的に行われる。
    """

    def reset(self):
        """STOP時にカメラを解放"""
        release_all_webcams()

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        global _caps
        camera_id = int(properties.get("camera_id", 0))
        resolution = properties.get("resolution", "1280x720")
        width, height = parse_resolution(resolution)

        cap_key = (camera_id, width, height)

        # 指定されたカメラID・解像度のキャプチャを取得または作成
        if cap_key not in _caps:
            # 古い解像度のキャプチャがあれば解放
            old_keys = [k for k in _caps.keys() if k[0] == camera_id]
            for old_key in old_keys:
                _caps[old_key].release()
                del _caps[old_key]

            # 新しいキャプチャを作成
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                return {"image_out": None}

            # 解像度を設定
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            _caps[cap_key] = cap

        cap = _caps[cap_key]
        ret, frame = cap.read()
        if not ret:
            # キャプチャ失敗時は再接続を試みる
            cap.release()
            del _caps[cap_key]
            return {"image_out": None}

        # cv画像をそのまま返す（Base64変換はcore.pyで自動処理）
        return {"image_out": frame}


atexit.register(release_all_webcams)
