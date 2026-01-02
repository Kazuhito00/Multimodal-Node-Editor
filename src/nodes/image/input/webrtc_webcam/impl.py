"""
WebRTCを使用してブラウザからカメラ画像を取得するノード。
Base64エンコードを経由しないため低遅延。
"""
from typing import Any, Dict

import numpy as np

from node_editor.node_def import ComputeLogic


class WebRTCWebcamNodeLogic(ComputeLogic):
    """
    WebRTC経由でブラウザからのカメラ画像を受け取るノードロジック。
    バックエンドのWebRTCFrameStoreから最新フレームを取得する。
    """

    def __init__(self):
        self._last_frame: np.ndarray | None = None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        # webrtc_webcamプロパティからconnection_idを取得
        # フロントエンドが "webrtc_webcam_node-xxx" 形式で設定
        connection_id = properties.get("webrtc_webcam", "")

        if not connection_id:
            return {"image_out": self._last_frame}

        # WebRTCFrameStoreからフレームを取得
        frame = None
        try:
            from src.gui.reactflow.backend.main import webrtc_frame_store
            frame = webrtc_frame_store.get_video_frame(connection_id)
        except Exception:
            pass

        if frame is not None:
            self._last_frame = frame
            return {"image_out": frame}

        # フレームがない場合は前回のフレームを返す
        if self._last_frame is not None:
            return {"image_out": self._last_frame}

        return {"image_out": None}
