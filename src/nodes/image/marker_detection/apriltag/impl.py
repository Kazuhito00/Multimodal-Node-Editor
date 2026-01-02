from typing import Dict, Any
import json

from node_editor.node_def import ComputeLogic

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class AprilTagLogic(ComputeLogic):
    """
    AprilTagマーカー検出ノードロジック。
    cv2.aruco.ArucoDetectorを使用してAprilTagマーカーを検出。
    """

    def __init__(self):
        self._detector = None
        self._current_dict_id = None

    def _get_detector(self, dict_id: int):
        """検出器を取得（辞書が変わった場合は再生成）"""
        if not CV2_AVAILABLE:
            return None

        if self._detector is None or self._current_dict_id != dict_id:
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            parameters = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            self._current_dict_id = dict_id

        return self._detector

    def reset(self):
        """状態をリセット"""
        pass

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {"image": None, "result_json": "[]", "__error__": "opencv-python is not installed"}
        if not NUMPY_AVAILABLE:
            return {"image": None, "result_json": "[]", "__error__": "numpy is not installed"}

        image = inputs.get("image")
        if image is None:
            return {"image": None, "result_json": "[]"}

        dict_id = properties.get("dictionary", 17)
        detector = self._get_detector(dict_id)
        if detector is None:
            return {"image": image, "result_json": "[]", "__error__": "Failed to create AprilTag detector"}

        # マーカー検出を実行
        corners, ids, rejected = detector.detectMarkers(image)

        results = []
        output_image = image.copy()

        if ids is not None:
            # 検出されたマーカーをカスタム描画（太い線、大きいテキスト）
            for marker_id, corner in zip(ids.flatten(), corners):
                pts = corner[0].astype(np.int32)

                # 枠線を描画（太さ3）
                cv2.polylines(output_image, [pts], True, (0, 255, 0), thickness=3)

                # 最初のコーナーに円を描画（半径6、太さ3）
                cv2.circle(output_image, tuple(pts[0]), 6, (0, 0, 255), thickness=3)

                # ID表示（フォントサイズ1.2、太さ3）
                center_x = int(pts[:, 0].mean())
                center_y = int(pts[:, 1].mean())
                text = str(int(marker_id))
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3

                # テキストサイズを取得してセンタリング
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = center_x - text_w // 2
                text_y = center_y + text_h // 2

                # 背景付きでテキストを描画（見やすさ向上）
                cv2.putText(output_image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(output_image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

                # 結果を記録
                result = {
                    "id": int(marker_id),
                    "corners": corner[0].tolist(),
                }
                results.append(result)

        result_json = json.dumps(results, ensure_ascii=False)
        return {"image": output_image, "result_json": result_json}
