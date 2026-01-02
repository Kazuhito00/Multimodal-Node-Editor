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


class QRCodeLogic(ComputeLogic):
    """
    QRコード検出ノードロジック。
    cv2.QRCodeDetectorArucoを使用して複数のQRコードを検出・デコード。
    """

    def __init__(self):
        self._detector = None

    def _get_detector(self):
        """検出器を遅延初期化"""
        if self._detector is None and CV2_AVAILABLE:
            self._detector = cv2.QRCodeDetectorAruco()
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

        detector = self._get_detector()
        if detector is None:
            return {"image": image, "result_json": "[]", "__error__": "Failed to create QR detector"}

        # 検出と復号を実行
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(image)

        results = []
        output_image = image.copy()

        if retval and points is not None:
            for info, point in zip(decoded_info, points):
                if point is None:
                    continue

                # 四隅の座標を整数に変換
                point = point.astype(int)

                # 結果を記録
                result = {
                    "content": info if info else "",
                    "points": point.tolist(),
                }
                results.append(result)

                # 検出範囲を描画（緑の四角形、太さ3）
                for i in range(4):
                    cv2.line(
                        output_image,
                        tuple(point[i]),
                        tuple(point[(i + 1) % 4]),
                        (0, 255, 0),
                        3,
                    )

                # デコード内容を画像上に表示
                if info:
                    cv2.putText(
                        output_image,
                        info,
                        tuple(point[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

        result_json = json.dumps(results, ensure_ascii=False)
        return {"image": output_image, "result_json": result_json}
