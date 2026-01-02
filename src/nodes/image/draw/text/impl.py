from typing import Dict, Any
from node_editor.node_def import ComputeLogic

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class DrawTextLogic(ComputeLogic):
    """テキスト描画ノードロジック（OpenCV使用、ASCII文字のみ対応）"""

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

        image = inputs.get("image")
        if image is None:
            return {"image": None}

        text = inputs.get("text", "") or ""
        if not text:
            return {"image": image}

        x = int(properties.get("x", 10))
        y = int(properties.get("y", 10))
        font_size = int(properties.get("font_size", 32))
        color_hex = properties.get("color", "#00ff00")

        # HEXカラーをBGRタプルに変換
        try:
            color_hex = color_hex.lstrip("#")
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            color_bgr = (b, g, r)
        except (ValueError, IndexError):
            color_bgr = (0, 255, 0)

        try:
            # 画像をコピー（元画像を変更しない）
            result = image.copy()

            # フォントスケールを計算（font_sizeをピクセル相当に変換）
            font_scale = font_size / 30.0
            thickness = max(1, int(font_size / 16))

            # 改行で分割して各行を描画
            lines = text.split('\n')
            line_height = int(font_size * 1.2)

            for i, line in enumerate(lines):
                if not line:
                    continue
                # OpenCVでテキスト描画
                cv2.putText(
                    result,
                    line,
                    (x, y + font_size + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color_bgr,
                    thickness,
                    cv2.LINE_AA
                )

            return {"image": result}

        except Exception as e:
            return {
                "image": image,
                "__error__": str(e),
            }
