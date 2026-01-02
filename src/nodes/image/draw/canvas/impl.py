from typing import Dict, Any, Optional
from node_editor.node_def import ComputeLogic
import numpy as np
import json

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


class DrawCanvasLogic(ComputeLogic):
    """
    キャンバス上に描画するノードロジック。
    入力画像がある場合はその上に描画、ない場合は白背景に描画。
    """

    # フロントエンドのキャンバスサイズ（16:9）
    CANVAS_WIDTH = 640
    CANVAS_HEIGHT = 360

    def __init__(self):
        self._canvas: Optional[np.ndarray] = None
        self._width: int = 0
        self._height: int = 0
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0

    def reset(self):
        """キャンバスをクリア"""
        self._canvas = None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {"image": None, "__error__": "opencv-python is not installed"}

        input_image = inputs.get("image")
        pen_size = int(properties.get("pen_size", 5))
        pen_color_hex = properties.get("pen_color", "#000000")
        draw_commands_str = properties.get("draw_commands", "")
        reset_clicked = properties.get("reset", False)

        # 色をHEXからBGRに変換
        pen_color = self._hex_to_bgr(pen_color_hex)

        # 入力画像がない場合は白背景（960x540）
        if input_image is None:
            width, height = 960, 540
            base_image = np.full((height, width, 3), 255, dtype=np.uint8)
        else:
            height, width = input_image.shape[:2]
            base_image = input_image.copy()

        # スケール係数を計算（キャンバス座標 → 実画像座標）
        self._scale_x = width / self.CANVAS_WIDTH
        self._scale_y = height / self.CANVAS_HEIGHT

        # リセットボタンが押された場合は入力画像をそのまま返す
        if reset_clicked:
            return {"image": base_image.copy()}

        # 毎フレーム入力画像を背景として使用し、描画コマンドを再適用
        # これにより入力画像が変わっても描画が維持される
        self._canvas = base_image.copy()
        self._width = width
        self._height = height

        # 描画コマンドを処理（毎フレーム再適用）
        if draw_commands_str:
            try:
                commands = json.loads(draw_commands_str)
                self._apply_commands(commands, pen_size, pen_color)
            except json.JSONDecodeError:
                pass

        return {"image": self._canvas.copy()}

    def _hex_to_bgr(self, hex_color: str) -> tuple:
        """HEX色をBGRタプルに変換"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return (0, 0, 0)
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (b, g, r)
        except ValueError:
            return (0, 0, 0)

    def _scale_point(self, x: float, y: float) -> tuple:
        """キャンバス座標を実画像座標にスケーリング"""
        return (
            int(x * self._scale_x),
            int(y * self._scale_y)
        )

    def _scale_size(self, size: int) -> int:
        """ペンサイズをスケーリング"""
        scale = (self._scale_x + self._scale_y) / 2
        return max(1, int(size * scale))

    def _apply_commands(self, commands: list, pen_size: int, pen_color: tuple):
        """描画コマンドをキャンバスに適用"""
        if not commands:
            return

        for cmd in commands:
            cmd_type = cmd.get("type")

            if cmd_type == "clear":
                # clearコマンドは無視（毎フレーム入力画像から開始するため）
                pass

            elif cmd_type == "stroke":
                # ストロークを描画
                points = cmd.get("points", [])
                size = cmd.get("size", pen_size)
                color_hex = cmd.get("color", None)
                color = self._hex_to_bgr(color_hex) if color_hex else pen_color
                self._draw_stroke(points, size, color)

            elif cmd_type == "point":
                # 単一点を描画
                x = cmd.get("x", 0)
                y = cmd.get("y", 0)
                size = cmd.get("size", pen_size)
                color_hex = cmd.get("color", None)
                color = self._hex_to_bgr(color_hex) if color_hex else pen_color
                scaled_x, scaled_y = self._scale_point(x, y)
                scaled_size = self._scale_size(size)
                cv2.circle(self._canvas, (scaled_x, scaled_y), scaled_size // 2, color, -1)

    def _draw_stroke(self, points: list, size: int, color: tuple):
        """ストローク（点の列）を描画"""
        if len(points) < 1:
            return

        scaled_size = self._scale_size(size)
        radius = scaled_size // 2

        # 最初の点
        if len(points) == 1:
            pt = points[0]
            x, y = self._scale_point(pt["x"], pt["y"])
            cv2.circle(self._canvas, (x, y), radius, color, -1)
            return

        # 連続する点を線で結ぶ
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i + 1]
            x1, y1 = self._scale_point(pt1["x"], pt1["y"])
            x2, y2 = self._scale_point(pt2["x"], pt2["y"])

            # 太い線を描画
            cv2.line(self._canvas, (x1, y1), (x2, y2), color, scaled_size)

            # 端点を円で描画（滑らかな線端）
            cv2.circle(self._canvas, (x1, y1), radius, color, -1)

        # 最後の点
        last_pt = points[-1]
        x, y = self._scale_point(last_pt["x"], last_pt["y"])
        cv2.circle(self._canvas, (x, y), radius, color, -1)
