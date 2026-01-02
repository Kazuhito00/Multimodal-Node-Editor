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


class CreateMaskLogic(ComputeLogic):
    """
    入力画像上にマウス描画でマスクを作成するノードロジック。
    入力画像をプレビューとして表示し、出力はBGR形式の白黒マスク画像。
    """

    # フロントエンドのキャンバスサイズ（16:9）
    CANVAS_WIDTH = 640
    CANVAS_HEIGHT = 360

    def __init__(self):
        self._mask: Optional[np.ndarray] = None
        self._width: int = 0
        self._height: int = 0
        self._last_commands_hash: str = ""
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0

    def reset(self):
        """マスクをクリア"""
        self._mask = None
        self._last_commands_hash = ""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not CV2_AVAILABLE:
            return {"mask": None, "__error__": "opencv-python is not installed"}

        input_image = inputs.get("image")
        pen_size = int(properties.get("pen_size", 10))
        draw_commands_str = properties.get("draw_commands", "")

        # 入力画像がない場合はデフォルトサイズ（960x540）で黒画像
        if input_image is None:
            width, height = 960, 540
        else:
            height, width = input_image.shape[:2]

        # スケール係数を計算（キャンバス座標 → 実画像座標）
        self._scale_x = width / self.CANVAS_WIDTH
        self._scale_y = height / self.CANVAS_HEIGHT

        # サイズ変更時はマスクを再作成
        if self._mask is None or self._width != width or self._height != height:
            self._mask = np.zeros((height, width, 3), dtype=np.uint8)
            self._width = width
            self._height = height
            self._last_commands_hash = ""

        # 描画コマンドを処理
        if draw_commands_str:
            commands_hash = hash(draw_commands_str)
            if commands_hash != self._last_commands_hash:
                self._last_commands_hash = commands_hash
                try:
                    commands = json.loads(draw_commands_str)
                    self._process_commands(commands, pen_size)
                except json.JSONDecodeError:
                    pass

        # 出力は白黒マスク
        return {"mask": self._mask.copy()}

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

    def _process_commands(self, commands: list, pen_size: int):
        """描画コマンドを処理"""
        if not commands:
            return

        for cmd in commands:
            cmd_type = cmd.get("type")

            if cmd_type == "clear":
                # マスクをクリア
                self._mask.fill(0)

            elif cmd_type == "stroke":
                # ストロークを描画（白）
                points = cmd.get("points", [])
                size = cmd.get("size", pen_size)
                self._draw_stroke(points, size, color=(255, 255, 255))

            elif cmd_type == "erase":
                # 消しゴム（黒で描画）
                points = cmd.get("points", [])
                size = cmd.get("size", pen_size)
                self._draw_stroke(points, size, color=(0, 0, 0))

            elif cmd_type == "point":
                # 単一点を描画
                x = cmd.get("x", 0)
                y = cmd.get("y", 0)
                size = cmd.get("size", pen_size)
                scaled_x, scaled_y = self._scale_point(x, y)
                scaled_size = self._scale_size(size)
                cv2.circle(self._mask, (scaled_x, scaled_y), scaled_size // 2, (255, 255, 255), -1)

    def _draw_stroke(self, points: list, size: int, color: tuple = (255, 255, 255)):
        """ストローク（点の列）を描画"""
        if len(points) < 1:
            return

        scaled_size = self._scale_size(size)
        radius = scaled_size // 2

        # 最初の点
        if len(points) == 1:
            pt = points[0]
            x, y = self._scale_point(pt["x"], pt["y"])
            cv2.circle(self._mask, (x, y), radius, color, -1)
            return

        # 連続する点を線で結ぶ
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i + 1]
            x1, y1 = self._scale_point(pt1["x"], pt1["y"])
            x2, y2 = self._scale_point(pt2["x"], pt2["y"])

            # 太い線を描画
            cv2.line(self._mask, (x1, y1), (x2, y2), color, scaled_size)

            # 端点を円で描画（滑らかな線端）
            cv2.circle(self._mask, (x1, y1), radius, color, -1)

        # 最後の点
        last_pt = points[-1]
        x, y = self._scale_point(last_pt["x"], last_pt["y"])
        cv2.circle(self._mask, (x, y), radius, color, -1)
