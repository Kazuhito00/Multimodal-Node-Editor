"""
LBP Histogram ノードの実装。
入力画像からLocal Binary Pattern (LBP) を計算し、ヒストグラムとして可視化する。
"""
from typing import Dict, Any, Tuple

import numpy as np
import cv2

from node_editor.node_def import ComputeLogic


# グラフサイズ（16:9）
GRAPH_WIDTH = 512
GRAPH_HEIGHT = 288

# 縦軸の最大値（ピクセル比率）
Y_AXIS_MAX = 0.1

# モード定義
MODE_LBP = 0
MODE_UNIFORM_LBP = 1

# LBPの8近傍オフセット（時計回り）
LBP_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, 1), (1, 1), (1, 0),
    (1, -1), (0, -1)
]


def calculate_lbp(image: np.ndarray) -> np.ndarray:
    """
    標準LBPを計算

    Args:
        image: グレースケール画像

    Returns:
        LBP画像（0-255の値）
    """
    height, width = image.shape
    lbp = np.zeros((height, width), dtype=np.uint8)

    # パディング
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)

    # LBP計算
    for i, (dy, dx) in enumerate(LBP_OFFSETS):
        neighbor = padded[1 + dy:height + 1 + dy, 1 + dx:width + 1 + dx]
        lbp += ((neighbor >= padded[1:height + 1, 1:width + 1]).astype(np.uint8) << i)

    return lbp


def get_uniform_lbp_table() -> np.ndarray:
    """
    Uniform LBP用のルックアップテーブルを生成

    Uniform patternは、ビット列中の0→1または1→0の遷移が2回以下のパターン。
    8ビットの場合、58個のuniformパターン + 1個の非uniformパターン = 59ビン

    Returns:
        256要素のルックアップテーブル（uniform pattern番号、非uniformは58）
    """
    table = np.zeros(256, dtype=np.uint8)
    uniform_count = 0

    for i in range(256):
        # ビット遷移回数をカウント
        bits = bin(i)[2:].zfill(8)
        circular_bits = bits + bits[0]  # 循環パターン
        transitions = sum(1 for j in range(8) if circular_bits[j] != circular_bits[j + 1])

        if transitions <= 2:
            table[i] = uniform_count
            uniform_count += 1
        else:
            table[i] = 58  # 非uniformパターン

    return table


def calculate_uniform_lbp(image: np.ndarray, uniform_table: np.ndarray) -> np.ndarray:
    """
    Uniform LBPを計算

    Args:
        image: グレースケール画像
        uniform_table: Uniform LBPルックアップテーブル

    Returns:
        Uniform LBP画像（0-58の値）
    """
    lbp = calculate_lbp(image)
    return uniform_table[lbp]


def calculate_lbp_histogram(
    lbp_image: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    """
    LBPヒストグラムを計算

    Args:
        lbp_image: LBP画像
        num_bins: ビン数

    Returns:
        正規化されたヒストグラム配列
    """
    hist = cv2.calcHist([lbp_image], [0], None, [num_bins], [0, num_bins])
    hist = hist.flatten()

    # 総ピクセル数で正規化
    total_pixels = lbp_image.shape[0] * lbp_image.shape[1]
    hist = hist / total_pixels

    return hist


def draw_lbp_histogram(
    histogram: np.ndarray,
    label: str,
    width: int = GRAPH_WIDTH,
    height: int = GRAPH_HEIGHT,
) -> np.ndarray:
    """
    LBPヒストグラムをグラフとして描画

    Args:
        histogram: ヒストグラム配列
        label: モードラベル
        width: グラフ幅
        height: グラフ高さ

    Returns:
        描画されたグラフ画像
    """
    # 背景を作成
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    graph[:] = (32, 32, 32)  # ダークグレー背景

    # グリッド線を描画
    grid_color = (64, 64, 64)
    for i in range(1, 4):
        y = int(height * i / 4)
        cv2.line(graph, (0, y), (width, y), grid_color, 1)
    for i in range(1, 4):
        x = int(width * i / 4)
        cv2.line(graph, (x, 0), (x, height), grid_color, 1)

    # ヒストグラムを描画
    bins = len(histogram)
    bin_width = width / bins
    color = (0, 200, 255)  # オレンジ系

    # ポイントを計算
    points = []
    for i, val in enumerate(histogram):
        x = int(i * bin_width + bin_width / 2)
        normalized_val = min(val / Y_AXIS_MAX, 1.0)
        y = int(height - normalized_val * (height - 10))
        points.append((x, y))

    # バーグラフとして描画
    for i, val in enumerate(histogram):
        x1 = int(i * bin_width) + 1
        x2 = int((i + 1) * bin_width) - 1
        if x2 <= x1:
            x2 = x1 + 1

        normalized_val = min(val / Y_AXIS_MAX, 1.0)
        y = int(height - normalized_val * (height - 10))

        # バーを描画
        cv2.rectangle(graph, (x1, y), (x2, height - 1), color, -1)

    # 枠線
    cv2.rectangle(graph, (0, 0), (width - 1, height - 1), (128, 128, 128), 1)

    # 右上にラベルを描画
    cv2.putText(graph, label, (width - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return graph


# Uniform LBPテーブル（モジュールロード時に生成）
UNIFORM_LBP_TABLE = get_uniform_lbp_table()


class LbpHistogramLogic(ComputeLogic):
    """
    LBP Histogramノードのロジック。
    入力画像からLBPヒストグラムを計算し、グラフ画像を出力。
    """

    def __init__(self):
        pass

    def reset(self):
        """ノードの状態をリセット"""
        pass

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        image = inputs.get("image")
        if image is None:
            return {"image": None}

        # キャンセルチェック
        if self.is_cancelled():
            self.clear_cancel()
            return {"image": None}

        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # モード取得
        mode = int(properties.get("mode", MODE_LBP))

        # LBP計算
        if mode == MODE_UNIFORM_LBP:
            lbp_image = calculate_uniform_lbp(gray, UNIFORM_LBP_TABLE)
            num_bins = 59  # 58 uniform + 1 non-uniform
            label = "Uniform LBP"
        else:
            lbp_image = calculate_lbp(gray)
            num_bins = 256
            label = "LBP"

        # ヒストグラム計算
        histogram = calculate_lbp_histogram(lbp_image, num_bins)

        # グラフを描画
        graph_image = draw_lbp_histogram(histogram, label)

        return {"image": graph_image}
