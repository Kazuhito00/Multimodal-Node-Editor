"""
Color Histogram ノードの実装。
入力画像からRGB/HSV/Labのヒストグラムを計算し、グラフとして可視化する。
"""
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2

from node_editor.node_def import ComputeLogic


# ヒストグラムのビン数
HIST_BINS = 256
HIST_BINS_HUE = 180  # OpenCVのHueは0-180

# グラフサイズ（16:9）
GRAPH_WIDTH = 512
GRAPH_HEIGHT = 288

# 縦軸の最大値（ピクセル比率）- 5%を最大とする
Y_AXIS_MAX = 0.05

# モード定義
MODE_RGB = 0
MODE_HSV = 1
MODE_LAB = 2

# 各モードのチャンネル設定
MODE_CONFIG = {
    MODE_RGB: {
        'channels': ['R', 'G', 'B'],
        'colors': [(0, 0, 255), (0, 255, 0), (255, 0, 0)],  # BGR形式
        'bins': [HIST_BINS, HIST_BINS, HIST_BINS],
        'ranges': [(0, 256), (0, 256), (0, 256)],
    },
    MODE_HSV: {
        'channels': ['H', 'S', 'V'],
        'colors': [(128, 0, 255), (0, 255, 255), (255, 255, 255)],  # マゼンタ、黄、白
        'bins': [HIST_BINS_HUE, HIST_BINS, HIST_BINS],
        'ranges': [(0, 180), (0, 256), (0, 256)],
    },
    MODE_LAB: {
        'channels': ['L', 'a', 'b'],
        'colors': [(255, 255, 255), (0, 128, 255), (255, 128, 0)],  # 白、オレンジ、青
        'bins': [HIST_BINS, HIST_BINS, HIST_BINS],
        'ranges': [(0, 256), (0, 256), (0, 256)],
    },
}


def calculate_histogram(
    image: np.ndarray,
    channel: int,
    bins: int,
    hist_range: Tuple[int, int],
) -> np.ndarray:
    """
    指定チャンネルのヒストグラムを計算

    Args:
        image: 変換済み画像
        channel: チャンネルインデックス
        bins: ビン数
        hist_range: ヒストグラムの範囲

    Returns:
        正規化されたヒストグラム配列（総ピクセル数で正規化）
    """
    hist = cv2.calcHist([image], [channel], None, [bins], list(hist_range))
    hist = hist.flatten()

    # 総ピクセル数で正規化（固定スケール）
    total_pixels = image.shape[0] * image.shape[1]
    hist = hist / total_pixels

    return hist


def draw_histogram_graph(
    histograms: List[np.ndarray],
    colors: List[Tuple[int, int, int]],
    labels: List[str],
    width: int = GRAPH_WIDTH,
    height: int = GRAPH_HEIGHT,
) -> np.ndarray:
    """
    ヒストグラムをグラフとして描画

    Args:
        histograms: ヒストグラム配列のリスト
        colors: 各チャンネルの色（BGR）
        labels: チャンネルラベル
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

    # 各チャンネルのヒストグラムを描画
    for hist, color in zip(histograms, colors):
        bins = len(hist)
        bin_width = width / bins

        # ポイントを計算
        points = []
        for i, val in enumerate(hist):
            x = int(i * bin_width)
            # 固定スケール: Y_AXIS_MAXを100%として描画
            normalized_val = min(val / Y_AXIS_MAX, 1.0)
            y = int(height - normalized_val * (height - 10))
            points.append((x, y))

        # ラインを描画（半透明効果のため、塗りつぶしポリゴンを使用）
        if len(points) > 1:
            # 塗りつぶし用のポリゴン
            fill_points = [(0, height)] + points + [(width, height)]
            fill_array = np.array(fill_points, dtype=np.int32)

            # 半透明の塗りつぶし
            overlay = graph.copy()
            cv2.fillPoly(overlay, [fill_array], color)
            cv2.addWeighted(overlay, 0.3, graph, 0.7, 0, graph)

            # ライン描画
            for i in range(len(points) - 1):
                cv2.line(graph, points[i], points[i + 1], color, 1, cv2.LINE_AA)

    # 枠線
    cv2.rectangle(graph, (0, 0), (width - 1, height - 1), (128, 128, 128), 1)

    # 右上にラベルを描画
    label_x = width - 25 * len(labels) - 10
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = label_x + i * 25
        cv2.putText(graph, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return graph


class ColorHistogramLogic(ComputeLogic):
    """
    Color Histogramノードのロジック。
    入力画像からRGB/HSV/Labヒストグラムを計算し、グラフ画像を出力。
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

        # グレースケール画像の場合はBGRに変換
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # モード取得
        mode = int(properties.get("mode", MODE_RGB))
        config = MODE_CONFIG.get(mode, MODE_CONFIG[MODE_RGB])

        # 色空間変換
        if mode == MODE_HSV:
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif mode == MODE_LAB:
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        else:
            # RGBモード: BGRのまま（チャンネル順序を考慮）
            converted = image

        # 各チャンネルのヒストグラムを計算
        histograms = []
        for i in range(3):
            if mode == MODE_RGB:
                # RGBの場合: B=0, G=1, R=2 → R, G, B の順で計算
                channel_idx = 2 - i
            else:
                channel_idx = i

            hist = calculate_histogram(
                converted,
                channel_idx,
                config['bins'][i],
                config['ranges'][i],
            )
            histograms.append(hist)

        # グラフを描画
        graph_image = draw_histogram_graph(
            histograms,
            config['colors'],
            config['channels'],
        )

        return {"image": graph_image}
