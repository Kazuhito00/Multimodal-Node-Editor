"""
Multi Image Concat ノードの実装。
複数の画像をグリッドレイアウトで連結する。
"""
from typing import Dict, Any, List

import numpy as np
import cv2

from node_editor.node_def import ComputeLogic


def get_grid_size(count: int) -> tuple:
    """
    画像数に応じたグリッドサイズを返す

    Args:
        count: 画像の数

    Returns:
        (rows, cols) のタプル
    """
    if count <= 1:
        return (1, 1)
    elif count == 2:
        return (1, 2)
    elif count <= 4:
        return (2, 2)
    elif count <= 6:
        return (2, 3)
    else:
        return (3, 3)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    画像をBGR形式に変換

    Args:
        image: 入力画像

    Returns:
        BGR画像
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def resize_to_cell_size(image: np.ndarray, cell_width: int, cell_height: int) -> np.ndarray:
    """
    画像をセルサイズにリサイズ（アスペクト比維持、余白は黒）

    Args:
        image: 入力画像
        cell_width: セルの幅
        cell_height: セルの高さ

    Returns:
        リサイズされた画像
    """
    h, w = image.shape[:2]

    # アスペクト比を維持してリサイズ
    scale = min(cell_width / w, cell_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # セルサイズの黒画像に中央配置
    result = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    x_offset = (cell_width - new_w) // 2
    y_offset = (cell_height - new_h) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return result


def create_grid_image(images: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
    """
    画像をグリッドレイアウトで結合

    Args:
        images: 画像のリスト
        rows: 行数
        cols: 列数

    Returns:
        結合された画像
    """
    if not images:
        return None

    # 最初の画像のサイズを基準にセルサイズを決定
    base_h, base_w = images[0].shape[:2]
    cell_width = base_w
    cell_height = base_h

    # グリッド画像を作成（黒で初期化）
    grid_height = cell_height * rows
    grid_width = cell_width * cols
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # 各セルに画像を配置
    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        y_start = row * cell_height
        x_start = col * cell_width

        if idx < len(images):
            img = ensure_bgr(images[idx])
            cell_img = resize_to_cell_size(img, cell_width, cell_height)
            grid[y_start:y_start + cell_height, x_start:x_start + cell_width] = cell_img

    return grid


class MultiImageConcatLogic(ComputeLogic):
    """
    Multi Image Concatノードのロジック。
    複数の画像をグリッドレイアウトで連結する。
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
        # キャンセルチェック
        if self.is_cancelled():
            self.clear_cancel()
            return {"image": None}

        # 接続された画像を順番に収集
        images = []
        for i in range(1, 10):
            img = inputs.get(f"Image {i}")
            if img is not None:
                images.append(img)

        # 画像がない場合
        if not images:
            return {"image": None}

        # 1枚のみの場合
        if len(images) == 1:
            return {"image": ensure_bgr(images[0])}

        # グリッドサイズを決定
        rows, cols = get_grid_size(len(images))

        # グリッド画像を作成
        result = create_grid_image(images, rows, cols)

        return {"image": result}
