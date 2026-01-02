"""
画像データの変換ユーティリティ。
Base64エンコード文字列とOpenCV画像（numpy配列）間の自動変換を提供。
"""
import base64
from typing import Any, Optional

import cv2
import numpy as np


def is_cv_image(value: Any) -> bool:
    """値がOpenCV画像（numpy配列）かどうかを判定"""
    return isinstance(value, np.ndarray)


def is_base64_image(value: Any) -> bool:
    """値がBase64エンコードされた画像文字列かどうかを判定"""
    return isinstance(value, str) and len(value) > 100


def base64_to_cv_image(b64_string: str) -> Optional[np.ndarray]:
    """Base64文字列をOpenCV画像に変換"""
    if not b64_string:
        return None

    try:
        img_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def resize_for_preview(img: np.ndarray, max_size: int) -> np.ndarray:
    """プレビュー用に画像をリサイズ（アスペクト比維持）"""
    h, w = img.shape[:2]
    if h <= max_size and w <= max_size:
        return img

    scale = min(max_size / w, max_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def cv_image_to_base64(img: np.ndarray, max_size: int = 0) -> Optional[str]:
    """
    OpenCV画像をBase64文字列に変換

    Args:
        img: OpenCV画像
        max_size: 最大サイズ（0の場合はリサイズなし）

    Returns:
        Base64文字列
    """
    if img is None:
        return None

    try:
        # max_sizeが指定されている場合はリサイズ
        if max_size > 0:
            img = resize_for_preview(img, max_size)

        _, buffer = cv2.imencode('.jpg', img)
        b64_string = base64.b64encode(buffer).decode('utf-8')
        return b64_string
    except Exception:
        return None


def ensure_cv_image(value: Any) -> Optional[np.ndarray]:
    """
    値をOpenCV画像に変換。
    - numpy配列の場合はそのまま返す
    - Base64文字列の場合は変換して返す
    - それ以外はNoneを返す
    """
    if value is None:
        return None

    if is_cv_image(value):
        return value

    if is_base64_image(value):
        return base64_to_cv_image(value)

    return None


def ensure_base64(value: Any, max_size: int = 0) -> Optional[str]:
    """
    値をBase64文字列に変換。
    - Base64文字列の場合はそのまま返す
    - numpy配列の場合は変換して返す
    - それ以外はNoneを返す

    Args:
        value: 変換対象の値
        max_size: 最大サイズ（0の場合はリサイズなし、プレビュー用）

    Returns:
        Base64文字列
    """
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if is_cv_image(value):
        return cv_image_to_base64(value, max_size)

    return None
