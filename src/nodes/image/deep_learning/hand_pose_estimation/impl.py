"""
Hand Pose Estimation ノードの実装。
MediaPipe GestureRecognizerを使用した手のランドマーク検出とジェスチャー認識を行う。
"""
from typing import Dict, Any, List, Tuple, Optional
from node_editor.node_def import ComputeLogic
import json
import os

import numpy as np
import cv2

# MediaPipe利用可能かチェック
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# モデル設定
MODEL_NAMES = [
    "MediaPipe HandLandmarker (full)",
]

# ランドマーク名（21点）
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# ランドマークの色（BGR）
LANDMARK_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 0),      # 手首 - 緑
    1: (255, 0, 0),      # 親指CM - 赤
    2: (0, 0, 255),      # 親指MCP - 青
    3: (255, 255, 0),    # 親指IP - 黄
    4: (0, 255, 255),    # 親指TIP - シアン
    5: (255, 0, 255),    # 人差し指MCP - マゼンタ
    6: (128, 128, 128),  # 人差し指PIP - グレー
    7: (255, 128, 0),    # 人差し指DIP - オレンジ
    8: (128, 0, 255),    # 人差し指TIP - 紫
    9: (0, 128, 255),    # 中指MCP - ライトブルー
    10: (128, 255, 0),   # 中指PIP - ライム
    11: (255, 128, 128), # 中指DIP - ライトレッド
    12: (128, 128, 0),   # 中指TIP - オリーブ
    13: (0, 128, 128),   # 薬指MCP - ティール
    14: (128, 0, 128),   # 薬指PIP - マルーン
    15: (64, 64, 64),    # 薬指DIP - ダークグレー
    16: (192, 192, 192), # 薬指TIP - シルバー
    17: (255, 69, 0),    # 小指MCP - レッドオレンジ
    18: (75, 0, 130),    # 小指PIP - インディゴ
    19: (173, 255, 47),  # 小指DIP - グリーンイエロー
    20: (220, 20, 60),   # 小指TIP - クリムゾン
}

# スケルトン接続
SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4],      # 親指
    [0, 5], [5, 6], [6, 7], [7, 8],      # 人差し指
    [0, 9], [9, 10], [10, 11], [11, 12], # 中指
    [0, 13], [13, 14], [14, 15], [15, 16], # 薬指
    [0, 17], [17, 18], [18, 19], [19, 20], # 小指
    [5, 9], [9, 13], [13, 17],           # 手のひら横方向
]


def draw_hand_landmarks(
    image: np.ndarray,
    landmarks: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    handedness: str = "",
    gesture: str = "",
    gesture_score: float = 0.0,
) -> np.ndarray:
    """
    手のランドマークとジェスチャーを描画

    Args:
        image: 描画対象画像
        landmarks: ランドマークリスト
        image_width: 画像幅
        image_height: 画像高さ
        handedness: 左右の判定
        gesture: ジェスチャー名
        gesture_score: ジェスチャーの信頼度

    Returns:
        描画後の画像
    """
    # ランドマーク座標を計算
    landmark_dict: Dict[int, List[int]] = {}
    for idx, lm in enumerate(landmarks):
        x = int(lm['x'] * image_width)
        y = int(lm['y'] * image_height)
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        landmark_dict[idx] = [x, y]

    # バウンディングボックスを計算
    if landmark_dict:
        xs = [p[0] for p in landmark_dict.values()]
        ys = [p[1] for p in landmark_dict.values()]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # バウンディングボックス描画
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 左右 + ジェスチャーラベル描画
        label_parts = []
        if handedness:
            label_parts.append(handedness)
        if gesture:
            label_parts.append(f"{gesture}({gesture_score:.2f})")
        if label_parts:
            label = ": ".join(label_parts)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # スケルトン描画
    for line in SKELETON:
        if line[0] in landmark_dict and line[1] in landmark_dict:
            pt1 = tuple(landmark_dict[line[0]])
            pt2 = tuple(landmark_dict[line[1]])
            cv2.line(image, pt1, pt2, (220, 220, 220), 3, cv2.LINE_AA)

    # ランドマーク描画
    for idx, point in landmark_dict.items():
        color = LANDMARK_COLORS.get(idx, (255, 255, 255))
        cv2.circle(image, (point[0], point[1]), 5, color, -1, cv2.LINE_AA)

    return image


class HandPoseEstimationLogic(ComputeLogic):
    """
    手のポーズ推定ノードのロジック。
    MediaPipe GestureRecognizerでランドマーク検出とジェスチャー認識を行い、
    結果をJSONと可視化画像で出力。
    """

    # クラス共有のモデルキャッシュ
    _model_cache: Dict[str, Any] = {}
    _model_errors: Dict[str, str] = {}

    def __init__(self):
        self._last_result_json: str = json.dumps({"hands": []})

    def reset(self):
        """ノードの状態をリセット"""
        self._last_result_json = json.dumps({"hands": []})

    def _get_model_path(self, model_index: int) -> str:
        """モデルファイルのパスを取得"""
        base_path = os.path.dirname(os.path.abspath(__file__))

        model_paths = {
            0: "MediaPipe/model/gesture_recognizer_float16.task",
        }

        return os.path.join(base_path, model_paths.get(model_index, ""))

    def _load_model(self, model_index: int) -> Tuple[Optional[Any], Optional[str]]:
        """
        モデルをロード
        """
        if not MEDIAPIPE_AVAILABLE:
            return None, "MediaPipe is not installed"

        cache_key = f"gesture_recognizer_{model_index}"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key], None

        if cache_key in self._model_errors:
            return None, self._model_errors[cache_key]

        model_path = self._get_model_path(model_index)
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                num_hands=4,
            )
            recognizer = vision.GestureRecognizer.create_from_options(options)
            self._model_cache[cache_key] = recognizer
            return recognizer, None
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

    def _detect_hands(
        self,
        recognizer: Any,
        image: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        手のランドマークとジェスチャーを検出

        Args:
            recognizer: MediaPipe GestureRecognizer
            image: BGR画像

        Returns:
            検出結果のリスト
        """
        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA),
        )
        recognition_result = recognizer.recognize(rgb_frame)

        hands = []
        for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
            # 左右判定
            handedness = ""
            if i < len(recognition_result.handedness):
                handedness = recognition_result.handedness[i][0].display_name

            # ジェスチャー認識結果
            gesture = ""
            gesture_score = 0.0
            if i < len(recognition_result.gestures) and len(recognition_result.gestures[i]) > 0:
                gesture = recognition_result.gestures[i][0].category_name
                gesture_score = float(recognition_result.gestures[i][0].score)

            # ランドマーク
            landmarks = []
            for idx, lm in enumerate(hand_landmarks):
                landmarks.append({
                    "name": LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"POINT_{idx}",
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                })

            hands.append({
                "handedness": handedness,
                "gesture": gesture,
                "gesture_score": gesture_score,
                "landmarks": landmarks,
            })

        return hands

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        image = inputs.get("image")
        if image is None:
            return {"image": None, "result_json": json.dumps({"hands": []})}

        # キャンセルチェック
        if self.is_cancelled():
            self.clear_cancel()
            return {"image": image, "result_json": self._last_result_json}

        # プロパティ取得
        model_index = int(properties.get("model", 0))
        model_name = MODEL_NAMES[model_index] if 0 <= model_index < len(MODEL_NAMES) else MODEL_NAMES[0]

        # モデルロード
        recognizer, load_error = self._load_model(model_index)
        if recognizer is None:
            raise RuntimeError(load_error or "Failed to load model")

        # 検出実行
        image_height, image_width = image.shape[:2]
        hands = self._detect_hands(recognizer, image)

        # 描画
        result_image = image.copy()
        for hand in hands:
            draw_hand_landmarks(
                result_image,
                hand["landmarks"],
                image_width,
                image_height,
                hand.get("handedness", ""),
                hand.get("gesture", ""),
                hand.get("gesture_score", 0.0),
            )

        # 結果を構築
        result = {
            "model": model_name,
            "hand_count": len(hands),
            "hands": hands,
        }

        result_json = json.dumps(result, ensure_ascii=False)
        self._last_result_json = result_json

        return {"image": result_image, "result_json": result_json}
