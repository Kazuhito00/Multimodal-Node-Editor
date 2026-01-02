"""
Pose Estimation ノードの実装。
MediaPipe Pose LandmarkerとViTPoseを統合したポーズ推定を行う。
"""
from typing import Dict, Any, List, Tuple, Optional
from node_editor.node_def import ComputeLogic
import json
import os

import numpy as np
import cv2

# GPU利用可能かチェック
try:
    import onnxruntime
    CUDA_AVAILABLE = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
except Exception:
    CUDA_AVAILABLE = False

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
    "MediaPipe Pose landmarker (lite)",
    "MediaPipe Pose landmarker (Full)",
    "MediaPipe Pose landmarker (Heavy)",
    "ViTPose S",
    "ViTPose B",
    "ViTPose L",
]

# MediaPipeランドマークの色（BGR）
MEDIAPIPE_LANDMARK_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 0),      # 鼻 - 緑
    1: (255, 0, 0),      # 左目内側 - 赤
    2: (0, 0, 255),      # 左目 - 青
    3: (255, 255, 0),    # 左目外側 - 黄
    4: (0, 255, 255),    # 右目内側 - シアン
    5: (255, 0, 255),    # 右目 - マゼンタ
    6: (128, 128, 128),  # 右目外側 - グレー
    7: (255, 128, 0),    # 左耳 - オレンジ
    8: (128, 0, 255),    # 右耳 - 紫
    9: (0, 128, 255),    # 口左 - ライトブルー
    10: (128, 255, 0),   # 口右 - ライム
    11: (255, 128, 128), # 左肩 - ライトレッド
    12: (128, 128, 0),   # 右肩 - オリーブ
    13: (0, 128, 128),   # 左肘 - ティール
    14: (128, 0, 128),   # 右肘 - マルーン
    15: (64, 64, 64),    # 左手首 - ダークグレー
    16: (192, 192, 192), # 右手首 - シルバー
    17: (255, 69, 0),    # 左小指 - レッドオレンジ
    18: (75, 0, 130),    # 右小指 - インディゴ
    19: (173, 255, 47),  # 左人差し指 - グリーンイエロー
    20: (220, 20, 60),   # 右人差し指 - クリムゾン
    21: (255, 0, 0),     # 左親指 - 赤
    22: (0, 0, 255),     # 右親指 - 青
    23: (0, 255, 0),     # 左腰 - 緑
    24: (255, 255, 0),   # 右腰 - 黄
    25: (0, 255, 255),   # 左膝 - シアン
    26: (255, 0, 255),   # 右膝 - マゼンタ
    27: (128, 128, 128), # 左足首 - グレー
    28: (255, 128, 0),   # 右足首 - オレンジ
    29: (128, 0, 255),   # 左かかと - 紫
    30: (0, 128, 255),   # 右かかと - ライトブルー
    31: (128, 255, 0),   # 左足指先 - ライム
    32: (255, 128, 128), # 右足指先 - ライトレッド
}

# MediaPipeスケルトン接続
MEDIAPIPE_SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 7],
    [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10], [11, 12],
    [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
    [23, 24],
    [23, 25], [25, 27], [27, 29], [29, 31],
    [24, 26], [26, 28], [28, 30], [30, 32],
    [11, 23], [12, 24],
]

# ViTPose COCO形式のキーポイント（17点）
VITPOSE_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# ViTPoseスケルトン接続
VITPOSE_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

# ViTPoseキーポイントの色（BGR）
VITPOSE_KEYPOINT_COLORS = [
    (0, 255, 0),    # nose
    (0, 255, 255),  # left_eye
    (0, 255, 255),  # right_eye
    (255, 255, 0),  # left_ear
    (255, 255, 0),  # right_ear
    (255, 0, 0),    # left_shoulder
    (255, 0, 0),    # right_shoulder
    (255, 128, 0),  # left_elbow
    (255, 128, 0),  # right_elbow
    (255, 0, 255),  # left_wrist
    (255, 0, 255),  # right_wrist
    (0, 128, 255),  # left_hip
    (0, 128, 255),  # right_hip
    (0, 255, 128),  # left_knee
    (0, 255, 128),  # right_knee
    (128, 0, 255),  # left_ankle
    (128, 0, 255),  # right_ankle
]


def draw_mediapipe_pose(
    image: np.ndarray,
    landmarks: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    """
    MediaPipe形式のポーズをMediaPipeの色付けで描画

    Args:
        image: 描画対象画像
        landmarks: ランドマークリスト
        image_width: 画像幅
        image_height: 画像高さ
        offset_x: X方向オフセット（検出領域用）
        offset_y: Y方向オフセット（検出領域用）

    Returns:
        描画後の画像
    """
    # ランドマーク座標を計算
    landmark_dict: Dict[int, List[int]] = {}
    for idx, lm in enumerate(landmarks):
        x = int(lm['x'] * image_width) + offset_x
        y = int(lm['y'] * image_height) + offset_y
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        landmark_dict[idx] = [x, y]

    # スケルトン描画
    for line in MEDIAPIPE_SKELETON:
        if line[0] in landmark_dict and line[1] in landmark_dict:
            pt1 = tuple(landmark_dict[line[0]])
            pt2 = tuple(landmark_dict[line[1]])
            cv2.line(image, pt1, pt2, (220, 220, 220), 3, cv2.LINE_AA)

    # ランドマーク描画
    for idx, point in landmark_dict.items():
        color = MEDIAPIPE_LANDMARK_COLORS.get(idx, (255, 255, 255))
        cv2.circle(image, (point[0], point[1]), 5, color, -1, cv2.LINE_AA)

    return image


def draw_vitpose_pose(
    image: np.ndarray,
    keypoints: List[Tuple[int, int, float]],
    threshold: float = 0.3,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    """
    ViTPose形式のキーポイントを描画

    Args:
        image: 描画対象画像
        keypoints: キーポイントリスト [(x, y, confidence), ...]
        threshold: 描画閾値
        offset_x: X方向オフセット
        offset_y: Y方向オフセット

    Returns:
        描画後の画像
    """
    # スケルトン描画
    for connection in VITPOSE_SKELETON:
        idx1, idx2 = connection
        if idx1 >= len(keypoints) or idx2 >= len(keypoints):
            continue

        x1, y1, conf1 = keypoints[idx1]
        x2, y2, conf2 = keypoints[idx2]

        if conf1 > threshold and conf2 > threshold:
            pt1 = (x1 + offset_x, y1 + offset_y)
            pt2 = (x2 + offset_x, y2 + offset_y)
            color = VITPOSE_KEYPOINT_COLORS[idx1]
            cv2.line(image, pt1, pt2, color, 3)

    # キーポイント描画
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > threshold:
            pt = (x + offset_x, y + offset_y)
            color = VITPOSE_KEYPOINT_COLORS[i]
            cv2.circle(image, pt, 5, color, -1)
            cv2.circle(image, pt, 7, (255, 255, 255), 2)

    return image


class PoseEstimationLogic(ComputeLogic):
    """
    ポーズ推定ノードのロジック。
    MediaPipeまたはViTPoseでポーズ推定を行い、結果をJSON文字列と可視化画像で出力。
    """

    # クラス共有のモデルキャッシュ
    _mediapipe_cache: Dict[str, Any] = {}
    _vitpose_cache: Dict[str, Any] = {}
    _model_errors: Dict[str, str] = {}

    def __init__(self):
        self._last_result_json: str = json.dumps({"poses": []})

    def reset(self):
        """ノードの状態をリセット"""
        self._last_result_json = json.dumps({"poses": []})

    def _get_model_path(self, model_index: int) -> str:
        """モデルファイルのパスを取得"""
        base_path = os.path.dirname(os.path.abspath(__file__))

        model_paths = {
            0: "MediaPipe/model/pose_landmarker_lite_float16.task",
            1: "MediaPipe/model/pose_landmarker_full_float16.task",
            2: "MediaPipe/model/pose_landmarker_heavy_float16.task",
            3: "ViTPose/model/vitpose_small.onnx",
            4: "ViTPose/model/vitpose_base.onnx",
            5: "ViTPose/model/vitpose_large.onnx",
        }

        return os.path.join(base_path, model_paths.get(model_index, ""))

    def _get_providers(self, use_gpu: bool) -> List[str]:
        """使用するプロバイダーリストを取得"""
        if use_gpu and CUDA_AVAILABLE:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_mediapipe_model(self, model_index: int) -> Tuple[Optional[Any], Optional[str]]:
        """
        MediaPipeモデルをロード
        """
        if not MEDIAPIPE_AVAILABLE:
            return None, "MediaPipe is not installed"

        cache_key = f"mediapipe_{model_index}"

        if cache_key in self._mediapipe_cache:
            return self._mediapipe_cache[cache_key], None

        if cache_key in self._model_errors:
            return None, self._model_errors[cache_key]

        model_path = self._get_model_path(model_index)
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
            )
            detector = vision.PoseLandmarker.create_from_options(options)
            self._mediapipe_cache[cache_key] = detector
            return detector, None
        except Exception as e:
            error_msg = f"Failed to load MediaPipe model: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

    def _load_vitpose_model(
        self,
        model_index: int,
        use_gpu: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        ViTPoseモデルをロード
        """
        cache_key = f"vitpose_{model_index}_gpu={use_gpu and CUDA_AVAILABLE}"

        if cache_key in self._vitpose_cache:
            return self._vitpose_cache[cache_key], None

        if cache_key in self._model_errors:
            return None, self._model_errors[cache_key]

        model_path = self._get_model_path(model_index)
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

        providers = self._get_providers(use_gpu)

        try:
            session = onnxruntime.InferenceSession(model_path, providers=providers)
            input_shape = session.get_inputs()[0].shape
            input_height = input_shape[2] if input_shape[2] is not None else 256
            input_width = input_shape[3] if input_shape[3] is not None else 192

            model_info = {
                "session": session,
                "input_name": session.get_inputs()[0].name,
                "output_name": session.get_outputs()[0].name,
                "input_size": (input_width, input_height),
            }
            self._vitpose_cache[cache_key] = model_info
            return model_info, None
        except Exception as e:
            error_msg = f"Failed to load ViTPose model: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

    def _detect_mediapipe(
        self,
        detector: Any,
        image: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        MediaPipeでポーズ検出を実行

        Args:
            detector: MediaPipe PoseLandmarker
            image: BGR画像

        Returns:
            検出結果のリスト
        """
        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA),
        )
        detection_result = detector.detect(rgb_frame)

        poses = []
        for pose_landmarks in detection_result.pose_landmarks:
            landmarks = []
            for lm in pose_landmarks:
                landmarks.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility) if hasattr(lm, 'visibility') else 1.0,
                })
            poses.append({"landmarks": landmarks, "model": "mediapipe"})

        return poses

    def _preprocess_vitpose(
        self,
        image: np.ndarray,
        input_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        ViTPose用の前処理
        """
        original_height, original_width = image.shape[:2]
        target_width, target_height = input_size

        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        resized = cv2.resize(image, (new_width, new_height))

        pad_left = (target_width - new_width) // 2
        pad_top = (target_height - new_height) // 2
        pad_right = target_width - new_width - pad_left
        pad_bottom = target_height - new_height - pad_top

        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb_image.astype(np.float32) / 255.0 - mean) / std

        tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]

        transform_info = {
            'scale': scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'original_size': (original_width, original_height),
            'input_size': input_size
        }

        return tensor.astype(np.float32), transform_info

    def _postprocess_vitpose(
        self,
        heatmaps: np.ndarray,
        transform_info: Dict[str, Any],
        threshold: float = 0.3,
    ) -> List[Tuple[int, int, float]]:
        """
        ViTPoseヒートマップからキーポイントを抽出
        """
        heatmaps = heatmaps[0]
        num_keypoints = heatmaps.shape[0]
        heatmap_height, heatmap_width = heatmaps.shape[1], heatmaps.shape[2]

        input_width, input_height = transform_info['input_size']
        scale = transform_info['scale']
        pad_left = transform_info['pad_left']
        pad_top = transform_info['pad_top']
        original_width, original_height = transform_info['original_size']

        keypoints = []

        for i in range(num_keypoints):
            heatmap = heatmaps[i]
            max_idx = np.argmax(heatmap)
            max_y = max_idx // heatmap_width
            max_x = max_idx % heatmap_width
            confidence = float(heatmap[max_y, max_x])

            if confidence < threshold:
                keypoints.append((0, 0, 0.0))
                continue

            x = max_x * (input_width / heatmap_width)
            y = max_y * (input_height / heatmap_height)

            x = x - pad_left
            y = y - pad_top

            x = x / scale
            y = y / scale

            x = max(0, min(x, original_width - 1))
            y = max(0, min(y, original_height - 1))

            keypoints.append((int(x), int(y), confidence))

        return keypoints

    def _detect_vitpose(
        self,
        model_info: Dict[str, Any],
        image: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        ViTPoseでポーズ検出を実行

        Args:
            model_info: モデル情報
            image: BGR画像

        Returns:
            検出結果のリスト
        """
        session = model_info["session"]
        input_name = model_info["input_name"]
        output_name = model_info["output_name"]
        input_size = model_info["input_size"]

        input_tensor, transform_info = self._preprocess_vitpose(image, input_size)
        outputs = session.run([output_name], {input_name: input_tensor})
        heatmaps = outputs[0]
        keypoints = self._postprocess_vitpose(heatmaps, transform_info)

        # COCO形式のキーポイントをリストに変換
        kp_list = []
        for i, (x, y, conf) in enumerate(keypoints):
            kp_list.append({
                "name": VITPOSE_KEYPOINT_NAMES[i],
                "x": int(x),
                "y": int(y),
                "confidence": float(conf),
            })

        return [{"keypoints": kp_list, "model": "vitpose"}]

    def _crop_detection(
        self,
        image: np.ndarray,
        bbox: List[int],
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """
        検出領域を切り出し

        Args:
            image: 入力画像
            bbox: バウンディングボックス [x1, y1, x2, y2]

        Returns:
            (切り出した画像, オフセット(x, y))
        """
        h, w = image.shape[:2]
        x1 = max(0, min(bbox[0], w - 1))
        y1 = max(0, min(bbox[1], h - 1))
        x2 = max(0, min(bbox[2], w))
        y2 = max(0, min(bbox[3], h))

        if x2 <= x1 or y2 <= y1:
            return None, (0, 0)

        return image[y1:y2, x1:x2].copy(), (x1, y1)

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        image = inputs.get("image")
        if image is None:
            return {"image": None, "result_json": json.dumps({"poses": []})}

        # キャンセルチェック
        if self.is_cancelled():
            self.clear_cancel()
            return {"image": image, "result_json": self._last_result_json}

        # プロパティ取得
        model_index = int(properties.get("model", 0))
        use_gpu = bool(properties.get("use_gpu", False)) and CUDA_AVAILABLE

        # モデルタイプ判定
        is_mediapipe = model_index < 3
        model_name = MODEL_NAMES[model_index] if 0 <= model_index < len(MODEL_NAMES) else MODEL_NAMES[0]

        # モデルロード
        if is_mediapipe:
            detector, load_error = self._load_mediapipe_model(model_index)
            if detector is None:
                raise RuntimeError(load_error or "Failed to load MediaPipe model")
        else:
            model_info, load_error = self._load_vitpose_model(model_index, use_gpu)
            if model_info is None:
                raise RuntimeError(load_error or "Failed to load ViTPose model")

        # 検出JSON入力を解析
        detection_json = inputs.get("detection_result")
        detections = []
        if detection_json:
            try:
                detection_data = json.loads(detection_json)
                detections = detection_data.get("detections", [])
            except (json.JSONDecodeError, TypeError):
                detections = []

        # 結果画像
        result_image = image.copy()
        image_height, image_width = image.shape[:2]
        all_poses = []

        if len(detections) > 0:
            # 検出領域ごとにポーズ推定
            for det in detections:
                if self.is_cancelled():
                    self.clear_cancel()
                    return {"image": image, "result_json": self._last_result_json}

                bbox = det.get("bbox", [])
                if len(bbox) != 4:
                    continue

                cropped, (offset_x, offset_y) = self._crop_detection(image, bbox)
                if cropped is None or cropped.size == 0:
                    continue

                crop_h, crop_w = cropped.shape[:2]

                if is_mediapipe:
                    poses = self._detect_mediapipe(detector, cropped)
                    for pose in poses:
                        # 座標をオフセット調整して描画
                        draw_mediapipe_pose(
                            result_image,
                            pose["landmarks"],
                            crop_w,
                            crop_h,
                            offset_x,
                            offset_y,
                        )
                        # 結果に検出情報を追加
                        pose["detection"] = {
                            "class_name": det.get("class_name", ""),
                            "bbox": bbox,
                        }
                        all_poses.append(pose)
                else:
                    poses = self._detect_vitpose(model_info, cropped)
                    for pose in poses:
                        # キーポイント座標をオフセット調整
                        keypoints = [
                            (kp["x"], kp["y"], kp["confidence"])
                            for kp in pose["keypoints"]
                        ]
                        draw_vitpose_pose(
                            result_image,
                            keypoints,
                            threshold=0.3,
                            offset_x=offset_x,
                            offset_y=offset_y,
                        )
                        # 座標を元画像基準に変換
                        for kp in pose["keypoints"]:
                            kp["x"] += offset_x
                            kp["y"] += offset_y
                        pose["detection"] = {
                            "class_name": det.get("class_name", ""),
                            "bbox": bbox,
                        }
                        all_poses.append(pose)

                # バウンディングボックスを描画
                x1, y1, x2, y2 = bbox
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 255, 255), 3)
        else:
            # 全体画像でポーズ推定
            if is_mediapipe:
                poses = self._detect_mediapipe(detector, image)
                for pose in poses:
                    draw_mediapipe_pose(
                        result_image,
                        pose["landmarks"],
                        image_width,
                        image_height,
                    )
                    all_poses.append(pose)
            else:
                poses = self._detect_vitpose(model_info, image)
                for pose in poses:
                    keypoints = [
                        (kp["x"], kp["y"], kp["confidence"])
                        for kp in pose["keypoints"]
                    ]
                    draw_vitpose_pose(result_image, keypoints, threshold=0.3)
                    all_poses.append(pose)

        # 結果を構築
        result = {
            "model": model_name,
            "mode": "detection" if len(detections) > 0 else "whole_image",
            "pose_count": len(all_poses),
            "poses": all_poses,
        }

        result_json = json.dumps(result, ensure_ascii=False)
        self._last_result_json = result_json

        return {"image": result_image, "result_json": result_json}
