"""
Face Detection ノードの実装。
MediaPipe BlazeFace / FaceLandmarker / YuNet を使用して顔検出を行う。
"""
from typing import Dict, Any, List, Tuple, Optional
from node_editor.node_def import ComputeLogic
import json
import os

import numpy as np
import cv2

# MediaPipeインポート
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"MediaPipe not available: {e}")

# motpyインポート
try:
    from motpy import MultiObjectTracker, Detection as MotpyDetection
    MOTPY_AVAILABLE = True
except ImportError as e:
    MOTPY_AVAILABLE = False
    MultiObjectTracker = None
    MotpyDetection = None

# SAHIインポート
try:
    from sahi.slicing import slice_image
    from sahi.prediction import ObjectPrediction
    from sahi.postprocess.combine import NMSPostprocess
    SAHI_AVAILABLE = True
except ImportError as e:
    SAHI_AVAILABLE = False
    slice_image = None
    ObjectPrediction = None
    NMSPostprocess = None

# モデル設定
MODEL_NAMES = [
    "BlazeFace (short-range)",
    "FaceLandmark (478 points)",
    "YuNet",
]

MODEL_FILES = {
    "BlazeFace (short-range)": "blaze_face_short_range_float16.tflite",
    "FaceLandmark (478 points)": "face_landmarker_float16.task",
    "YuNet": "face_detection_yunet_2023mar.onnx",
}

# 顔の色
FACE_COLOR = (0, 255, 0)
LANDMARK_COLOR = (0, 255, 0)
EYE_COLOR = (0, 0, 255)

# ランドマーク色（右目、左目、鼻、右口角、左口角）
LANDMARK_COLORS = [
    (255, 0, 0),    # 右目（青）
    (0, 0, 255),    # 左目（赤）
    (0, 255, 0),    # 鼻（緑）
    (255, 0, 255),  # 右口角（マゼンタ）
    (0, 255, 255),  # 左口角（シアン）
]

# 顔のパーツ接続定義（FaceLandmark用）
FACE_CONNECTIONS = {
    "left_eyebrow": [(55, 65), (65, 52), (52, 53), (53, 46)],
    "right_eyebrow": [(285, 295), (295, 282), (282, 283), (283, 276)],
    "left_eye": [
        (133, 173), (173, 157), (157, 158), (158, 159), (159, 160), (160, 161), (161, 246),
        (246, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
    ],
    "right_eye": [
        (362, 398), (398, 384), (384, 385), (385, 386), (386, 387), (387, 388), (388, 466),
        (466, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362),
    ],
    "mouth": [
        (308, 415), (415, 310), (310, 311), (311, 312), (312, 13), (13, 82), (82, 81),
        (81, 80), (80, 191), (191, 78), (78, 95), (95, 88), (88, 178), (178, 87),
        (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    ],
}

# 目の中心ランドマークインデックス（FaceLandmark用）
EYE_CENTER_LANDMARKS = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]


def draw_face_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    normalized_keypoints: bool = True,
) -> np.ndarray:
    """顔検出結果を画像に描画"""
    result_image = image.copy()
    image_height, image_width = image.shape[:2]

    for det in detections:
        bbox = det["bbox"]
        score = det["score"]

        # バウンディングボックス描画
        x1, y1, x2, y2 = bbox
        cv2.rectangle(result_image, (x1, y1), (x2, y2), FACE_COLOR, 3)

        # スコアラベル（track_idがあれば表示）
        if "track_id" in det:
            label = f"TID:{det['track_id']} {score:.2f}"
        else:
            label = f"{score:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            result_image,
            (x1, y1 - label_h - baseline - 4),
            (x1 + label_w, y1),
            FACE_COLOR,
            -1
        )
        cv2.putText(
            result_image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        # キーポイント描画（色分け）
        if "keypoints" in det:
            for idx, kp in enumerate(det["keypoints"]):
                if normalized_keypoints:
                    kp_x = int(kp["x"] * image_width)
                    kp_y = int(kp["y"] * image_height)
                else:
                    kp_x = int(kp["x"])
                    kp_y = int(kp["y"])
                color = LANDMARK_COLORS[idx] if idx < len(LANDMARK_COLORS) else FACE_COLOR
                cv2.circle(result_image, (kp_x, kp_y), 4, color, -1, cv2.LINE_AA)

    return result_image


def draw_face_landmarks(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """FaceLandmark結果を画像に描画（478点ランドマーク）"""
    result_image = image.copy()
    image_height, image_width = image.shape[:2]

    for det in detections:
        landmarks = det.get("landmarks", [])
        if not landmarks:
            continue

        # ランドマークを座標辞書に変換
        landmark_coords = {}
        for i, lm in enumerate(landmarks):
            x = int(lm["x"] * image_width)
            y = int(lm["y"] * image_height)
            landmark_coords[i] = (x, y)

        # 全ランドマークを小さい点で描画
        for idx, (x, y) in landmark_coords.items():
            cv2.circle(result_image, (x, y), 1, LANDMARK_COLOR, -1, cv2.LINE_AA)

        # パーツの接続線を描画
        for part_name, connections in FACE_CONNECTIONS.items():
            for start_idx, end_idx in connections:
                if start_idx in landmark_coords and end_idx in landmark_coords:
                    cv2.line(
                        result_image,
                        landmark_coords[start_idx],
                        landmark_coords[end_idx],
                        LANDMARK_COLOR,
                        2
                    )

        # 目の中心を赤で描画
        for idx in EYE_CENTER_LANDMARKS:
            if idx in landmark_coords:
                cv2.circle(result_image, landmark_coords[idx], 2, EYE_COLOR, -1)

    return result_image


class FaceDetectionLogic(ComputeLogic):
    """
    顔検出ノードのロジック。
    MediaPipe BlazeFace/FaceLandmarker/YuNetで顔検出を行い、結果をJSONと可視化画像で出力。
    """

    # クラス共有のモデルキャッシュ
    _model_cache: Dict[str, Any] = {}
    _model_errors: Dict[str, str] = {}

    def __init__(self):
        self._tracker = None
        self._last_model_index: int = -1
        self._last_use_motpy: bool = False
        self._last_result_json: str = json.dumps({"detections": []})
        self._last_detections: List[Dict[str, Any]] = []
        self._last_image_size: Tuple[int, int] = (0, 0)
        self._track_id_map: Dict[str, int] = {}
        self._next_track_id: int = 1

    def reset(self):
        """ノードの状態をリセット"""
        self._tracker = None
        self._last_result_json = json.dumps({"detections": []})
        self._last_detections = []
        self._last_image_size = (0, 0)
        self._track_id_map = {}
        self._next_track_id = 1

    def _get_model_path(self, model_name: str) -> str:
        """モデルファイルのパスを取得"""
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_file = MODEL_FILES.get(model_name, "")

        if model_name == "YuNet":
            return os.path.join(base_path, "YuNet", "model", model_file)
        else:
            return os.path.join(base_path, "MediaPipe", "model", model_file)

    def _is_landmark_model(self, model_name: str) -> bool:
        """FaceLandmarkモデルかどうか"""
        return "Landmark" in model_name

    def _is_yunet_model(self, model_name: str) -> bool:
        """YuNetモデルかどうか"""
        return model_name == "YuNet"

    def _init_tracker(self, min_iou: float, multi_match_min_iou: float):
        """motpyトラッカーを初期化"""
        self._tracker = MultiObjectTracker(
            dt=0.1,
            tracker_kwargs={
                'max_staleness': 5,
            },
            matching_fn_kwargs={
                'min_iou': min_iou,
                'multi_match_min_iou': multi_match_min_iou,
            },
        )
        self._track_id_map = {}
        self._next_track_id = 1

    def _get_sequential_track_id(self, uuid_id: str) -> int:
        """UUIDのtrack_idを連番に変換"""
        if uuid_id not in self._track_id_map:
            self._track_id_map[uuid_id] = self._next_track_id
            self._next_track_id += 1
        return self._track_id_map[uuid_id]

    def _calculate_iou(self, box1: List[int], box2: List[float]) -> float:
        """2つのバウンディングボックス間のIOUを計算"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0
        return intersection / union

    def _apply_tracking(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """検出結果にトラッキングを適用"""
        motpy_detections = []
        for det in detections:
            bbox = det["bbox"]
            motpy_det = MotpyDetection(
                box=[bbox[0], bbox[1], bbox[2], bbox[3]],
                score=det["score"],
                class_id=det["class_id"]
            )
            motpy_detections.append(motpy_det)

        self._tracker.step(detections=motpy_detections)
        tracks = self._tracker.active_tracks()

        if len(tracks) == 0:
            return []

        tracked_results = []
        for track in tracks:
            track_bbox = track.box
            track_id = self._get_sequential_track_id(str(track.id))

            best_iou = 0.0
            best_det = None
            for det in detections:
                det_bbox = det["bbox"]
                iou = self._calculate_iou(det_bbox, track_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det is not None and best_iou > 0.1:
                tracked_item = {
                    "class_id": best_det["class_id"],
                    "class_name": best_det.get("class_name", "face"),
                    "score": best_det["score"],
                    "bbox": [int(track_bbox[0]), int(track_bbox[1]),
                             int(track_bbox[2]), int(track_bbox[3])],
                    "track_id": track_id,
                }
                if "keypoints" in best_det:
                    tracked_item["keypoints"] = best_det["keypoints"]
                tracked_results.append(tracked_item)

        return tracked_results

    def _load_model(self, model_name: str) -> Tuple[Optional[Any], Optional[str]]:
        """モデルをロード。(detector, error_message)を返す。"""
        if self._is_yunet_model(model_name):
            return self._load_yunet_model(model_name)

        if not MEDIAPIPE_AVAILABLE:
            return None, "MediaPipe is not installed"

        if model_name in self._model_cache:
            return self._model_cache[model_name], None

        if model_name in self._model_errors:
            return None, self._model_errors[model_name]

        model_path = self._get_model_path(model_name)
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            self._model_errors[model_name] = error_msg
            return None, error_msg

        try:
            base_options = python.BaseOptions(model_asset_path=model_path)

            if self._is_landmark_model(model_name):
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=10,
                )
                detector = vision.FaceLandmarker.create_from_options(options)
            else:
                options = vision.FaceDetectorOptions(base_options=base_options)
                detector = vision.FaceDetector.create_from_options(options)

            self._model_cache[model_name] = detector
            return detector, None
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            self._model_errors[model_name] = error_msg
            return None, error_msg

    def _load_yunet_model(self, model_name: str) -> Tuple[Optional[Any], Optional[str]]:
        """YuNetモデルをロード"""
        if model_name in self._model_cache:
            return self._model_cache[model_name], None

        if model_name in self._model_errors:
            return None, self._model_errors[model_name]

        model_path = self._get_model_path(model_name)
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            self._model_errors[model_name] = error_msg
            return None, error_msg

        try:
            detector = cv2.FaceDetectorYN.create(
                model_path,
                "",
                (320, 320),
                0.9,
                0.3,
                5000,
            )
            self._model_cache[model_name] = detector
            return detector, None
        except Exception as e:
            error_msg = f"Failed to load YuNet model: {e}"
            self._model_errors[model_name] = error_msg
            return None, error_msg

    def _detect_single(
        self,
        detector: Any,
        image: np.ndarray,
        model_name: str,
        score_th: float,
    ) -> List[Dict[str, Any]]:
        """単一画像での検出"""
        if self._is_yunet_model(model_name):
            return self._detect_faces_yunet(detector, image, score_th)
        elif self._is_landmark_model(model_name):
            return self._detect_faces_landmark(detector, image, score_th)
        else:
            return self._detect_faces_blazeface(detector, image, score_th)

    def _detect_with_sahi(
        self,
        detector: Any,
        image: np.ndarray,
        model_name: str,
        score_th: float,
        slice_w: int,
        slice_h: int,
        overlap: float,
    ) -> List[Dict[str, Any]]:
        """SAHIを使用した検出"""
        h, w = image.shape[:2]

        # スライス画像を生成
        slice_result = slice_image(
            image=image,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
        )

        all_predictions = []

        # 各スライスで検出
        for slice_data in slice_result.sliced_image_list:
            slice_img = slice_data.image
            shift_x = slice_data.starting_pixel[0]
            shift_y = slice_data.starting_pixel[1]

            slice_detections = self._detect_single(detector, slice_img, model_name, score_th)

            for det in slice_detections:
                bbox = det["bbox"]
                shifted_bbox = [
                    bbox[0] + shift_x,
                    bbox[1] + shift_y,
                    bbox[2] + shift_x,
                    bbox[3] + shift_y,
                ]
                obj_pred = ObjectPrediction(
                    bbox=shifted_bbox,
                    score=det["score"],
                    category_id=det["class_id"],
                    category_name=det["class_name"],
                )
                all_predictions.append(obj_pred)

        # 全体画像での検出も追加
        full_detections = self._detect_single(detector, image, model_name, score_th)
        for det in full_detections:
            obj_pred = ObjectPrediction(
                bbox=det["bbox"],
                score=det["score"],
                category_id=det["class_id"],
                category_name=det["class_name"],
            )
            all_predictions.append(obj_pred)

        # NMSで重複除去
        if len(all_predictions) > 0:
            nms = NMSPostprocess(
                match_metric="IOU",
                match_threshold=0.5,
            )
            merged_predictions = nms(all_predictions)
        else:
            merged_predictions = []

        # 結果を辞書形式に変換
        result_detections = []
        for pred in merged_predictions:
            det = {
                "class_id": pred.category.id,
                "class_name": pred.category.name,
                "score": pred.score.value,
                "bbox": [int(pred.bbox.minx), int(pred.bbox.miny),
                         int(pred.bbox.maxx), int(pred.bbox.maxy)],
            }
            result_detections.append(det)

        return result_detections

    def _detect_faces_blazeface(
        self,
        detector: Any,
        image: np.ndarray,
        score_th: float,
    ) -> List[Dict[str, Any]]:
        """BlazeFaceで顔検出を実行"""
        image_height, image_width = image.shape[:2]

        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA),
        )

        detection_result = detector.detect(rgb_frame)

        detections = []
        for detection_info in detection_result.detections:
            score = detection_info.categories[0].score
            if score < score_th:
                continue

            bbox = detection_info.bounding_box
            x1 = bbox.origin_x
            y1 = bbox.origin_y
            x2 = x1 + bbox.width
            y2 = y1 + bbox.height

            keypoints = []
            for kp in detection_info.keypoints:
                keypoints.append({
                    "x": float(kp.x),
                    "y": float(kp.y),
                })

            detections.append({
                "class_id": 0,
                "class_name": "face",
                "score": float(score),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "keypoints": keypoints,
            })

        return detections

    def _detect_faces_landmark(
        self,
        detector: Any,
        image: np.ndarray,
        score_th: float,
    ) -> List[Dict[str, Any]]:
        """FaceLandmarkerで顔検出＋ランドマーク検出を実行"""
        image_height, image_width = image.shape[:2]

        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA),
        )

        detection_result = detector.detect(rgb_frame)

        detections = []
        for face_landmarks in detection_result.face_landmarks:
            xs = [lm.x * image_width for lm in face_landmarks]
            ys = [lm.y * image_height for lm in face_landmarks]
            x1 = int(min(xs))
            y1 = int(min(ys))
            x2 = int(max(xs))
            y2 = int(max(ys))

            landmarks = []
            for lm in face_landmarks:
                landmarks.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z) if hasattr(lm, 'z') else 0.0,
                })

            detections.append({
                "class_id": 0,
                "class_name": "face",
                "score": 1.0,
                "bbox": [x1, y1, x2, y2],
                "landmarks": landmarks,
                "landmark_count": len(landmarks),
            })

        return detections

    def _detect_faces_yunet(
        self,
        detector: Any,
        image: np.ndarray,
        score_th: float,
    ) -> List[Dict[str, Any]]:
        """YuNetで顔検出を実行"""
        h, w = image.shape[:2]

        if (w, h) != self._last_image_size:
            detector.setInputSize((w, h))
            self._last_image_size = (w, h)

        detector.setScoreThreshold(score_th)

        _, results = detector.detect(image)

        detections = []
        if results is not None:
            for det in results:
                x = int(det[0])
                y = int(det[1])
                box_w = int(det[2])
                box_h = int(det[3])
                score = float(det[14])

                keypoints = []
                for i in range(5):
                    kp_x = float(det[4 + i * 2])
                    kp_y = float(det[5 + i * 2])
                    keypoints.append({"x": kp_x, "y": kp_y})

                detections.append({
                    "class_id": 0,
                    "class_name": "face",
                    "score": score,
                    "bbox": [x, y, x + box_w, y + box_h],
                    "keypoints": keypoints,
                })

        return detections

    def _check_cancel_and_return(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """キャンセルされていたら早期リターン用の結果を返す"""
        if self.is_cancelled():
            self.clear_cancel()
            return {"image": image, "result_json": self._last_result_json}
        return None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        image = inputs.get("image")
        if image is None:
            return {"image": None, "result_json": json.dumps({"detections": []})}

        cancelled_result = self._check_cancel_and_return(image)
        if cancelled_result is not None:
            return cancelled_result

        # プロパティ取得
        model_index = int(properties.get("model", 0))
        score_th = float(properties.get("score_threshold", 0.5))
        use_motpy = bool(properties.get("use_motpy", False))
        motpy_min_iou = float(properties.get("motpy_min_iou", 0.2))
        motpy_multi_iou = float(properties.get("motpy_multi_match_min_iou", 0.9))
        use_sahi = bool(properties.get("use_sahi", False))
        sahi_slice_w = int(properties.get("sahi_slice_width", 320))
        sahi_slice_h = int(properties.get("sahi_slice_height", 320))
        sahi_overlap = float(properties.get("sahi_overlap_ratio", 0.2))

        # モデル名取得
        if model_index < 0 or model_index >= len(MODEL_NAMES):
            model_index = 0
        model_name = MODEL_NAMES[model_index]
        is_landmark = self._is_landmark_model(model_name)
        is_yunet = self._is_yunet_model(model_name)

        # モデルロード
        detector, load_error = self._load_model(model_name)

        if detector is None:
            error_msg = load_error or f"Model not found: {model_name}"
            raise RuntimeError(error_msg)

        # motpy使用時のチェック
        if use_motpy and not MOTPY_AVAILABLE:
            raise RuntimeError("motpy is not installed. Please install: pip install motpy")

        # SAHI使用時のチェック
        if use_sahi and not SAHI_AVAILABLE:
            raise RuntimeError("sahi is not installed. Please install: pip install sahi")

        # トラッカー初期化（モデル変更時またはuse_motpy変更時）
        if model_index != self._last_model_index or use_motpy != self._last_use_motpy:
            if use_motpy and MOTPY_AVAILABLE:
                self._init_tracker(motpy_min_iou, motpy_multi_iou)
            else:
                self._tracker = None
            self._last_model_index = model_index
            self._last_use_motpy = use_motpy

        # 検出実行
        if use_sahi and not is_landmark:
            detections = self._detect_with_sahi(
                detector, image, model_name, score_th,
                sahi_slice_w, sahi_slice_h, sahi_overlap
            )
        else:
            detections = self._detect_single(detector, image, model_name, score_th)

        # motpyトラッキング（FaceLandmark以外）
        if use_motpy and MOTPY_AVAILABLE and self._tracker is not None and len(detections) > 0 and not is_landmark:
            detections = self._apply_tracking(detections)

        # 可視化
        if is_landmark:
            result_image = draw_face_landmarks(image, detections)
        elif is_yunet:
            result_image = draw_face_detections(image, detections, normalized_keypoints=False)
        else:
            result_image = draw_face_detections(image, detections, normalized_keypoints=True)

        self._last_detections = detections

        result = {
            "model": model_name,
            "score_threshold": score_th,
            "detections": detections,
            "count": len(detections),
        }
        result_json = json.dumps(result, ensure_ascii=False)
        self._last_result_json = result_json

        return {"image": result_image, "result_json": result_json}
