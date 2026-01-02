from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
from node_editor.node_def import ComputeLogic
import json
import os

import numpy as np
import cv2

# オプショナルな依存関係のインポート
IMPORT_ERRORS: List[str] = []

# GPU利用可能かチェック
try:
    import onnxruntime
    CUDA_AVAILABLE = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
except Exception:
    CUDA_AVAILABLE = False

try:
    from motpy import MultiObjectTracker, Detection as MotpyDetection
    MOTPY_AVAILABLE = True
except ImportError as e:
    MOTPY_AVAILABLE = False
    IMPORT_ERRORS.append(f"motpy: {e}")
    MultiObjectTracker = None
    MotpyDetection = None

try:
    from sahi.slicing import slice_image
    from sahi.prediction import ObjectPrediction
    from sahi.postprocess.combine import NMSPostprocess
    SAHI_AVAILABLE = True
except ImportError as e:
    SAHI_AVAILABLE = False
    IMPORT_ERRORS.append(f"sahi: {e}")
    slice_image = None
    ObjectPrediction = None
    NMSPostprocess = None

# クラス名辞書をインポート
from src.nodes.image.deep_learning.object_detection.coco_class_names import coco_class_names
from src.nodes.image.deep_learning.object_detection.wholebody34_class_names import (
    wholebody34_class_names,
    HEAD_POSE_COLORS,
    HEAD_POSE_NAMES,
    WHOLEBODY34_EDGES,
)

# モデル設定
MODEL_NAMES = [
    "DEIMv2-Atto(COCO)",
    "DEIMv2-Femto(COCO)",
    "DEIMv2-Pico(COCO)",
    "DEIMv2-N(COCO)",
    "DEIMv2-S(COCO)",
    "DEIMv2-Wholebody34-Atto",
    "DEIMv2-Wholebody34-Femto",
    "DEIMv2-Wholebody34-Pico",
    "DEIMv2-Wholebody34-N",
    "DEIMv2-Wholebody34-S",
]

WHOLEBODY34_MODELS = {
    "DEIMv2-Wholebody34-Atto",
    "DEIMv2-Wholebody34-Femto",
    "DEIMv2-Wholebody34-Pico",
    "DEIMv2-Wholebody34-N",
    "DEIMv2-Wholebody34-S",
}

# クラスごとの色（BGR）
CLASS_COLORS = {}


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """クラスIDに対応する色を取得（なければ生成）"""
    if class_id not in CLASS_COLORS:
        np.random.seed(class_id)
        CLASS_COLORS[class_id] = (
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
        )
    return CLASS_COLORS[class_id]


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    class_names: Dict[int, str]
) -> np.ndarray:
    """検出結果を画像に描画（COCO用）"""
    result_image = image.copy()

    for det in detections:
        bbox = det["bbox"]
        class_id = det["class_id"]
        score = det["score"]
        class_name = det.get("class_name", class_names.get(class_id, "unknown"))

        # 色を取得
        color = get_class_color(class_id)

        # バウンディングボックス描画
        x1, y1, x2, y2 = bbox
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)

        # ラベル作成
        if "track_id" in det:
            label = f"TID:{det['track_id']} {class_name} {score:.2f}"
        else:
            label = f"{class_name} {score:.2f}"

        # ラベル背景
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            result_image,
            (x1, y1 - label_h - baseline - 4),
            (x1 + label_w, y1),
            color,
            -1
        )

        # ラベルテキスト
        cv2.putText(
            result_image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return result_image


# Wholebody34用のクラスID別色
WHOLEBODY34_COLORS = {
    5: (0, 200, 255),    # Body-With-Wheelchair
    6: (83, 36, 179),    # Body-With-Crutches
    16: (0, 200, 255),   # Face
    17: (255, 0, 0),     # Eye
    18: (0, 255, 0),     # Nose
    19: (0, 0, 255),     # Mouth
    20: (203, 192, 255), # Ear
    21: (0, 0, 255),     # Collarbone
    22: (255, 0, 0),     # Shoulder
    23: (252, 189, 107), # Solar_plexus
    24: (0, 255, 0),     # Elbow
    25: (0, 0, 255),     # Wrist
    29: (0, 0, 255),     # Abdomen
    30: (255, 0, 0),     # Hip_joint
    31: (0, 0, 255),     # Knee
    32: (255, 0, 0),     # Ankle
    33: (250, 0, 136),   # Foot
}

WHOLEBODY34_KEYPOINT_CLASSES = {21, 22, 23, 24, 25, 29, 30, 31, 32}


def draw_wholebody34_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """Wholebody34検出結果を画像に描画（デモファイル準拠）"""
    result_image = image.copy()
    image_h, image_w = result_image.shape[:2]

    # person_idを割り当て（Bodyのバウンディングボックス内のキーポイントをグルーピング）
    body_boxes = [d for d in detections if d["class_id"] == 0]
    for i, body in enumerate(body_boxes):
        body["person_id"] = i

    # キーポイントにperson_idを割り当て
    for det in detections:
        if det["class_id"] in WHOLEBODY34_KEYPOINT_CLASSES:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            det["person_id"] = -1
            for body in body_boxes:
                bb = body["bbox"]
                if bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3]:
                    det["person_id"] = body["person_id"]
                    break

    # スケルトン描画（同じperson_id同士のみ接続）
    classid_to_dets = {}
    for det in detections:
        cid = det["class_id"]
        if cid not in classid_to_dets:
            classid_to_dets[cid] = []
        classid_to_dets[cid].append(det)

    edge_counts = Counter(WHOLEBODY34_EDGES)

    for (pid, cid), repeat_count in edge_counts.items():
        parent_list = classid_to_dets.get(pid, [])
        child_list = classid_to_dets.get(cid, [])

        if not parent_list or not child_list:
            continue

        for_parent = repeat_count if pid in [21, 29] else 1
        parent_capacity = [for_parent] * len(parent_list)
        child_used = [False] * len(child_list)

        pair_candidates = []
        for i, pdet in enumerate(parent_list):
            for j, cdet in enumerate(child_list):
                p_pid = pdet.get("person_id", -1)
                c_pid = cdet.get("person_id", -1)
                if p_pid >= 0 and p_pid == c_pid:
                    pb = pdet["bbox"]
                    cb = cdet["bbox"]
                    pcx, pcy = (pb[0] + pb[2]) // 2, (pb[1] + pb[3]) // 2
                    ccx, ccy = (cb[0] + cb[2]) // 2, (cb[1] + cb[3]) // 2
                    dist = ((pcx - ccx)**2 + (pcy - ccy)**2)**0.5
                    if dist <= 300:
                        pair_candidates.append((dist, i, j, pcx, pcy, ccx, ccy))

        pair_candidates.sort(key=lambda x: x[0])
        for dist, i, j, pcx, pcy, ccx, ccy in pair_candidates:
            if parent_capacity[i] > 0 and not child_used[j]:
                cv2.line(result_image, (pcx, pcy), (ccx, ccy), (0, 255, 255), 2)
                parent_capacity[i] -= 1
                child_used[j] = True

    # キーポイント描画
    for det in detections:
        class_id = det["class_id"]
        if class_id not in WHOLEBODY34_KEYPOINT_CLASSES:
            continue
        bbox = det["bbox"]
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        color = WHOLEBODY34_COLORS.get(class_id, (255, 255, 255))
        cv2.circle(result_image, (cx, cy), 4, (255, 255, 255), -1)
        cv2.circle(result_image, (cx, cy), 3, color, -1)

    # バウンディングボックス描画
    for det in detections:
        class_id = det["class_id"]
        if class_id in WHOLEBODY34_KEYPOINT_CLASSES:
            continue

        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        generation = det.get("generation", -1)
        gender = det.get("gender", -1)
        head_pose = det.get("head_pose", -1)
        handedness = det.get("handedness", -1)

        # 色の決定
        if class_id == 0:  # Body
            if gender == 0:
                color = (255, 0, 0)  # Male: Blue
            elif gender == 1:
                color = (139, 116, 225)  # Female: Pink
            else:
                color = (0, 200, 255)  # Unknown: Yellow-Orange
        elif class_id == 7:  # Head
            color = HEAD_POSE_COLORS.get(head_pose, (216, 67, 21))
        elif class_id == 26:  # Hand
            if handedness == 0:
                color = (0, 128, 0)  # Left: Green
            elif handedness == 1:
                color = (255, 0, 255)  # Right: Magenta
            else:
                color = (0, 255, 0)  # Unknown: Green
        else:
            color = WHOLEBODY34_COLORS.get(class_id, (255, 255, 255))

        # バウンディングボックス
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        # 属性テキスト
        attr_parts = []
        if class_id == 0:
            if generation == 0:
                attr_parts.append("Adult")
            elif generation == 1:
                attr_parts.append("Child")
            if gender == 0:
                attr_parts.append("(M)")
            elif gender == 1:
                attr_parts.append("(F)")
        elif class_id == 7 and head_pose >= 0:
            attr_parts.append(HEAD_POSE_NAMES.get(head_pose, ""))
        elif class_id == 26:
            if handedness == 0:
                attr_parts.append("L")
            elif handedness == 1:
                attr_parts.append("R")

        if "track_id" in det:
            attr_parts.insert(0, f"TID:{det['track_id']}")

        attr_txt = " ".join(attr_parts)
        if attr_txt:
            text_x = x1 if x1 + 50 < image_w else image_w - 50
            text_y = y1 - 10 if y1 - 25 > 0 else 20
            cv2.putText(result_image, attr_txt, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_image, attr_txt, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)

    return result_image


class ObjectDetectionLogic(ComputeLogic):
    """
    物体検出ノードのロジック。
    DEIMv2等のモデルで物体検出を行い、結果をJSON文字列と可視化画像で出力。
    """

    # クラス共有のモデルキャッシュ
    _model_cache: Dict[str, Any] = {}
    _model_errors: Dict[str, str] = {}

    def __init__(self):
        self._tracker = None
        self._last_model_index: int = -1
        self._last_use_motpy: bool = False
        self._last_image = None  # キャンセル時に返す直前の画像
        self._last_result_json: str = json.dumps({"detections": []})  # 最後の有効な結果JSON
        self._last_detections: List[Dict[str, Any]] = []  # 最後の検出結果
        self._last_is_wholebody34: bool = False  # 最後のモデルタイプ
        self._track_id_map: Dict[str, int] = {}  # UUID -> 連番ID のマッピング
        self._next_track_id: int = 1  # 次に割り当てる連番ID

    def reset(self):
        """ノードの状態をリセット"""
        self._tracker = None
        self._last_image = None
        self._last_result_json = json.dumps({"detections": []})
        self._last_detections = []
        self._last_is_wholebody34 = False
        self._track_id_map = {}
        self._next_track_id = 1

    def _get_model_path(self, model_name: str) -> str:
        """モデルファイルのパスを取得"""
        base_path = os.path.dirname(os.path.abspath(__file__))

        model_paths = {
            "DEIMv2-Atto(COCO)": "DEIMv2/model/deimv2_hgnetv2_atto_coco.onnx",
            "DEIMv2-Femto(COCO)": "DEIMv2/model/deimv2_hgnetv2_femto_coco.onnx",
            "DEIMv2-Pico(COCO)": "DEIMv2/model/deimv2_hgnetv2_pico_coco.onnx",
            "DEIMv2-N(COCO)": "DEIMv2/model/deimv2_hgnetv2_n_coco.onnx",
            "DEIMv2-S(COCO)": "DEIMv2/model/deimv2_dinov3_s_coco.onnx",
            "DEIMv2-Wholebody34-Atto": "DEIMv2Wholebody34/model/deimv2_hgnetv2_atto_wholebody34_340query_n_batch_320x320.onnx",
            "DEIMv2-Wholebody34-Femto": "DEIMv2Wholebody34/model/deimv2_hgnetv2_femto_wholebody34_340query_n_batch_416x416.onnx",
            "DEIMv2-Wholebody34-Pico": "DEIMv2Wholebody34/model/deimv2_hgnetv2_pico_wholebody34_340query_n_batch_640x640.onnx",
            "DEIMv2-Wholebody34-N": "DEIMv2Wholebody34/model/deimv2_hgnetv2_n_wholebody34_680query_n_batch_640x640.onnx",
            "DEIMv2-Wholebody34-S": "DEIMv2Wholebody34/model/deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx",
        }

        return os.path.join(base_path, model_paths.get(model_name, ""))

    def _get_providers(self, use_gpu: bool) -> List[str]:
        """使用するプロバイダーリストを取得"""
        if use_gpu and CUDA_AVAILABLE:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_model(self, model_name: str, use_gpu: bool = False) -> Tuple[Optional[Any], Optional[str]]:
        """
        モデルをロード。(model, error_message)を返す。
        """
        # キャッシュキーはモデル名とGPU設定の組み合わせ
        cache_key = f"{model_name}_gpu={use_gpu and CUDA_AVAILABLE}"

        # キャッシュ済みモデルを返す
        if cache_key in self._model_cache:
            return self._model_cache[cache_key], None

        # 以前のエラーを返す
        if cache_key in self._model_errors:
            return None, self._model_errors[cache_key]

        # モデルパス確認
        model_path = self._get_model_path(model_name)
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

        # プロバイダー選択
        providers = self._get_providers(use_gpu)

        # モデルロード
        try:
            if model_name in WHOLEBODY34_MODELS:
                from src.nodes.image.deep_learning.object_detection.DEIMv2Wholebody34.deimv2_wholebody34 import (
                    DEIMv2Wholebody34,
                )
                model = DEIMv2Wholebody34(model_path, providers=providers)
            else:
                from src.nodes.image.deep_learning.object_detection.DEIMv2.deimv2 import DEIMv2
                model = DEIMv2(model_path, providers=providers)
            self._model_cache[cache_key] = model
            return model, None
        except ImportError as e:
            error_msg = f"Import error: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

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
        # トラッカー初期化時にIDマッピングもリセット
        self._track_id_map = {}
        self._next_track_id = 1

    def _get_sequential_track_id(self, uuid_id: str) -> int:
        """UUIDのtrack_idを連番に変換"""
        if uuid_id not in self._track_id_map:
            self._track_id_map[uuid_id] = self._next_track_id
            self._next_track_id += 1
        return self._track_id_map[uuid_id]

    def _detect_single(
        self,
        model,
        image: np.ndarray,
        score_th: float,
        is_wholebody34: bool,
        class_names: Dict[int, str]
    ) -> List[Dict[str, Any]]:
        """単一画像での検出"""
        detections = []

        if is_wholebody34:
            boxes = model(image)
            for box in boxes:
                if box.score >= score_th:
                    detections.append({
                        "class_id": int(box.classid),
                        "class_name": class_names.get(box.classid, "unknown"),
                        "score": float(box.score),
                        "bbox": [int(box.x1), int(box.y1), int(box.x2), int(box.y2)],
                        "generation": box.generation,
                        "gender": box.gender,
                        "handedness": box.handedness,
                        "head_pose": box.head_pose,
                    })
        else:
            bboxes, scores, class_ids = model(image)
            if len(bboxes) > 0:
                for bbox, score, class_id in zip(bboxes, scores, class_ids):
                    if score >= score_th:
                        detections.append({
                            "class_id": int(class_id),
                            "class_name": class_names.get(int(class_id), "unknown"),
                            "score": float(score),
                            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        })

        return detections

    def _detect_with_sahi(
        self,
        model,
        image: np.ndarray,
        score_th: float,
        is_wholebody34: bool,
        class_names: Dict[int, str],
        slice_width: int,
        slice_height: int,
        overlap_ratio: float
    ) -> List[Dict[str, Any]]:
        """SAHIパッケージによるスライス推論"""
        # sahiで画像をスライス
        slice_result = slice_image(
            image=image,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
        )

        all_object_predictions = []

        # 各スライスで検出（キャンセルチェック付き）
        for sliced_image in slice_result.sliced_image_list:
            # スライス間でキャンセルチェック（キャンセル時は空リストを返す）
            if self.is_cancelled():
                return []
            slice_img = sliced_image.image
            shift_x = sliced_image.starting_pixel[0]
            shift_y = sliced_image.starting_pixel[1]

            # 検出実行
            slice_detections = self._detect_single(
                model, slice_img, score_th, is_wholebody34, class_names
            )

            # ObjectPredictionに変換（座標をシフト）
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
                # Wholebody34の属性を保持
                if is_wholebody34:
                    obj_pred.extra_data = {
                        "generation": det.get("generation", -1),
                        "gender": det.get("gender", -1),
                        "handedness": det.get("handedness", -1),
                        "head_pose": det.get("head_pose", -1),
                    }
                all_object_predictions.append(obj_pred)

        # 全体画像での検出も追加
        full_detections = self._detect_single(
            model, image, score_th, is_wholebody34, class_names
        )
        for det in full_detections:
            obj_pred = ObjectPrediction(
                bbox=det["bbox"],
                score=det["score"],
                category_id=det["class_id"],
                category_name=det["class_name"],
            )
            if is_wholebody34:
                obj_pred.extra_data = {
                    "generation": det.get("generation", -1),
                    "gender": det.get("gender", -1),
                    "handedness": det.get("handedness", -1),
                    "head_pose": det.get("head_pose", -1),
                }
            all_object_predictions.append(obj_pred)

        # sahiのNMSで重複除去
        nms_postprocess = NMSPostprocess(
            match_threshold=0.5,
            match_metric="IOU",
        )
        merged_predictions = nms_postprocess(all_object_predictions)

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
            if is_wholebody34 and hasattr(pred, 'extra_data') and pred.extra_data:
                det["generation"] = pred.extra_data.get("generation", -1)
                det["gender"] = pred.extra_data.get("gender", -1)
                det["handedness"] = pred.extra_data.get("handedness", -1)
                det["head_pose"] = pred.extra_data.get("head_pose", -1)
            result_detections.append(det)

        return result_detections

    def _check_cancel_and_return(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """キャンセルされていたら早期リターン用の結果を返す、そうでなければNone"""
        if self.is_cancelled():
            self.clear_cancel()  # キャンセル検出時にフラグをクリア
            # 最後の検出結果を現在の入力画像に描画
            if len(self._last_detections) > 0:
                if self._last_is_wholebody34:
                    return_image = draw_wholebody34_detections(image, self._last_detections)
                else:
                    return_image = draw_detections(image, self._last_detections, coco_class_names)
            else:
                return_image = image
            return {"image": return_image, "result_json": self._last_result_json}
        return None

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
        """
        検出結果にトラッキングを適用。
        トラックのバウンディングボックスとIDを使用した結果を返す。
        """
        # MotpyDetectionオブジェクトのリストを作成
        motpy_detections = []
        for det in detections:
            bbox = det["bbox"]
            motpy_det = MotpyDetection(
                box=[bbox[0], bbox[1], bbox[2], bbox[3]],
                score=det["score"],
                class_id=det["class_id"]
            )
            motpy_detections.append(motpy_det)

        # トラッカーを更新
        self._tracker.step(detections=motpy_detections)
        tracks = self._tracker.active_tracks()

        if len(tracks) == 0:
            return []

        # トラックから結果を生成（検出ではなくトラックのbboxを使用）
        tracked_results = []
        for track in tracks:
            track_bbox = track.box
            track_id = self._get_sequential_track_id(str(track.id))

            # 最もIOUが高い検出を見つけて属性を継承
            best_iou = 0.0
            best_det = None
            for det in detections:
                det_bbox = det["bbox"]
                iou = self._calculate_iou(det_bbox, track_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            # トラック結果を作成
            if best_det is not None and best_iou > 0.1:
                tracked_item = {
                    "class_id": best_det["class_id"],
                    "class_name": best_det.get("class_name", "unknown"),
                    "score": best_det["score"],
                    "bbox": [int(track_bbox[0]), int(track_bbox[1]),
                             int(track_bbox[2]), int(track_bbox[3])],
                    "track_id": track_id,
                }
                # Wholebody34の属性を継承
                for attr in ["generation", "gender", "handedness", "head_pose"]:
                    if attr in best_det:
                        tracked_item[attr] = best_det[attr]
                tracked_results.append(tracked_item)

        return tracked_results

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        image = inputs.get("image")
        if image is None:
            return {"image": None, "result_json": json.dumps({"detections": []})}

        # キャンセルチェック（開始時）
        cancelled_result = self._check_cancel_and_return(image)
        if cancelled_result is not None:
            return cancelled_result

        # プロパティ取得
        use_gpu = bool(properties.get("use_gpu", False)) and CUDA_AVAILABLE
        model_index = int(properties.get("model", 0))
        score_th = float(properties.get("score_threshold", 0.3))
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

        # モデルロード
        model, load_error = self._load_model(model_name, use_gpu)

        # エラーの場合
        if model is None:
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

        # クラス名辞書
        is_wholebody34 = model_name in WHOLEBODY34_MODELS
        class_names = wholebody34_class_names if is_wholebody34 else coco_class_names

        # 検出実行
        if use_sahi:
            detections = self._detect_with_sahi(
                model, image, score_th, is_wholebody34, class_names,
                sahi_slice_w, sahi_slice_h, sahi_overlap
            )
        else:
            detections = self._detect_single(
                model, image, score_th, is_wholebody34, class_names
            )

        # motpyトラッキング
        if use_motpy and MOTPY_AVAILABLE and self._tracker is not None and len(detections) > 0:
            detections = self._apply_tracking(detections)

        # 可視化
        if is_wholebody34:
            result_image = draw_wholebody34_detections(image, detections)
        else:
            result_image = draw_detections(image, detections, class_names)

        # 成功時は結果を保存（ポーズ時に使用）
        self._last_image = result_image
        self._last_detections = detections
        self._last_is_wholebody34 = is_wholebody34

        # 結果を返す
        result = {
            "model": model_name,
            "score_threshold": score_th,
            "detections": detections,
            "count": len(detections),
        }
        result_json = json.dumps(result, ensure_ascii=False)
        self._last_result_json = result_json

        return {"image": result_image, "result_json": result_json}
