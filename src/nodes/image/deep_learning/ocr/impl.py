"""
OCRノードの実装。
PaddleOCR v5を使用して画像からテキストを認識する。
"""
from typing import Dict, Any, List, Tuple, Optional
from node_editor.node_def import ComputeLogic
import json
import os
import math

import numpy as np
import cv2

# GPU利用可能かチェック
try:
    import onnxruntime
    CUDA_AVAILABLE = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
except Exception:
    CUDA_AVAILABLE = False
    onnxruntime = None

# モデル設定
MODEL_NAMES = [
    "PaddleOCRv5(Mobile)",
    "PaddleOCRv5(Server)",
]

# 検出パラメータ
DET_LIMIT_SIDE = 960
DET_LIMIT_TYPE = "max"
DET_THRESH = 0.3
DET_BOX_THRESH = 0.6
DET_UNCLIP_RATIO = 1.5

# 認識パラメータ
REC_IMAGE_HEIGHT = 48
REC_IMAGE_WIDTH = 320


# ========== 前処理 ==========
def resize_for_det(img: np.ndarray, limit_side: int = 960, limit_type: str = "max"):
    """検出用に画像をリサイズ"""
    h, w = img.shape[:2]

    # 小さすぎる画像のパディング
    if h + w < 64:
        pad_h = max(32, h)
        pad_w = max(32, w)
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded[:h, :w, :] = img
        img = padded
        h, w = img.shape[:2]

    # リサイズ比率の計算
    if limit_type == "max":
        if max(h, w) > limit_side:
            ratio = float(limit_side) / max(h, w)
        else:
            ratio = 1.0
    elif limit_type == "min":
        if min(h, w) < limit_side:
            ratio = float(limit_side) / min(h, w)
        else:
            ratio = 1.0
    else:
        ratio = float(limit_side) / max(h, w)

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    # 32の倍数に調整
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    resized = cv2.resize(img, (resize_w, resize_h))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return resized, (h, w, ratio_h, ratio_w)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """画像の正規化（検出用）"""
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img


def preprocess_det(img: np.ndarray, limit_side: int = 960, limit_type: str = "max"):
    """検出モデル用の前処理"""
    resized, shape_info = resize_for_det(img, limit_side, limit_type)
    normalized = normalize_image(resized)

    # HWC -> CHW
    transposed = normalized.transpose(2, 0, 1)

    # バッチ次元を追加
    batched = np.expand_dims(transposed, axis=0).astype(np.float32)

    return batched, shape_info


def resize_norm_img_rec(
    img: np.ndarray,
    img_height: int = 48,
    img_width: int = 320,
    max_wh_ratio: float = None
) -> np.ndarray:
    """認識用に画像をリサイズして正規化"""
    img_channel = 3
    if max_wh_ratio is not None:
        img_width = int(img_height * max_wh_ratio)

    h, w = img.shape[:2]
    ratio = w / float(h)

    if math.ceil(img_height * ratio) > img_width:
        resized_w = img_width
    else:
        resized_w = int(math.ceil(img_height * ratio))

    resized_image = cv2.resize(img, (resized_w, img_height))
    resized_image = resized_image.astype("float32")

    # 正規化: (x / 255 - 0.5) / 0.5
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5

    # パディング
    padding_im = np.zeros((img_channel, img_height, img_width), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image

    return padding_im


def preprocess_rec_batch(
    img_list: List[np.ndarray],
    img_height: int = 48,
    img_width: int = 320
) -> np.ndarray:
    """認識モデル用のバッチ前処理"""
    # 各画像のアスペクト比を計算
    width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

    # 最大アスペクト比を計算
    max_wh_ratio = img_width / img_height
    for w_ratio in width_list:
        max_wh_ratio = max(max_wh_ratio, w_ratio)

    # 動的な画像幅を計算
    dynamic_width = int(img_height * max_wh_ratio)

    # バッチを作成
    norm_img_batch = []
    for img in img_list:
        norm_img = resize_norm_img_rec(img, img_height, dynamic_width, max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)

    norm_img_batch = np.concatenate(norm_img_batch)
    return norm_img_batch.astype(np.float32)


def preprocess_rec(img: np.ndarray, target_height: int = 48, max_width: int = 320):
    """認識モデル用の前処理（単一画像）"""
    return preprocess_rec_batch([img], target_height, max_width)


# ========== 後処理 (検出) ==========
def unclip(box: np.ndarray, unclip_ratio: float = 1.5) -> np.ndarray:
    """検出ボックスを拡張"""
    box = np.array(box, dtype=np.float32)

    if len(box) < 3:
        return box.reshape(1, -1, 2)

    # 最小外接矩形を取得
    rect = cv2.minAreaRect(box.reshape(-1, 1, 2))
    center, (width, height), angle = rect

    # 幅と高さが0の場合は元のボックスを返す
    if width < 1e-6 or height < 1e-6:
        return box.reshape(1, -1, 2)

    # 面積と周長を計算
    area = width * height
    perimeter = 2 * (width + height)

    # オフセット距離を計算
    distance = area * unclip_ratio / perimeter
    distance *= 1.10

    # 各辺をdistance分だけ外側に移動
    new_width = width + 2 * distance
    new_height = height + 2 * distance

    # 拡張した矩形を再構築
    new_rect = (center, (new_width, new_height), angle)
    expanded = cv2.boxPoints(new_rect)

    return expanded.astype(np.float32).reshape(1, -1, 2)


def box_score_fast(bitmap: np.ndarray, _box: np.ndarray) -> float:
    """高速なボックススコア計算"""
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
    return cv2.mean(bitmap[ymin: ymax + 1, xmin: xmax + 1], mask)[0]


def get_mini_boxes(contour: np.ndarray) -> Tuple[np.ndarray, float]:
    """輪郭から最小外接矩形を取得"""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0

    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return np.array(box), min(bounding_box[1])


def postprocess_det(
    pred: np.ndarray,
    shape_info: Tuple,
    thresh: float = 0.3,
    box_thresh: float = 0.6,
    unclip_ratio: float = 1.5,
    max_candidates: int = 1000
) -> Tuple[List[np.ndarray], List[float]]:
    """検出結果の後処理"""
    ori_h, ori_w, ratio_h, ratio_w = shape_info
    pred = pred[0, 0]

    height, width = pred.shape

    # 二値化
    segmentation = pred > thresh
    mask = (segmentation * 255).astype(np.uint8)

    # 輪郭検出
    outs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(outs) == 3:
        _, contours, _ = outs
    else:
        contours, _ = outs

    num_contours = min(len(contours), max_candidates)

    boxes = []
    scores = []

    for index in range(num_contours):
        contour = contours[index]

        # 最小外接矩形
        points, sside = get_mini_boxes(contour)
        if sside < 3:
            continue
        points = np.array(points)

        # スコア計算
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue

        # ボックスを拡張
        box = unclip(points, unclip_ratio)
        if box is None or len(box) == 0:
            continue
        box = box.reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)
        if sside < 5:
            continue
        box = np.array(box)

        # 元のサイズにスケール
        box[:, 0] = np.clip(np.round(box[:, 0] / width * ori_w), 0, ori_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * ori_h), 0, ori_h)

        boxes.append(box.astype("int32"))
        scores.append(score)

    return boxes, scores


# ========== 後処理 (認識) ==========
def load_dict(dict_path: str) -> List[str]:
    """文字辞書を読み込む"""
    with open(dict_path, "r", encoding="utf-8") as f:
        chars = [line.strip() for line in f]
    chars = ["blank"] + chars
    return chars


def postprocess_rec(pred: np.ndarray, char_dict: List[str]):
    """認識結果の後処理（CTCデコード）"""
    preds_idx = pred.argmax(axis=2)
    preds_prob = pred.max(axis=2)

    results = []
    batch_size = len(preds_idx)

    for batch_idx in range(batch_size):
        text_index = preds_idx[batch_idx]
        text_prob = preds_prob[batch_idx]

        char_list = []
        conf_list = []

        selection = np.ones(len(text_index), dtype=bool)
        selection[1:] = text_index[1:] != text_index[:-1]
        selection &= text_index != 0

        for i, selected in enumerate(selection):
            if selected and text_index[i] < len(char_dict):
                char_list.append(char_dict[text_index[i]])
                conf_list.append(text_prob[i])

        text = "".join(char_list)
        confidence = float(np.mean(conf_list)) if len(conf_list) > 0 else 0.0
        results.append((text, confidence))

    if batch_size == 1:
        return results[0]
    return results


# ========== ユーティリティ ==========
def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """検出ボックスから画像を切り出して回転補正"""
    points = np.array(points, dtype=np.float32)

    width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )

    dst_points = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )

    M = cv2.getPerspectiveTransform(points, dst_points)
    cropped = cv2.warpPerspective(
        img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
    )

    if height > width * 1.5:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    return cropped


def sort_boxes(boxes: List[np.ndarray]) -> List[np.ndarray]:
    """ボックスを上から下、左から右の順にソート"""
    if len(boxes) == 0:
        return []

    boxes_with_y = [(box, np.mean(box[:, 1])) for box in boxes]
    boxes_with_y.sort(key=lambda x: x[1])

    return [box for box, _ in boxes_with_y]


# ========== OCRエンジン ==========
class PaddleOCREngine:
    """PaddleOCR v5 ONNXエンジン"""

    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        dict_path: str,
        providers: List[str]
    ):
        self.det_session = onnxruntime.InferenceSession(
            det_model_path, providers=providers
        )
        self.rec_session = onnxruntime.InferenceSession(
            rec_model_path, providers=providers
        )
        self.char_dict = load_dict(dict_path)

    def detect(self, img: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """テキスト領域を検出"""
        input_data, shape_info = preprocess_det(img, DET_LIMIT_SIDE, DET_LIMIT_TYPE)

        input_name = self.det_session.get_inputs()[0].name
        output = self.det_session.run(None, {input_name: input_data})[0]

        boxes, scores = postprocess_det(
            output, shape_info, DET_THRESH, DET_BOX_THRESH, DET_UNCLIP_RATIO
        )

        return boxes, scores

    def recognize(self, img: np.ndarray) -> Tuple[str, float]:
        """テキストを認識"""
        input_data = preprocess_rec(img, REC_IMAGE_HEIGHT, REC_IMAGE_WIDTH)

        input_name = self.rec_session.get_inputs()[0].name
        output = self.rec_session.run(None, {input_name: input_data})[0]

        text, confidence = postprocess_rec(output, self.char_dict)

        return text, confidence

    def ocr(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """OCR実行（検出 + 認識）"""
        boxes, det_scores = self.detect(img)

        if len(boxes) == 0:
            return []

        boxes = sort_boxes(boxes)

        results = []
        for box in boxes:
            cropped = get_rotate_crop_image(img, box)
            text, confidence = self.recognize(cropped)

            if text:
                results.append({
                    "box": box.tolist(),
                    "text": text,
                    "confidence": confidence
                })

        return results


def draw_ocr_results(
    image: np.ndarray,
    results: List[Dict[str, Any]]
) -> np.ndarray:
    """OCR結果を画像に描画（検出ボックス、ID、スコア）"""
    result_image = image.copy()

    for idx, result in enumerate(results):
        box = np.array(result["box"], dtype=np.int32)
        conf = result["confidence"]
        detection_id = idx + 1

        # ボックス描画
        cv2.polylines(result_image, [box], True, (0, 255, 0), 2)

        # ラベル作成（ID: スコア）
        label = f"#{detection_id} {conf:.2f}"

        # ラベル位置
        x, y = box[0]
        label_y = max(y - 8, 20)

        # ラベル背景
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            result_image,
            (x, label_y - label_h - baseline),
            (x + label_w, label_y + baseline),
            (0, 255, 0),
            -1
        )

        # ラベルテキスト
        cv2.putText(
            result_image,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    return result_image


class OCRLogic(ComputeLogic):
    """
    OCRノードのロジック。
    PaddleOCR v5でテキスト検出・認識を行い、結果をJSON文字列と可視化画像で出力。
    """

    # クラス共有のモデルキャッシュ
    _model_cache: Dict[str, PaddleOCREngine] = {}
    _model_errors: Dict[str, str] = {}

    def __init__(self):
        self._last_model_index: int = -1
        self._last_image: Optional[np.ndarray] = None
        self._last_result_json: str = json.dumps({"results": []})
        self._last_text_list: str = ""
        self._last_results: List[Dict[str, Any]] = []

    def reset(self):
        """ノードの状態をリセット"""
        self._last_image = None
        self._last_result_json = json.dumps({"results": []})
        self._last_text_list = ""
        self._last_results = []

    def _get_model_paths(self, model_name: str) -> Tuple[str, str, str]:
        """モデルファイルのパスを取得（det, rec, dict）"""
        base_path = os.path.dirname(os.path.abspath(__file__))

        if model_name == "PaddleOCRv5(Mobile)":
            det_path = os.path.join(
                base_path, "PaddleOCRv5/model/PP-OCRv5_mobile_det_infer.onnx"
            )
            rec_path = os.path.join(
                base_path, "PaddleOCRv5/model/PP-OCRv5_mobile_rec_infer.onnx"
            )
        else:  # Server
            det_path = os.path.join(
                base_path, "PaddleOCRv5/model/PP-OCRv5_server_det_infer.onnx"
            )
            rec_path = os.path.join(
                base_path, "PaddleOCRv5/model/PP-OCRv5_server_rec_infer.onnx"
            )

        dict_path = os.path.join(base_path, "PaddleOCRv5/ppocrv5_dict.txt")

        return det_path, rec_path, dict_path

    def _get_providers(self, use_gpu: bool) -> List[str]:
        """使用するプロバイダーリストを取得"""
        if use_gpu and CUDA_AVAILABLE:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_model(
        self,
        model_name: str,
        use_gpu: bool = False
    ) -> Tuple[Optional[PaddleOCREngine], Optional[str]]:
        """モデルをロード。(model, error_message)を返す。"""
        # キャッシュキーはモデル名とGPU設定の組み合わせ
        cache_key = f"{model_name}_gpu={use_gpu and CUDA_AVAILABLE}"

        # キャッシュ済みモデルを返す
        if cache_key in self._model_cache:
            return self._model_cache[cache_key], None

        # 以前のエラーを返す
        if cache_key in self._model_errors:
            return None, self._model_errors[cache_key]

        # モデルパス確認
        det_path, rec_path, dict_path = self._get_model_paths(model_name)

        if not os.path.exists(det_path):
            error_msg = f"Detection model not found: {det_path}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

        if not os.path.exists(rec_path):
            error_msg = f"Recognition model not found: {rec_path}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

        if not os.path.exists(dict_path):
            error_msg = f"Dictionary not found: {dict_path}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

        # プロバイダー選択
        providers = self._get_providers(use_gpu)

        # モデルロード
        try:
            model = PaddleOCREngine(det_path, rec_path, dict_path, providers)
            self._model_cache[cache_key] = model
            return model, None
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

    def _check_cancel_and_return(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """キャンセルされていたら早期リターン用の結果を返す"""
        if self.is_cancelled():
            self.clear_cancel()
            # 最後の検出結果を現在の入力画像に描画
            if len(self._last_results) > 0:
                return_image = draw_ocr_results(image, self._last_results)
            else:
                return_image = image
            return {
                "image": return_image,
                "result_json": self._last_result_json,
                "text_list": self._last_text_list
            }
        return None

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        image = inputs.get("image")
        if image is None:
            return {
                "image": None,
                "result_json": json.dumps({"results": []}),
                "text_list": ""
            }

        # キャンセルチェック（開始時）
        cancelled_result = self._check_cancel_and_return(image)
        if cancelled_result is not None:
            return cancelled_result

        # プロパティ取得
        use_gpu = bool(properties.get("use_gpu", False)) and CUDA_AVAILABLE
        model_index = int(properties.get("model", 0))

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

        # OCR実行
        ocr_results = model.ocr(image)

        # 結果にIDを付与
        results_with_id = []
        for idx, result in enumerate(ocr_results):
            results_with_id.append({
                "id": idx + 1,
                "text": result["text"],
                "confidence": result["confidence"],
                "box": result["box"]
            })

        # 可視化
        result_image = draw_ocr_results(image, ocr_results)

        # text_listを生成（ID: テキスト形式）
        text_list_lines = [f"{r['id']}: {r['text']}" for r in results_with_id]
        text_list = "\n".join(text_list_lines)

        # 成功時は結果を保存
        self._last_image = result_image
        self._last_results = ocr_results
        self._last_text_list = text_list

        # 結果を返す
        result = {
            "model": model_name,
            "results": results_with_id,
            "count": len(results_with_id),
        }
        result_json = json.dumps(result, ensure_ascii=False)
        self._last_result_json = result_json

        return {
            "image": result_image,
            "result_json": result_json,
            "text_list": text_list
        }
