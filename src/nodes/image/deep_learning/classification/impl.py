"""
Classification ノードの実装。
MobileNetV3、EfficientNetLite4 などのモデルで画像分類を行う。
物体検出結果と連携して、検出領域ごとの分類も可能。
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

# クラス名辞書をインポート
from src.nodes.image.deep_learning.classification.imagenet_class_names import imagenet_class_names

# モデル設定
MODEL_NAMES = [
    "MobileNetV3-Small",
    "MobileNetV3-Large",
    "EfficientNetLite4",
    "MNIST",
]

# モデルごとの入力サイズ
MODEL_INPUT_SIZES = {
    "MobileNetV3-Small": (224, 224),
    "MobileNetV3-Large": (224, 224),
    "EfficientNetLite4": (224, 224),
    "MNIST": (28, 28),
}

# MNISTクラス名（0-9の数字）
MNIST_CLASS_NAMES = {i: str(i) for i in range(10)}


def draw_classification_results(
    image: np.ndarray,
    results: List[Dict[str, Any]],
    position: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    """
    分類結果を画像に描画（左上にTop-K表示）

    Args:
        image: 入力画像
        results: 分類結果のリスト（class_name, score）
        position: テキスト開始位置 (x, y)

    Returns:
        描画後の画像
    """
    result_image = image.copy()
    x, y = position
    line_height = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    for i, res in enumerate(results):
        class_name = res["class_name"]
        score = res["score"]
        text = f"#{i+1}: {class_name} ({score:.1%})"

        # 背景を描画
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(
            result_image,
            (x - 2, y - text_h - 2),
            (x + text_w + 2, y + baseline + 2),
            (0, 0, 0),
            -1,
        )

        # テキストを描画
        cv2.putText(
            result_image,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height

    return result_image


def draw_detection_classifications(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    classifications: List[List[Dict[str, Any]]],
) -> np.ndarray:
    """
    検出領域ごとの分類結果を画像に描画

    Args:
        image: 入力画像
        detections: 検出結果のリスト（bbox含む）
        classifications: 各検出に対する分類結果のリスト

    Returns:
        描画後の画像
    """
    result_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for det, cls_results in zip(detections, classifications):
        if len(cls_results) == 0:
            continue

        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox

        # 検出クラス名（Object Detection由来）
        det_class = det.get("class_name", "")

        # Top-1の分類結果
        top1 = cls_results[0]
        cls_name = top1["class_name"]
        cls_score = top1["score"]

        # バウンディングボックス描画
        color = (0, 255, 255)  # Yellow
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)

        # ラベル作成（検出クラス + 分類結果）
        if det_class:
            label = f"{det_class}: {cls_name} ({cls_score:.1%})"
        else:
            label = f"{cls_name} ({cls_score:.1%})"

        # ラベル背景
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(
            result_image,
            (x1, y1 - label_h - baseline - 4),
            (x1 + label_w + 4, y1),
            color,
            -1,
        )

        # ラベルテキスト
        cv2.putText(
            result_image,
            label,
            (x1 + 2, y1 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return result_image


class ClassificationLogic(ComputeLogic):
    """
    画像分類ノードのロジック。
    MobileNetV3等のモデルで画像分類を行い、結果をJSON文字列と可視化画像で出力。
    """

    # クラス共有のモデルキャッシュ
    _model_cache: Dict[str, Any] = {}
    _model_errors: Dict[str, str] = {}

    def __init__(self):
        self._last_model_index: int = -1
        self._last_result_json: str = json.dumps({"classifications": []})
        self._last_classifications: List[Dict[str, Any]] = []

    def reset(self):
        """ノードの状態をリセット"""
        self._last_result_json = json.dumps({"classifications": []})
        self._last_classifications = []

    def _get_model_path(self, model_name: str) -> str:
        """モデルファイルのパスを取得"""
        base_path = os.path.dirname(os.path.abspath(__file__))

        model_paths = {
            "MobileNetV3-Small": "MobileNetV3/model/MobileNetV3Small.onnx",
            "MobileNetV3-Large": "MobileNetV3/model/MobileNetV3Large.onnx",
            "EfficientNetLite4": "EfficientNetLite4/model/efficientnet-lite4-11.onnx",
            "MNIST": "MNIST/model/mnist-12.onnx",
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
            import onnxruntime
            session = onnxruntime.InferenceSession(
                model_path,
                providers=providers,
            )
            input_size = MODEL_INPUT_SIZES.get(model_name, (224, 224))
            model_info = {
                "session": session,
                "input_name": session.get_inputs()[0].name,
                "input_size": input_size,
            }
            self._model_cache[cache_key] = model_info
            return model_info, None
        except ImportError as e:
            error_msg = f"Import error: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

    def _classify_image(
        self,
        model_info: Dict[str, Any],
        image: np.ndarray,
        model_name: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        単一画像の分類を実行

        Args:
            model_info: モデル情報（session, input_name, input_size）
            image: 入力画像（BGR形式）
            model_name: モデル名
            top_k: 上位K件を返す

        Returns:
            分類結果のリスト
        """
        session = model_info["session"]
        input_name = model_info["input_name"]
        input_size = model_info["input_size"]
        is_mnist = model_name == "MNIST"

        # 前処理
        input_image = cv2.resize(image, (input_size[1], input_size[0]))

        if is_mnist:
            # MNIST: グレースケール変換、正規化、形状変換
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            input_image = input_image.astype("float32") / 255.0
            input_image = np.expand_dims(input_image, axis=0)  # (1, 28, 28)
            input_image = np.expand_dims(input_image, axis=0)  # (1, 1, 28, 28)
        else:
            # ImageNet系: BGR->RGB、float32変換
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = np.expand_dims(input_image, axis=0).astype("float32")

        # 推論実行
        result = session.run(None, {input_name: input_image})

        # 結果をソート
        result = np.array(result).squeeze()
        sorted_indices = np.argsort(result)[::-1][:top_k]
        class_scores = result[sorted_indices]
        class_ids = sorted_indices

        # クラス名辞書を選択
        class_names = MNIST_CLASS_NAMES if is_mnist else imagenet_class_names

        # 結果を辞書形式に変換
        classifications = []
        for class_id, score in zip(class_ids, class_scores):
            class_name = class_names.get(int(class_id), "unknown")
            classifications.append({
                "class_id": int(class_id),
                "class_name": class_name,
                "score": float(score),
            })

        return classifications

    def _crop_detection(
        self,
        image: np.ndarray,
        bbox: List[int],
    ) -> Optional[np.ndarray]:
        """
        検出領域を切り出し

        Args:
            image: 入力画像
            bbox: バウンディングボックス [x1, y1, x2, y2]

        Returns:
            切り出した画像、または領域が無効な場合はNone
        """
        h, w = image.shape[:2]
        x1 = max(0, min(bbox[0], w - 1))
        y1 = max(0, min(bbox[1], h - 1))
        x2 = max(0, min(bbox[2], w))
        y2 = max(0, min(bbox[3], h))

        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2].copy()

    def _check_cancel_and_return(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """キャンセルされていたら早期リターン用の結果を返す、そうでなければNone"""
        if self.is_cancelled():
            self.clear_cancel()
            # 最後の分類結果を使用
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
            return {"image": None, "result_json": json.dumps({"classifications": []})}

        # キャンセルチェック
        cancelled_result = self._check_cancel_and_return(image)
        if cancelled_result is not None:
            return cancelled_result

        # プロパティ取得
        model_index = int(properties.get("model", 0))
        top_k = int(properties.get("top_k", 5))
        use_gpu = bool(properties.get("use_gpu", False)) and CUDA_AVAILABLE

        # モデル名取得
        if model_index < 0 or model_index >= len(MODEL_NAMES):
            model_index = 0
        model_name = MODEL_NAMES[model_index]

        # モデルロード
        model_info, load_error = self._load_model(model_name, use_gpu)

        # エラーの場合
        if model_info is None:
            error_msg = load_error or f"Model not found: {model_name}"
            raise RuntimeError(error_msg)

        # 検出JSON入力を解析
        detection_json = inputs.get("detection_json")
        detections = []
        if detection_json:
            try:
                detection_data = json.loads(detection_json)
                detections = detection_data.get("detections", [])
            except (json.JSONDecodeError, TypeError):
                detections = []

        # 分類実行
        result_image = image.copy()
        is_mnist = model_name == "MNIST"

        if len(detections) > 0:
            # 検出領域ごとに分類
            all_classifications = []

            for det in detections:
                # キャンセルチェック
                if self.is_cancelled():
                    self.clear_cancel()
                    return {"image": image, "result_json": self._last_result_json}

                bbox = det.get("bbox", [])
                if len(bbox) != 4:
                    all_classifications.append([])
                    continue

                cropped = self._crop_detection(image, bbox)
                if cropped is None or cropped.size == 0:
                    all_classifications.append([])
                    continue

                cls_results = self._classify_image(model_info, cropped, model_name, top_k)
                all_classifications.append(cls_results)

            # 検出領域に分類結果を重畳（MNIST以外）
            if not is_mnist:
                result_image = draw_detection_classifications(image, detections, all_classifications)

            # 結果を構築
            result = {
                "model": model_name,
                "mode": "detection",
                "detection_count": len(detections),
                "classifications": [
                    {
                        "detection": {
                            "class_name": det.get("class_name", ""),
                            "bbox": det.get("bbox", []),
                        },
                        "top_k": cls_list,
                    }
                    for det, cls_list in zip(detections, all_classifications)
                ],
            }
        else:
            # 全体画像を分類
            classifications = self._classify_image(model_info, image, model_name, top_k)
            # 結果を重畳（MNIST以外）
            if not is_mnist:
                result_image = draw_classification_results(image, classifications)

            # 結果を構築
            result = {
                "model": model_name,
                "mode": "whole_image",
                "top_k": classifications,
            }

        result_json = json.dumps(result, ensure_ascii=False)
        self._last_result_json = result_json

        return {"image": result_image, "result_json": result_json}
