"""
Semantic Segmentation ノードの実装。
MediaPipe、PP-LiteSeg、Road Segmentationなどのモデルでセマンティックセグメンテーションを行う。
"""
from typing import Dict, Any, List, Tuple, Optional
from node_editor.node_def import ComputeLogic
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
    "MediaPipe selfie_segmentation (square)",
    "MediaPipe selfie_segmentation (landscape)",
    "MediaPipe Hair Segmentation",
    "MediaPipe Selfie Multiclass",
    "PP-LiteSeg (STDC1)",
    "PP-LiteSeg (STDC2)",
    "Road Segmentation (ADAS 0001)",
]

# MediaPipeセグメンテーション用カラーテーブル
MEDIAPIPE_COLORTABLE = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Red
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (128, 128, 128),  # Gray
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Light Blue
    (128, 255, 0),    # Lime
    (255, 128, 128),  # Light Red
    (128, 128, 0),    # Olive
    (0, 128, 128),    # Teal
    (128, 0, 128),    # Maroon
    (64, 64, 64),     # Dark Gray
]

# Cityscapesカラーマップ（19クラス）
CITYSCAPES_COLORMAP = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
], dtype=np.uint8)


def get_road_color_map(num_classes: int) -> List[int]:
    """Road Segmentation用カラーマップを生成"""
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3 + 2] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    return color_map[3:]


def draw_mediapipe_segmentation(
    image: np.ndarray,
    category_mask: np.ndarray,
) -> np.ndarray:
    """MediaPipeセグメンテーション結果を描画"""
    result = image.copy()

    # 255を除く最大値を取得
    temp_mask = category_mask[category_mask < 255]
    if temp_mask.size > 0:
        max_value = int(np.max(temp_mask)) + 1
    else:
        max_value = 0

    # セグメンテーション色分け
    for index in range(0, max_value):
        mask = np.where(category_mask == index, 0, 1)
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        color_index = index % len(MEDIAPIPE_COLORTABLE)
        color = MEDIAPIPE_COLORTABLE[color_index]
        bg_image[:] = (color[2], color[1], color[0])  # RGB to BGR

        mask = np.stack((mask,) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, result, bg_image)
        result = cv2.addWeighted(result, 0.5, mask_image, 0.5, 1.0)

    return result


def draw_cityscapes_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Cityscapesカラーマップでセグメンテーション結果を描画"""
    # マスクをカラー画像に変換
    colored = CITYSCAPES_COLORMAP[mask % len(CITYSCAPES_COLORMAP)]
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    # ブレンド
    result = cv2.addWeighted(image, 1 - alpha, colored_bgr, alpha, 0)
    return result


def draw_road_segmentation(
    image: np.ndarray,
    segmentation_map: np.ndarray,
    score_threshold: float = 0.5,
) -> np.ndarray:
    """Road Segmentation結果を描画"""
    result = image.copy()
    num_classes = segmentation_map.shape[0]
    color_map = get_road_color_map(num_classes)

    # 閾値処理
    segmentation_map = np.where(segmentation_map > score_threshold, 0, 1)

    for index, mask in enumerate(segmentation_map):
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (
            color_map[index * 3 + 0],
            color_map[index * 3 + 1],
            color_map[index * 3 + 2],
        )

        mask = np.stack((mask,) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, result, bg_image)
        result = cv2.addWeighted(result, 0.5, mask_image, 0.5, 1.0)

    return result


class SemanticSegmentationLogic(ComputeLogic):
    """
    セマンティックセグメンテーションノードのロジック。
    各種モデルでセグメンテーションを行い、可視化画像を出力。
    """

    # クラス共有のモデルキャッシュ
    _mediapipe_cache: Dict[str, Any] = {}
    _onnx_cache: Dict[str, Any] = {}
    _model_errors: Dict[str, str] = {}

    def __init__(self):
        pass

    def reset(self):
        """ノードの状態をリセット"""
        pass

    def _get_model_path(self, model_index: int) -> str:
        """モデルファイルのパスを取得"""
        base_path = os.path.dirname(os.path.abspath(__file__))

        model_paths = {
            0: "MediaPipe/model/selfie_segmenter_float16.tflite",
            1: "MediaPipe/model/selfie_segmenter_landscape_float16.tflite",
            2: "MediaPipe/model/hair_segmenter_float32.tflite",
            3: "MediaPipe/model/selfie_multiclass_256x256_float32.tflite",
            4: "PP-LiteSeg/model/ppliteseg_stdc1.onnx",
            5: "PP-LiteSeg/model/ppliteseg_stdc2.onnx",
            6: "road_segmentation_adas_0001/model/road_segmentation_adas_0001.onnx",
        }

        return os.path.join(base_path, model_paths.get(model_index, ""))

    def _get_providers(self, use_gpu: bool) -> List[str]:
        """使用するプロバイダーリストを取得"""
        if use_gpu and CUDA_AVAILABLE:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_mediapipe_model(self, model_index: int) -> Tuple[Optional[Any], Optional[str]]:
        """MediaPipeモデルをロード"""
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
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                output_category_mask=True,
            )
            segmenter = vision.ImageSegmenter.create_from_options(options)
            self._mediapipe_cache[cache_key] = segmenter
            return segmenter, None
        except Exception as e:
            error_msg = f"Failed to load MediaPipe model: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

    def _load_onnx_model(
        self,
        model_index: int,
        use_gpu: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """ONNXモデルをロード"""
        cache_key = f"onnx_{model_index}_gpu={use_gpu and CUDA_AVAILABLE}"

        if cache_key in self._onnx_cache:
            return self._onnx_cache[cache_key], None

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
            input_detail = session.get_inputs()[0]
            output_detail = session.get_outputs()[0]

            model_info = {
                "session": session,
                "input_name": input_detail.name,
                "input_shape": input_detail.shape,
                "output_name": output_detail.name,
                "model_type": "ppliteseg" if model_index in [4, 5] else "road",
            }
            self._onnx_cache[cache_key] = model_info
            return model_info, None
        except Exception as e:
            error_msg = f"Failed to load ONNX model: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

    def _segment_mediapipe(
        self,
        segmenter: Any,
        image: np.ndarray,
    ) -> np.ndarray:
        """MediaPipeでセグメンテーション実行"""
        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGBA),
        )
        result = segmenter.segment(rgb_frame)
        category_mask = result.category_mask
        return np.squeeze(category_mask.numpy_view())

    def _segment_ppliteseg(
        self,
        model_info: Dict[str, Any],
        image: np.ndarray,
    ) -> np.ndarray:
        """PP-LiteSegでセグメンテーション実行"""
        session = model_info["session"]
        input_name = model_info["input_name"]
        output_name = model_info["output_name"]

        original_h, original_w = image.shape[:2]
        input_size = (512, 1024)  # (H, W)

        # 前処理
        resized = cv2.resize(image, (input_size[1], input_size[0]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
        tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # 推論
        output = session.run([output_name], {input_name: tensor})[0]

        # 後処理
        prediction = np.argmax(output[0], axis=0)
        mask = cv2.resize(
            prediction.astype(np.uint8),
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )

        return mask

    def _segment_road(
        self,
        model_info: Dict[str, Any],
        image: np.ndarray,
    ) -> np.ndarray:
        """Road Segmentationでセグメンテーション実行"""
        session = model_info["session"]
        input_name = model_info["input_name"]
        input_shape = model_info["input_shape"]

        original_h, original_w = image.shape[:2]

        # 前処理
        input_h, input_w = input_shape[1], input_shape[2]
        resized = cv2.resize(image, (input_w, input_h))
        input_tensor = np.expand_dims(resized, axis=0).astype('float32')

        # 推論
        result = session.run(None, {input_name: input_tensor})

        # 後処理
        segmentation_map = np.squeeze(result[0])
        segmentation_map = cv2.resize(
            segmentation_map,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        segmentation_map = segmentation_map.transpose(2, 0, 1)

        return segmentation_map

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
            return {"image": image}

        # プロパティ取得
        model_index = int(properties.get("model", 0))
        use_gpu = bool(properties.get("use_gpu", False)) and CUDA_AVAILABLE

        # MediaPipeモデル (0-3)
        is_mediapipe = model_index < 4

        if is_mediapipe:
            segmenter, load_error = self._load_mediapipe_model(model_index)
            if segmenter is None:
                raise RuntimeError(load_error or "Failed to load MediaPipe model")

            category_mask = self._segment_mediapipe(segmenter, image)
            result_image = draw_mediapipe_segmentation(image, category_mask)

        elif model_index in [4, 5]:
            # PP-LiteSeg
            model_info, load_error = self._load_onnx_model(model_index, use_gpu)
            if model_info is None:
                raise RuntimeError(load_error or "Failed to load ONNX model")

            mask = self._segment_ppliteseg(model_info, image)
            result_image = draw_cityscapes_segmentation(image, mask)

        else:
            # Road Segmentation
            model_info, load_error = self._load_onnx_model(model_index, use_gpu)
            if model_info is None:
                raise RuntimeError(load_error or "Failed to load ONNX model")

            segmentation_map = self._segment_road(model_info, image)
            result_image = draw_road_segmentation(image, segmentation_map)

        return {"image": result_image}
