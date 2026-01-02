"""
Monocular Depth Estimation ノードの実装。
Lite-Mono、RT-MonoDepth、Depth-Anything-V2 モデルで単眼深度推定を行う。
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

# ImageNet正規化パラメータ（Depth-Anything-V2用）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# モデル設定（Lite-Mono → RT-MonoDepth → Depth-Anything-V2 の順）
# (model_path, height, width, model_type)
# model_type: "standard" = 通常の正規化, "imagenet" = ImageNet正規化
MODEL_CONFIGS = [
    # Lite-Mono
    # 0: Lite-Mono-Tiny 640x192
    ("Lite-Mono/model/lite-mono-tiny_640x192.onnx", 192, 640, "standard"),
    # 1: Lite-Mono-Tiny 1024x320
    ("Lite-Mono/model/lite-mono-tiny_1024x320.onnx", 320, 1024, "standard"),
    # 2: Lite-Mono-Small 640x192
    ("Lite-Mono/model/lite-mono-small_640x192.onnx", 192, 640, "standard"),
    # 3: Lite-Mono-Small 1024x320
    ("Lite-Mono/model/lite-mono-small_1024x320.onnx", 320, 1024, "standard"),
    # 4: Lite-Mono 640x192 - デフォルト
    ("Lite-Mono/model/lite-mono_640x192.onnx", 192, 640, "standard"),
    # 5: Lite-Mono 1024x320
    ("Lite-Mono/model/lite-mono_1024x320.onnx", 320, 1024, "standard"),
    # 6: Lite-Mono-8M 640x192
    ("Lite-Mono/model/lite-mono-8m_640x192.onnx", 192, 640, "standard"),
    # 7: Lite-Mono-8M 1024x320
    ("Lite-Mono/model/lite-mono-8m_1024x320.onnx", 320, 1024, "standard"),
    # Depth-Anything-V2
    # 8: Depth Anything V2 ViT-S
    ("Depth-Anything-V2/model/depth_anything_v2_vits.onnx", 518, 518, "imagenet"),
    # 9: Depth Anything V2 ViT-B
    ("Depth-Anything-V2/model/depth_anything_v2_vitb.onnx", 518, 518, "imagenet"),
    # 10: Depth Anything V2 ViT-L
    ("Depth-Anything-V2/model/depth_anything_v2_vitl.onnx", 518, 518, "imagenet"),
]

MODEL_NAMES = [
    "Lite-Mono-Tiny 640x192",
    "Lite-Mono-Tiny 1024x320",
    "Lite-Mono-Small 640x192",
    "Lite-Mono-Small 1024x320",
    "Lite-Mono 640x192",
    "Lite-Mono 1024x320",
    "Lite-Mono-8M 640x192",
    "Lite-Mono-8M 1024x320",
    "Depth Anything V2 ViT-S",
    "Depth Anything V2 ViT-B",
    "Depth Anything V2 ViT-L",
]

class MonocularDepthEstimationLogic(ComputeLogic):
    """
    単眼深度推定ノードのロジック。
    Lite-Mono、RT-MonoDepth、Depth-Anything-V2 モデルで深度マップを生成する。
    """

    # クラス共有のモデルキャッシュ
    _model_cache: Dict[str, Any] = {}
    _model_errors: Dict[str, str] = {}

    def __init__(self):
        self._last_model_index: int = -1
        self._last_image: Optional[np.ndarray] = None

    def reset(self):
        """ノードの状態をリセット"""
        self._last_image = None

    def _get_model_info(self, model_index: int) -> Tuple[str, int, int, str]:
        """モデルファイルのパス、入力サイズ、モデルタイプを取得"""
        base_path = os.path.dirname(os.path.abspath(__file__))

        if model_index < 0 or model_index >= len(MODEL_CONFIGS):
            model_index = 4  # デフォルト: Lite-Mono 640x192

        rel_path, height, width, model_type = MODEL_CONFIGS[model_index]
        full_path = os.path.join(base_path, rel_path)
        return full_path, height, width, model_type

    def _get_providers(self, use_gpu: bool) -> List[str]:
        """使用するプロバイダーリストを取得"""
        if use_gpu and CUDA_AVAILABLE:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_model(
        self, model_index: int, use_gpu: bool = False
    ) -> Tuple[Optional[Any], Optional[str]]:
        """モデルをロード。(session, error_message)を返す。"""
        # Depth-Anything-V2は入力サイズが異なるだけで同じモデルファイルを使うため
        # キャッシュキーにはモデルパスを使用
        model_path, _, _, _ = self._get_model_info(model_index)
        cache_key = f"{model_path}_gpu={use_gpu and CUDA_AVAILABLE}"

        # キャッシュ済みモデルを返す
        if cache_key in self._model_cache:
            return self._model_cache[cache_key], None

        # 以前のエラーを返す
        if cache_key in self._model_errors:
            return None, self._model_errors[cache_key]

        # モデルパス確認
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

        # プロバイダー選択
        providers = self._get_providers(use_gpu)

        # モデルロード
        try:
            import onnxruntime
            session = onnxruntime.InferenceSession(model_path, providers=providers)
            self._model_cache[cache_key] = session
            return session, None
        except ImportError as e:
            error_msg = f"Import error: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            self._model_errors[cache_key] = error_msg
            return None, error_msg

    def _preprocess_standard(
        self, frame: np.ndarray, target_height: int, target_width: int
    ) -> np.ndarray:
        """
        標準前処理: リサイズ、正規化、CHW変換（Lite-Mono、RT-MonoDepth用）

        Args:
            frame: OpenCVフレーム (H, W, 3) BGR uint8
            target_height: モデル入力高さ
            target_width: モデル入力幅

        Returns:
            input_tensor: (1, 3, H, W) float32 [0, 1]
        """
        # リサイズ
        resized = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 正規化 (0-255 -> 0-1)
        normalized = rgb.astype(np.float32) / 255.0

        # HWC -> CHW、バッチ次元追加 (1, 3, H, W)
        input_tensor = normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor

    def _preprocess_imagenet(
        self, frame: np.ndarray, target_height: int, target_width: int
    ) -> np.ndarray:
        """
        ImageNet正規化前処理（Depth-Anything-V2用）

        Args:
            frame: OpenCVフレーム (H, W, 3) BGR uint8
            target_height: モデル入力高さ
            target_width: モデル入力幅

        Returns:
            input_tensor: (1, 3, H, W) float32
        """
        # リサイズ（INTER_CUBICを使用）
        resized = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC
        )

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 正規化 (0-255 -> 0-1)
        normalized = rgb.astype(np.float32) / 255.0

        # ImageNet正規化
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

        # HWC -> CHW、バッチ次元追加 (1, 3, H, W)
        input_tensor = normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor.astype(np.float32)

    def _postprocess(
        self,
        disparity: np.ndarray,
        original_height: int,
        original_width: int,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        後処理: グレースケール深度マップ、リサイズ

        Args:
            disparity: モデル出力 (1, 1, H, W) or (1, H, W)
            original_height: 元画像の高さ
            original_width: 元画像の幅
            normalize: 相対値で正規化するかどうか

        Returns:
            depth_image: (H, W, 3) BGR uint8
        """
        # [1, 1, H, W] or [1, H, W] -> [H, W]
        disp = disparity.squeeze()

        if normalize:
            # 相対値変換（フレームごとのmin/maxで正規化）
            disp_min = disp.min()
            disp_max = disp.max()
            if disp_max - disp_min > 1e-6:
                disp_normalized = (disp - disp_min) / (disp_max - disp_min)
                disp_uint8 = (disp_normalized * 255).astype(np.uint8)
            else:
                disp_uint8 = np.zeros_like(disp, dtype=np.uint8)
        else:
            # 絶対値変換 (0-1 -> 0-255、範囲外はクリップ)
            disp_clipped = np.clip(disp, 0, 1)
            disp_uint8 = (disp_clipped * 255).astype(np.uint8)

        # 元のサイズにリサイズ
        resized = cv2.resize(
            disp_uint8, (original_width, original_height), interpolation=cv2.INTER_LINEAR
        )

        # グレースケールをBGRに変換（3チャンネル）
        result = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        return result

    def _check_cancel_and_return(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """キャンセルされていたら早期リターン用の結果を返す"""
        if self.is_cancelled():
            self.clear_cancel()
            return_image = self._last_image if self._last_image is not None else image
            return {"image": return_image}
        return None

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
        cancelled_result = self._check_cancel_and_return(image)
        if cancelled_result is not None:
            return cancelled_result

        # プロパティ取得
        model_index = int(properties.get("model", 4))
        normalize = bool(properties.get("normalize", False))
        use_gpu = bool(properties.get("use_gpu", False)) and CUDA_AVAILABLE

        # モデル名取得
        if model_index < 0 or model_index >= len(MODEL_NAMES):
            model_index = 4
        model_name = MODEL_NAMES[model_index]

        # モデルロード
        session, load_error = self._load_model(model_index, use_gpu)

        if session is None:
            error_msg = load_error or f"Model not found: {model_name}"
            raise RuntimeError(error_msg)

        # モデル情報取得
        _, target_height, target_width, model_type = self._get_model_info(model_index)

        # 元サイズを保存
        h, w = image.shape[:2]
        original_height, original_width = h, w

        # 前処理（モデルタイプに応じて選択）
        if model_type == "imagenet":
            input_tensor = self._preprocess_imagenet(image, target_height, target_width)
        else:
            input_tensor = self._preprocess_standard(image, target_height, target_width)

        # 推論
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_tensor})
        disparity = output[0]

        # 後処理
        result_image = self._postprocess(
            disparity, original_height, original_width, normalize
        )

        # 結果を保存（キャンセル時に使用）
        self._last_image = result_image

        return {"image": result_image}
