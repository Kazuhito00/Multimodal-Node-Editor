"""
Low Light Image Enhancement ノードの実装。
MobileIE、TBEFN、CPGA-Net モデルで低照度画像を改善する。
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

# モデル設定
MODEL_NAMES = ["MobileIE", "TBEFN", "CPGA-Net"]

# 処理サイズ設定
PROCESS_SIZES = {
    0: None,           # Original
    1: (1280, 720),    # 1280x720
    2: (640, 360),     # 640x360
    3: (320, 180),     # 320x180
}


class LowLightEnhancementLogic(ComputeLogic):
    """
    低照度画像改善ノードのロジック。
    MobileIE、TBEFN、CPGA-Net モデルで画像を改善する。
    """

    # クラス共有のモデルキャッシュ
    _model_cache: Dict[str, Any] = {}
    _model_errors: Dict[str, str] = {}

    def __init__(self):
        self._last_model_index: int = -1
        self._last_image: Optional[np.ndarray] = None
        # CPGA-Net用のガンマ状態（フレーム間で引き継ぐ）
        self._cpga_gamma: Optional[np.ndarray] = None

    def reset(self):
        """ノードの状態をリセット"""
        self._last_image = None
        self._cpga_gamma = None

    def _get_model_path(self, model_name: str) -> str:
        """モデルファイルのパスを取得"""
        base_path = os.path.dirname(os.path.abspath(__file__))

        model_paths = {
            "MobileIE": "MobileIE/model/mobileie_lolv1.onnx",
            "TBEFN": "TBEFN/model/tbefn.onnx",
            "CPGA-Net": "CPGA-Net/model/cpga_net.onnx",
        }

        return os.path.join(base_path, model_paths.get(model_name, ""))

    def _get_providers(self, use_gpu: bool) -> List[str]:
        """使用するプロバイダーリストを取得"""
        if use_gpu and CUDA_AVAILABLE:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_model(
        self, model_name: str, use_gpu: bool = False
    ) -> Tuple[Optional[Any], Optional[str]]:
        """モデルをロード。(session, error_message)を返す。"""
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

    def _preprocess_nchw(
        self, frame: np.ndarray, target_size: Optional[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        NCHW形式の前処理（MobileIE、CPGA-Net用）

        Args:
            frame: OpenCVフレーム (H, W, 3) BGR uint8
            target_size: (width, height) リサイズ先、Noneでオリジナル

        Returns:
            input_tensor: (1, 3, H, W) float32 [0, 1]
            resized_frame: リサイズ後のフレーム
        """
        if target_size is not None:
            resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 正規化
        normalized = rgb.astype(np.float32) / 255.0
        # HWC -> CHW、バッチ次元追加 (1, 3, H, W)
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor, resized

    def _preprocess_nhwc(
        self, frame: np.ndarray, target_size: Optional[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        NHWC形式の前処理（TBEFN用）

        Args:
            frame: OpenCVフレーム (H, W, 3) BGR uint8
            target_size: (width, height) リサイズ先、Noneでオリジナル

        Returns:
            input_tensor: (1, H, W, 3) float32 [0, 1]
            resized_frame: リサイズ後のフレーム
        """
        if target_size is not None:
            resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 正規化
        normalized = rgb.astype(np.float32) / 255.0
        # バッチ次元追加 (1, H, W, 3)
        input_tensor = np.expand_dims(normalized, axis=0)
        return input_tensor, resized

    def _postprocess_nchw(
        self, output: np.ndarray, original_size: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """
        NCHW形式の後処理

        Args:
            output: (1, 3, H, W) float32
            original_size: (width, height) 元サイズに戻す場合

        Returns:
            frame: (H, W, 3) BGR uint8
        """
        # バッチ次元削除、CHW -> HWC変換
        result = output[0]  # (3, H, W)
        result = np.transpose(result, (1, 2, 0))  # (H, W, 3)
        # [0, 1]にクリップ
        result = np.clip(result, 0, 1)
        # [0, 255]にスケール
        result = (result * 255.0).astype(np.uint8)
        # RGB -> BGR
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        # 元サイズにリサイズ
        if original_size is not None:
            bgr = cv2.resize(bgr, original_size, interpolation=cv2.INTER_LINEAR)
        return bgr

    def _postprocess_nhwc(
        self, output: np.ndarray, original_size: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """
        NHWC形式の後処理

        Args:
            output: (1, H, W, 3) float32
            original_size: (width, height) 元サイズに戻す場合

        Returns:
            frame: (H, W, 3) BGR uint8
        """
        # バッチ次元削除
        result = output[0]  # (H, W, 3)
        # [0, 1]にクリップ
        result = np.clip(result, 0, 1)
        # [0, 255]にスケール
        result = (result * 255.0).astype(np.uint8)
        # RGB -> BGR
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        # 元サイズにリサイズ
        if original_size is not None:
            bgr = cv2.resize(bgr, original_size, interpolation=cv2.INTER_LINEAR)
        return bgr

    def _enhance_mobileie(
        self,
        session,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]],
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """MobileIEモデルで画像を改善"""
        # 前処理（NCHW）
        input_tensor, _ = self._preprocess_nchw(image, target_size)

        # 推論
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: input_tensor})[0]

        # 後処理
        restore_size = original_size if target_size else None
        return self._postprocess_nchw(output, restore_size)

    def _enhance_tbefn(
        self,
        session,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]],
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """TBEFNモデルで画像を改善"""
        # 前処理（NHWC）
        input_tensor, _ = self._preprocess_nhwc(image, target_size)

        # 推論
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: input_tensor})[0]

        # 後処理
        restore_size = original_size if target_size else None
        return self._postprocess_nhwc(output, restore_size)

    def _enhance_cpga_net(
        self,
        session,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]],
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """CPGA-Netモデルで画像を改善（ガンマ状態を引き継ぐ）"""
        # 前処理（NCHW）
        input_tensor, _ = self._preprocess_nchw(image, target_size)

        # ガンマ値の初期化（最初のフレームは1.0を使用）
        if self._cpga_gamma is None:
            self._cpga_gamma = np.ones((1, 1, 1, 1), dtype=np.float32)

        # 推論（2入力: 画像とガンマ）
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        outputs = session.run(
            output_names,
            {input_names[0]: input_tensor, input_names[1]: self._cpga_gamma}
        )

        # ガンマ値を更新（次フレーム用）
        self._cpga_gamma = outputs[1]

        # 後処理
        restore_size = original_size if target_size else None
        return self._postprocess_nchw(outputs[0], restore_size)

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
        model_index = int(properties.get("model", 0))
        process_size_index = int(properties.get("process_size", 0))
        use_gpu = bool(properties.get("use_gpu", False)) and CUDA_AVAILABLE

        # モデル名取得
        if model_index < 0 or model_index >= len(MODEL_NAMES):
            model_index = 0
        model_name = MODEL_NAMES[model_index]

        # 処理サイズ取得
        target_size = PROCESS_SIZES.get(process_size_index, None)

        # モデル変更時にCPGA-Netのガンマをリセット
        if model_index != self._last_model_index:
            self._cpga_gamma = None
            self._last_model_index = model_index

        # モデルロード
        session, load_error = self._load_model(model_name, use_gpu)

        if session is None:
            error_msg = load_error or f"Model not found: {model_name}"
            raise RuntimeError(error_msg)

        # 元サイズを保存
        h, w = image.shape[:2]
        original_size = (w, h)

        # モデル別に処理
        if model_name == "MobileIE":
            result_image = self._enhance_mobileie(session, image, target_size, original_size)
        elif model_name == "TBEFN":
            result_image = self._enhance_tbefn(session, image, target_size, original_size)
        elif model_name == "CPGA-Net":
            result_image = self._enhance_cpga_net(session, image, target_size, original_size)
        else:
            result_image = image

        # 結果を保存（キャンセル時に使用）
        self._last_image = result_image

        return {"image": result_image}
