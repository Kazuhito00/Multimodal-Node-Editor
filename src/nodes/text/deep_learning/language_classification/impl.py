"""
Language Classification ノードの実装。
MediaPipe Language Detector を使用してテキストの言語を判定する。
"""
from typing import Dict, Any, List, Optional
from node_editor.node_def import ComputeLogic
import json
import os


# モデル設定
MODEL_NAMES = [
    "MediaPipe Language Detector",
]

MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite",
]


class LanguageClassificationLogic(ComputeLogic):
    """
    テキスト言語分類ノードのロジック。
    MediaPipe Language Detector でテキストの言語を判定し、結果をJSONで出力。
    """

    # クラス共有のモデルキャッシュ
    _model_cache: Dict[int, Any] = {}
    _model_errors: Dict[int, str] = {}

    def __init__(self):
        self._last_result_json: str = json.dumps({"detections": []})

    def reset(self):
        """ノードの状態をリセット"""
        self._last_result_json = json.dumps({"detections": []})

    def _get_model_path(self, model_index: int) -> str:
        """モデルファイルのパスを取得"""
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_url = MODEL_URLS[model_index]

        # ダウンロードファイル名生成
        model_name = model_url.split("/")[-1]
        quantize_type = model_url.split("/")[-3]
        split_name = model_name.split(".")
        model_name = f"{split_name[0]}_{quantize_type}.{split_name[1]}"

        return os.path.join(base_path, "mediapipe_language_detector", "model", model_name)

    def _download_model(self, model_index: int) -> Optional[str]:
        """モデルをダウンロード"""
        model_path = self._get_model_path(model_index)
        if os.path.exists(model_path):
            return model_path

        # modelディレクトリ作成
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)

        # ダウンロード
        try:
            from utils.download_file import download_file
            download_file(url=MODEL_URLS[model_index], save_path=model_path)
            return model_path
        except Exception as e:
            return None

    def _load_model(self, model_index: int) -> tuple:
        """
        モデルをロード。(detector, error_message)を返す。
        """
        # キャッシュ済みモデルを返す
        if model_index in self._model_cache:
            return self._model_cache[model_index], None

        # 以前のエラーを返す
        if model_index in self._model_errors:
            return None, self._model_errors[model_index]

        # モデルパス確認・ダウンロード
        model_path = self._download_model(model_index)
        if model_path is None or not os.path.exists(model_path):
            error_msg = f"Failed to download model: {MODEL_NAMES[model_index]}"
            self._model_errors[model_index] = error_msg
            return None, error_msg

        # モデルロード
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import text

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = text.LanguageDetectorOptions(base_options=base_options)
            detector = text.LanguageDetector.create_from_options(options)

            self._model_cache[model_index] = detector
            return detector, None
        except ImportError as e:
            error_msg = f"MediaPipe not installed: {e}"
            self._model_errors[model_index] = error_msg
            return None, error_msg
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            self._model_errors[model_index] = error_msg
            return None, error_msg

    def _detect_language(
        self,
        detector: Any,
        input_text: str,
    ) -> List[Dict[str, Any]]:
        """
        言語検出を実行

        Args:
            detector: MediaPipe Language Detector
            input_text: 入力テキスト

        Returns:
            検出結果のリスト
        """
        detection_result = detector.detect(input_text)

        detections = []
        for detection in detection_result.detections:
            detections.append({
                "language_code": detection.language_code,
                "probability": float(detection.probability),
            })

        return detections

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        input_text = inputs.get("text_in")
        if input_text is None or input_text == "":
            return {"result_json": json.dumps({"detections": []})}

        # キャンセルチェック
        if self.is_cancelled():
            self.clear_cancel()
            return {"result_json": self._last_result_json}

        # プロパティ取得
        model_index = int(properties.get("model", 0))
        if model_index < 0 or model_index >= len(MODEL_NAMES):
            model_index = 0

        # モデルロード
        detector, load_error = self._load_model(model_index)

        # エラーの場合
        if detector is None:
            error_msg = load_error or f"Model not found: {MODEL_NAMES[model_index]}"
            raise RuntimeError(error_msg)

        # 言語検出実行
        detections = self._detect_language(detector, input_text)

        # 結果を構築
        result = {
            "model": MODEL_NAMES[model_index],
            "input_text": input_text,
            "detections": detections,
        }

        result_json = json.dumps(result, ensure_ascii=False)
        self._last_result_json = result_json

        return {"result_json": result_json}
