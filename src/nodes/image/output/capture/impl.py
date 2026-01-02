import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from node_editor.node_def import ComputeLogic


def get_timestamp_string() -> str:
    """yyyymmdd_hhmmss形式のタイムスタンプを取得"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None


class CaptureLogic(ComputeLogic):
    """
    キャプチャノードのロジック。
    トリガー入力またはボタン押下で画像をキャプチャし、
    temp/capturesに個別画像を保存。
    STOP時にzipを作成してダウンロード。
    """

    def __init__(self):
        self._captured_paths: List[Path] = []
        self._is_streaming: bool = False
        self._output_path: Optional[str] = None
        self._download_ready: bool = False
        self._last_trigger_value: float = 0.0
        self._last_button_value: bool = False
        self._session_id: str = ""
        self._capture_count: int = 0

    def reset(self):
        """ノードの状態をリセット"""
        self._captured_paths = []
        self._is_streaming = False
        self._output_path = None
        self._download_ready = False
        self._last_trigger_value = 0.0
        self._last_button_value = False
        self._session_id = ""
        self._capture_count = 0

    def _get_temp_dir(self) -> Path:
        """キャプチャ保存用ディレクトリを取得"""
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        temp_dir = project_root / "temp" / "captures"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def _save_image(self, image) -> Optional[Path]:
        """画像をPNGファイルとして保存"""
        if not CV2_AVAILABLE:
            return None

        temp_dir = self._get_temp_dir()
        filename = f"{self._session_id}_{self._capture_count:04d}.png"
        output_path = temp_dir / filename

        success = cv2.imwrite(str(output_path), image)
        if success:
            self._capture_count += 1
            return output_path
        return None

    def _save_zip(self) -> Optional[str]:
        """保存した画像をzipファイルにまとめる"""
        if not self._captured_paths:
            return None

        temp_dir = self._get_temp_dir()
        zip_path = temp_dir / f"{self._session_id}.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, img_path in enumerate(self._captured_paths):
                if img_path.exists():
                    # zip内のファイル名は連番
                    arcname = f"capture_{i:04d}.png"
                    zf.write(img_path, arcname)

        print(f"Captures zip: {zip_path} ({len(self._captured_paths)} images)")
        return str(zip_path)

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if context is None:
            context = {}

        if not CV2_AVAILABLE:
            return {"__error__": "opencv-python is not installed"}

        image = inputs.get("image")
        trigger_value = inputs.get("capture") or 0.0
        capture_button = properties.get("capture", False)

        is_streaming = context.get("is_streaming", False)

        # START/STOP状態の変化を検出
        was_streaming = self._is_streaming
        self._is_streaming = is_streaming

        # START時にセッションを開始
        if is_streaming and not was_streaming:
            self._captured_paths = []
            self._download_ready = False
            self._session_id = f"capture_{get_timestamp_string()}"
            self._capture_count = 0
            print(f"Capture: Session started ({self._session_id})")

        # STOP時にzipを作成（キャプチャがある場合）
        if not is_streaming and was_streaming:
            if self._captured_paths:
                self._output_path = self._save_zip()
                if self._output_path:
                    self._download_ready = True
                    print(f"Capture: Zip ready: {self._output_path}")
            else:
                print("Capture: No images to save")

        # キャプチャトリガーの検出
        should_capture = False

        # トリガー入力（立ち上がりエッジ検出）
        if trigger_value > 0.5 and self._last_trigger_value <= 0.5:
            should_capture = True
        self._last_trigger_value = trigger_value

        # ボタン押下（立ち上がりエッジ検出）
        if capture_button and not self._last_button_value:
            should_capture = True
        self._last_button_value = capture_button

        # 画像をキャプチャ（ファイルに保存）
        if should_capture and image is not None:
            if isinstance(image, np.ndarray) and len(image.shape) >= 2:
                # セッションIDがない場合は生成
                if not self._session_id:
                    self._session_id = f"capture_{get_timestamp_string()}"
                    self._capture_count = 0

                saved_path = self._save_image(image)
                if saved_path:
                    self._captured_paths.append(saved_path)
                    print(f"Capture: Saved {saved_path.name} ({len(self._captured_paths)} total)")

        # 結果を返す
        result: Dict[str, Any] = {}

        # ダウンロード準備完了時にダウンロード情報を返す
        if self._download_ready and self._output_path:
            result["__download__"] = {
                "path": self._output_path,
                "filename": Path(self._output_path).name,
                "type": "application/zip",
            }
            # ダウンロード情報は一度だけ送信
            self._download_ready = False

        return result
