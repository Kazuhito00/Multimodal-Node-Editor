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


class VideoWriterLogic(ComputeLogic):
    """
    動画書き出しノードのロジック。
    入力画像をフレームとしてバッファに蓄積し、録画停止時にmp4ファイルとして保存する。
    """

    def __init__(self):
        self._frames: List[Any] = []
        self._is_recording: bool = False
        self._resolution: Optional[tuple] = None
        self._output_path: Optional[str] = None
        self._download_ready: bool = False

    def reset(self):
        """ノードの状態をリセット"""
        self._frames = []
        self._is_recording = False
        self._resolution = None
        self._output_path = None
        self._download_ready = False

    def _save_video(self, fps: int) -> Optional[str]:
        """蓄積したフレームを動画ファイルとして保存"""
        if not self._frames or not CV2_AVAILABLE:
            return None

        if self._resolution is None:
            return None

        # プロジェクトルートのtempディレクトリに保存
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        temp_dir = project_root / "temp" / "videos"
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_dir / f"video_{get_timestamp_string()}.mp4"

        width, height = self._resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"Failed to open VideoWriter: {output_path}")
            return None

        for frame in self._frames:
            # フレームが正しいサイズであることを確認
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)

        writer.release()
        print(f"Video saved: {output_path} ({len(self._frames)} frames)")

        return str(output_path)

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

        # プレビューモードでは録画処理をスキップ
        # ただしSTOP実行時（録画中にis_streaming=False）は保存処理を実行
        is_preview = context.get("preview", False)
        is_streaming = context.get("is_streaming", False)
        if is_preview and not self._is_recording:
            return {}

        image = inputs.get("image")

        # FPSをinterval_msから逆算（interval_ms = 100 → FPS = 10）
        interval_ms = context.get("interval_ms", 100)
        fps = max(1, round(1000 / interval_ms))

        # 録画開始（START時）
        if is_streaming and not self._is_recording:
            self._frames = []
            self._resolution = None
            self._is_recording = True
            self._download_ready = False
            print("Recording started")

        # 録画停止 → 動画保存（STOP時）
        if not is_streaming and self._is_recording:
            self._is_recording = False
            self._output_path = self._save_video(fps)
            if self._output_path:
                self._download_ready = True
                print(f"Recording stopped. Video ready: {self._output_path}")
            else:
                print("Recording stopped but no frames to save")

        # 録画中はフレームを蓄積
        if self._is_recording and image is not None:
            if isinstance(image, np.ndarray) and len(image.shape) >= 2:
                # 解像度を最初のフレームから取得
                if self._resolution is None:
                    height, width = image.shape[:2]
                    self._resolution = (width, height)
                    print(f"Recording resolution: {width}x{height}")

                # フレームをコピーして蓄積
                self._frames.append(image.copy())

        # 結果を返す
        result: Dict[str, Any] = {}

        # ダウンロード準備完了時にダウンロード情報を返す
        if self._download_ready and self._output_path:
            result["__download__"] = {
                "path": self._output_path,
                "filename": Path(self._output_path).name,
                "type": "video/mp4",
            }
            # ダウンロード情報は一度だけ送信
            self._download_ready = False

        # 録画状態情報
        if self._is_recording:
            result["__status__"] = f"Recording: {len(self._frames)} frames"

        return result
