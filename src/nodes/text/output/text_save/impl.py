import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from node_editor.node_def import ComputeLogic


def get_timestamp_string() -> str:
    """yyyymmdd_hhmmss形式のタイムスタンプを取得"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class TextSaveLogic(ComputeLogic):
    """
    テキスト保存ノード。
    トリガー入力またはボタン押下でテキストを保存し、
    temp/textsに個別ファイルを保存。
    STOP時にzipを作成してダウンロード。
    """

    def __init__(self):
        self._saved_paths: List[Path] = []
        self._is_streaming: bool = False
        self._output_path: Optional[str] = None
        self._download_ready: bool = False
        self._last_trigger_value: float = 0.0
        self._last_button_value: bool = False
        self._session_id: str = ""
        self._save_count: int = 0

    def reset(self):
        """ノードの状態をリセット"""
        self._saved_paths = []
        self._is_streaming = False
        self._output_path = None
        self._download_ready = False
        self._last_trigger_value = 0.0
        self._last_button_value = False
        self._session_id = ""
        self._save_count = 0

    def _get_temp_dir(self) -> Path:
        """テキスト保存用ディレクトリを取得"""
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        temp_dir = project_root / "temp" / "texts"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def _save_text(self, text: str) -> Optional[Path]:
        """テキストをファイルに保存"""
        temp_dir = self._get_temp_dir()
        filename = f"{self._session_id}_{self._save_count:04d}.txt"
        output_path = temp_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        self._save_count += 1
        return output_path

    def _save_zip(self) -> Optional[str]:
        """保存したテキストをzipファイルにまとめる"""
        if not self._saved_paths:
            return None

        temp_dir = self._get_temp_dir()
        zip_path = temp_dir / f"{self._session_id}.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, txt_path in enumerate(self._saved_paths):
                if txt_path.exists():
                    # zip内のファイル名は連番
                    arcname = f"text_{i:04d}.txt"
                    zf.write(txt_path, arcname)

        print(f"Text Save: Zip created {zip_path} ({len(self._saved_paths)} files)")
        return str(zip_path)

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if context is None:
            context = {}

        text = inputs.get("text", "")
        trigger_value = inputs.get("save") or 0.0
        save_button = properties.get("save", False)

        is_streaming = context.get("is_streaming", False)

        # START/STOP状態の変化を検出
        was_streaming = self._is_streaming
        self._is_streaming = is_streaming

        # START時にセッションを開始
        if is_streaming and not was_streaming:
            self._saved_paths = []
            self._download_ready = False
            self._session_id = f"text_{get_timestamp_string()}"
            self._save_count = 0
            print(f"Text Save: Session started ({self._session_id})")

        # STOP時にzipを作成（保存がある場合）
        if not is_streaming and was_streaming:
            if self._saved_paths:
                self._output_path = self._save_zip()
                if self._output_path:
                    self._download_ready = True
                    print(f"Text Save: Zip ready: {self._output_path}")
            else:
                print("Text Save: No texts to save")

        # 保存トリガーの検出
        should_save = False

        # トリガー入力（立ち上がりエッジ検出）
        if trigger_value > 0.5 and self._last_trigger_value <= 0.5:
            should_save = True
        self._last_trigger_value = trigger_value

        # ボタン押下（立ち上がりエッジ検出）
        if save_button and not self._last_button_value:
            should_save = True
        self._last_button_value = save_button

        # テキストを保存（ファイルに保存）
        if should_save and text:
            # セッションIDがない場合は生成
            if not self._session_id:
                self._session_id = f"text_{get_timestamp_string()}"
                self._save_count = 0

            saved_path = self._save_text(text)
            if saved_path:
                self._saved_paths.append(saved_path)
                print(f"Text Save: Saved {saved_path.name} ({len(self._saved_paths)} total)")

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
