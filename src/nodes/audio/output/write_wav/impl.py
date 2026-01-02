import struct
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from node_editor.node_def import ComputeLogic


def get_timestamp_string() -> str:
    """yyyymmdd_hhmmss形式のタイムスタンプを取得"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class WavWriterLogic(ComputeLogic):
    """
    WAV書き出しノードのロジック。
    入力オーディオをサンプルとしてバッファに蓄積し、録音停止時にWAVファイルとして保存する。
    """

    # 最小録音サンプル数（これ以下の場合は録音停止を無視）
    MIN_SAMPLES_TO_SAVE = 1600  # 0.1秒 @ 16kHz

    def __init__(self):
        self._samples: List[float] = []
        self._is_recording: bool = False
        self._sample_rate: int = 16000
        self._output_path: str = None
        self._download_ready: bool = False

    def reset(self):
        """ノードの状態をリセット"""
        self._samples = []
        self._is_recording = False
        self._sample_rate = 16000
        self._output_path = None
        self._download_ready = False

    def _save_wav(self) -> str:
        """蓄積したサンプルをWAVファイルとして保存"""
        if not self._samples:
            return None

        # プロジェクトルートのtempディレクトリに保存
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        temp_dir = project_root / "temp" / "audio"
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_dir / f"audio_{get_timestamp_string()}.wav"

        # WAVファイルを書き出し
        with wave.open(str(output_path), 'w') as wav_file:
            wav_file.setnchannels(1)  # モノラル
            wav_file.setsampwidth(2)  # 16bit
            wav_file.setframerate(self._sample_rate)

            # float (-1.0 ~ 1.0) を 16bit PCM に変換
            for sample in self._samples:
                # クリッピング
                sample = max(-1.0, min(1.0, sample))
                # 16bit整数に変換
                int_sample = int(sample * 32767)
                wav_file.writeframes(struct.pack('<h', int_sample))

        print(f"WAV saved: {output_path} ({len(self._samples)} samples, {self._sample_rate}Hz)")
        return str(output_path)

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if context is None:
            context = {}

        # プレビューモードでは録音処理をスキップ
        # ただしSTOP実行時（録音中にis_streaming=False）は保存処理を実行
        is_preview = context.get("preview", False)
        is_streaming = context.get("is_streaming", False)
        if is_preview and not self._is_recording:
            return {}

        audio = inputs.get("audio")

        # 録音開始（START時）
        if is_streaming and not self._is_recording:
            self._samples = []
            self._is_recording = True
            self._download_ready = False
            print("Audio recording started")

        # 録音停止 → WAV保存（STOP時）
        # 最小サンプル数未満の場合は誤操作と見なして無視（グラフロード時の一瞬のSTOP等）
        if not is_streaming and self._is_recording:
            if len(self._samples) >= self.MIN_SAMPLES_TO_SAVE:
                self._is_recording = False
                self._output_path = self._save_wav()
                if self._output_path:
                    self._download_ready = True
                    print(f"Audio recording stopped. WAV ready: {self._output_path}")
                else:
                    print("Audio recording stopped but no samples to save")

        # 録音中はサンプルを蓄積
        if self._is_recording and audio is not None:
            # audioデータからdeltaサンプルを取得
            if isinstance(audio, dict):
                delta = audio.get("delta", [])
                sample_rate = audio.get("sample_rate", 16000)
                self._sample_rate = sample_rate

                if delta:
                    self._samples.extend(delta)

        # 結果を返す
        result: Dict[str, Any] = {}

        # ダウンロード準備完了時にダウンロード情報を返す
        if self._download_ready and self._output_path:
            result["__download__"] = {
                "path": self._output_path,
                "filename": Path(self._output_path).name,
                "type": "audio/wav",
            }
            # ダウンロード情報は一度だけ送信
            self._download_ready = False

        # 録音状態情報
        if self._is_recording:
            duration = len(self._samples) / self._sample_rate if self._sample_rate > 0 else 0
            result["__status__"] = f"Recording: {duration:.1f}s"

        return result
