import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import tempfile
import shutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# WebRTC
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
    from aiortc.contrib.media import MediaRelay
    from av import VideoFrame, AudioFrame
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("Warning: aiortc not available, WebRTC features disabled")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    # ImportError, OSError (PortAudio not found), etc.
    SOUNDDEVICE_AVAILABLE = False
    sd = None


# WebRTC グローバルフレームストア
# ノードからアクセスしてWebRTC経由の最新フレームを取得する
class WebRTCFrameStore:
    """WebRTCで受信したフレームを保持するグローバルストア"""

    def __init__(self):
        self._video_frames: Dict[str, np.ndarray] = {}  # connection_id -> frame
        self._audio_buffers: Dict[str, Dict[str, Any]] = {}  # connection_id -> {samples: List, sample_rate: int}
        self._lock = threading.Lock()

    def set_video_frame(self, connection_id: str, frame: np.ndarray):
        """ビデオフレームを保存"""
        with self._lock:
            self._video_frames[connection_id] = frame

    def get_video_frame(self, connection_id: str) -> Optional[np.ndarray]:
        """最新のビデオフレームを取得"""
        with self._lock:
            return self._video_frames.get(connection_id)

    def set_audio_buffer(self, connection_id: str, samples: np.ndarray, sample_rate: int = 48000):
        """オーディオサンプルを蓄積（上書きではなく追加）"""
        with self._lock:
            if connection_id not in self._audio_buffers:
                self._audio_buffers[connection_id] = {"samples": [], "sample_rate": sample_rate}
            self._audio_buffers[connection_id]["samples"].append(samples)
            self._audio_buffers[connection_id]["sample_rate"] = sample_rate

    def get_audio_buffer(self, connection_id: str, max_duration_ms: float = 100.0) -> Optional[Dict[str, Any]]:
        """蓄積されたオーディオサンプルを取得してクリア

        max_duration_ms: 最大取得時間（ミリ秒）。これを超える古いサンプルは破棄。
        """
        with self._lock:
            buffer_data = self._audio_buffers.get(connection_id)
            if not buffer_data or not buffer_data["samples"]:
                return None
            # 蓄積されたサンプルを結合
            combined = np.concatenate(buffer_data["samples"])
            sample_rate = buffer_data["sample_rate"]

            # 最大サンプル数を計算（max_duration_ms分）
            max_samples = int(sample_rate * max_duration_ms / 1000.0)

            # バッファが大きすぎる場合は最新部分だけを保持
            if len(combined) > max_samples:
                combined = combined[-max_samples:]

            # バッファをクリア
            self._audio_buffers[connection_id] = {"samples": [], "sample_rate": sample_rate}
            return {"samples": combined, "sample_rate": sample_rate}

    def clear_audio_buffer(self, connection_id: str):
        """オーディオバッファをクリア（接続は維持）"""
        with self._lock:
            if connection_id in self._audio_buffers:
                sample_rate = self._audio_buffers[connection_id].get("sample_rate", 16000)
                self._audio_buffers[connection_id] = {"samples": [], "sample_rate": sample_rate}

    def clear_all_audio_buffers(self):
        """全てのオーディオバッファをクリア"""
        with self._lock:
            for connection_id in self._audio_buffers:
                sample_rate = self._audio_buffers[connection_id].get("sample_rate", 16000)
                self._audio_buffers[connection_id] = {"samples": [], "sample_rate": sample_rate}

    def remove_connection(self, connection_id: str):
        """接続に関連するデータを削除"""
        with self._lock:
            self._video_frames.pop(connection_id, None)
            self._audio_buffers.pop(connection_id, None)

    def get_all_video_connections(self) -> List[str]:
        """ビデオフレームを持つ接続IDのリストを取得"""
        with self._lock:
            return list(self._video_frames.keys())

    def get_all_audio_connections(self) -> List[str]:
        """オーディオバッファを持つ接続IDのリストを取得"""
        with self._lock:
            return list(self._audio_buffers.keys())


# グローバルフレームストアのインスタンス
webrtc_frame_store = WebRTCFrameStore()

# WebRTC接続管理
_webrtc_connections: Dict[str, "RTCPeerConnection"] = {}
_webrtc_lock = threading.Lock()
# 接続処理の直列化用asyncioロック（同時に1つの接続のみ処理）
_webrtc_offer_lock: Optional[asyncio.Lock] = None

def get_webrtc_offer_lock() -> asyncio.Lock:
    """asyncio.Lockを遅延初期化して取得"""
    global _webrtc_offer_lock
    if _webrtc_offer_lock is None:
        _webrtc_offer_lock = asyncio.Lock()
    return _webrtc_offer_lock

# CUDA利用可能かチェック（実際にプロバイダーをロードして確認）
def check_cuda_available() -> bool:
    """CUDAが実際に利用可能かチェック"""
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False

        import numpy as np
        from onnx import TensorProto, helper

        # 最小限のONNXモデルを作成
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
        node = helper.make_node("Identity", ["X"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y])

        # IR versionをONNX Runtimeが読める範囲に固定
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=11,
        )

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4  # エラーログを抑制

        # CUDAで実行テスト
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options,
            providers=["CUDAExecutionProvider"],
        )
        session.run(None, {"X": np.array([1.0], dtype=np.float32)})
        return True
    except Exception as e:
        print(f"CUDA not available: {e}")
        return False

CUDA_AVAILABLE = check_cuda_available()
print(f"CUDA available: {CUDA_AVAILABLE}")

# srcディレクトリをパスに追加
# main.py は src/gui/reactflow/backend/main.py にある
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

import node_editor.node_def as node_def
from node_editor.node_def import discover_nodes, get_all_categories, reset_all_nodes, reset_node_by_id, get_all_nodes_for_gui, cancel_all_nodes
from node_editor.core import Graph
from node_editor.settings import init_settings

# 設定ファイルを初期化（環境変数NODE_EDITOR_CONFIGがあればそれを使用）
config_env = os.environ.get("NODE_EDITOR_CONFIG")
if config_env:
    config_file = Path(config_env)
else:
    config_file = project_root / "config.json"
settings_manager = init_settings(config_file)


def is_colab() -> bool:
    """Google Colaboratory環境かどうかを判定"""
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


# --- FastAPI App Setup ---
app = FastAPI(title="ReactFlow Node Editor Backend")

# CORS Middleware（ローカル環境のみ）
if not is_colab():
    origins = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ノード定義を検出・登録（config.jsonのnode_search_pathsから読み込み）
node_search_paths = settings_manager.get("node_search_paths", [])
for path_str in node_search_paths:
    path = Path(path_str)
    # 相対パスの場合はプロジェクトルートからの相対パスとして解決
    if not path.is_absolute():
        path = project_root / path
    if path.exists():
        discover_nodes(path)
        print(f"Discovered nodes from: {path}")
    else:
        print(f"Warning: node_search_path not found: {path}")

print(f"Discovered {len(node_def._node_definition_registry)} node definitions.")


def scan_available_cameras(max_cameras: int = 2) -> List[Dict[str, Any]]:
    """利用可能なカメラをスキャンしてリストを返す"""
    available = []
    # DirectShowバックエンドを使用してOrbbecSDKのスキャンを回避（高速化）
    backend = cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else cv2.CAP_ANY
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            available.append({"value": i, "label": f"Camera {i}"})
            cap.release()
    # カメラが見つからない場合はデフォルトで0を追加
    if not available:
        available.append({"value": 0, "label": "Camera 0 (default)"})
    return available


# カメラリストをキャッシュ（起動時にスキャン）
_camera_options_cache: List[Dict[str, Any]] = []

# 起動時にカメラをスキャン（初回リクエストの遅延を回避）
if not is_colab():
    max_cameras = settings_manager.get("camera.max_scan_count", 2)
    _camera_options_cache = scan_available_cameras(max_cameras)
    print(f"Scanned cameras: {_camera_options_cache}")


def get_camera_options() -> List[Dict[str, Any]]:
    """キャッシュされたカメラオプションを返す"""
    return _camera_options_cache


def truncate_device_name(name: str, max_length: int = 20) -> str:
    """デバイス名を省略する"""
    if len(name) <= max_length:
        return name
    return name[:max_length - 3] + "..."


def scan_audio_devices() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """利用可能なオーディオデバイスをスキャンして入力/出力リストを返す"""
    inputs = [{"value": -1, "label": "Default"}]
    outputs = [{"value": -1, "label": "Default"}]

    if not SOUNDDEVICE_AVAILABLE:
        print("Warning: sounddevice is not available")
        return inputs, outputs

    try:
        devices = sd.query_devices()
        print(f"Found {len(devices)} audio devices")
        for i, dev in enumerate(devices):
            name = truncate_device_name(dev['name'])
            if dev['max_input_channels'] > 0:
                inputs.append({"value": i, "label": name})
            if dev['max_output_channels'] > 0:
                outputs.append({"value": i, "label": name})
        print(f"Audio inputs: {len(inputs)}, outputs: {len(outputs)}")
    except Exception as e:
        print(f"Error scanning audio devices: {e}")

    return inputs, outputs


# オーディオデバイスリストをキャッシュ（起動時にスキャン、Colabではスキップ）
if is_colab():
    _audio_inputs_cache: List[Dict[str, Any]] = []
    _audio_outputs_cache: List[Dict[str, Any]] = []
else:
    _audio_inputs_cache, _audio_outputs_cache = scan_audio_devices()


def get_audio_input_options() -> List[Dict[str, Any]]:
    """キャッシュされた入力デバイスオプションを返す"""
    return _audio_inputs_cache


def get_audio_output_options() -> List[Dict[str, Any]]:
    """キャッシュされた出力デバイスオプションを返す"""
    return _audio_outputs_cache


# ローカル環境のみルートエンドポイントを定義（Colabでは静的ファイル配信）
if not is_colab():
    @app.get("/")
    async def read_root():
        return {"message": "Welcome to the ReactFlow Node Editor Backend"}


# --- WebRTC エンドポイント ---

class WebRTCTrackInfo(BaseModel):
    """WebRTCトラック情報"""
    kind: str  # "video" or "audio"
    node_id: str  # ノードID


class WebRTCOffer(BaseModel):
    """WebRTC SDP Offerリクエスト"""
    sdp: str
    type: str
    connection_id: str
    track_type: str  # "video" or "audio" or "combined"
    tracks: Optional[List[WebRTCTrackInfo]] = None  # 複合接続の場合のトラック情報


class WebRTCAnswer(BaseModel):
    """WebRTC SDP Answerレスポンス"""
    sdp: str
    type: str


if WEBRTC_AVAILABLE:
    class VideoTrackHandler(MediaStreamTrack):
        """ビデオトラックを受信してフレームストアに保存するハンドラ"""
        kind = "video"

        def __init__(self, track: MediaStreamTrack, connection_id: str):
            super().__init__()
            self._track = track
            self._connection_id = connection_id

        async def recv(self):
            frame = await self._track.recv()
            # VideoFrameをnumpy配列(BGR)に変換
            img = frame.to_ndarray(format="bgr24")
            webrtc_frame_store.set_video_frame(self._connection_id, img)
            return frame

    class AudioTrackHandler(MediaStreamTrack):
        """オーディオトラックを受信してバッファに保存するハンドラ"""
        kind = "audio"

        def __init__(self, track: MediaStreamTrack, connection_id: str):
            super().__init__()
            self._track = track
            self._connection_id = connection_id

        async def recv(self):
            frame = await self._track.recv()
            # AudioFrameをnumpy配列に変換
            samples = frame.to_ndarray()
            webrtc_frame_store.set_audio_buffer(self._connection_id, samples)
            return frame

    @app.post("/api/webrtc/offer", response_model=WebRTCAnswer)
    async def webrtc_offer(offer: WebRTCOffer):
        """
        WebRTC Offerを受信してAnswerを返す。
        ブラウザからのメディアストリームを受信する準備をする。
        接続処理を直列化して競合を防ぐ。
        """
        # 接続処理を直列化（複数接続の同時処理で競合が発生するため）
        async with get_webrtc_offer_lock():
            connection_id = offer.connection_id

            # イベントループを事前にキャプチャ（on_trackコールバック用）
            loop = asyncio.get_running_loop()

            # 既存の接続があれば閉じる（ロック外でawait）
            old_pc = None
            with _webrtc_lock:
                if connection_id in _webrtc_connections:
                    old_pc = _webrtc_connections.pop(connection_id)

            if old_pc:
                await old_pc.close()
                webrtc_frame_store.remove_connection(connection_id)

            # 新しいPeerConnectionを作成
            pc = RTCPeerConnection()

            with _webrtc_lock:
                _webrtc_connections[connection_id] = pc

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                print(f"[WebRTC] Connection {connection_id}: {pc.connectionState}")
                if pc.connectionState in ["failed", "closed", "disconnected"]:
                    with _webrtc_lock:
                        _webrtc_connections.pop(connection_id, None)
                    webrtc_frame_store.remove_connection(connection_id)

            # トラック情報を保持（複合接続の場合に使用）
            track_infos = offer.tracks or []
            video_node_ids = [t.node_id for t in track_infos if t.kind == "video"]
            audio_node_ids = [t.node_id for t in track_infos if t.kind == "audio"]
            video_track_index = [0]  # 可変オブジェクトで参照渡し
            audio_track_index = [0]

            @pc.on("track")
            def on_track(track):
                # 複合接続の場合はトラック情報からconnection_idを決定
                if offer.track_type == "combined" and track_infos:
                    if track.kind == "video" and video_track_index[0] < len(video_node_ids):
                        node_id = video_node_ids[video_track_index[0]]
                        track_connection_id = f"webrtc_webcam_{node_id}"
                        video_track_index[0] += 1
                    elif track.kind == "audio" and audio_track_index[0] < len(audio_node_ids):
                        node_id = audio_node_ids[audio_track_index[0]]
                        track_connection_id = f"webrtc_microphone_{node_id}"
                        audio_track_index[0] += 1
                    else:
                        track_connection_id = connection_id
                else:
                    track_connection_id = connection_id

                print(f"[WebRTC] Track received: {track.kind} -> {track_connection_id}")
                try:
                    if track.kind == "video":
                        loop.create_task(process_video_track(track, track_connection_id))
                    elif track.kind == "audio":
                        loop.create_task(process_audio_track(track, track_connection_id))
                except Exception as e:
                    print(f"[WebRTC] Error creating track task: {e}")

            # Offerを設定
            await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

            # Answerを作成
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return WebRTCAnswer(sdp=pc.localDescription.sdp, type=pc.localDescription.type)

    # CPU処理用スレッドプール（イベントループをブロックしないため）
    # ビデオとオーディオで別々のプールを使用して競合を防ぐ
    _video_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="video")
    _audio_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio")

    def _convert_video_frame(frame) -> np.ndarray:
        """ビデオフレームをnumpy配列に変換（スレッドプールで実行）"""
        return frame.to_ndarray(format="bgr24")

    # 設定からターゲットサンプルレートを取得
    from node_editor.settings import get_setting
    _target_sample_rate = get_setting("audio.sample_rate", 16000)
    print(f"[WebRTC] Target sample rate for audio resampling: {_target_sample_rate}Hz")

    def _convert_audio_frame_with_rate(frame, effective_source_rate: int) -> Dict[str, Any]:
        """オーディオフレームをnumpy配列に変換してリサンプリング（スレッドプールで実行）

        effective_source_rate: キャリブレーション済みの実効サンプルレート
        """
        samples = frame.to_ndarray()

        # aiortcは (channels, samples) 形式で返す
        # ステレオの場合はモノラルに変換（平均を取る）
        if samples.ndim > 1:
            if samples.shape[0] == 2:
                # ステレオ -> モノラル（両チャンネルの平均）
                samples = samples.mean(axis=0)
            else:
                # その他の場合は最初のチャンネルを使用
                samples = samples[0]

        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        # リサンプリング（実効レート -> ターゲットレート）
        if effective_source_rate != _target_sample_rate:
            ratio = effective_source_rate / _target_sample_rate
            if ratio > 1:
                output_length = int(len(samples) / ratio)
                output = np.zeros(output_length, dtype=np.float32)
                for i in range(output_length):
                    src_index = i * ratio
                    src_floor = int(src_index)
                    src_ceil = min(src_floor + 1, len(samples) - 1)
                    t = src_index - src_floor
                    # 線形補間
                    output[i] = samples[src_floor] * (1 - t) + samples[src_ceil] * t
                samples = output

        return {"samples": samples, "sample_rate": _target_sample_rate}

    async def process_video_track(track: MediaStreamTrack, connection_id: str):
        """ビデオトラックからフレームを継続的に取得"""
        loop = asyncio.get_running_loop()
        try:
            while True:
                frame = await track.recv()
                # CPU処理をビデオ専用スレッドプールで実行（イベントループをブロックしない）
                img = await loop.run_in_executor(_video_executor, _convert_video_frame, frame)
                webrtc_frame_store.set_video_frame(connection_id, img)
        except Exception:
            pass

    async def process_audio_track(track: MediaStreamTrack, connection_id: str):
        """オーディオトラックからサンプルを継続的に取得

        サンプルレートの自動キャリブレーション機能付き:
        - 報告されるレート（通常48kHz）と実際の受信レートを比較
        - 差異が大きい場合（ratio > 1.5）は実効レートを調整
        """
        loop = asyncio.get_running_loop()
        frame_count = 0
        last_calibration_time = 0.0
        total_raw_samples = 0
        calibration_done = False
        effective_source_rate = 48000

        try:
            while True:
                frame = await track.recv()
                reported_rate = frame.sample_rate
                raw_samples = frame.to_ndarray()
                now = time.time()

                # 最初のフレームで計測開始
                if last_calibration_time == 0.0:
                    last_calibration_time = now
                    continue

                # リサンプリング前のサンプル数をカウント
                if raw_samples.ndim > 1:
                    raw_count = raw_samples.shape[1]
                else:
                    raw_count = len(raw_samples)
                total_raw_samples += raw_count
                frame_count += 1

                # 1秒ごとにキャリブレーション（初回のみ）
                elapsed = now - last_calibration_time
                if elapsed >= 1.0 and frame_count > 0 and not calibration_done:
                    measured_rate = total_raw_samples / elapsed
                    ratio = measured_rate / reported_rate

                    if ratio > 1.5:
                        effective_source_rate = int(reported_rate * round(ratio))
                    else:
                        effective_source_rate = reported_rate
                    calibration_done = True

                    frame_count = 0
                    total_raw_samples = 0
                    last_calibration_time = now

                # リサンプリング（オーディオ専用スレッドプールで実行）
                result = await loop.run_in_executor(
                    _audio_executor,
                    lambda f=frame, r=effective_source_rate: _convert_audio_frame_with_rate(f, r)
                )

                webrtc_frame_store.set_audio_buffer(
                    connection_id, result["samples"], result["sample_rate"]
                )
        except Exception as e:
            print(f"[WebRTC Audio] Error processing track: {e}")

    @app.post("/api/webrtc/close")
    async def webrtc_close(connection_id: str):
        """WebRTC接続を閉じる"""
        with _webrtc_lock:
            pc = _webrtc_connections.pop(connection_id, None)

        if pc:
            await pc.close()
            webrtc_frame_store.remove_connection(connection_id)
            return {"status": "closed"}

        return {"status": "not_found"}

    @app.get("/api/webrtc/connections")
    async def webrtc_connections():
        """アクティブなWebRTC接続のリストを返す"""
        with _webrtc_lock:
            connections = list(_webrtc_connections.keys())
        return {
            "connections": connections,
            "video_connections": webrtc_frame_store.get_all_video_connections(),
            "audio_connections": webrtc_frame_store.get_all_audio_connections(),
        }


@app.get("/api/settings/runtime")
async def get_runtime():
    """実行環境情報を返す"""
    return {"is_colab": is_colab(), "webrtc_available": WEBRTC_AVAILABLE}


@app.get("/api/settings/theme")
async def get_theme():
    """現在のテーマ設定を返す"""
    theme = settings_manager.get("ui.theme", "dark")
    return {"theme": theme}


@app.get("/api/settings/graph")
async def get_graph_settings():
    """グラフ実行設定を返す"""
    interval_ms = settings_manager.get("graph.interval_ms", 100)
    return {"interval_ms": interval_ms}


@app.get("/api/settings/sidebar")
async def get_sidebar_settings():
    """サイドバー表示設定を返す"""
    show_edit = settings_manager.get("ui.sidebar.show_edit", True)
    show_file = settings_manager.get("ui.sidebar.show_file", True)
    show_auto_layout = settings_manager.get("ui.sidebar.show_auto_layout", True)
    return {"show_edit": show_edit, "show_file": show_file, "show_auto_layout": show_auto_layout}


@app.get("/api/settings/audio")
async def get_audio_settings():
    """オーディオ設定を返す"""
    sample_rate = settings_manager.get("audio.sample_rate", 16000)
    return {"sample_rate": sample_rate}


@app.get("/api/settings/auto_download")
async def get_auto_download_settings():
    """自動ダウンロード設定を返す"""
    video = settings_manager.get("auto_download.video", False)
    wav = settings_manager.get("auto_download.wav", False)
    capture = settings_manager.get("auto_download.capture", False)
    text = settings_manager.get("auto_download.text", False)
    return {"video": video, "wav": wav, "capture": capture, "text": text}


@app.get("/api/settings/api_keys_status")
async def get_api_keys_status():
    """APIキーが設定されているかどうかを返す（キー自体は返さない）"""
    openai_key = settings_manager.get("api_keys.openai", "")
    return {
        "openai": bool(openai_key and len(openai_key) > 0),
    }


# アップロードファイル用のテンポラリディレクトリ（プロジェクトルート/temp/uploads）
_upload_temp_dir = project_root / "temp" / "uploads"
_upload_temp_dir.mkdir(parents=True, exist_ok=True)


def resize_for_preview(image, max_size: int = 400):
    """プレビュー用にリサイズ（アスペクト比維持）"""
    h, w = image.shape[:2]
    if w <= max_size and h <= max_size:
        return image
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_video_first_frame_base64(file_path: str, max_size: int = 400) -> str | None:
    """動画ファイルの最初のフレームをBase64で取得（プレビュー用にリサイズ）"""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        # プレビュー用にリサイズ
        preview = resize_for_preview(frame, max_size)
        # JPEGにエンコードしてBase64変換
        _, buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        import base64
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error getting first frame: {e}")
        return None


def get_image_base64(file_path: str, max_size: int = 400) -> str | None:
    """画像ファイルをBase64で取得（プレビュー用にリサイズ）"""
    try:
        image = cv2.imread(file_path)
        if image is None:
            return None
        # プレビュー用にリサイズ
        preview = resize_for_preview(image, max_size)
        # JPEGにエンコードしてBase64変換
        _, buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        import base64
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error reading image: {e}")
        return None


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """ファイルをアップロードして保存先パスを返す"""
    try:
        # ファイル名をサニタイズ
        safe_filename = Path(file.filename).name
        file_path = _upload_temp_dir / safe_filename
        print(f"Uploading file: {file.filename} -> {file_path}")

        # チャンク単位で書き込み（大きなファイル対応）
        total_size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)
                total_size += len(chunk)

        print(f"File saved: {file_path} ({total_size} bytes)")

        # 動画ファイルの場合は最初のフレームを取得、画像ファイルはそのまま返す
        result: Dict[str, Any] = {"path": str(file_path)}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        suffix_lower = file_path.suffix.lower()

        if suffix_lower in video_extensions:
            first_frame = get_video_first_frame_base64(str(file_path))
            if first_frame:
                result["first_frame"] = first_frame
            # 総フレーム数を取得
            cap = cv2.VideoCapture(str(file_path))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count > 0:
                    result["frame_count"] = frame_count
                cap.release()
        elif suffix_lower in image_extensions:
            image_data = get_image_base64(str(file_path))
            if image_data:
                result["first_frame"] = image_data

        return result
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 動画ダウンロード用のテンポラリディレクトリ（プロジェクトルート/temp/videos）
_video_temp_dir = project_root / "temp" / "videos"
_video_temp_dir.mkdir(parents=True, exist_ok=True)

# オーディオダウンロード用のテンポラリディレクトリ（プロジェクトルート/temp/audio）
_audio_temp_dir = project_root / "temp" / "audio"
_audio_temp_dir.mkdir(parents=True, exist_ok=True)

# キャプチャダウンロード用のテンポラリディレクトリ（プロジェクトルート/temp/captures）
_captures_temp_dir = project_root / "temp" / "captures"
_captures_temp_dir.mkdir(parents=True, exist_ok=True)

# テキストダウンロード用のテンポラリディレクトリ（プロジェクトルート/temp/texts）
_texts_temp_dir = project_root / "temp" / "texts"
_texts_temp_dir.mkdir(parents=True, exist_ok=True)


@app.get("/api/download")
async def download_file(path: str):
    """
    指定されたファイルをダウンロードする。
    セキュリティのため、許可されたディレクトリからのみダウンロード可能。
    """
    try:
        file_path = Path(path)

        # セキュリティチェック: 許可されたディレクトリ内のファイルのみ
        allowed_dirs = [_upload_temp_dir, _video_temp_dir, _audio_temp_dir, _captures_temp_dir, _texts_temp_dir]
        is_allowed = any(
            file_path.resolve().is_relative_to(allowed_dir.resolve())
            for allowed_dir in allowed_dirs
            if allowed_dir.exists()
        )

        if not is_allowed:
            raise HTTPException(status_code=403, detail="Access denied")

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # ファイル名を取得してダウンロード
        filename = file_path.name
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/nodes/definitions")
async def get_node_definitions():
    """登録されているノード定義の一覧を返す（reactflow対応のみ、order順でソート）"""
    # reactflow対応ノードのみ取得
    node_definitions = get_all_nodes_for_gui("reactflow")

    defs = []
    for node_def_instance in node_definitions:
        # プロパティを処理（動的オプションを解決、GPU条件フィルタ）
        properties = []
        for prop in node_def_instance.properties:
            # requires_gpu=TrueのプロパティはCUDA利用不可時にスキップ
            if prop.requires_gpu and not CUDA_AVAILABLE:
                continue
            prop_dict = prop.model_dump()
            # options_sourceに基づいて動的にオプションを設定
            if prop.options_source == "cameras":
                prop_dict["options"] = get_camera_options()
            elif prop.options_source == "audio_inputs":
                prop_dict["options"] = get_audio_input_options()
            elif prop.options_source == "audio_outputs":
                prop_dict["options"] = get_audio_output_options()
            properties.append(prop_dict)

        defs.append({
            "definition_id": node_def_instance.definition_id,
            "version": node_def_instance.version,
            "display_name": node_def_instance.display_name,
            "description": node_def_instance.description,
            "order": node_def_instance.order,
            "resizable": node_def_instance.resizable,
            "no_duplicate": node_def_instance.no_duplicate,
            "dynamic_ports": node_def_instance.dynamic_ports,
            "inputs": [p.model_dump() for p in node_def_instance.inputs],
            "outputs": [p.model_dump() for p in node_def_instance.outputs],
            "properties": properties,
        })
    # order順でソート
    defs.sort(key=lambda x: x["order"])
    return defs


@app.get("/api/nodes/categories")
async def get_categories():
    """登録されているカテゴリの一覧を返す（order順）"""
    categories = get_all_categories()
    return [
        {
            "category_id": c.category_id,
            "display_name": c.display_name,
            "order": c.order,
            "default_open": c.default_open,
        }
        for c in categories
    ]


@app.post("/api/graph/execute")
async def execute_graph(graph_data: Dict[str, Any]):
    """グラフを1回実行する"""
    try:
        graph = Graph(**graph_data)
        results, elapsed_ms, node_times, node_errors, connected_props, gui_overhead_ms = graph.execute()
        return {
            "status": "success",
            "results": results,
            "elapsed_ms": elapsed_ms,
            "node_times": node_times,
            "node_errors": node_errors,
            "connected_properties": connected_props,
            "gui_overhead_ms": gui_overhead_ms,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StreamState:
    """WebSocketストリーミングの状態管理"""
    def __init__(self):
        self.graph = None
        self.running = False
        self.interval_ms = 100
        self.loop = True


@app.websocket("/api/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocketでグラフを連続実行し、結果をストリーミング。
    メッセージ受信と実行を別タスクで並行処理。
    """
    await websocket.accept()
    state = StreamState()

    async def receive_messages():
        """クライアントからのメッセージを受信"""
        try:
            while True:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)
                msg_type = message.get("type")

                if msg_type == "set_graph":
                    graph_data = message.get("graph")
                    state.graph = Graph(**graph_data)

                elif msg_type == "start":
                    # START時にWebRTCオーディオバッファをクリア（古いデータを破棄）
                    print("[Backend] START received - clearing audio buffers")
                    webrtc_frame_store.clear_all_audio_buffers()
                    state.running = True
                    state.interval_ms = message.get("interval_ms", 100)
                    state.loop = message.get("loop", True)
                    print(f"[Backend] START complete - running={state.running}")

                elif msg_type == "set_loop":
                    state.loop = message.get("loop", True)

                elif msg_type == "stop":
                    print("[Backend] STOP received")
                    state.running = False
                    # 処理中のノードをキャンセル
                    cancel_all_nodes()
                    # STOP時にis_streaming=falseで一度実行（録画停止処理など）
                    # preview=Trueでrun_when_stopped=falseのノードはスキップ
                    if state.graph is not None:
                        try:
                            stop_context = {
                                "is_streaming": False,
                                "interval_ms": state.interval_ms,
                                "preview": True,
                            }
                            results, elapsed_ms, node_times, node_errors, connected_props, gui_overhead_ms = state.graph.execute(stop_context)
                            await websocket.send_json({
                                "type": "result",
                                "results": results,
                                "elapsed_ms": elapsed_ms,
                                "node_times": node_times,
                                "node_errors": node_errors,
                                "connected_properties": connected_props,
                                "gui_overhead_ms": gui_overhead_ms,
                            })
                        except Exception as e:
                            print(f"Error in stop execution: {e}")

                elif msg_type == "reset":
                    # 全ノードの状態をリセット
                    reset_all_nodes()

                elif msg_type == "release_webcams":
                    # Webcamカメラを解放（ノードのreset()を呼び出す）
                    reset_node_by_id("image.input.webcam")

                elif msg_type == "execute_once":
                    # 1回だけ実行（プレビュー用）
                    graph_data = message.get("graph")
                    if graph_data:
                        try:
                            graph = Graph(**graph_data)
                            # プレビューモードフラグを渡す
                            preview_context = {"preview": True}
                            results, elapsed_ms, node_times, node_errors, connected_props, gui_overhead_ms = graph.execute(preview_context)
                            await websocket.send_json({
                                "type": "result",
                                "results": results,
                                "elapsed_ms": elapsed_ms,
                                "node_times": node_times,
                                "node_errors": node_errors,
                                "connected_properties": connected_props,
                                "gui_overhead_ms": gui_overhead_ms,
                            })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e)
                            })

        except WebSocketDisconnect:
            state.running = False

    async def execute_loop():
        """グラフを連続実行"""
        try:
            while True:
                # 実行前に他のタスク（メッセージ受信等）に制御を渡す
                await asyncio.sleep(0)

                if state.running and state.graph is not None:
                    try:
                        # コンテキストにループ設定とインターバルと再生状態を渡す
                        context = {
                            "loop": state.loop,
                            "interval_ms": state.interval_ms,
                            "is_streaming": True,
                        }

                        # 現在のグラフを保持（実行中に変更される可能性があるため）
                        current_graph = state.graph

                        # 重い処理を別スレッドで実行してイベントループをブロックしない
                        results, elapsed_ms, node_times, node_errors, connected_props, gui_overhead_ms = await asyncio.to_thread(
                            current_graph.execute, context
                        )

                        # 実行中にSTOPされた場合は結果を送信しない
                        if not state.running:
                            continue

                        # 終了シグナルをチェック
                        ended = results.pop("__ended__", False)

                        await websocket.send_json({
                            "type": "result",
                            "results": results,
                            "elapsed_ms": elapsed_ms,
                            "node_times": node_times,
                            "node_errors": node_errors,
                            "connected_properties": connected_props,
                            "gui_overhead_ms": gui_overhead_ms,
                        })

                        # 終了シグナルがあればフロントエンドに通知してSTOP
                        if ended:
                            state.running = False
                            await websocket.send_json({"type": "ended"})

                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })

                    # 処理時間を差し引いた残り時間だけスリープ
                    # 処理がインターバルを超えた場合は即座に次を実行
                    remaining_ms = state.interval_ms - elapsed_ms
                    if remaining_ms > 0:
                        await asyncio.sleep(remaining_ms / 1000.0)
                else:
                    await asyncio.sleep(0.05)

        except WebSocketDisconnect:
            pass

    # 受信タスクと実行タスクを並行実行
    receive_task = asyncio.create_task(receive_messages())
    execute_task = asyncio.create_task(execute_loop())

    try:
        await asyncio.gather(receive_task, execute_task)
    except Exception:
        pass
    finally:
        receive_task.cancel()
        execute_task.cancel()
        print("WebSocket disconnected")


# --- Colab環境: フロントエンド静的ファイル配信 ---
if is_colab():
    frontend_dist = project_root / "src" / "gui" / "reactflow" / "frontend" / "dist"
    index_html = frontend_dist / "index.html"

    # assetsディレクトリをマウント
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    async def serve_index():
        """index.htmlを配信"""
        if not index_html.exists():
            raise HTTPException(status_code=404, detail="Frontend not built. Run 'npm run build' first.")
        return FileResponse(index_html)

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        """SPAフォールバック（API・assets以外はindex.htmlを返す）"""
        if full_path.startswith(("api/", "assets/")):
            raise HTTPException(status_code=404, detail="Not found")
        if not index_html.exists():
            raise HTTPException(status_code=404, detail="Frontend not built. Run 'npm run build' first.")
        return FileResponse(index_html)
