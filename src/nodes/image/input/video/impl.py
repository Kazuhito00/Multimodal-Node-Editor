from typing import Dict, Any, Optional, List
from collections import OrderedDict
from node_editor.node_def import ComputeLogic
from node_editor.settings import get_setting
import numpy as np
import threading
import time
from math import gcd

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    av = None


def get_sample_rate() -> int:
    return get_setting("audio.sample_rate", 16000)


def get_interval_ms() -> int:
    return get_setting("graph.interval_ms", 50)


class VideoFileBuffer:
    """
    動画ファイルのストリーミングバッファ（時間ベース同期）
    オーディオは事前に全部読み込み、ビデオは時間ベースで取得
    フレーム先読み機能付き（カクつき防止）
    """

    def __init__(self, sample_rate: int, buffer_seconds: float = 5.0, prefetch_seconds: float = 1.0):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.buffer_size = int(sample_rate * buffer_seconds)
        self.prefetch_seconds = prefetch_seconds

        # 表示用循環バッファ（オーディオ波形表示用）
        self.display_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.valid_samples = 0

        # ビデオキャプチャ
        self.video_cap: Optional[Any] = None
        self.video_fps: float = 30.0
        self.video_total_frames: int = 0
        self.video_width: int = 0
        self.video_height: int = 0

        # オーディオデータ（事前読み込み）
        self.audio_data: Optional[np.ndarray] = None
        self.audio_play_pos: int = 0
        self.has_audio: bool = False

        # 再生状態
        self.file_path: str = ""
        self.duration_seconds: float = 0.0
        self.is_loaded: bool = False

        # 時間ベース再生管理
        self.playback_start_time: Optional[float] = None  # 再生開始した実時間
        self.playback_start_position: float = 0.0  # 再生開始した動画内位置（秒）
        self.last_playback_time: float = 0.0  # 最後に取得した動画内時間

        # フレーム位置管理（内部追跡）
        self.current_frame_index: int = 0  # 現在読み込んだフレーム番号
        self.last_displayed_frame: int = -1  # 最後に表示したフレーム番号

        # キャッシュされたフレーム
        self.current_frame: Optional[np.ndarray] = None
        self.first_frame: Optional[np.ndarray] = None

        # フレーム先読みバッファ
        self.frame_cache: Dict[int, np.ndarray] = {}
        self.prefetch_frame_count: int = 0  # 先読みするフレーム数（FPSベースで計算）
        self.prefetch_thread: Optional[threading.Thread] = None
        self.prefetch_stop_event = threading.Event()
        self.prefetch_target_frame: int = 0  # 先読み先頭位置

        # バックグラウンド読み込み用
        self.reader_thread: Optional[threading.Thread] = None
        self.reader_stop_event = threading.Event()
        self.reader_target_frame: int = 0  # 読み込み目標フレーム
        self.reader_running: bool = False

        # 全フレームプリロード済みフラグ
        self.all_frames_preloaded: bool = False

        self.lock = threading.Lock()
        self.cache_lock = threading.Lock()  # キャッシュ専用ロック

    def _stop_prefetch_thread(self):
        """先読みスレッドを停止"""
        if self.prefetch_thread is not None:
            self.prefetch_stop_event.set()
            self.prefetch_thread.join(timeout=1.0)
            self.prefetch_thread = None
            self.prefetch_stop_event.clear()

    def _stop_reader_thread(self):
        """バックグラウンドリーダースレッドを停止"""
        self.reader_running = False
        if self.reader_thread is not None:
            self.reader_stop_event.set()
            self.reader_thread.join(timeout=2.0)
            self.reader_thread = None
            self.reader_stop_event.clear()

    def _start_reader_thread(self):
        """バックグラウンドリーダースレッドを開始"""
        if self.reader_thread is not None and self.reader_thread.is_alive():
            return
        self.reader_stop_event.clear()
        self.reader_running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def _reader_loop(self):
        """バックグラウンドでフレームを先読みするループ"""
        # 読み込み専用のVideoCaptureを作成
        reader_cap = cv2.VideoCapture(self.file_path, cv2.CAP_ANY)
        if not reader_cap.isOpened():
            print("Failed to open video for background reading")
            return

        reader_frame_index = 0

        while self.reader_running and not self.reader_stop_event.is_set():
            # 現在の目標フレームを取得
            with self.lock:
                target = self.reader_target_frame
                total_frames = self.video_total_frames
                prefetch_count = self.prefetch_frame_count

            # 先読み範囲を計算
            prefetch_start = target
            prefetch_end = min(target + prefetch_count, total_frames)

            # キャッシュにないフレームを読み込み
            frames_read = 0
            for frame_idx in range(prefetch_start, prefetch_end):
                if self.reader_stop_event.is_set():
                    break

                # 既にキャッシュにあればスキップ
                with self.cache_lock:
                    if frame_idx in self.frame_cache:
                        continue

                # シークが必要な場合
                if reader_frame_index != frame_idx:
                    reader_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    reader_frame_index = frame_idx

                # フレームを読み込み
                ret, frame = reader_cap.read()
                if ret and frame is not None:
                    with self.cache_lock:
                        self.frame_cache[frame_idx] = frame
                    reader_frame_index = frame_idx + 1
                    frames_read += 1
                else:
                    break

            # 古いフレームをクリア（現在位置より前）
            with self.cache_lock:
                keys_to_remove = [k for k in self.frame_cache.keys() if k < target - 5]
                for k in keys_to_remove:
                    del self.frame_cache[k]

            # 読み込むフレームがなければ少し待機
            if frames_read == 0:
                time.sleep(0.01)

        reader_cap.release()

    def _release_captures(self):
        """キャプチャを解放（ロックなしで呼び出すこと）"""
        self._stop_reader_thread()
        self._stop_prefetch_thread()
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        with self.cache_lock:
            self.frame_cache.clear()

    def load_file(self, file_path: str, preload_all: bool = False) -> bool:
        """動画ファイルを読み込む（オーディオは事前に全部読み込み）"""
        if not CV2_AVAILABLE or not file_path:
            return False

        try:
            # 既存のキャプチャを解放
            self._release_captures()

            # 映像キャプチャを開く
            video_cap = cv2.VideoCapture(file_path, cv2.CAP_ANY)
            if not video_cap.isOpened():
                print(f"Failed to open video: {file_path}")
                return False

            fps = video_cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0.0

            print(f"Video: {width}x{height}, {fps:.2f}fps, {duration:.2f}s, {total_frames} frames")

            # 最初のフレームを読み込んでキャッシュ
            first_frame = None
            ret, frame = video_cap.read()
            if ret and frame is not None:
                first_frame = frame.copy()
            # 位置を確実に0にリセット
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # オーディオを事前に全部読み込む
            audio_data = None
            has_audio = False
            audio_data = self._load_all_audio(file_path)
            if audio_data is not None and len(audio_data) > 0:
                has_audio = True
                print(f"Audio: {len(audio_data)} samples loaded")

            # 先読みフレーム数を計算
            if preload_all:
                # 全フレームを先読み
                prefetch_frame_count = total_frames
            else:
                # 1秒分のみ先読み
                prefetch_frame_count = int(fps * self.prefetch_seconds)
                prefetch_frame_count = min(prefetch_frame_count, total_frames)

            with self.lock:
                self.video_cap = video_cap
                self.video_fps = fps
                self.video_total_frames = total_frames
                self.video_width = width
                self.video_height = height

                self.audio_data = audio_data
                self.audio_play_pos = 0
                self.has_audio = has_audio

                self.file_path = file_path
                self.duration_seconds = duration
                self.is_loaded = True
                self.first_frame = first_frame
                self.current_frame = first_frame

                # 再生状態をリセット
                self.playback_start_time = None
                self.playback_start_position = 0.0
                self.last_playback_time = 0.0
                self.current_frame_index = 0
                self.last_displayed_frame = -1

                # 表示バッファをリセット
                self.display_buffer = np.zeros(self.buffer_size, dtype=np.float32)
                self.write_pos = 0
                self.valid_samples = 0

                # フレーム先読み設定
                self.prefetch_frame_count = prefetch_frame_count
                self.frame_cache.clear()
                self.prefetch_target_frame = 0

            # フレームを先読み（同期的に実行）
            if preload_all:
                print(f"Preloading all {total_frames} frames...")
                self._prefetch_frames_sync(0, total_frames)
                self.all_frames_preloaded = True
                print(f"Preloaded all frames")
            else:
                self._prefetch_frames_sync(0, prefetch_frame_count)
                self.all_frames_preloaded = False
                # バックグラウンドリーダースレッドを開始（preload_allでない場合のみ）
                self._start_reader_thread()

            print(f"Video loaded: {file_path} (prefetched {prefetch_frame_count} frames)")
            return True

        except Exception as e:
            print(f"Error loading video file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_all_audio(self, file_path: str) -> Optional[np.ndarray]:
        """PyAVを使って動画からオーディオを全部読み込む"""
        if not AV_AVAILABLE:
            print("PyAV is not available, audio will not be loaded")
            return None

        try:
            container = av.open(file_path)

            # オーディオストリームを取得
            audio_streams = [s for s in container.streams if s.type == 'audio']
            if not audio_streams:
                container.close()
                return None

            audio_stream = audio_streams[0]

            # リサンプラーを設定（指定サンプルレート、モノラル、float32）
            resampler = av.AudioResampler(
                format='flt',
                layout='mono',
                rate=self.sample_rate,
            )

            # 全オーディオフレームを読み込む
            all_samples: List[np.ndarray] = []

            for frame in container.decode(audio_stream):
                # リサンプリング
                resampled_frames = resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    if resampled_frame is not None:
                        # NumPy配列に変換（float32, モノラル）
                        samples = resampled_frame.to_ndarray().flatten()
                        all_samples.append(samples)

            # 残りのバッファをフラッシュ
            flushed_frames = resampler.resample(None)
            for flushed_frame in flushed_frames:
                if flushed_frame is not None:
                    samples = flushed_frame.to_ndarray().flatten()
                    all_samples.append(samples)

            container.close()

            if not all_samples:
                return None

            # 全サンプルを結合
            audio_data = np.concatenate(all_samples).astype(np.float32)
            return audio_data

        except Exception as e:
            print(f"Error loading audio with PyAV: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _prefetch_frames_sync(self, start_frame: int, count: int):
        """指定位置から指定数のフレームを先読み（同期実行、初回読み込み用）"""
        if not CV2_AVAILABLE or self.video_cap is None:
            return

        with self.lock:
            # キャプチャ位置を設定
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.current_frame_index = start_frame

        # フレームを読み込んでキャッシュ
        for i in range(count):
            frame_idx = start_frame + i
            if frame_idx >= self.video_total_frames:
                break

            with self.lock:
                ret, frame = self.video_cap.read()
                if ret and frame is not None:
                    with self.cache_lock:
                        self.frame_cache[frame_idx] = frame.copy()
                    self.current_frame_index = frame_idx + 1
                else:
                    break

        with self.lock:
            self.prefetch_target_frame = start_frame + count

    def _get_cached_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """キャッシュからフレームを取得（なければNone）"""
        with self.cache_lock:
            return self.frame_cache.get(frame_idx)

    def start_playback(self):
        """再生を開始（時間計測開始）"""
        with self.lock:
            if self.playback_start_time is None:
                self.playback_start_time = time.perf_counter()
                self.playback_start_position = self.last_playback_time

    def seek_to_position(self, position_percent: float):
        """指定位置にシーク（0-100%）"""
        with self.lock:
            if not self.is_loaded:
                return

            position_percent = max(0.0, min(100.0, position_percent))
            position_seconds = (position_percent / 100.0) * self.duration_seconds

            # ビデオ位置を更新
            frame_number = 0
            if self.video_cap is not None:
                frame_number = int(position_seconds * self.video_fps)
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                self.current_frame_index = frame_number
                # 次に表示するフレームがframe_numberになるように設定
                self.last_displayed_frame = frame_number - 1

            # オーディオ再生位置を更新
            if self.audio_data is not None:
                self.audio_play_pos = int(position_seconds * self.sample_rate)
                self.audio_play_pos = min(self.audio_play_pos, len(self.audio_data))

            # 再生位置を更新
            self.last_playback_time = position_seconds
            if self.playback_start_time is not None:
                # 再生中ならタイマーをリセット
                self.playback_start_time = time.perf_counter()
                self.playback_start_position = position_seconds

            # 表示バッファをリセット
            self.display_buffer = np.zeros(self.buffer_size, dtype=np.float32)
            self.write_pos = 0
            self.valid_samples = 0

            # 全フレームプリロード済みの場合はキャッシュをクリアしない
            if not self.all_frames_preloaded:
                with self.cache_lock:
                    self.frame_cache.clear()
                self.reader_target_frame = frame_number

    def get_frame_and_audio_for_time(
        self, num_audio_samples: int, loop: bool = True, realtime_sync: bool = True,
        frame_step: int = 1
    ) -> tuple:
        """
        フレームと音声を取得
        realtime_sync=True: リアルタイム同期（フレームスキップあり）
        realtime_sync=False: フレーム順次再生（frame_stepずつ進める）
        frame_step: フレーム順次再生時の進行ステップ数（1=毎フレーム、2=2フレームごと）
        戻り値: (frame, audio_samples, ended)
        """
        with self.lock:
            if not self.is_loaded:
                zeros = np.zeros(num_audio_samples, dtype=np.float32)
                return self.first_frame, zeros, False

            # 終了判定フラグ
            ended = False

            if realtime_sync:
                # リアルタイム同期モード：時間ベースでフレームを計算
                if self.playback_start_time is None:
                    self.playback_start_time = time.perf_counter()
                    self.playback_start_position = self.last_playback_time

                elapsed_real_time = time.perf_counter() - self.playback_start_time
                current_video_time = self.playback_start_position + elapsed_real_time

                # 終端到達時の処理
                if current_video_time >= self.duration_seconds:
                    if loop:
                        current_video_time = current_video_time % self.duration_seconds
                        self.playback_start_time = time.perf_counter()
                        self.playback_start_position = current_video_time
                        self.audio_play_pos = 0
                        self.last_displayed_frame = -1
                        # 全フレームプリロード済みの場合はキャッシュをクリアしない
                        if not self.all_frames_preloaded:
                            with self.cache_lock:
                                self.frame_cache.clear()
                    else:
                        ended = True
                        current_video_time = self.duration_seconds

                target_frame = int(current_video_time * self.video_fps)
                target_frame = min(target_frame, self.video_total_frames - 1)
            else:
                # フレーム順次再生モード：frame_stepずつ進める
                if self.last_displayed_frame < 0:
                    target_frame = 0
                else:
                    target_frame = self.last_displayed_frame + frame_step

                if target_frame >= self.video_total_frames:
                    if loop:
                        target_frame = 0
                        self.last_displayed_frame = -1
                        self.audio_play_pos = 0
                        # 全フレームプリロード済みの場合はキャッシュをクリアしない
                        if not self.all_frames_preloaded:
                            with self.cache_lock:
                                self.frame_cache.clear()
                    else:
                        ended = True
                        target_frame = self.video_total_frames - 1

                current_video_time = target_frame / self.video_fps if self.video_fps > 0 else 0.0

            # バックグラウンドリーダーに目標フレームを通知
            self.reader_target_frame = target_frame

        # フレームを取得（キャッシュから）
        frame = self._get_cached_frame(target_frame)

        # キャッシュにない場合の待機処理
        if frame is None:
            if realtime_sync:
                # リアルタイム同期：短い待機
                max_wait_ms = 50
            else:
                # フレーム順次：長めに待機（スキップしない）
                max_wait_ms = 500

            wait_interval = 0.002
            waited = 0.0
            while frame is None and waited < max_wait_ms / 1000:
                time.sleep(wait_interval)
                waited += wait_interval
                frame = self._get_cached_frame(target_frame)

        # フレーム更新
        if frame is None:
            frame = self.current_frame if self.current_frame is not None else self.first_frame
        else:
            self.current_frame = frame
            self.last_displayed_frame = target_frame

        with self.lock:
            audio_samples = self._get_next_audio_samples(num_audio_samples)
            self.last_playback_time = current_video_time
            self._write_to_display_buffer(audio_samples)

        return frame, audio_samples, ended

    def _get_next_audio_samples(self, num_samples: int) -> np.ndarray:
        """事前読み込みしたオーディオから次のサンプルを取得（ループ再生）"""
        if not self.has_audio or self.audio_data is None:
            return np.zeros(num_samples, dtype=np.float32)

        result = np.zeros(num_samples, dtype=np.float32)
        remaining = num_samples
        write_idx = 0

        while remaining > 0:
            available = len(self.audio_data) - self.audio_play_pos
            to_copy = min(remaining, available)

            if to_copy > 0:
                result[write_idx:write_idx + to_copy] = \
                    self.audio_data[self.audio_play_pos:self.audio_play_pos + to_copy]

            self.audio_play_pos += to_copy
            write_idx += to_copy
            remaining -= to_copy

            # ファイル終端でループ
            if self.audio_play_pos >= len(self.audio_data):
                self.audio_play_pos = 0

        return result

    def _write_to_display_buffer(self, data: np.ndarray):
        """表示用バッファに書き込む"""
        n = len(data)
        if n >= self.buffer_size:
            self.display_buffer[:] = data[-self.buffer_size:]
            self.write_pos = 0
            self.valid_samples = self.buffer_size
        else:
            end_pos = self.write_pos + n
            if end_pos <= self.buffer_size:
                self.display_buffer[self.write_pos:end_pos] = data
            else:
                first_part = self.buffer_size - self.write_pos
                self.display_buffer[self.write_pos:] = data[:first_part]
                self.display_buffer[:n - first_part] = data[first_part:]
            self.write_pos = end_pos % self.buffer_size
            self.valid_samples = min(self.valid_samples + n, self.buffer_size)

    def get_waveform_display(self, display_width: int = 200) -> list:
        """表示用のmin/maxペアを生成"""
        with self.lock:
            if self.valid_samples == 0:
                return [0.0] * (display_width * 2)

            if self.valid_samples >= self.buffer_size:
                waveform = np.concatenate([
                    self.display_buffer[self.write_pos:],
                    self.display_buffer[:self.write_pos]
                ])
            else:
                start_pos = (self.write_pos - self.valid_samples) % self.buffer_size
                if start_pos < self.write_pos:
                    waveform = self.display_buffer[start_pos:self.write_pos].copy()
                else:
                    waveform = np.concatenate([
                        self.display_buffer[start_pos:],
                        self.display_buffer[:self.write_pos]
                    ])

            samples_per_pixel = self.buffer_size // display_width
            data_pixels = len(waveform) // samples_per_pixel if samples_per_pixel > 0 else 0
            empty_pixels = display_width - data_pixels

            result = []
            for _ in range(empty_pixels):
                result.extend([0.0, 0.0])

            for i in range(data_pixels):
                start = i * samples_per_pixel
                end = start + samples_per_pixel
                segment = waveform[start:end]
                if len(segment) > 0:
                    result.append(float(np.min(segment)))
                    result.append(float(np.max(segment)))
                else:
                    result.extend([0.0, 0.0])

            return result

    def get_current_position_percent(self) -> float:
        """現在の再生位置をパーセントで取得"""
        with self.lock:
            if not self.is_loaded or self.duration_seconds <= 0:
                return 0.0
            return (self.last_playback_time / self.duration_seconds) * 100.0

    def get_first_frame(self) -> Optional[np.ndarray]:
        """キャッシュされた最初のフレームを取得"""
        with self.lock:
            return self.first_frame

    def stop_playback(self):
        """再生を停止（時間計測停止）"""
        with self.lock:
            if self.playback_start_time is not None:
                # 現在位置を保存
                elapsed = time.perf_counter() - self.playback_start_time
                self.last_playback_time = self.playback_start_position + elapsed
                if self.last_playback_time >= self.duration_seconds:
                    self.last_playback_time = self.last_playback_time % self.duration_seconds
            self.playback_start_time = None

    def release(self):
        """リソースを解放"""
        # デッドロック防止: スレッド停止はロック外で行う
        self._stop_reader_thread()
        with self.lock:
            self._stop_prefetch_thread()
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
            with self.cache_lock:
                self.frame_cache.clear()
            self.is_loaded = False


class VideoPlayerLogic(ComputeLogic):
    """動画再生ノードロジック"""

    def __init__(self):
        self._buffer: Optional[VideoFileBuffer] = None
        self._current_file: str = ""
        self._current_preload_all: bool = False
        self._just_loaded: bool = False
        self._last_compute_time: Optional[float] = None

    def reset(self):
        """再生位置を先頭にリセット"""
        if self._buffer is not None:
            self._buffer.seek_to_position(0.0)
        # 次のcomputeで安定した時間計測を開始
        self._last_compute_time = time.perf_counter()

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if context is None:
            context = {}

        if not CV2_AVAILABLE:
            return {
                "image_out": None,
                "audio_out": None,
                "__error__": "opencv-python is not installed",
            }

        sample_rate = get_sample_rate()
        interval_ms = context.get("interval_ms", get_interval_ms())
        loop = context.get("loop", True)
        file_path = properties.get("file_path", "")
        frame_step = max(1, properties.get("frame_step", 1))
        realtime_sync = properties.get("realtime_sync", False)
        preload_all = properties.get("preload_all", False)

        # バッファ初期化
        if self._buffer is None:
            self._buffer = VideoFileBuffer(sample_rate, buffer_seconds=5.0)

        # ファイルパスが空になった場合（Undoなど）はリセット
        if not file_path:
            if self._current_file:
                self._current_file = ""
                self._just_loaded = False
            self._last_compute_time = None
            return {"image_out": None, "audio_out": None}

        # ファイルまたはpreload_all設定が変更されたら読み込み
        need_reload = file_path != self._current_file or preload_all != self._current_preload_all
        if need_reload:
            if self._buffer.load_file(file_path, preload_all=preload_all):
                self._current_file = file_path
                self._current_preload_all = preload_all
                self._last_property_position = 0.0
                self._just_loaded = True
                self._last_compute_time = None
            else:
                self._current_file = ""

        # ファイルが読み込まれていない場合
        if not self._current_file:
            self._last_compute_time = None
            return {"image_out": None, "audio_out": None}

        # 読み込み直後は最初のフレームを返す
        if self._just_loaded:
            self._just_loaded = False
            self._last_compute_time = time.perf_counter()
            first_frame = self._buffer.get_first_frame()
            return {
                "image_out": first_frame,
                "audio_out": {
                    "delta": [],
                    "waveform": [0.0] * 400,
                    "sample_rate": sample_rate,
                    "duration": 5.0,
                }
            }

        # STOP時は現在のフレームを表示するだけ（再生しない）
        is_streaming = context.get("is_streaming", False)
        if not is_streaming:
            # 再生を停止
            self._buffer.stop_playback()
            self._last_compute_time = None
            # 現在のフレーム（または最初のフレーム）を返す
            current_frame = self._buffer.current_frame if self._buffer.current_frame is not None else self._buffer.get_first_frame()
            return {
                "image_out": current_frame,
                "audio_out": {
                    "delta": [],
                    "waveform": self._buffer.get_waveform_display(200),
                    "sample_rate": sample_rate,
                    "duration": 5.0,
                }
            }

        # 実際の経過時間からサンプル数を計算
        current_time = time.perf_counter()
        if self._last_compute_time is None:
            # 初回は interval_ms を使用
            elapsed_ms = interval_ms
        else:
            elapsed_ms = (current_time - self._last_compute_time) * 1000
            # 異常値を制限（最小: interval_msの半分、最大: interval_msの3倍）
            elapsed_ms = max(interval_ms * 0.5, min(elapsed_ms, interval_ms * 3))
        self._last_compute_time = current_time

        num_samples = int(sample_rate * elapsed_ms / 1000)

        # フレームとオーディオを取得
        frame, delta_samples, ended = self._buffer.get_frame_and_audio_for_time(
            num_samples, loop=loop, realtime_sync=realtime_sync, frame_step=frame_step
        )

        # 表示用waveformを生成
        waveform_display = self._buffer.get_waveform_display(200)

        result = {
            "image_out": frame,
            "audio_out": {
                "delta": delta_samples.tolist(),
                "waveform": waveform_display,
                "sample_rate": sample_rate,
                "duration": 5.0,
            }
        }

        # 終了シグナルを追加（ループがオフで動画が終了した場合）
        if ended:
            result["__ended__"] = True

        return result
