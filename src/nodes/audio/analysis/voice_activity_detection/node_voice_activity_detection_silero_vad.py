#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import warnings
from collections import deque
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np
import onnxruntime  # type: ignore
from node.node_abc import DpgNodeABC  # type: ignore
from node_editor.util import (  # type: ignore
    dpg_set_value,
    get_tag_name_list,
)


class SileroVadOnnxWrapper:
    def __init__(self, path, force_onnx_cpu=False):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if (
            force_onnx_cpu
            and "CPUExecutionProvider" in onnxruntime.get_available_providers()
        ):
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts
            )
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()
        if "16k" in path:
            warnings.warn("This model supports only 16000 sampling rate!")
            self.sample_rates = [16000]
        else:
            self.sample_rates = [8000, 16000]

    def _validate_input(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        if x.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk: {x.ndim}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(
                f"Supported sampling rates: {self.sample_rates} (or multiple of 16000)"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size: int = 1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = None
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x: np.ndarray, sr: int) -> np.ndarray:
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} (Supported: 256 for 8kHz, 512 for 16kHz)"
            )

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if (
            self._last_sr != sr
            or self._last_batch_size != batch_size
            or self._last_batch_size == 0
        ):
            self.reset_states(batch_size)

        if self._context is None:
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x = np.concatenate([self._context, x], axis=1)

        ort_inputs = {
            "input": x.astype(np.float32),
            "state": self._state.astype(np.float32),
            "sr": np.array(sr, dtype=np.int64),
        }

        out, new_state = self.session.run(None, ort_inputs)
        self._state = new_state
        self._context = x[:, -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return out

    def audio_forward(self, x: np.ndarray, sr: int) -> np.ndarray:
        outs = []
        x, sr = self._validate_input(x, sr)
        self.reset_states()
        num_samples = 512 if sr == 16000 else 256

        if x.shape[1] % num_samples != 0:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = np.pad(x, ((0, 0), (0, pad_num)), mode="constant")

        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i : i + num_samples]
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)

        return np.concatenate(outs, axis=1)


class SileroVadIterator:
    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError("VADIterator supports only 8000 and 16000 Hz")

        self.min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x: np.ndarray, return_seconds=False, time_resolution: int = 1):
        """
        x: np.ndarray, shape [1, N] or [N]
            Audio chunk
        return_seconds: bool
            If True, output start/end in seconds instead of samples
        time_resolution: int
            Decimal places for seconds
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input audio must be a NumPy array")

        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        elif x.ndim != 2:
            raise ValueError("Input must be 1D or 2D NumPy array")

        window_size_samples = x.shape[1]
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if speech_prob >= self.threshold and self.temp_end:
            self.temp_end = 0

        if speech_prob >= self.threshold and not self.triggered:
            self.triggered = True
            speech_start = max(
                0, self.current_sample - self.speech_pad_samples - window_size_samples
            )
            return {
                "start": int(speech_start)
                if not return_seconds
                else round(speech_start / self.sampling_rate, time_resolution)
            }

        if speech_prob < self.threshold - 0.15 and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = (
                    self.temp_end + self.speech_pad_samples - window_size_samples
                )
                self.temp_end = 0
                self.triggered = False
                return {
                    "end": int(speech_end)
                    if not return_seconds
                    else round(speech_end / self.sampling_rate, time_resolution)
                }

        return None


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Voice Activity Detection(Silero VAD)"
    node_tag = "SileroVAD"

    def __init__(self):
        self._node_data = {}

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        setting_dict=None,
        callback=None,
    ):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_SIGNAL_CHUNK],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_FLOAT, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]
        output_tag_list = tag_name_list[2]

        # 設定
        self._setting_dict = setting_dict or {}
        waveform_w: int = self._setting_dict.get("waveform_width", 200)
        waveform_h: int = self._setting_dict.get("waveform_height", 400)
        self._default_sampling_rate: int = self._setting_dict.get(
            "default_sampling_rate", 16000
        )
        self._chunk_size: int = self._setting_dict.get("chunk_size", 1024)
        self._use_pref_counter: bool = self._setting_dict["use_pref_counter"]

        # モデル読み込み
        model = SileroVadOnnxWrapper(
            "node/time_domain_node/model/silero_vad_v6.onnx", force_onnx_cpu=True
        )
        self._node_data[str(node_id)] = {
            "buffer": np.zeros(0, dtype=np.float32),
            "display_x_buffer": np.array([]),
            "display_y_buffer": np.array([]),
            "chunk": np.array([]),
            "current_chunk_index": 0,
            "vad_iterator": SileroVadIterator(
                model, sampling_rate=self._default_sampling_rate
            ),
            "speech_state": 0,
        }

        # 表示用バッファ用意
        buffer_len: int = self._default_sampling_rate * 5
        self._node_data[str(node_id)]["display_y_buffer"] = deque(
            [0.0] * buffer_len, maxlen=buffer_len
        )
        self._node_data[str(node_id)]["display_x_buffer"] = (
            np.arange(len(self._node_data[str(node_id)]["display_y_buffer"]))
            / self._default_sampling_rate
        )

        # ノード
        with dpg.node(
            tag=tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            # 入力端子
            with dpg.node_attribute(
                tag=input_tag_list[0][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=input_tag_list[0][1],
                    default_value="Input Chunk",
                )
            # プロットエリア
            with dpg.node_attribute(
                tag=output_tag_list[0][0],
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                with dpg.plot(
                    height=waveform_h,
                    width=waveform_w,
                    no_inputs=False,
                    tag=f"{node_id}:audio_plot_area",
                ):
                    dpg.add_plot_axis(
                        dpg.mvXAxis,
                        label="Time(s)",
                        no_label=True,
                        no_tick_labels=True,
                        tag=f"{node_id}:xaxis",
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis,
                        label="Amplitude",
                        no_label=True,
                        no_tick_labels=True,
                        tag=f"{node_id}:yaxis",
                    )
                    dpg.set_axis_limits(f"{node_id}:xaxis", 0.0, 5.0)
                    dpg.set_axis_limits(f"{node_id}:yaxis", -0.5, 1.5)

                    dpg.add_line_series(
                        self._node_data[str(node_id)]["display_x_buffer"].tolist(),
                        list(self._node_data[str(node_id)]["display_y_buffer"]),
                        parent=f"{node_id}:yaxis",
                        tag=f"{node_id}:audio_line_series",
                    )
            # VAD結果
            with dpg.node_attribute(
                tag=output_tag_list[1][0],
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                # dpg.add_input_text(
                #     tag=output_tag_list[1][1],
                #     width=waveform_w - 190,
                #     label="Activity (0:OFF 1:ON)",
                #     default_value="0",
                #     callback=None,
                #     readonly=True,
                # )
                dpg.add_slider_int(
                    tag=output_tag_list[1][1],
                    width=waveform_w - 190,
                    label="Activity (0:OFF 1:ON)",
                    default_value=0,
                    min_value=0,
                    max_value=1,
                    enabled=False,
                )
            # 処理時間
            if self._use_pref_counter:
                with dpg.node_attribute(
                    tag=output_tag_list[2][0],
                    attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=output_tag_list[2][1],
                        default_value="elapsed time(ms)",
                    )

        self._prev_time = time.perf_counter()

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        player_status_dict,
        node_result_dict,
    ):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_SIGNAL_CHUNK],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_FLOAT, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        output_tag_list = tag_name_list[2]

        # 計測開始
        if self._use_pref_counter:
            start_time = time.perf_counter()

        chunk: Optional[np.ndarray] = np.array([])
        chunk_index: int = -1

        # 接続情報確認
        chunk: Optional[np.ndarray] = np.zeros([self._chunk_size], dtype=np.float32)
        for connection_info in connection_list:
            connection_type = connection_info[0].split(":")[2]
            if connection_type == self.TYPE_SIGNAL_CHUNK:
                # 接続タグ取得
                source_node = ":".join(connection_info[0].split(":")[:2])
                destination_node = ":".join(connection_info[1].split(":")[:2])
                if tag_node_name == destination_node:
                    chunk_index = node_result_dict[source_node].get("chunk_index", -1)
                    chunk = node_result_dict[source_node].get("chunk", np.array([]))
                    break

        # プロット
        current_status = player_status_dict.get("current_status", False)
        if current_status == "play" and chunk_index >= 0 and len(chunk) > 0:
            if len(chunk) < self._chunk_size:
                chunk = np.pad(
                    chunk, (0, self._chunk_size - len(chunk)), constant_values=0
                )
            if self._node_data[str(node_id)]["current_chunk_index"] < chunk_index:
                if (
                    chunk_index
                    != self._node_data[str(node_id)]["current_chunk_index"] + 1
                    and self._node_data[str(node_id)]["current_chunk_index"] != -1
                ):
                    print(
                        f"    [Warning] Silero VAD Node Chunk Index Gap: {chunk_index - self._node_data[str(node_id)]['current_chunk_index']} (Index: {self._node_data[str(node_id)]['current_chunk_index']} -> {chunk_index})"
                    )

                # チャンク
                self._node_data[str(node_id)]["chunk"] = chunk
                self._node_data[str(node_id)]["current_chunk_index"] = chunk_index

                # バッファ更新
                self._node_data[str(node_id)]["buffer"] = np.concatenate(
                    (self._node_data[str(node_id)]["buffer"], chunk)
                )

                while len(self._node_data[str(node_id)]["buffer"]) > 512:
                    # VAD用チャンク取り出し
                    temp_buffer: np.ndarray = self._node_data[str(node_id)]["buffer"][
                        :512
                    ]

                    # VAD
                    speech_dict = self._node_data[str(node_id)]["vad_iterator"](
                        temp_buffer, return_seconds=False
                    )
                    if speech_dict:
                        if "start" in speech_dict:
                            self._node_data[str(node_id)]["speech_state"] = 1
                        elif "end" in speech_dict:
                            self._node_data[str(node_id)]["speech_state"] = 0

                    # 先頭を削除
                    self._node_data[str(node_id)]["buffer"] = self._node_data[
                        str(node_id)
                    ]["buffer"][512:]

                # 表示チャンク用意
                speech_state = self._node_data[str(node_id)]["speech_state"]
                display_chunk: Optional[np.ndarray] = np.full(
                    [self._chunk_size], fill_value=speech_state, dtype=np.float32
                )

                # プロット
                temp_display_y_buffer = self._node_data[str(node_id)][
                    "display_y_buffer"
                ]
                temp_display_y_buffer = np.roll(
                    temp_display_y_buffer, -self._chunk_size
                )
                temp_display_y_buffer[-self._chunk_size :] = display_chunk
                self._node_data[str(node_id)]["display_y_buffer"] = (
                    temp_display_y_buffer
                )
                dpg.set_value(
                    f"{node_id}:audio_line_series",
                    [
                        self._node_data[str(node_id)]["display_x_buffer"],
                        temp_display_y_buffer,
                    ],
                )

                # 値設定
                dpg_set_value(output_tag_list[1][1], speech_state)
        elif current_status == "stop":
            # バッファ初期化
            self._node_data[str(node_id)]["buffer"] = np.zeros(0, dtype=np.float32)
            self._node_data[str(node_id)]["speech_state"] = 0

            self._node_data[str(node_id)]["vad_iterator"].reset_states()

            # プロットエリア初期化
            self._node_data[str(node_id)]["current_chunk_index"] = -1

            buffer_len: int = self._default_sampling_rate * 5
            self._node_data[str(node_id)]["display_y_buffer"] = deque(
                [0.0] * buffer_len, maxlen=buffer_len
            )
            self._node_data[str(node_id)]["display_x_buffer"] = (
                np.arange(len(self._node_data[str(node_id)]["display_y_buffer"]))
                / self._default_sampling_rate
            )
            dpg.set_value(
                f"{node_id}:audio_line_series",
                [
                    self._node_data[str(node_id)]["display_x_buffer"].tolist(),
                    list(self._node_data[str(node_id)]["display_y_buffer"]),
                ],
            )

        result_dict = {
            "chunk_index": self._node_data[str(node_id)]["current_chunk_index"],
            "chunk": self._node_data[str(node_id)]["chunk"],
        }

        # 計測終了
        if self._use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_tag_list[2][1], str(elapsed_time).zfill(4) + "ms")

        return result_dict

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_SIGNAL_CHUNK],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_FLOAT, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]

        pos: List[int] = dpg.get_item_pos(tag_node_name)

        setting_dict: Dict[str, Any] = {
            "ver": self._ver,
            "pos": pos,
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        pass
