#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from collections import deque
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg
import numpy as np
import webrtcvad
from node.node_abc import DpgNodeABC
from node_editor.util import dpg_set_value, get_tag_name_list


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Voice Activity Detection(WebRTC VAD)"
    node_tag = "WebRTC_VAD"

    _VAD_FRAME_MS = 30  # 10, 20, or 30
    _VAD_AGGRESSIVENESS = 0  # 0 to 3

    def __init__(self):
        self._node_data = {}

    def _change_aggressiveness(self, sender, app_data, user_data):
        node_id = user_data
        aggressiveness = int(app_data)
        self._node_data[str(node_id)]["vad"].set_mode(aggressiveness)
        self._node_data[str(node_id)]["aggressiveness"] = aggressiveness

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
        aggressiveness: int = self._setting_dict.get(
            "aggressiveness", self._VAD_AGGRESSIVENESS
        )

        # VADインスタンス作成
        vad = webrtcvad.Vad()
        vad.set_mode(aggressiveness)

        # VADフレームサイズ計算
        self._vad_frame_size = int(
            self._default_sampling_rate * (self._VAD_FRAME_MS / 1000.0)
        )

        self._node_data[str(node_id)] = {
            "vad": vad,
            "aggressiveness": aggressiveness,
            "buffer": np.zeros(0, dtype=np.float32),
            "display_x_buffer": np.array([]),
            "display_y_buffer": np.array([]),
            "chunk": np.array([]),
            "current_chunk_index": 0,
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
                tag=f"{tag_node_name}:plot",
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
            # VAD Aggressiveness
            with dpg.node_attribute(
                tag=f"{tag_node_name}:aggressiveness_combo",
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    tag=f"{node_id}:aggressiveness",
                    label="Aggressiveness",
                    items=["0", "1", "2", "3"],
                    width=(waveform_w - 100) // 2,
                    default_value=str(aggressiveness),
                    callback=self._change_aggressiveness,
                    user_data=node_id,
                )
            # VAD結果
            with dpg.node_attribute(
                tag=output_tag_list[1][0],
                attribute_type=dpg.mvNode_Attr_Output,
            ):
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
                        f"    [Warning] WebRTC VAD Node Chunk Index Gap: {chunk_index - self._node_data[str(node_id)]['current_chunk_index']} (Index: {self._node_data[str(node_id)]['current_chunk_index']} -> {chunk_index})"
                    )

                # チャンク
                self._node_data[str(node_id)]["chunk"] = chunk
                self._node_data[str(node_id)]["current_chunk_index"] = chunk_index

                # バッファ更新
                self._node_data[str(node_id)]["buffer"] = np.concatenate(
                    (self._node_data[str(node_id)]["buffer"], chunk)
                )

                while (
                    len(self._node_data[str(node_id)]["buffer"]) >= self._vad_frame_size
                ):
                    # VAD用フレーム取り出し
                    frame_data = self._node_data[str(node_id)]["buffer"][
                        : self._vad_frame_size
                    ]

                    # float32 -> int16に変換
                    frame_int16 = (frame_data * 32767).astype(np.int16)

                    # VAD実行
                    try:
                        vad = self._node_data[str(node_id)]["vad"]
                        is_speech = vad.is_speech(
                            frame_int16.tobytes(), self._default_sampling_rate
                        )
                        self._node_data[str(node_id)]["speech_state"] = (
                            1 if is_speech else 0
                        )
                    except Exception as e:
                        # サンプリングレートやフレームサイズが不正な場合
                        print(f"WebRTC VAD Error: {e}")
                        self._node_data[str(node_id)]["speech_state"] = 0

                    # 処理済みデータをバッファから削除
                    self._node_data[str(node_id)]["buffer"] = self._node_data[
                        str(node_id)
                    ]["buffer"][self._vad_frame_size :]

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
            self._node_data[str(node_id)]["current_chunk_index"] = -1

            # プロットエリア初期化
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
        aggressiveness = self._node_data[str(node_id)]["aggressiveness"]

        setting_dict: Dict[str, Any] = {
            "ver": self._ver,
            "pos": pos,
            "aggressiveness": aggressiveness,
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        aggressiveness = setting_dict.get("aggressiveness", self._VAD_AGGRESSIVENESS)
        dpg_set_value(f"{node_id}:aggressiveness", str(aggressiveness))
        self._node_data[str(node_id)]["vad"].set_mode(aggressiveness)
        self._node_data[str(node_id)]["aggressiveness"] = aggressiveness
