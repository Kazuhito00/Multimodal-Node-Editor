import time
from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class TimerLogic(ComputeLogic):
    """指定間隔で1を出力し、それ以外は0を出力するタイマーノード"""

    def __init__(self):
        self._last_trigger_time: float = 0.0
        self._start_time: float = 0.0
        self._last_trigger: float = 0.0

    def reset(self):
        """タイマーをリセット"""
        self._last_trigger_time = 0.0
        self._start_time = 0.0
        self._last_trigger = 0.0

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        interval = int(properties.get("interval", 1))
        unit = properties.get("unit", "sec")

        # 秒に変換
        if unit == "ms":
            interval_sec = interval / 1000.0
        else:
            interval_sec = float(interval)

        current_time = time.perf_counter()

        # 初回実行時に開始時間を記録（トリガーは出さない）
        if self._start_time == 0.0:
            self._start_time = current_time
            self._last_trigger_time = current_time
            self._last_trigger = 0.0
            return {"trigger": 0.0}

        # 前回のトリガーから指定時間経過したかチェック
        elapsed = current_time - self._last_trigger_time
        remaining = max(0.0, interval_sec - elapsed)

        # タイマー情報を共通で含める
        timer_info = {
            "__timer_interval__": interval_sec,
            "__timer_remaining__": remaining,
            "__timer_unit__": unit,
        }

        if elapsed >= interval_sec:
            self._last_trigger_time = current_time
            self._last_trigger = 1.0
            return {"trigger": 1.0, **timer_info}

        self._last_trigger = 0.0
        return {"trigger": 0.0, **timer_info}
