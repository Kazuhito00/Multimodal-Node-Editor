import time
from typing import Dict, Any
from node_editor.node_def import ComputeLogic


class ElapsedTimeLogic(ComputeLogic):
    """START からの経過時間を出力するノード"""

    def __init__(self):
        self._start_time: float = 0.0
        self._last_elapsed: float = 0.0

    def reset(self):
        """開始時間をリセット"""
        self._start_time = 0.0
        self._last_elapsed = 0.0

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        unit = properties.get("unit", "sec")
        current_time = time.perf_counter()

        # 初回実行時に開始時間を記録
        if self._start_time == 0.0:
            self._start_time = current_time

        elapsed_sec = current_time - self._start_time
        if unit == "ms":
            self._last_elapsed = float(int(elapsed_sec * 1000))
        else:
            self._last_elapsed = float(int(elapsed_sec))

        return {"time": self._last_elapsed}
