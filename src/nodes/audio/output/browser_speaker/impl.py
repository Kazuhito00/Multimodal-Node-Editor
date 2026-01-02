"""
ブラウザのWeb Audio APIを使用してスピーカー出力するノード。
フロントエンドでaudioDataを直接再生する（バックエンドでは処理不要）。
"""
from typing import Any, Dict

from node_editor.node_def import ComputeLogic


class BrowserSpeakerNodeLogic(ComputeLogic):
    """
    ブラウザでオーディオを再生するノードロジック。
    入力ポートのみのため、バックエンドでは処理不要。
    フロントエンドがaudioDataを直接再生する。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        return {}
