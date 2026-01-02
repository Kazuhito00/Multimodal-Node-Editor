"""
graph.json をヘッドレスで実行するスクリプト

使用方法:
    python run_headless.py <graph.json>
    python run_headless.py graph.json --interval 100 --count 10
    python run_headless.py graph.json --config path/to/config.json
"""

import argparse
import sys
from pathlib import Path

# srcディレクトリをパスに追加
project_root = Path(__file__).resolve().parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from gui.headless.headless_main import run_headless


def main():
    parser = argparse.ArgumentParser(description="graph.json をヘッドレスで実行")
    parser.add_argument("graph_file", help="実行するgraph.jsonファイルのパス")
    parser.add_argument("--interval", type=int, default=100, help="実行間隔（ミリ秒）")
    parser.add_argument("--count", type=int, default=0, help="実行回数（0=無限、1=1回実行）")
    parser.add_argument("--show-all", action="store_true", help="全ての終端出力を表示（デフォルトは最終ノードのみ）")
    parser.add_argument("--no-resize", action="store_true", help="大きい画像のリサイズを無効化（デフォルトは1280x720にリサイズ）")
    parser.add_argument("--config", type=str, default=None, help="設定ファイルのパス（デフォルトはconfig.json）")
    args = parser.parse_args()

    graph_file = Path(args.graph_file)
    if not graph_file.exists():
        print(f"Error: File not found: {graph_file}")
        sys.exit(1)

    config_file = Path(args.config) if args.config else None
    if config_file and not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)

    run_headless(
        graph_file=graph_file,
        project_root=project_root,
        interval_ms=args.interval,
        count=args.count,
        show_all=args.show_all,
        resize_display=not args.no_resize,
        config_file=config_file,
    )


if __name__ == "__main__":
    main()
