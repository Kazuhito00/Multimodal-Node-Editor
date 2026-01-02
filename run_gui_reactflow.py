#!/usr/bin/env python
"""
ReactFlow Node Editorの起動スクリプト
バックエンド（FastAPI）とフロントエンド（Vite）を起動し、ブラウザを自動で開く

使用方法:
    python run_gui_reactflow.py
    python run_gui_reactflow.py --config path/to/config.json
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser
import socket
from pathlib import Path

# 設定
BACKEND_PORT = 8000
FRONTEND_PORT = 5173
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

# パス設定
PROJECT_ROOT = Path(__file__).parent
FRONTEND_DIR = PROJECT_ROOT / "src" / "gui" / "reactflow" / "frontend"


def is_port_open(port: int, host: str = "localhost") -> bool:
    """指定ポートが開いているかチェック"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result == 0


def wait_for_port(port: int, timeout: int = 30) -> bool:
    """ポートが開くまで待機"""
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open(port):
            return True
        time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="ReactFlow Node Editorの起動")
    parser.add_argument("--config", type=str, default=None, help="設定ファイルのパス（デフォルトはconfig.json）")
    args = parser.parse_args()

    # 設定ファイルのパスを環境変数に設定
    env = os.environ.copy()
    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        env["NODE_EDITOR_CONFIG"] = str(config_path)
        print(f"Using config: {config_path}")

    processes = []

    try:
        # バックエンド起動
        print(f"Starting backend on port {BACKEND_PORT}...")
        backend_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.gui.reactflow.backend.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(BACKEND_PORT),
        ]
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=PROJECT_ROOT,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0,
        )
        processes.append(backend_proc)

        # フロントエンド起動
        print(f"Starting frontend on port {FRONTEND_PORT}...")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        frontend_proc = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=FRONTEND_DIR,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0,
        )
        processes.append(frontend_proc)

        # 少し待ってからブラウザを開く（サービス起動中でもブラウザ側でリトライされる）
        time.sleep(5)
        print(f"Opening browser: {FRONTEND_URL}")
        webbrowser.open(FRONTEND_URL)

        print("\nPress Ctrl+C to stop all services...")

        # プロセス終了を待機
        for proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # 全プロセスを終了
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        print("All services stopped.")


if __name__ == "__main__":
    main()
