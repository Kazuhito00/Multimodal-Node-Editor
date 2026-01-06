#!/usr/bin/env python3
"""
モデルウェイトファイルをGitHub Releasesからダウンロードして配置するスクリプト。

使用方法:
    python download_weights.py              # 既存ファイルはスキップ
    python download_weights.py --force      # 全ファイルを強制的に上書き
    python download_weights.py --max-size 150  # 150MB超のファイルはスキップ

出力:
    各モデルファイルが元の格納場所に配置される
"""

import argparse
import sys
import urllib.request
import urllib.error
import zipfile
from pathlib import Path


# GitHubリリースのベースURL
GITHUB_RELEASE_BASE = "https://github.com/Kazuhito00/Multimodal-Node-Editor/releases/download"

# ダウンロード対象ファイルのマッピング
# 形式: (バージョン, ファイル名, 格納先相対パス)
# モデル更新時は該当ファイルのバージョンを変更する
MODEL_FILES = [
    ("v0.0.0", "MobileNetV3Large.onnx", "src/nodes/image/deep_learning/classification/MobileNetV3/model/MobileNetV3Large.onnx"),
    ("v0.0.0", "MobileNetV3Small.onnx", "src/nodes/image/deep_learning/classification/MobileNetV3/model/MobileNetV3Small.onnx"),
    ("v0.0.0", "silero_vad_v5.onnx", "src/nodes/audio/analysis/voice_activity_detection/model/silero_vad_v5.onnx"),
    ("v0.0.0", "gtcrn_simple.onnx", "src/nodes/audio/deep_learning/speech_enhancement/gtcrn/model/gtcrn_simple.onnx"),
    ("v1.1.0", "fastenhancer_t.onnx", "src/nodes/audio/deep_learning/speech_enhancement/fastenhancer/model/fastenhancer_t.onnx"),
    ("v1.1.0", "fastenhancer_s.onnx", "src/nodes/audio/deep_learning/speech_enhancement/fastenhancer/model/fastenhancer_s.onnx"),
    ("v1.1.0", "fastenhancer_m.onnx", "src/nodes/audio/deep_learning/speech_enhancement/fastenhancer/model/fastenhancer_m.onnx"),
    ("v1.1.0", "fastenhancer_b.onnx", "src/nodes/audio/deep_learning/speech_enhancement/fastenhancer/model/fastenhancer_b.onnx"),
    ("v1.1.0", "fastenhancer_l.onnx", "src/nodes/audio/deep_learning/speech_enhancement/fastenhancer/model/fastenhancer_l.onnx"),
    ("v0.0.0", "silero_vad_v6.onnx", "src/nodes/audio/analysis/voice_activity_detection/model/silero_vad_v6.onnx"),
    ("v0.0.0", "deimv2_dinov3_s_coco.onnx", "src/nodes/image/deep_learning/object_detection/DEIMv2/model/deimv2_dinov3_s_coco.onnx"),
    ("v0.0.0", "deimv2_hgnetv2_atto_coco.onnx", "src/nodes/image/deep_learning/object_detection/DEIMv2/model/deimv2_hgnetv2_atto_coco.onnx"),
    ("v0.0.0", "deimv2_hgnetv2_femto_coco.onnx", "src/nodes/image/deep_learning/object_detection/DEIMv2/model/deimv2_hgnetv2_femto_coco.onnx"),
    ("v0.0.0", "deimv2_hgnetv2_n_coco.onnx", "src/nodes/image/deep_learning/object_detection/DEIMv2/model/deimv2_hgnetv2_n_coco.onnx"),
    ("v0.0.0", "deimv2_hgnetv2_pico_coco.onnx", "src/nodes/image/deep_learning/object_detection/DEIMv2/model/deimv2_hgnetv2_pico_coco.onnx"),
    ("v0.0.0", "deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx", "src/nodes/image/deep_learning/object_detection/DEIMv2Wholebody34/model/deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx"),
    ("v0.0.0", "deimv2_hgnetv2_atto_wholebody34_340query_n_batch_320x320.onnx", "src/nodes/image/deep_learning/object_detection/DEIMv2Wholebody34/model/deimv2_hgnetv2_atto_wholebody34_340query_n_batch_320x320.onnx"),
    ("v0.0.0", "deimv2_hgnetv2_femto_wholebody34_340query_n_batch_416x416.onnx", "src/nodes/image/deep_learning/object_detection/DEIMv2Wholebody34/model/deimv2_hgnetv2_femto_wholebody34_340query_n_batch_416x416.onnx"),
    ("v0.0.0", "deimv2_hgnetv2_n_wholebody34_680query_n_batch_640x640.onnx", "src/nodes/image/deep_learning/object_detection/DEIMv2Wholebody34/model/deimv2_hgnetv2_n_wholebody34_680query_n_batch_640x640.onnx"),
    ("v0.0.0", "deimv2_hgnetv2_pico_wholebody34_340query_n_batch_640x640.onnx", "src/nodes/image/deep_learning/object_detection/DEIMv2Wholebody34/model/deimv2_hgnetv2_pico_wholebody34_340query_n_batch_640x640.onnx"),
    ("v0.0.0", "road_segmentation_adas_0001.onnx", "src/nodes/image/deep_learning/semantic_segmentation/road_segmentation_adas_0001/model/road_segmentation_adas_0001.onnx"),
    ("v0.0.0", "efficientnet-lite4-11.onnx", "src/nodes/image/deep_learning/classification/EfficientNetLite4/model/efficientnet-lite4-11.onnx"),
    ("v0.0.0", "mnist-12.onnx", "src/nodes/image/deep_learning/classification/MNIST/model/mnist-12.onnx"),
    ("v0.0.0", "face_detection_yunet_2023mar.onnx", "src/nodes/image/deep_learning/face_detection/YuNet/model/face_detection_yunet_2023mar.onnx"),
    ("v0.0.0", "mobileie_lolv1.onnx", "src/nodes/image/deep_learning/low_light_image_enhancement/MobileIE/model/mobileie_lolv1.onnx"),
    ("v0.0.0", "cpga_net.onnx", "src/nodes/image/deep_learning/low_light_image_enhancement/CPGA-Net/model/cpga_net.onnx"),
    ("v0.0.0", "tbefn.onnx", "src/nodes/image/deep_learning/low_light_image_enhancement/TBEFN/model/tbefn.onnx"),
    ("v0.0.0", "lite-mono-8m_1024x320.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/Lite-Mono/model/lite-mono-8m_1024x320.onnx"),
    ("v0.0.0", "lite-mono-8m_640x192.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/Lite-Mono/model/lite-mono-8m_640x192.onnx"),
    ("v0.0.0", "lite-mono-small_1024x320.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/Lite-Mono/model/lite-mono-small_1024x320.onnx"),
    ("v0.0.0", "lite-mono-small_640x192.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/Lite-Mono/model/lite-mono-small_640x192.onnx"),
    ("v0.0.0", "lite-mono-tiny_1024x320.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/Lite-Mono/model/lite-mono-tiny_1024x320.onnx"),
    ("v0.0.0", "lite-mono-tiny_640x192.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/Lite-Mono/model/lite-mono-tiny_640x192.onnx"),
    ("v0.0.0", "lite-mono_1024x320.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/Lite-Mono/model/lite-mono_1024x320.onnx"),
    ("v0.0.0", "lite-mono_640x192.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/Lite-Mono/model/lite-mono_640x192.onnx"),
    ("v0.0.0", "rtmonodepth_full_m_640x192.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/RT-MonoDepth/model/rtmonodepth_full_m_640x192.onnx"),
    ("v0.0.0", "rtmonodepth_full_ms_640x192.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/RT-MonoDepth/model/rtmonodepth_full_ms_640x192.onnx"),
    ("v0.0.0", "rtmonodepth_s_m_640x192.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/RT-MonoDepth/model/rtmonodepth_s_m_640x192.onnx"),
    ("v0.0.0", "rtmonodepth_s_ms_640x192.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/RT-MonoDepth/model/rtmonodepth_s_ms_640x192.onnx"),
    ("v0.0.0", "depth_anything_v2_vits.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/Depth-Anything-V2/model/depth_anything_v2_vits.onnx"),
    ("v0.0.0", "depth_anything_v2_vitb.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/Depth-Anything-V2/model/depth_anything_v2_vitb.onnx"),
    ("v0.0.0", "depth_anything_v2_vitl.onnx", "src/nodes/image/deep_learning/monocular_depth_estimation/Depth-Anything-V2/model/depth_anything_v2_vitl.onnx"),
    ("v0.0.0", "vitpose_small.onnx", "src/nodes/image/deep_learning/pose_estimation/ViTPose/model/vitpose_small.onnx"),
    ("v0.0.0", "vitpose_base.onnx", "src/nodes/image/deep_learning/pose_estimation/ViTPose/model/vitpose_base.onnx"),
    ("v0.0.0", "vitpose_large.onnx", "src/nodes/image/deep_learning/pose_estimation/ViTPose/model/vitpose_large.onnx"),
    ("v0.0.0", "ppliteseg_stdc1.onnx", "src/nodes/image/deep_learning/semantic_segmentation/PP-LiteSeg/model/ppliteseg_stdc1.onnx"),
    ("v0.0.0", "ppliteseg_stdc2.onnx", "src/nodes/image/deep_learning/semantic_segmentation/PP-LiteSeg/model/ppliteseg_stdc2.onnx"),
    ("v0.0.0", "blaze_face_short_range_float16.tflite", "src/nodes/image/deep_learning/face_detection/MediaPipe/model/blaze_face_short_range_float16.tflite"),
    ("v0.0.0", "selfie_segmenter_float16.tflite", "src/nodes/image/deep_learning/semantic_segmentation/MediaPipe/model/selfie_segmenter_float16.tflite"),
    ("v0.0.0", "selfie_segmenter_landscape_float16.tflite", "src/nodes/image/deep_learning/semantic_segmentation/MediaPipe/model/selfie_segmenter_landscape_float16.tflite"),
    ("v0.0.0", "hair_segmenter_float32.tflite", "src/nodes/image/deep_learning/semantic_segmentation/MediaPipe/model/hair_segmenter_float32.tflite"),
    ("v0.0.0", "selfie_multiclass_256x256_float32.tflite", "src/nodes/image/deep_learning/semantic_segmentation/MediaPipe/model/selfie_multiclass_256x256_float32.tflite"),
    ("v0.0.0", "yamnet_float32.tflite", "src/nodes/audio/deep_learning/classification/yamnet/model/yamnet_float32.tflite"),
    ("v0.0.0", "language_detector_float32.tflite", "src/nodes/text/deep_learning/language_classification/mediapipe_language_detector/model/language_detector_float32.tflite"),
    ("v0.0.0", "face_landmarker_float16.task", "src/nodes/image/deep_learning/face_detection/MediaPipe/model/face_landmarker_float16.task"),
    ("v0.0.0", "pose_landmarker_lite_float16.task", "src/nodes/image/deep_learning/pose_estimation/MediaPipe/model/pose_landmarker_lite_float16.task"),
    ("v0.0.0", "pose_landmarker_full_float16.task", "src/nodes/image/deep_learning/pose_estimation/MediaPipe/model/pose_landmarker_full_float16.task"),
    ("v0.0.0", "pose_landmarker_heavy_float16.task", "src/nodes/image/deep_learning/pose_estimation/MediaPipe/model/pose_landmarker_heavy_float16.task"),
    ("v0.0.0", "gesture_recognizer_float16.task", "src/nodes/image/deep_learning/hand_pose_estimation/MediaPipe/model/gesture_recognizer_float16.task"),
    ("v0.0.0", "PP-OCRv5_mobile_det_infer.onnx", "src/nodes/image/deep_learning/ocr/PaddleOCRv5/model/PP-OCRv5_mobile_det_infer.onnx"),
    ("v0.0.0", "PP-OCRv5_mobile_rec_infer.onnx", "src/nodes/image/deep_learning/ocr/PaddleOCRv5/model/PP-OCRv5_mobile_rec_infer.onnx"),
    ("v0.0.0", "PP-OCRv5_server_det_infer.onnx", "src/nodes/image/deep_learning/ocr/PaddleOCRv5/model/PP-OCRv5_server_det_infer.onnx"),
    ("v0.0.0", "PP-OCRv5_server_rec_infer.onnx", "src/nodes/image/deep_learning/ocr/PaddleOCRv5/model/PP-OCRv5_server_rec_infer.onnx"),
]

# ZIPファイルのダウンロード対象（ダウンロード後に解凍）
# 形式: (バージョン, ファイル名, 格納先ディレクトリ相対パス, 解凍後のディレクトリ名)
ZIP_FILES = [
    ("v1.1.0", "vosk-model-small-ja-0.22.zip", "src/nodes/audio/recognition/vosk/model", "vosk-model-small-ja-0.22"),
    ("v1.1.0", "vosk-model-ja-0.22.zip", "src/nodes/audio/recognition/vosk/model", "vosk-model-ja-0.22"),
    ("v1.1.0", "vosk-model-small-en-us-0.15.zip", "src/nodes/audio/recognition/vosk/model", "vosk-model-small-en-us-0.15"),
    ("v1.1.0", "vosk-model-en-us-0.22.zip", "src/nodes/audio/recognition/vosk/model", "vosk-model-en-us-0.22"),
]


def format_size(size_bytes: int) -> str:
    """バイト数を人間が読みやすい形式に変換"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_remote_file_size(url: str) -> int:
    """HEADリクエストでリモートファイルのサイズを取得（取得失敗時は0）"""
    try:
        request = urllib.request.Request(url, method='HEAD')
        response = urllib.request.urlopen(request)
        return int(response.headers.get('Content-Length', 0))
    except Exception:
        return 0


def download_file_with_progress(
    url: str,
    dest_path: Path,
    prefix: str = ""
) -> tuple[bool, str, int]:
    """ファイルをダウンロード（進捗表示付き）"""
    try:
        # 親ディレクトリを作成
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # URLを開いてファイルサイズを取得
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('Content-Length', 0))

        # チャンクサイズ
        chunk_size = 8192
        downloaded = 0

        with open(dest_path, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                # 進捗表示（行末をスペースで埋めて前の文字を消す）
                if total_size > 0:
                    # 単位は最後だけ表示（19.5/38.8 MB）
                    dl_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    progress_str = f"{dl_mb:.1f}/{total_mb:.1f} MB"
                else:
                    progress_str = format_size(downloaded)
                line = f"\r{prefix} {progress_str}"
                print(f"{line:<80}", end="", flush=True)

        return True, "", total_size

    except urllib.error.HTTPError as e:
        return False, f"HTTP Error {e.code}: {e.reason}", 0
    except urllib.error.URLError as e:
        return False, f"URL Error: {e.reason}", 0
    except Exception as e:
        return False, str(e), 0


def extract_zip(zip_path: Path, extract_dir: Path) -> tuple[bool, str]:
    """ZIPファイルを解凍"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return True, ""
    except zipfile.BadZipFile:
        return False, "Invalid ZIP file"
    except Exception as e:
        return False, str(e)


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="モデルウェイトファイルをGitHub Releasesからダウンロード"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="既存ファイルを強制的に上書きする"
    )
    parser.add_argument(
        "--max-size", "-m",
        type=float,
        default=None,
        metavar="MB",
        help="指定サイズ（MB）を超えるファイルはスキップ"
    )
    return parser.parse_args()


def main():
    # コマンドライン引数をパース
    args = parse_arguments()
    force_overwrite = args.force
    max_size_mb = args.max_size
    max_size_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else None

    # プロジェクトルートを取得（このスクリプトの親ディレクトリ）
    project_root = Path(__file__).resolve().parent

    total_files = len(MODEL_FILES)
    total_zips = len(ZIP_FILES)
    success_count = 0
    skipped_exists_count = 0
    skipped_size_count = 0
    failed_files = []

    # ZIPファイル用カウンタ
    zip_success_count = 0
    zip_skipped_count = 0
    zip_failed_files = []

    print("=" * 60)
    print("Model Weights Downloader")
    print("=" * 60)
    print(f"Source: {GITHUB_RELEASE_BASE}")
    print(f"Target: {project_root}")
    print(f"Total files: {total_files} (+ {total_zips} ZIP archives)")
    print(f"Force overwrite: {force_overwrite}")
    print(f"Max file size: {f'{max_size_mb} MB' if max_size_mb else 'unlimited'}")
    print("=" * 60)
    print()

    # 通常ファイルのダウンロード
    for i, (version, filename, dest_rel_path) in enumerate(MODEL_FILES, 1):
        url = f"{GITHUB_RELEASE_BASE}/{version}/{filename}"
        dest_path = project_root / dest_rel_path

        # 既存ファイルのスキップ判定
        if dest_path.exists() and not force_overwrite:
            skipped_exists_count += 1
            file_size = dest_path.stat().st_size
            print(f"({i}/{total_files}) SKIP: {filename} ({format_size(file_size)}) - already exists")
            continue

        # ファイルサイズによるスキップ判定
        if max_size_bytes:
            remote_size = get_remote_file_size(url)
            if remote_size > max_size_bytes:
                skipped_size_count += 1
                print(f"({i}/{total_files}) SKIP: {filename} ({format_size(remote_size)}) - exceeds max size")
                continue

        # ダウンロード（進捗表示付き）
        prefix = f"({i}/{total_files}) {filename}"
        success, error_msg, file_size = download_file_with_progress(url, dest_path, prefix)

        if success:
            success_count += 1
            # 完了行を出力（進捗表示を上書き）
            line = f"\r({i}/{total_files}) OK: {filename} ({format_size(file_size)})"
            print(f"{line:<80}")
        else:
            failed_files.append((filename, dest_rel_path, error_msg))
            line = f"\r({i}/{total_files}) NG: {filename} - {error_msg}"
            print(f"{line:<80}")

    # ZIPファイルのダウンロードと解凍
    if total_zips > 0:
        print()
        print("-" * 60)
        print("ZIP Archives (download and extract)")
        print("-" * 60)
        print()

        for i, (version, filename, dest_dir_rel, extract_dirname) in enumerate(ZIP_FILES, 1):
            url = f"{GITHUB_RELEASE_BASE}/{version}/{filename}"
            dest_dir = project_root / dest_dir_rel
            extract_path = dest_dir / extract_dirname

            # 解凍済みディレクトリが存在する場合はスキップ
            if extract_path.exists() and not force_overwrite:
                zip_skipped_count += 1
                print(f"[ZIP {i}/{total_zips}] SKIP: {filename} - already extracted")
                continue

            # ファイルサイズによるスキップ判定
            if max_size_bytes:
                remote_size = get_remote_file_size(url)
                if remote_size > max_size_bytes:
                    zip_skipped_count += 1
                    print(f"[ZIP {i}/{total_zips}] SKIP: {filename} ({format_size(remote_size)}) - exceeds max size")
                    continue

            # ZIPファイルをダウンロード
            zip_path = dest_dir / filename
            prefix = f"[ZIP {i}/{total_zips}] {filename}"
            success, error_msg, file_size = download_file_with_progress(url, zip_path, prefix)

            if not success:
                zip_failed_files.append((filename, dest_dir_rel, error_msg))
                line = f"\r[ZIP {i}/{total_zips}] NG: {filename} - {error_msg}"
                print(f"{line:<80}")
                continue

            # 解凍
            print(f"\r[ZIP {i}/{total_zips}] Extracting {filename}...".ljust(80), end="", flush=True)
            extract_success, extract_error = extract_zip(zip_path, dest_dir)

            if extract_success:
                zip_success_count += 1
                line = f"\r[ZIP {i}/{total_zips}] OK: {filename} ({format_size(file_size)}) - extracted"
                print(f"{line:<80}")
                # 解凍成功後、ZIPファイルを削除
                try:
                    zip_path.unlink()
                except Exception:
                    pass
            else:
                zip_failed_files.append((filename, dest_dir_rel, f"Extract failed: {extract_error}"))
                line = f"\r[ZIP {i}/{total_zips}] NG: {filename} - {extract_error}"
                print(f"{line:<80}")

    # 結果サマリー
    skipped_total = skipped_exists_count + skipped_size_count
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Downloaded:       {success_count}/{total_files}")
    print(f"  Skipped (exists): {skipped_exists_count}/{total_files}")
    print(f"  Skipped (size):   {skipped_size_count}/{total_files}")
    print(f"  Failed:           {len(failed_files)}/{total_files}")

    if total_zips > 0:
        print()
        print(f"  ZIP extracted:    {zip_success_count}/{total_zips}")
        print(f"  ZIP skipped:      {zip_skipped_count}/{total_zips}")
        print(f"  ZIP failed:       {len(zip_failed_files)}/{total_zips}")

    if failed_files:
        print()
        print("[FAILED FILES]")
        for filename, dest_path, error_msg in failed_files:
            print(f"  - {filename}")
            print(f"    -> {dest_path}")
            print(f"    Error: {error_msg}")

    if zip_failed_files:
        print()
        print("[FAILED ZIP FILES]")
        for filename, dest_path, error_msg in zip_failed_files:
            print(f"  - {filename}")
            print(f"    -> {dest_path}")
            print(f"    Error: {error_msg}")

    print()
    completed_count = success_count + skipped_total
    zip_completed_count = zip_success_count + zip_skipped_count
    all_success = (completed_count == total_files) and (zip_completed_count == total_zips)

    if all_success:
        print("All files are ready!")
        return 0
    else:
        print("Some downloads failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
