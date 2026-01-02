#!/usr/bin/env python3
"""
モデルウェイトファイルをGitHub Releasesからダウンロードして配置するスクリプト。

使用方法:
    python download_weights.py

出力:
    各モデルファイルが元の格納場所に配置される
"""

import sys
import urllib.request
import urllib.error
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
    ("v0.0.0", "gtcrn_simple.onnx", "src/nodes/audio/deep_learning/speech_enhancement/model/gtcrn_simple.onnx"),
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
    ("v0.0.0", "yamnet_float32.tflite", "src/nodes/audio/deep_learning/classification/model/yamnet_float32.tflite"),
    ("v0.0.0", "language_detector_float32.tflite", "src/nodes/text/deep_learning/language_classification/model/language_detector_float32.tflite"),
    ("v0.0.0", "face_landmarker_float16.task", "src/nodes/image/deep_learning/face_detection/MediaPipe/model/face_landmarker_float16.task"),
    ("v0.0.0", "pose_landmarker_lite_float16.task", "src/nodes/image/deep_learning/pose_estimation/MediaPipe/model/pose_landmarker_lite_float16.task"),
    ("v0.0.0", "pose_landmarker_full_float16.task", "src/nodes/image/deep_learning/pose_estimation/MediaPipe/model/pose_landmarker_full_float16.task"),
    ("v0.0.0", "pose_landmarker_heavy_float16.task", "src/nodes/image/deep_learning/pose_estimation/MediaPipe/model/pose_landmarker_heavy_float16.task"),
    ("v0.0.0", "gesture_recognizer_float16.task", "src/nodes/image/deep_learning/hand_pose_estimation/MediaPipe/model/gesture_recognizer_float16.task"),
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


def download_file(url: str, dest_path: Path) -> tuple[bool, str]:
    """ファイルをダウンロード"""
    try:
        # 親ディレクトリを作成
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # ダウンロード
        urllib.request.urlretrieve(url, dest_path)
        return True, ""
    except urllib.error.HTTPError as e:
        return False, f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, f"URL Error: {e.reason}"
    except Exception as e:
        return False, str(e)


def main():
    # プロジェクトルートを取得（このスクリプトの親ディレクトリ）
    project_root = Path(__file__).resolve().parent

    total_files = len(MODEL_FILES)
    success_count = 0
    failed_files = []

    print("=" * 60)
    print("Model Weights Downloader")
    print("=" * 60)
    print(f"Source: {GITHUB_RELEASE_BASE}")
    print(f"Target: {project_root}")
    print(f"Total files: {total_files}")
    print("=" * 60)
    print()

    for i, (version, filename, dest_rel_path) in enumerate(MODEL_FILES, 1):
        url = f"{GITHUB_RELEASE_BASE}/{version}/{filename}"
        dest_path = project_root / dest_rel_path

        # ダウンロード
        success, error_msg = download_file(url, dest_path)

        if success:
            success_count += 1
            file_size = dest_path.stat().st_size
            print(f"({i}/{total_files}) OK: {filename} ({format_size(file_size)})")
        else:
            failed_files.append((filename, dest_rel_path, error_msg))
            print(f"({i}/{total_files}) NG: {filename} - {error_msg}")

    # 結果サマリー
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Success: {success_count}/{total_files}")
    print(f"  Failed:  {len(failed_files)}/{total_files}")

    if failed_files:
        print()
        print("[FAILED FILES]")
        for filename, dest_path, error_msg in failed_files:
            print(f"  - {filename}")
            print(f"    -> {dest_path}")
            print(f"    Error: {error_msg}")

    print()
    if success_count == total_files:
        print("All downloads completed successfully!")
        return 0
    else:
        print("Some downloads failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
