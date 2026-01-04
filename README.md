[[Japanese](README.md)/[English](README_EN.md)]

# Multimodal-Node-Editor
ノードエディターベースのマルチモーダル処理アプリケーションです。<br>
画像・音声・テキストなどをノードを繋げ処理することが可能で、処理検討や処理比較での利用を想定しています。<br>


<img src="https://github.com/user-attachments/assets/264acff2-4b6c-460f-b6a0-77fb959a6f66" width="100%">

# Features
- 画像・音声・テキスト・深層学習など、100 以上のノードを標準搭載
- Web カメラやマイク入力を用いたリアルタイム処理に対応
- TOML + Python による新規ノードの簡単な追加が可能（GUI を自作しない場合）
- 保存したグラフを GUI なしでヘッドレス実行可能
- Google Colaboratory バックエンドでのシステム起動可能<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/Multimodal-Node-Editor/blob/main/run_gui_reactflow_colab.ipynb)
  
# Note
ノードは作成者(高橋)が必要になった順に追加しているため、<br>
画像処理、オーディオ処理、テキスト処理における基本的な処理を担うノードが不足していることがあります。

# Requirement
<details>
<summary>フロントエンド</summary>
  
```
Node.js v18 or later

react                ^18.3.1
react-dom            ^18.3.1
@xyflow/react        ^12.3.6
dagre                ^0.8.5 
@types/dagre         ^0.7.53
vite                 ^6.0.5 
typescript           ~5.6.2 
@vitejs/plugin-react ^4.3.4 
@types/react         ^18.3.18
@types/react-dom     ^18.3.5
```
</details>

<details>
<summary>バックエンド</summary>
  
```
Python 3.10 or later

pydantic            2.12.5    or later
platformdirs        4.5.1     or later
fastapi             0.128.0   or later
python-multipart    0.0.21    or later
uvicorn[standard]   0.40.0    or later
opencv-python       4.11.0.86 or later
motpy               0.0.10    or later
sahi                0.11.36   or later
onnx                1.20.0    or later
onnxruntime         1.23.2    or later ※GPU を使用する場合は onnxruntime-gpu
mediapipe           0.10.31   or later
sounddevice         0.5.3     or later
soundfile           0.13.1    or later
webrtcvad-wheels    2.0.14    or later
scipy               1.16.3    or later
av                  16.0.1    or later
openai              2.14.0    or later
aiortc              1.14.0    or later
websocket-client    1.9.0     or later
google-cloud-speech 2.35.0    or later
```
</details>

# Installation
Google Colaboratory起動を行う方は、以下の作業は不要です。ノートブックの処理に従ってください。<br>
また、以下の手順は、Pythonやnode.jsがインストールされている前提です。<br>

```bash
# リポジトリクローン
git clone https://github.com/Kazuhito00/Multimodal-Node-Editor
cd Multimodal-Node-Editor

# Python パッケージインストール
pip install -r requirements.txt

# モデルウェイトダウンロード
python download_weights.py  # 全ファイルをダウンロード(既にファイルがある場合はスキップ)
# python download_weights.py --force  # 全ファイルを強制的に上書き
# python download_weights.py --max-size 150  # 指定サイズMB以上のファイルのダウンロードはスキップ

# Node.js パッケージインストール
cd src/gui/reactflow/frontend
npm install
cd ../../../../

# コンフィグをコピー
cp config.example.json config.json
```

一部のノードはAPIキーなどを設定しないと使用できません。<br> 
必要に応じて、`config.json` の以下キーを設定してください。

```json
{
  "api_keys": {
    "openai": "SET_YOUR_OPENAI_API_KEY",
    "google_stt": "PATH_TO_GOOGLE_CREDENTIALS_JSON"
  }
}
```

<details>
<summary>コンフィグ詳細</summary>
  
| キー                        | 型       | デフォルト    | 説明                                   |
|-----------------------------|----------|---------------|----------------------------------------|
| node_search_paths           | string[] | ["src/nodes"] | ノード定義を検索するディレクトリ       |
| ui.theme                    | string   | "light"       | テーマ（light / dark）                 |
| ui.sidebar.show_edit        | bool     | false         | サイドバーにアンドゥリドゥメニューを表示         |
| ui.sidebar.show_file        | bool     | true          | サイドバーにグラフ保存(json)メニューを表示     |
| ui.sidebar.show_auto_layout | bool     | true          | サイドバーに自動レイアウトボタンを表示 |
| graph.interval_ms           | int      | 50            | グラフ実行間隔（ミリ秒）               |
| audio.sample_rate           | int      | 16000         | オーディオ処理のサンプルレート（Hz）       |
| camera.max_scan_count       | int      | 2             | カメラデバイスの最大スキャン数         |
| auto_download.video         | bool     | false         | 動画キャプチャ時の自動ダウンロード         |
| auto_download.wav           | bool     | false         | 録音時の自動ダウンロード         |
| auto_download.capture       | bool     | false         | 画像キャプチャ時の自動ダウンロード       |
| auto_download.text          | bool     | false         | テキスト保存時の自動ダウンロード     |
| api_keys.openai             | string   | ""            | OpenAI APIキー      |
| api_keys.google_stt         | string   | ""            | Google Speech-to-Text 用 クレデンシャルjson格納パス         |
</details>

# Launch the application
* <b>ローカルPCでの起動</b><br>
  以下のスクリプトを実行してください。起動に成功するとブラウザが立ち上がります。
  ```
  python run_gui_reactflow.py
  ```
  | オプション | 説明 |
  |-----------|------|
  | `--config <path>` | 設定ファイルのパス（デフォルト: config.json） |
  <br>
  または、以下をそれぞれ別コンソールにて実行し、ブラウザにて`http://localhost:5173/`にアクセスしてください。
  ```
  uvicorn src.gui.reactflow.backend.main:app --reload
  ```
  ```
  cd src/gui/reactflow/frontend
  npm run dev
  ```
* <b>Google Colaboratoryでの起動</b><br>  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/Multimodal-Node-Editor/blob/main/run_gui_reactflow_colab.ipynb)<br>
Colaboratoryでノートブックを開き、上から順に実行してください。<br>
  最終セルの出力結果に`https://localhost:8000/`と表示されるため、そのリンクをクリックしてください<br><img src="https://github.com/user-attachments/assets/cbfa6eaa-8411-4e56-9940-1fc144c127ba" width="75%">

* <b>ヘッドレス実行</b><br>
ReactFlowフロントエンドを起動せずにコマンドラインでグラフ実行が可能です<br>
  ```
  python run_headless.py graph.json
  ```
  | オプション | 説明 |
  |-----------|------|
  | `--config <path>` | 設定ファイルのパス（デフォルト: config.json） |
  | `--count <n>` | 実行回数（0=無限ループ、1=1回実行、デフォルト: 0） |

# Usage
<details>
<summary>ノード配置 & 実行</summary>

1. サイドバーからノードをキャンバスにドラッグ＆ドロップ<br><img src="https://github.com/user-attachments/assets/186dccbf-cecb-48ac-86e3-975eaf6f9f7c" width="50%">
2. ノード間のポートを接続<br>※同色同士のポートで接続可能<br><img src="https://github.com/user-attachments/assets/39594af8-8b7c-4fe7-be23-a68eaa968840" width="50%">
3. Start ボタンをクリックして実行(ショートカットキー：Ctrl + Enter)<br>※実行中はノード配置やポート接続の変更は不可<br><img src="https://github.com/user-attachments/assets/c262bb0c-04a6-434f-92ac-00a43afe2864" width="50%">
</details>

<details>
<summary>エッジ削除、ノード削除</summary>
  
* 削除したいエッジやノードを選択してDeleteキー<br><img src="https://github.com/user-attachments/assets/3554273d-a265-4cf1-8eae-59bdbbd2b6d5" width="50%">
</details>

<details>
<summary>グラフjsonエクスポート、インポート</summary>

* グラフjsonエクスポート：Save ボタンをクリック(ショートカットキー：Ctrl + S)<br><img src="https://github.com/user-attachments/assets/e13218ae-9af6-494c-b97c-a9b1b3558736" width="50%">
* グラフjsonインポート：Load ボタンをクリック(ショートカットキー：Ctrl + L)<br><img src="https://github.com/user-attachments/assets/17df193c-ab7f-43f7-ac3b-6b14b2d8bb0a" width="50%">
</details>

<details>
<summary>オートレイアウト</summary>

* Auto Layout ボタンをクリック(ショートカットキー：Ctrl + A)<br><img src="https://github.com/user-attachments/assets/4f8c6417-a8df-40a7-904f-4a943c4b66e9" width="50%">
</details>

<details>
<summary>ノードへのコメント</summary>

* ノードを右クリックして、Add Commentをクリック<br><img src="https://github.com/user-attachments/assets/14a18fcf-51c5-4d27-bd33-122cd5a44d5b" width="50%">
</details>

# Keyboard Shortcuts
| ショートカット                   | 動作                 | 備考                                        |
|----------------------------------|----------------------|---------------------------------------------|
| Ctrl + Enter                     | START/STOP切り替え   | 実行中ならSTOP、停止中ならSTART             |
| Escape                           | STOP                 | グラフ実行を停止                            |
| Ctrl + P                         | Pause/Resume切り替え | 実行中ならPause、一時停止中ならResume       |
| Ctrl + Z                         | Undo                 | 元に戻す                                    |
| Ctrl + Y                         | Redo                 | やり直す                                    |
| Ctrl + A                         | Auto Layout          | ノードを自動配置                            |
| Ctrl + S                         | Save                 | グラフをJSONで保存（エクスポート）          |
| Ctrl + L                         | Load                 | グラフJSONを読み込み（インポート）          |
| Delete                           | 削除                 | 選択中のノード/エッジを削除（実行中は無効） |

# Nodes

### Image

<details>
<summary>Image > Input</summary>

| ノード名 | 説明 |
|:--|:--|
| Image | 静止画像ファイル（jpg, png, bmp, gif）を読み込む |
| Webcam | Webカメラからリアルタイム映像を取得<br>Colaboratoryバックエンドで利用不可|
| Webcam (WebSocket) | ブラウザのgetUserMedia() API経由でWebカメラ映像を取得<br>Colaboratoryバックエンドで利用可 |
| Video | 動画ファイル（mp4, avi）を読み込んでフレーム再生<br>・Realtime Syncチェックボックス：処理時間に同期してフレームを読み出すオプション<br>・Frame Step：読み込みフレーム間隔（realtime_sync=false時のみ）<br>・Preload All Frameチェックボックス：全フレームをプリロードする<br>※サイドバーの「Loop Playback」がONの場合、ループ再生を行う |
| Video Frame | 動画の指定フレーム位置の画像を出力 |
| RTSP | ネットワークカメラのRTSP入力から映像を取得 |
| Solid Color | 単色画像を生成<br>・width：画像幅（1-4096、デフォルト: 640）<br>・height：画像高さ（1-4096、デフォルト: 360）<br>・color：色（カラーピッカー、デフォルト: #ff0000） |
| URL Image | URLから画像をダウンロードして取得 |

</details>

<details>
<summary>Image > Transform</summary>

| ノード名 | 説明 |
|:--|:--|
| Crop | 正規化座標（0.0-1.0）で指定した領域を切り抜く<br>画像エリアをドラッグして領域を指定可能 |
| Flip | 画像を水平/垂直方向に反転する |
| Resize | 指定解像度・補間方法でリサイズする |
| Rotate | 指定角度で画像を回転する（90度の倍数以外では余白が発生） |
| 3D Rotate | 3次元空間でのピッチ・ヨー・ロール回転を行う |
| Click Perspective | 画像クリックでの4点指定による透視変換を行う |

</details>

<details>
<summary>Image > Filter</summary>

| ノード名 | 説明 |
|:--|:--|
| Apply Color Map | グレースケール画像に疑似カラーを適用する |
| Background Subtraction | 背景差分で前景を検出する |
| Blur | 各種ぼかしフィルタを適用する |
| Morphology | モルフォロジー変換を行う |
| Brightness | 輝度を加算調整する |
| Canny | Canny法によるエッジ検出を行う |
| Contrast | コントラストを調整する |
| Equalize Hist | HSVのVチャンネルにヒストグラム平坦化を適用する |
| Filter 2D (3x3) | 任意の3x3カーネルによる畳み込みフィルタを適用する |
| Gamma | ガンマ補正を適用する（LUTテーブル使用） |
| Grayscale | 画像をグレースケールに変換する（3チャンネル維持） |
| RGB Extract | 指定したRGBチャンネルを抽出する |
| RGB Adjust | RGB各チャンネルに値を加算する |
| HSV Adjust | HSV色空間で色相・彩度・明度を調整する |
| Inpaint | マスクを用いてインペイントする |
| Omnidirectional Viewer | 正距円筒図法の360度画像を回転表示する<br>画像のドラッグで視点を変更可能 |
| Sepia | セピア調エフェクトを適用する |
| Threshold | 各種アルゴリズムで2値化する |

</details>

<details>
<summary>Image > Marker Detection</summary>

| ノード名 | 説明 |
|:--|:--|
| QR Code | QRコードを検出・デコードし、結果をJSON出力する |
| ArUco Marker | ArUcoマーカーを検出し、ID・四隅座標をJSON出力する |
| AprilTag | AprilTagを検出し、ID・四隅座標をJSON出力する |

</details>

<details>
<summary>Image > Deep Learning</summary>

| ノード名 | 説明 |
|:--|:--|
| Image Classification | ImageNet 1000クラスで画像を分類する<br>・Model：使用するモデルを選択する（ドロップダウン）|
| Object Detection | 物体検出を行う<br>motpyによるマルチオブジェクトトラッキング、SAHIによるスライス処理検出にも対応 |
| Face Detection | 顔検出を行う<br>motpyによるマルチオブジェクトトラッキング、SAHIによるスライス処理検出にも対応 |
| Low-Light Image Enhancement | 暗所画像の画像強調を行う |
| Depth Estimation | 単眼深度推定を行う |
| Pose Estimation | 人体姿勢推定を行う |
| Hand Pose Estimation | 手の姿勢推定を行う |
| Semantic Segmentation | セマンティックセグメンテーションを行う |
| OCR | 光学文字認識を行う |

</details>

<details>
<summary>Image > Analysis</summary>

| ノード名 | 説明 |
|:--|:--|
| Color Histogram | 各チャンネルのヒストグラムをグラフ表示する |
| LBP Histogram | Local Binary Patternヒストグラムをバーグラフ表示する |
| FFT | FFTマグニチュードスペクトラムを対数スケールで可視化する |

</details>

<details>
<summary>Image > Draw</summary>

| ノード名 | 説明 |
|:--|:--|
| Draw Text (ASCII) | OpenCVでASCIIテキストを描画する（改行対応） |
| Draw Canvas | 入力画像上にフリーハンド描画を行う |
| Draw Mask | 入力画像上にフリーハンド描画を行い二値マスクを生成する |
| Simple Concat | 2枚の画像を連結する |
| Multi Image Concat | 最大9枚の画像をグリッドレイアウトで連結する |
| Comparison Slider | 2枚の画像を比較スライダーで表示する |
| Picture In Picture | Image 2をImage 1の指定領域に重ね合わせる<br>画像上のドラッグで領域指定が可能 |
| Blend | 各種ブレンドモードで合成する |
| Alpha Blend | 重み付きアルファブレンドを行う |

</details>

<details>
<summary>Image > Output</summary>

| ノード名 | 説明 |
|:--|:--|
| Image Display | 入力画像をノード上に表示する（ノードリサイズ可能） |
| Capture | ボタン押下で画像をキャプチャし保存する |
| Write Video | 入力画像をMP4動画として保存する（STOP時に保存） |

</details>

<details>
<summary>Image > Other</summary>

| ノード名 | 説明 |
|:--|:--|
| Execute Python | ユーザー入力のPythonコードを実行する（input_image → output_image）<br>OpenAI APIキーを設定している場合、生成AIによるコード生成も可能 |

</details>

### Audio

<details>
<summary>Audio > Input</summary>

| ノード名 | 説明 |
|:--|:--|
| Mic | マイクからリアルタイム音声を取得する |
| Mic (WebSocket) | ブラウザのgetUserMedia() API経由でマイク音声を取得する<br>Echo Cancellationは、 Speaker (Browser)ノードで音声出力を行った場合のみ有効 |
| Audio File | 音声ファイル（wav, mp3, ogg）を再生する<br>※サイドバーの「Loop Playback」がONの場合、ループ再生を行う |
| Noise | 各種ノイズ信号を生成する |
| Zero | 無音（ゼロデータ）を出力する |

</details>

<details>
<summary>Audio > Dynamics</summary>

| ノード名 | 説明 |
|:--|:--|
| Volume (Hard Limit) | 音量をスケール調整し、±1.0でハードクリッピングする |
| Volume (Soft Limit : tanh) | 音量をスケール調整し、tanh関数で滑らかにクリッピングする |
| Dynamic Range Compression | 閾値を超えた信号を圧縮する |
| Expander | 閾値未満の信号を減衰する |
| Noise Gate | 閾値未満の信号をカットする |

</details>

<details>
<summary>Audio > Filter</summary>

| ノード名 | 説明 |
|:--|:--|
| Lowpass Filter | カットオフ周波数より高い成分を除去する（Butterworth IIR） |
| Highpass Filter | カットオフ周波数より低い成分を除去する（Butterworth IIR）|
| Bandpass Filter | 指定周波数範囲のみ通過させる（Butterworth IIR） |
| Bandstop Filter | 指定周波数範囲を除去する（Butterworth IIR） |
| Equalizer | 指定周波数帯域をブースト/カットする |


</details>

<details>
<summary>Audio > Deep Learning</summary>

| ノード名 | 説明 |
|:--|:--|
| Speech Enhancement | 音声を強調する（ノイズ除去） |
| Audio Classification | 音声イベントを分類する |


</details>

<details>
<summary>Audio > Recognition</summary>

| ノード名 | 説明 |
|:--|:--|
| Google STT | Google Cloud Speech-to-Text APIでストリーミング音声認識を行う<br>※コンフィグの api_keys.google_stt にGoogleクレデンシャルjsonを指定している場合のみ有効 |

</details>

<details>
<summary>Audio > Utility</summary>

| ノード名 | 説明 |
|:--|:--|
| Delay | 音声信号を指定時間遅延させる |
| Mixer | 2つの音声信号を加算ミックスする |
| Waveform to Image | 音声波形からウェーブフォーム画像を作成する |

</details>

<details>
<summary>Audio > Analysis</summary>

| ノード名 | 説明 |
|:--|:--|
| Spectrogram | 音声信号のスペクトログラムを表示する |
| Power Spectrum | 音声信号のパワースペクトルを表示する |
| VAD | 音声区間検出を行う |
| MSC | 2信号間のMagnitude Squared Coherence（周波数別類似度）を計算する |

</details>

<details>
<summary>Audio > Output</summary>

| ノード名 | 説明 |
|:--|:--|
| Speaker | スピーカーから音声を再生する |
| Speaker (Browser) | ブラウザのWeb Audio APIで音声を再生する |
| Write WAV | 入力音声をWAVファイルとして記録する（STOP時に保存） |

</details>

### Text

<details>
<summary>Text > Input</summary>

| ノード名 | 説明 |
|:--|:--|
| Text | テキストを出力 |

</details>

<details>
<summary>Text > Process</summary>

| ノード名 | 説明 |
|:--|:--|
| Text Replace | 文字列を置換する |
| Text Join | 2つのテキストを結合する |
| Text Format | テンプレートのプレースホルダー {1}〜{10} を入力値で置換する |
| JSON Parse | JSON文字列をパースしてキー/パスで値を抽出する |
| JSON Array Format | JSON配列からフィールドを抽出してテキストにフォーマットする |

</details>

<details>
<summary>Text > Deep Learning</summary>

| ノード名 | 説明 |
|:--|:--|
| Language Classification | テキストの言語を判定する |

</details>

<details>
<summary>Text > Output</summary>

| ノード名 | 説明 |
|:--|:--|
| Text Display | テキスト内容をノード上に表示する |
| Text Save | テキストをファイルに保存する |

</details>

### OpenAI

<details>
<summary>OpenAI</summary>

| ノード名 | 説明 |
|:--|:--|
| OpenAI LLM | OpenAI LLM APIを呼び出す<br>Executeボタンを押したタイミングで実行する<br>※コンフィグの api_keys.openai にOpenAI APIキーを指定している場合のみ有効 |
| OpenAI VLM | OpenAI LLM APIを呼び出す（画像入力）<br>Executeボタンを押したタイミングで実行する<br>※コンフィグの api_keys.openai にOpenAI APIキーを指定している場合のみ有効 |
| OpenAI STT | OpenAI Realtime APIで音声を文字起こしする<br>※コンフィグの api_keys.openai にOpenAI APIキーを指定している場合のみ有効 |
| OpenAI Image Generation | OpenAI Image Generation APIで画像を生成する<br>※コンフィグの api_keys.openai にOpenAI APIキーを指定している場合のみ有効 |

</details>

## Math

<details>
<summary>Math > Value</summary>

| ノード名 | 説明 |
|:--|:--|
| Int | 整数値を出力 |
| Float | 浮動小数点値を出力 |
| Clamp | 値を指定範囲内に制限する |
| Float2Int | 浮動小数点を整数に変換 |

</details>

<details>
<summary>Math > Operation</summary>

| ノード名 | 説明 |
|:--|:--|
| Add | 2つの数値を加算（a + b） |
| Sub | 2つの数値を減算（a - b） |
| Mul | 2つの数値を乗算（a × b） |
| Div | 2つの数値を除算（a ÷ b）、ゼロ除算時は0を返す |
| Mod | 剰余を計算（a % b） |
| Abs | 絶対値を計算 |
| Sin | 度数法の角度からサイン値を計算<br>・degree：角度（-360〜360度） |

</details>

<details>
<summary>Math > Logic</summary>

| ノード名 | 説明 |
|:--|:--|
| AND | 論理積（両方が0以外で1、それ以外は0） |
| OR | 論理和（どちらかが0以外で1、両方0で0） |
| NOT | 論理否定（0で1、0以外で0） |
| XOR | 排他的論理和（一方のみ0以外で1） |

</details>

### Utility

<details>
<summary>Utility</summary>

| ノード名 | 説明 |
|:--|:--|
| Elapsed Time | Start からの経過時間を出力 |
| Timer Trigger | 指定間隔でトリガー（1）を出力、それ以外は0 |
| Trigger Button | ボタン押下時にトリガー（1）を出力、それ以外は0 |

</details>

# Directory Structure
```
├── src/
│   ├── node_editor/              # コアライブラリ
│   │   ├── core.py               # グラフ実行エンジン
│   │   ├── models.py             # データモデル（Node, Port, Connection）
│   │   ├── node_def.py           # ノード定義システム
│   │   ├── commands.py           # Undo/Redo
│   │   ├── settings.py           # 設定管理
│   │   └── image_utils.py        # 画像ユーティリティ
│   ├── nodes/                    # ノード実装
│   │   ├── image/                # 画像ノード
│   │   ├── audio/                # 音声ノード
│   │   ├── math/                 # 数値演算ノード
│   │   ├── text/                 # テキストノード
│   │   ├── openai/               # OpenAI連携ノード
│   │   └── utility/              # ユーティリティノード
│   └── gui/
│       ├── reactflow/            # ReactFlow
│       │   ├── backend/          # FastAPI バックエンド
│       │   └── frontend/         # React フロントエンド
│       └── headless/             # ヘッドレス実行
├── config.example.json           # アプリケーション設定
├── download_weights.py           # モデルダウンロードスクリプト
├── run_gui_reactflow.py          # GUI起動スクリプト
├── run_gui_reactflow_colab.ipynb # Colaboratory用ノートブック
├── run_headless.py               # ヘッドレス実行スクリプト
├── requirements.txt              # Python依存パッケージ
└── requirements-gpu.txt          # GPU用追加パッケージ
```

# Custom Node Development
新しいノードを作成する場合は `src/nodes/<category>/<node_name>/` に作成します。<br>
各ノードは、`node.toml`と`impl.py`の2つのファイルで構成されます。<br>
以下は、Image/Filter/Cannyノードの構成例です。
```bash
src/nodes/
└── image/
    ├── category.toml             # カテゴリ設定
    └── filter/
        ├── category.toml         # サブカテゴリ設定
        └── canny/
            ├── node.toml         # ノードメタデータ
            └── impl.py           # 実装
```

<details>
<summary>カテゴリ定義（category.toml）</summary>

各カテゴリフォルダに `category.toml` を配置してサイドバーの表示を制御します。

```toml
display_name = "Image"    # サイドバーに表示する名前
order = 10                # 表示順序（小さいほど上）
default_open = false      # サイドバーでデフォルトで展開するか
```

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `display_name` | string | フォルダ名 | サイドバーに表示する名前 |
| `order` | int | 100 | 表示順序（小さいほど上） |
| `default_open` | bool | true | デフォルトで展開するか |
| `requires_config` | string | null | 必要な設定キー（null指定の場合は常に表示） |

</details>

<details>
<summary>ノード定義（node.toml）</summary>

##### 基本構成

```toml
name = "image.filter.canny"
version = "1.0.0"
display_name = "Canny"
description = "Applies Canny edge detection to an image."
order = 50
gui = ["reactflow", "headless"]

[[ports]]
name = "image"
data_type = "image"
direction = "inout"

[[ports]]
name = "low_threshold"
data_type = "float"
direction = "in"

[[ports]]
name = "high_threshold"
data_type = "float"
direction = "in"

[[properties]]
name = "low_threshold"
display_name = "Low"
type = "int"
default = 50
widget = "slider"
min = 0
max = 255

[[properties]]
name = "high_threshold"
display_name = "High"
type = "int"
default = 150
widget = "slider"
min = 0
max = 255
```

##### ノード設定オプション

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `name` | string | 必須 | ノードID（`category.subcategory.name`形式） |
| `version` | string | 必須 |  |
| `display_name` | string | name | サイドバー/ノードに表示する名前 |
| `description` | string | "" | ノードの説明 |
| `order` | int | 100 | カテゴリ内での表示順序 |
| `gui` | string[] | [] | 対応GUI（空=全対応、`reactflow`, `headless`） |
| `measure_time` | bool | true | 処理時間計測の対象か |
| `run_when_stopped` | bool | false | STOP中も実行するか |

##### ポート定義（`[[ports]]`）

```toml
[[ports]]
name = "image"
data_type = "image"
direction = "inout"
preview = true
```

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `name` | string | 必須 | ポート名 |
| `data_type` | string | 必須 | データ型（下記参照） |
| `direction` | string | "in" | `in`, `out`, `inout` |
| `display_name` | string | name | UIに表示する名前 |
| `preview` | bool | true | プレビュー表示するか |

**データ型一覧:**

| データ型 | 説明 |
|---------|------|
| `image` | 画像（numpy配列）。接続可能: image |
| `audio` | 音声データ。接続可能: audio |
| `int` | 整数。接続可能: int, float |
| `float` | 浮動小数点。接続可能: int, float |
| `string` | 文字列。接続可能: string |
| `trigger` | トリガー信号（0/1）。接続可能: trigger |
| `any` | 任意の型。接続可能: すべて |

</details>

<details>
<summary>プロパティ定義（[[properties]]）</summary>

##### 基本構成

```toml
[[properties]]
name = "low_threshold"
display_name = "Low"
type = "int"
default = 50
widget = "slider"
min = 0
max = 255

[[properties]]
name = "high_threshold"
display_name = "High"
type = "int"
default = 150
widget = "slider"
min = 0
max = 255
```

#### プロパティオプション

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `name` | string | 必須 | プロパティ名 |
| `display_name` | string | name | 表示名 |
| `type` | string | "float" | データ型（`int`, `float`, `string`, `bool`） |
| `default` | any | null | デフォルト値 |
| `widget` | string | "input" | UIウィジェット（下記参照） |
| `min` | float | null | 最小値（slider/number_input用） |
| `max` | float | null | 最大値（slider/number_input用） |
| `step` | float | null | ステップ値 |
| `options` | array | [] | dropdown用の選択肢 |
| `options_source` | string | null | 動的オプションソース（`cameras`, `audio_inputs`） |
| `accept` | string | null | file_pickerで許可するファイルタイプ |
| `button_label` | string | null | buttonウィジェットのラベル |
| `rows` | int | null | text_areaの行数 |
| `visible_when` | object | null | 条件付き表示 |
| `disabled_while_streaming` | bool | false | 実行中は編集不可 |
| `requires_streaming` | bool | false | 実行中のみ有効（ボタン用） |
| `requires_gpu` | bool | false | GPU利用可能時のみ表示 |
| `requires_api_key` | string | null | 指定APIキーが設定されている時のみ表示 |


##### ウィジェット一覧

**標準ウィジェット:**

| ウィジェット | 説明 |
|-------------|------|
| `slider` | スライダー。`min`, `max`, `step` |
| `number_input` | 数値入力。`min`, `max`, `step` |
| `text_input` | テキスト入力（1行） |
| `text_area` | テキストエリア（複数行）。`rows` |
| `text_display` | テキスト表示（読み取り専用） |
| `dropdown` | ドロップダウン。`options`, `options_source` |
| `checkbox` | チェックボックス |
| `color_picker` | 色選択 |
| `file_picker` | ファイル選択。`accept` |
| `button` | ボタン。`button_label`, `requires_streaming` |
| `xy_input` | XY座標入力 |
| `matrix3x3` | 3x3行列入力 |

##### ウィジェット例

**ドロップダウン:**
```toml
[[properties]]
name = "mode"
display_name = "Mode"
type = "string"
default = "auto"
widget = "dropdown"
options = [
    { value = "auto", label = "Auto" },
    { value = "manual", label = "Manual" }
]
```

**ボタン:**
```toml
[[properties]]
name = "reset"
display_name = ""
type = "bool"
default = false
widget = "button"
button_label = "Reset"
```

**条件付き表示:**
```toml
[[properties]]
name = "custom_value"
display_name = "Custom Value"
type = "int"
default = 100
widget = "slider"
visible_when = { property = "mode", values = ["manual"] }
```

**ファイル選択:**
```toml
[[properties]]
name = "file_path"
display_name = "File"
type = "string"
default = ""
widget = "file_picker"
accept = "image/*"
```

**GPU依存プロパティ:**
```toml
[[properties]]
name = "use_gpu"
display_name = "Use GPU"
type = "bool"
default = true
widget = "checkbox"
disabled_while_streaming = true
requires_gpu = true
```

**実行中のみ有効なボタン:**
```toml
[[properties]]
name = "capture"
display_name = ""
type = "bool"
default = false
widget = "button"
button_label = "Capture"
requires_streaming = true
```

</details>

<details>
<summary>実装（impl.py）</summary>

##### 基本構成（Cannyノードの例）

```python
from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2


class CannyNodeLogic(ComputeLogic):
    """
    Cannyエッジ検出を実行するノードロジック。
    入力・出力ともにOpenCV画像（numpy配列）。Base64変換はcore.pyで自動処理。
    """

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        img = inputs.get("image")
        if img is None:
            return {"image": None}

        low_threshold = int(properties.get("low_threshold", 50))
        high_threshold = int(properties.get("high_threshold", 150))

        # グレースケールに変換
        if len(img.shape) == 2 or img.shape[2] == 1:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # 出力をBGRに変換（他ノードとの互換性のため）
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return {"image": edges_bgr}
```

**ポイント:**
- クラス名は任意（`ComputeLogic`を継承）
- `compute()`メソッドが必須
- 入力は`inputs`辞書から取得（ポート名がキー）
- プロパティは`properties`辞書から取得
- 戻り値は出力ポート名をキーとした辞書

##### コンテキスト情報

`context` 引数には以下の情報が含まれます：

| キー | 型 | 説明 |
|------|---|------|
| `is_streaming` | bool | START中かどうか |
| `preview` | bool | プレビューモード（STOP状態）かどうか |
| `loop` | bool | ループ再生が有効かどうか |
| `interval_ms` | int | 実行間隔（ミリ秒） |
| `node_id` | string | 現在のノードID |
| `encode_base64` | bool | 画像をBase64エンコードするか |

##### エラーハンドリングの例

```python
from typing import Dict, Any
from node_editor.node_def import ComputeLogic

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class BlurNodeLogic(ComputeLogic):
    """ガウシアンブラーを適用"""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        # 依存ライブラリのチェック
        if not CV2_AVAILABLE:
            return {"image": None, "__error__": "opencv-python is not installed"}

        image = inputs.get("image")
        if image is None:
            return {"image": None}

        kernel_size = int(properties.get("kernel_size", 5))

        # カーネルサイズは奇数である必要がある
        if kernel_size % 2 == 0:
            kernel_size += 1

        try:
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            return {"image": blurred}
        except Exception as e:
            return {"image": None, "__error__": str(e)}
```

##### 特殊な出力キー

| キー | 説明 |
|------|------|
| `__error__` | エラーメッセージ（ノードに赤く表示） |
| `__is_busy__` | ビジー状態（trueでボタンを無効化） |
| `__update_property__` | プロパティ値の更新（プロパティ名→値のdict） |
| `__display_text__` | テキスト表示ノード用の表示テキスト |

</details>

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Multimodal-Node-Editor is under [Apache-2.0 license](LICENSE).<br>
Multimodal-Node-Editorのソースコード自体は[Apache-2.0 license](LICENSE)ですが、<br>
各AIモデルのライセンスは、それぞれのライセンスに従います。
