[[Japanese](README.md)/[English](README_EN.md)]

# Multimodal-Node-Editor
A node editor-based multimodal processing application.<br>
Process images, audio, text, and more by connecting nodes - designed for experimentation and comparison of processing pipelines.<br>


<img src="https://github.com/user-attachments/assets/264acff2-4b6c-460f-b6a0-77fb959a6f66" width="100%">

# Features
- Over 100 built-in nodes for image, audio, text, and deep learning
- Real-time processing with webcam and microphone input
- Easy node creation with TOML + Python (no custom GUI required)
- Headless execution of saved graphs without GUI
- Google Colaboratory backend support<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/Multimodal-Node-Editor/blob/main/run_gui_reactflow_colab.ipynb)

# Note
Nodes are added as the author (Takahashi) needs them,<br>
so some basic processing nodes for image, audio, and text may be missing.

# Requirement
<details>
<summary>Frontend</summary>

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
<summary>Backend</summary>

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
onnxruntime         1.23.2    or later  # Use onnxruntime-gpu for GPU support
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
If using Google Colaboratory, skip the following steps and follow the notebook instructions.<br>
The following assumes Python and Node.js are already installed.<br>

```bash
# Clone repository
git clone https://github.com/Kazuhito00/Multimodal-Node-Editor
cd Multimodal-Node-Editor

# Install Python packages
pip install -r requirements.txt

# Download model weights
python download_weights.py  # Download all files (skip if already exists)
# python download_weights.py --force  # Force overwrite all files
# python download_weights.py --max-size 150  # Skip files larger than specified MB

# Install Node.js packages
cd src/gui/reactflow/frontend
npm install
cd ../../../../

# Copy config
cp config.example.json config.json
```

Some nodes require API keys to function.<br>
Set the following keys in `config.json` as needed.

```json
{
  "api_keys": {
    "openai": "SET_YOUR_OPENAI_API_KEY",
    "google_stt": "PATH_TO_GOOGLE_CREDENTIALS_JSON"
  }
}
```

<details>
<summary>Config Details</summary>

| Key                         | Type     | Default       | Description                            |
|-----------------------------|----------|---------------|----------------------------------------|
| node_search_paths           | string[] | ["src/nodes"] | Directories to search for node definitions |
| ui.theme                    | string   | "light"       | Theme (light / dark)                   |
| ui.sidebar.show_edit        | bool     | false         | Show undo/redo menu in sidebar         |
| ui.sidebar.show_file        | bool     | true          | Show graph save (json) menu in sidebar |
| ui.sidebar.show_auto_layout | bool     | true          | Show auto layout button in sidebar     |
| graph.interval_ms           | int      | 50            | Graph execution interval (milliseconds)|
| audio.sample_rate           | int      | 16000         | Audio processing sample rate (Hz)      |
| camera.max_scan_count       | int      | 2             | Maximum camera device scan count       |
| auto_download.video         | bool     | false         | Auto download on video capture         |
| auto_download.wav           | bool     | false         | Auto download on audio recording       |
| auto_download.capture       | bool     | false         | Auto download on image capture         |
| auto_download.text          | bool     | false         | Auto download on text save             |
| api_keys.openai             | string   | ""            | OpenAI API key                         |
| api_keys.google_stt         | string   | ""            | Path to Google Speech-to-Text credentials JSON |
</details>

# Launch the application
* <b>Local PC</b><br>
  Run the following script. A browser will open on successful launch.
  ```
  python run_gui_reactflow.py
  ```
  | Option | Description |
  |--------|-------------|
  | `--config <path>` | Path to config file (default: config.json) |
  <br>
  Or run the following in separate terminals and access `http://localhost:5173/` in your browser.<br>
  
  ```
  uvicorn src.gui.reactflow.backend.main:app --reload
  ```
  ```
  cd src/gui/reactflow/frontend
  npm run dev
  ```
* <b>Google Colaboratory</b><br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/Multimodal-Node-Editor/blob/main/run_gui_reactflow_colab.ipynb)<br>
  Open the notebook in Colaboratory and run cells from top to bottom.<br>
  Click the `https://localhost:8000/` link shown in the final cell output<br><img src="https://github.com/user-attachments/assets/04313de9-535e-48cc-a605-2551eda35e16" width="75%">

* <b>Headless Execution</b><br>
  Execute graphs from command line without ReactFlow frontend<br>
  ```
  python run_headless.py graph.json
  ```
  | Option | Description |
  |--------|-------------|
  | `--config <path>` | Path to config file (default: config.json) |
  | `--count <n>` | Execution count (0=infinite loop, 1=run once, default: 0) |

# Usage
<details>
<summary>Node Placement & Execution</summary>

1. Drag and drop nodes from sidebar to canvas<br><img src="https://github.com/user-attachments/assets/186dccbf-cecb-48ac-86e3-975eaf6f9f7c" width="50%">
2. Connect ports between nodes<br>*Ports of same color can be connected<br><img src="https://github.com/user-attachments/assets/39594af8-8b7c-4fe7-be23-a68eaa968840" width="50%">
3. Click Start button to execute (Shortcut: Ctrl + Enter)<br>*Node placement and port connections cannot be changed during execution<br><img src="https://github.com/user-attachments/assets/c262bb0c-04a6-434f-92ac-00a43afe2864" width="50%">
</details>

<details>
<summary>Edge/Node Deletion</summary>

* Select edges or nodes to delete and press Delete key<br><img src="https://github.com/user-attachments/assets/3554273d-a265-4cf1-8eae-59bdbbd2b6d5" width="50%">
</details>

<details>
<summary>Graph JSON Export/Import</summary>

* Graph JSON Export: Click Save button (Shortcut: Ctrl + S)<br><img src="https://github.com/user-attachments/assets/e13218ae-9af6-494c-b97c-a9b1b3558736" width="50%">
* Graph JSON Import: Click Load button (Shortcut: Ctrl + L)<br><img src="https://github.com/user-attachments/assets/17df193c-ab7f-43f7-ac3b-6b14b2d8bb0a" width="50%">
</details>

<details>
<summary>Auto Layout</summary>

* Click Auto Layout button (Shortcut: Ctrl + A)<br><img src="https://github.com/user-attachments/assets/4f8c6417-a8df-40a7-904f-4a943c4b66e9" width="50%">
</details>

<details>
<summary>Node Comments</summary>

* Right-click node and click Add Comment<br><img src="https://github.com/user-attachments/assets/14a18fcf-51c5-4d27-bd33-122cd5a44d5b" width="50%">
</details>

# Keyboard Shortcuts
| Shortcut                         | Action               | Notes                                       |
|----------------------------------|----------------------|---------------------------------------------|
| Ctrl + Enter                     | Toggle START/STOP    | STOP if running, START if stopped           |
| Escape                           | STOP                 | Stop graph execution                        |
| Ctrl + P                         | Toggle Pause/Resume  | Pause if running, Resume if paused          |
| Ctrl + Z                         | Undo                 | Undo last action                            |
| Ctrl + Y                         | Redo                 | Redo last action                            |
| Ctrl + A                         | Auto Layout          | Auto-arrange nodes                          |
| Ctrl + S                         | Save                 | Save graph as JSON (export)                 |
| Ctrl + L                         | Load                 | Load graph JSON (import)                    |
| Delete                           | Delete               | Delete selected nodes/edges (disabled during execution) |

# Nodes

### Image

<details>
<summary>Image > Input</summary>

| Node Name | Description |
|:--|:--|
| Image | Load still image files (jpg, png, bmp, gif) |
| Webcam | Capture real-time video from webcam<br>Not available on Colaboratory backend |
| Webcam (WebSocket) | Capture webcam video via browser getUserMedia() API<br>Available on Colaboratory backend |
| Video | Load and play video files (mp4, avi) frame by frame<br>- Realtime Sync checkbox: Read frames synchronized with processing time<br>- Frame Step: Frame interval (only when realtime_sync=false)<br>- Preload All Frame checkbox: Preload all frames<br>*Loops when sidebar "Loop Playback" is ON |
| Video Frame | Output image at specified frame position of video |
| RTSP | Capture video from network camera RTSP input |
| Solid Color | Generate solid color image<br>- width: Image width (1-4096, default: 640)<br>- height: Image height (1-4096, default: 360)<br>- color: Color (color picker, default: #ff0000) |
| URL Image | Download and load image from URL |

</details>

<details>
<summary>Image > Transform</summary>

| Node Name | Description |
|:--|:--|
| Crop | Crop region specified by normalized coordinates (0.0-1.0)<br>Drag on image area to specify region |
| Flip | Flip image horizontally/vertically |
| Resize | Resize with specified resolution and interpolation method |
| Rotate | Rotate image by specified angle (margins appear for non-90-degree multiples) |
| 3D Rotate | Perform pitch/yaw/roll rotation in 3D space |
| Click Perspective | Perspective transform by 4-point specification via image click |

</details>

<details>
<summary>Image > Filter</summary>

| Node Name | Description |
|:--|:--|
| Apply Color Map | Apply pseudo-color to grayscale image |
| Background Subtraction | Detect foreground using background subtraction |
| Blur | Apply various blur filters |
| Morphology | Perform morphological transformations |
| Brightness | Adjust brightness by addition |
| Canny | Perform Canny edge detection |
| Contrast | Adjust contrast |
| Equalize Hist | Apply histogram equalization to HSV V channel |
| Filter 2D (3x3) | Apply convolution filter with custom 3x3 kernel |
| Gamma | Apply gamma correction (using LUT table) |
| Grayscale | Convert image to grayscale (maintains 3 channels) |
| RGB Extract | Extract specified RGB channel |
| RGB Adjust | Add values to each RGB channel |
| HSV Adjust | Adjust hue, saturation, and value in HSV color space |
| Inpaint | Inpaint using mask |
| Omnidirectional Viewer | Display and rotate equirectangular 360-degree image<br>Drag on image to change viewpoint |
| Sepia | Apply sepia effect |
| Threshold | Binarize with various algorithms |

</details>

<details>
<summary>Image > Marker Detection</summary>

| Node Name | Description |
|:--|:--|
| QR Code | Detect and decode QR codes, output results as JSON |
| ArUco Marker | Detect ArUco markers, output ID and corner coordinates as JSON |
| AprilTag | Detect AprilTags, output ID and corner coordinates as JSON |

</details>

<details>
<summary>Image > Deep Learning</summary>

| Node Name | Description |
|:--|:--|
| Image Classification | Classify images with ImageNet 1000 classes<br>- Model: Select model (dropdown) |
| Object Detection | Perform object detection<br>Supports multi-object tracking with motpy and slice detection with SAHI |
| Face Detection | Perform face detection<br>Supports multi-object tracking with motpy and slice detection with SAHI |
| Low-Light Image Enhancement | Enhance low-light images |
| Depth Estimation | Perform monocular depth estimation |
| Pose Estimation | Perform human pose estimation |
| Hand Pose Estimation | Perform hand pose estimation |
| Semantic Segmentation | Perform semantic segmentation |
| OCR | Perform optical character recognition |

</details>

<details>
<summary>Image > Analysis</summary>

| Node Name | Description |
|:--|:--|
| Color Histogram | Display histogram graph for each channel |
| LBP Histogram | Display Local Binary Pattern histogram as bar graph |
| FFT | Visualize FFT magnitude spectrum in logarithmic scale |

</details>

<details>
<summary>Image > Draw</summary>

| Node Name | Description |
|:--|:--|
| Draw Text (ASCII) | Draw ASCII text with OpenCV (supports newlines) |
| Draw Canvas | Freehand drawing on input image |
| Draw Mask | Generate binary mask by freehand drawing on input image |
| Simple Concat | Concatenate 2 images |
| Multi Image Concat | Concatenate up to 9 images in grid layout |
| Comparison Slider | Display 2 images with comparison slider |
| Picture In Picture | Overlay Image 2 on specified region of Image 1<br>Drag on image to specify region |
| Blend | Composite with various blend modes |
| Alpha Blend | Perform weighted alpha blending |

</details>

<details>
<summary>Image > Output</summary>

| Node Name | Description |
|:--|:--|
| Image Display | Display input image on node (resizable) |
| Capture | Capture and save image on button press |
| Write Video | Save input images as MP4 video (saved on STOP) |

</details>

<details>
<summary>Image > Other</summary>

| Node Name | Description |
|:--|:--|
| Execute Python | Execute user-input Python code (input_image -> output_image)<br>AI code generation available when OpenAI API key is set |

</details>

### Audio

<details>
<summary>Audio > Input</summary>

| Node Name | Description |
|:--|:--|
| Mic | Capture real-time audio from microphone |
| Mic (WebSocket) | Capture microphone audio via browser getUserMedia() API<br>Echo Cancellation only works when audio is output via Speaker (Browser) node |
| Audio File | Play audio files (wav, mp3, ogg)<br>*Loops when sidebar "Loop Playback" is ON |
| Noise | Generate various noise signals |
| Zero | Output silence (zero data) |

</details>

<details>
<summary>Audio > Dynamics</summary>

| Node Name | Description |
|:--|:--|
| Volume (Hard Limit) | Scale volume with hard clipping at +/-1.0 |
| Volume (Soft Limit : tanh) | Scale volume with smooth clipping using tanh function |
| Dynamic Range Compression | Compress signals exceeding threshold |
| Expander | Attenuate signals below threshold |
| Noise Gate | Cut signals below threshold |

</details>

<details>
<summary>Audio > Filter</summary>

| Node Name | Description |
|:--|:--|
| Lowpass Filter | Remove components above cutoff frequency (Butterworth IIR) |
| Highpass Filter | Remove components below cutoff frequency (Butterworth IIR) |
| Bandpass Filter | Pass only specified frequency range (Butterworth IIR) |
| Bandstop Filter | Remove specified frequency range (Butterworth IIR) |
| Equalizer | Boost/cut specified frequency band |

</details>

<details>
<summary>Audio > Deep Learning</summary>

| Node Name | Description |
|:--|:--|
| Speech Enhancement | Enhance speech (noise removal) |
| Audio Classification | Classify audio events |

</details>

<details>
<summary>Audio > Recognition</summary>

| Node Name | Description |
|:--|:--|
| Google STT | Perform streaming speech recognition with Google Cloud Speech-to-Text API<br>*Only works when api_keys.google_stt is set in config |

</details>

<details>
<summary>Audio > Utility</summary>

| Node Name | Description |
|:--|:--|
| Delay | Delay audio signal by specified time |
| Mixer | Mix two audio signals by addition |
| Waveform to Image | Create waveform image from audio |

</details>

<details>
<summary>Audio > Analysis</summary>

| Node Name | Description |
|:--|:--|
| Spectrogram | Display spectrogram of audio signal |
| Power Spectrum | Display power spectrum of audio signal |
| VAD | Perform voice activity detection |
| MSC | Calculate Magnitude Squared Coherence between 2 signals (frequency-wise similarity) |

</details>

<details>
<summary>Audio > Output</summary>

| Node Name | Description |
|:--|:--|
| Speaker | Play audio through speaker |
| Speaker (Browser) | Play audio using browser Web Audio API |
| Write WAV | Record input audio as WAV file (saved on STOP) |

</details>

### Text

<details>
<summary>Text > Input</summary>

| Node Name | Description |
|:--|:--|
| Text | Output text |

</details>

<details>
<summary>Text > Process</summary>

| Node Name | Description |
|:--|:--|
| Text Replace | Replace strings |
| Text Join | Concatenate two texts |
| Text Format | Replace template placeholders {1}-{10} with input values |
| JSON Parse | Parse JSON string and extract value by key/path |
| JSON Array Format | Extract fields from JSON array and format as text |

</details>

<details>
<summary>Text > Deep Learning</summary>

| Node Name | Description |
|:--|:--|
| Language Classification | Detect text language |

</details>

<details>
<summary>Text > Output</summary>

| Node Name | Description |
|:--|:--|
| Text Display | Display text content on node |
| Text Save | Save text to file |

</details>

### OpenAI

<details>
<summary>OpenAI</summary>

| Node Name | Description |
|:--|:--|
| OpenAI LLM | Call OpenAI LLM API<br>Executed when Execute button is pressed<br>*Only works when api_keys.openai is set in config |
| OpenAI VLM | Call OpenAI LLM API (with image input)<br>Executed when Execute button is pressed<br>*Only works when api_keys.openai is set in config |
| OpenAI STT | Transcribe audio with OpenAI Realtime API<br>*Only works when api_keys.openai is set in config |
| OpenAI Image Generation | Generate images with OpenAI Image Generation API<br>*Only works when api_keys.openai is set in config |

</details>

## Math

<details>
<summary>Math > Value</summary>

| Node Name | Description |
|:--|:--|
| Int | Output integer value |
| Float | Output floating-point value |
| Clamp | Clamp value within specified range |
| Float2Int | Convert floating-point to integer |

</details>

<details>
<summary>Math > Operation</summary>

| Node Name | Description |
|:--|:--|
| Add | Add two numbers (a + b) |
| Sub | Subtract two numbers (a - b) |
| Mul | Multiply two numbers (a x b) |
| Div | Divide two numbers (a / b), returns 0 on division by zero |
| Mod | Calculate modulo (a % b) |
| Abs | Calculate absolute value |
| Sin | Calculate sine value from angle in degrees<br>- degree: Angle (-360 to 360 degrees) |

</details>

<details>
<summary>Math > Logic</summary>

| Node Name | Description |
|:--|:--|
| AND | Logical AND (1 if both non-zero, otherwise 0) |
| OR | Logical OR (1 if either non-zero, 0 if both zero) |
| NOT | Logical NOT (1 if zero, 0 if non-zero) |
| XOR | Exclusive OR (1 if only one is non-zero) |

</details>

### Utility

<details>
<summary>Utility</summary>

| Node Name | Description |
|:--|:--|
| Elapsed Time | Output elapsed time since Start |
| Timer Trigger | Output trigger (1) at specified intervals, otherwise 0 |
| Trigger Button | Output trigger (1) on button press, otherwise 0 |

</details>

# Directory Structure
```
src/
  node_editor/                # Core library
    core.py                   # Graph execution engine
    models.py                 # Data models (Node, Port, Connection)
    node_def.py               # Node definition system
    commands.py               # Undo/Redo
    settings.py               # Settings management
    image_utils.py            # Image utilities
  nodes/                      # Node implementations
    image/                    # Image nodes
    audio/                    # Audio nodes
    math/                     # Math operation nodes
    text/                     # Text nodes
    openai/                   # OpenAI integration nodes
    utility/                  # Utility nodes
  gui/
    reactflow/                # ReactFlow
      backend/                # FastAPI backend
      frontend/               # React frontend
    headless/                 # Headless execution
config.example.json           # Application settings
download_weights.py           # Model download script
run_gui_reactflow.py          # GUI launch script
run_gui_reactflow_colab.ipynb # Colaboratory notebook
run_headless.py               # Headless execution script
requirements.txt              # Python dependencies
requirements-gpu.txt          # GPU additional packages
```

# Custom Node Development
Create new nodes in `src/nodes/<category>/<node_name>/`.<br>
Each node consists of two files: `node.toml` and `impl.py`.<br>
Below is an example of the Image/Filter/Canny node structure.
```bash
src/nodes/
  image/
    category.toml         # Category settings
    filter/
      category.toml       # Subcategory settings
      canny/
        node.toml         # Node metadata
        impl.py           # Implementation
```

<details>
<summary>Category Definition (category.toml)</summary>

Place `category.toml` in each category folder to control sidebar display.

```toml
display_name = "Image"    # Name displayed in sidebar
order = 10                # Display order (smaller = higher)
default_open = false      # Whether to expand by default in sidebar
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `display_name` | string | folder name | Name displayed in sidebar |
| `order` | int | 100 | Display order (smaller = higher) |
| `default_open` | bool | true | Whether to expand by default |
| `requires_config` | string | null | Required config key (always shown if null) |

</details>

<details>
<summary>Node Definition (node.toml)</summary>

##### Basic Structure

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

##### Node Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Node ID (`category.subcategory.name` format) |
| `version` | string | required | |
| `display_name` | string | name | Name displayed in sidebar/node |
| `description` | string | "" | Node description |
| `order` | int | 100 | Display order within category |
| `gui` | string[] | [] | Supported GUIs (empty=all, `reactflow`, `headless`) |
| `measure_time` | bool | true | Whether to measure processing time |
| `run_when_stopped` | bool | false | Whether to run when STOPPED |

##### Port Definition (`[[ports]]`)

```toml
[[ports]]
name = "image"
data_type = "image"
direction = "inout"
preview = true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Port name |
| `data_type` | string | required | Data type (see below) |
| `direction` | string | "in" | `in`, `out`, `inout` |
| `display_name` | string | name | Name displayed in UI |
| `preview` | bool | true | Whether to show preview |

**Data Types:**

| Data Type | Description |
|-----------|-------------|
| `image` | Image (numpy array). Compatible: image |
| `audio` | Audio data. Compatible: audio |
| `int` | Integer. Compatible: int, float |
| `float` | Floating-point. Compatible: int, float |
| `string` | String. Compatible: string |
| `trigger` | Trigger signal (0/1). Compatible: trigger |
| `any` | Any type. Compatible: all |

</details>

<details>
<summary>Property Definition ([[properties]])</summary>

##### Basic Structure

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

#### Property Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Property name |
| `display_name` | string | name | Display name |
| `type` | string | "float" | Data type (`int`, `float`, `string`, `bool`) |
| `default` | any | null | Default value |
| `widget` | string | "input" | UI widget (see below) |
| `min` | float | null | Minimum value (for slider/number_input) |
| `max` | float | null | Maximum value (for slider/number_input) |
| `step` | float | null | Step value |
| `options` | array | [] | Choices for dropdown |
| `options_source` | string | null | Dynamic option source (`cameras`, `audio_inputs`) |
| `accept` | string | null | Allowed file types for file_picker |
| `button_label` | string | null | Label for button widget |
| `rows` | int | null | Number of rows for text_area |
| `visible_when` | object | null | Conditional display |
| `disabled_while_streaming` | bool | false | Disable editing during execution |
| `requires_streaming` | bool | false | Only active during execution (for buttons) |
| `requires_gpu` | bool | false | Only show when GPU is available |
| `requires_api_key` | string | null | Only show when specified API key is set |


##### Widget List

**Standard Widgets:**

| Widget | Description |
|--------|-------------|
| `slider` | Slider. `min`, `max`, `step` |
| `number_input` | Number input. `min`, `max`, `step` |
| `text_input` | Text input (single line) |
| `text_area` | Text area (multiple lines). `rows` |
| `text_display` | Text display (read-only) |
| `dropdown` | Dropdown. `options`, `options_source` |
| `checkbox` | Checkbox |
| `color_picker` | Color picker |
| `file_picker` | File picker. `accept` |
| `button` | Button. `button_label`, `requires_streaming` |
| `xy_input` | XY coordinate input |
| `matrix3x3` | 3x3 matrix input |

##### Widget Examples

**Dropdown:**
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

**Button:**
```toml
[[properties]]
name = "reset"
display_name = ""
type = "bool"
default = false
widget = "button"
button_label = "Reset"
```

**Conditional Display:**
```toml
[[properties]]
name = "custom_value"
display_name = "Custom Value"
type = "int"
default = 100
widget = "slider"
visible_when = { property = "mode", values = ["manual"] }
```

**File Picker:**
```toml
[[properties]]
name = "file_path"
display_name = "File"
type = "string"
default = ""
widget = "file_picker"
accept = "image/*"
```

**GPU-dependent Property:**
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

**Button Only Active During Execution:**
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
<summary>Implementation (impl.py)</summary>

##### Basic Structure (Canny node example)

```python
from typing import Dict, Any
from node_editor.node_def import ComputeLogic
import cv2


class CannyNodeLogic(ComputeLogic):
    """
    Node logic for Canny edge detection.
    Both input and output are OpenCV images (numpy arrays). Base64 conversion is handled automatically by core.py.
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

        # Convert to grayscale
        if len(img.shape) == 2 or img.shape[2] == 1:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Convert output to BGR (for compatibility with other nodes)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return {"image": edges_bgr}
```

**Key Points:**
- Class name is arbitrary (inherits from `ComputeLogic`)
- `compute()` method is required
- Get inputs from `inputs` dict (port names as keys)
- Get properties from `properties` dict
- Return value is a dict with output port names as keys

##### Context Information

The `context` argument contains the following information:

| Key | Type | Description |
|-----|------|-------------|
| `is_streaming` | bool | Whether currently running (START) |
| `preview` | bool | Whether in preview mode (STOP state) |
| `loop` | bool | Whether loop playback is enabled |
| `interval_ms` | int | Execution interval (milliseconds) |
| `node_id` | string | Current node ID |
| `encode_base64` | bool | Whether to Base64 encode images |

##### Error Handling Example

```python
from typing import Dict, Any
from node_editor.node_def import ComputeLogic

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class BlurNodeLogic(ComputeLogic):
    """Apply Gaussian blur"""

    def compute(
        self,
        inputs: Dict[str, Any],
        properties: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        # Check for required library
        if not CV2_AVAILABLE:
            return {"image": None, "__error__": "opencv-python is not installed"}

        image = inputs.get("image")
        if image is None:
            return {"image": None}

        kernel_size = int(properties.get("kernel_size", 5))

        # Kernel size must be odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        try:
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            return {"image": blurred}
        except Exception as e:
            return {"image": None, "__error__": str(e)}
```

##### Special Output Keys

| Key | Description |
|-----|-------------|
| `__error__` | Error message (displayed in red on node) |
| `__is_busy__` | Busy state (disables buttons when true) |
| `__update_property__` | Property value update (dict of property name -> value) |
| `__display_text__` | Display text for text display nodes |

</details>

# Author
Kazuhito Takahashi (https://x.com/KzhtTkhs)

# License
Multimodal-Node-Editor is under [Apache-2.0 license](LICENSE).<br>
The source code of Multimodal-Node-Editor itself is under [Apache-2.0 license](LICENSE),<br>
but each AI model's license follows its respective license.
