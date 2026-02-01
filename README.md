# YOLOv9 Aerial Human Detection System

A web-based application for detecting humans and damaged buildings in aerial videos using YOLOv9 and a custom YOLO damage model. The system processes uploaded videos, detects persons and damaged buildings, draws bounding boxes, and displays the processed video.

## Features

- 🚁 Aerial video processing with YOLOv9
- 👤 Human/person detection (COCO person class)
- 🏚️ Damaged building detection (custom trained model, optional)
- 📦 Bounding box visualization with confidence scores
- 🌐 Modern Flask web interface
- 📤 Video upload with drag & drop support
- ⏳ Real-time processing status with loader
- 🎥 Video playback of processed results
- 📱 **Live streaming from mobile devices** (DroidCam support)
- 🔴 **Real-time human detection** on live video streams

## Requirements

- Python 3.9 or higher
- pip (Python package manager)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd drone-based-arial-detection
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add damage detection model (optional)**  
   To also detect damaged buildings, place your trained `best.pt` in the project root (or set `DAMAGE_MODEL_PATH` to its path). If the file is missing, only person detection runs.

5. **Install FFmpeg (Optional but Recommended)**
   FFmpeg ensures better browser compatibility for video playback. If not installed, the system will use OpenCV's codecs.
   
   **macOS:**
   ```bash
   brew install ffmpeg
   ```
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get install ffmpeg
   ```
   
   **Windows:**
   Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH

## Usage

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Upload a video**
   - Click "Choose Video File" or drag and drop a video file
   - Supported formats: MP4, AVI, MOV, MKV, FLV, WMV
   - Maximum file size: 500MB

4. **Process the video**
   - Click "Detect Humans" button
   - Wait for processing to complete (a loader will be shown)
   - The processed video with bounding boxes will be displayed automatically

## Live Streaming from Mobile Device

You can also stream live video from your mobile device using DroidCam:

1. **Install DroidCam on your phone**
   - Download DroidCam from Google Play Store or App Store
   - Install the DroidCam client on your computer (optional, for USB connection)

2. **Connect your phone to the same WiFi network** as your computer

3. **Start DroidCam on your phone**
   - Open the DroidCam app
   - Note the IP address and port shown in the app (default port is 4747)

4. **Start live streaming in the web interface**
   - Enter the IP address from DroidCam (e.g., `192.168.1.100`)
   - Enter the port (default: `4747`)
   - Click "Start Live Stream"
   - The live video feed with human detection will appear in real-time

5. **Stop streaming**
   - Click "Stop Stream" when done

**Note**: Make sure both devices are on the same WiFi network for the connection to work.

## Live Streaming from Raspberry Pi

You can stream video from a **Raspberry Pi** (same WiFi as laptop) and run the same detection:

1. **On the Pi:** Copy the `raspberry_pi` folder to the Pi, install dependencies, and run the stream server (see **[README-RASPBERRY-PI.md](README-RASPBERRY-PI.md)** for full setup).
2. **On the laptop:** Open the app → **Live Stream from Raspberry Pi** → enter the Pi’s IP and port (default **8080**) → click **Start Pi Stream**.

Detection (humans & damaged buildings) is the same as for uploads and mobile streaming. See **README-RASPBERRY-PI.md** for Pi setup, camera options, and troubleshooting.

## How It Works

1. **Video Upload**: User uploads an aerial video file through the web interface
2. **Model Loading**: YOLOv9 model (yolov9c.pt) is loaded (automatically downloaded on first use)
3. **Frame Processing**: Each frame is processed through YOLOv9 to detect objects
4. **Human Filtering**: Only detections with class "person" (class 0) are kept
5. **Bounding Box Drawing**: Green bounding boxes with confidence scores are drawn around detected humans
6. **Video Output**: Processed frames are compiled into a new video file
7. **Display**: The final video is shown in the web interface

## Project Structure

```
drone-based-arial-detection/
├── app.py                 # Flask backend application
├── templates/
│   └── index.html         # Frontend HTML/JS interface
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── uploads/              # Uploaded video files (auto-created)
└── outputs/              # Processed video files (auto-created)
```

## Technical Details

- **Person model**: YOLOv9 (yolov9c.pt) or YOLOv8 (yolov8n.pt) via Ultralytics — detects person (COCO class 0).
- **Damage model**: Custom YOLO `best.pt` (trained with `damage.yaml`, class `damaged_building`) — loaded from project root or `DAMAGE_MODEL_PATH`.
- **Confidence threshold**: 0.25 for both models.
- **Visualization**: Green boxes = person, red boxes = damaged building.
- **Video Codec**: MP4V
- **Backend**: Flask
- **Frontend**: HTML5, CSS3, JavaScript

## Notes

- The YOLOv9 model will be automatically downloaded on first run (~50-100MB)
- Processing time depends on video length and resolution
- Larger videos will take longer to process
- The system processes videos frame by frame for accurate detection

## Troubleshooting

- **Model download issues**: Ensure you have internet connection for first-time model download
- **Video not processing**: Check that the video format is supported and file size is under 500MB
- **Memory errors**: Try with smaller videos or lower resolution
- **Port already in use**: Change the port in `app.py` (default: 5000)
