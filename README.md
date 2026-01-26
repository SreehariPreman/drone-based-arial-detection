# YOLOv9 Aerial Human Detection System

A web-based application for detecting humans in aerial videos using YOLOv9 pre-trained model. The system processes uploaded videos, detects only humans/persons, draws bounding boxes around them, and displays the processed video.

## Features

- 🚁 Aerial video processing with YOLOv9
- 👤 Human/person detection only (filters out other objects)
- 📦 Bounding box visualization with confidence scores
- 🌐 Modern Flask web interface
- 📤 Video upload with drag & drop support
- ⏳ Real-time processing status with loader
- 🎥 Video playback of processed results

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

4. **Install FFmpeg (Optional but Recommended)**
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

- **Model**: YOLOv9 (yolov9c.pt) via Ultralytics
- **Detection Class**: Person (class 0 from COCO dataset)
- **Confidence Threshold**: 0.25
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
