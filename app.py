import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
import torch.serialization

# Fix for PyTorch 2.6+ compatibility with YOLOv9
# Patch torch.load to allow loading YOLO models
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    # For YOLO model files, use weights_only=False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Apply patch before importing YOLO
torch.load = _patched_torch_load

# Also add safe globals for ultralytics classes
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass

from ultralytics import YOLO
import tempfile
import shutil
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLOv9 model (will download automatically if not present)
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    global model
    if model is None:
        # Using YOLOv9 model - ultralytics supports YOLOv9
        # For person detection, we use COCO pretrained model which includes 'person' class (class 0)
        # Try YOLOv9 first, fallback to YOLOv8 if there are compatibility issues
        try:
            model = YOLO('yolov9c.pt')  # or 'yolov9t.pt' for smaller model
        except Exception as e:
            print(f"Warning: YOLOv9 loading failed ({e}), trying YOLOv8 instead...")
            # Fallback to YOLOv8 which is more stable
            model = YOLO('yolov8n.pt')  # YOLOv8 nano - fast and accurate
    return model

def reencode_video_for_browser(input_path, output_path):
    """Re-encode video using ffmpeg for browser compatibility"""
    try:
        # Use ffmpeg to re-encode with H.264 codec (browser-compatible)
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',  # Enable streaming
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def process_video(input_path, output_path):
    """Process video to detect humans and draw bounding boxes"""
    model = load_model()
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use temporary file for initial encoding
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    
    # Try browser-compatible codecs in order of preference
    codecs_to_try = [
        ('avc1', 'H.264/AVC'),  # Best browser support
        ('H264', 'H.264'),
        ('mp4v', 'MPEG-4'),     # Fallback
    ]
    
    out = None
    codec_used = None
    
    for fourcc_str, codec_name in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            if out.isOpened():
                codec_used = codec_name
                break
        except:
            if out:
                out.release()
            continue
    
    if out is None or not out.isOpened():
        raise ValueError("Failed to initialize video writer with any codec")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv9 detection
            results = model(frame, verbose=False)
            
            # Process results and draw bounding boxes only for 'person' class (class 0)
            annotated_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class ID and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Only draw boxes for 'person' class (class 0 in COCO dataset)
                    if cls == 0 and conf > 0.25:  # Confidence threshold
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label with confidence
                        label = f'Person {conf:.2f}'
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Write frame to output video
            out.write(annotated_frame)
            frame_count += 1
            
    finally:
        cap.release()
        out.release()
    
    # Re-encode with ffmpeg for better browser compatibility if available
    if reencode_video_for_browser(temp_output, output_path):
        # Remove temp file if re-encoding succeeded
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"Video re-encoded with H.264 for browser compatibility")
    else:
        # If ffmpeg not available, just rename temp file
        if os.path.exists(temp_output):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output, output_path)
        print(f"Video encoded with {codec_used} codec")
    
    return frame_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Generate output filename
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_detected.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        try:
            # Process video
            frame_count = process_video(input_path, output_path)
            
            return jsonify({
                'success': True,
                'output_file': output_filename,
                'frames_processed': frame_count
            })
        except Exception as e:
            return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/output/<filename>')
def get_output_video(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        # Use proper MIME type and enable range requests for video streaming
        response = send_file(file_path, mimetype='video/mp4')
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Type'] = 'video/mp4'
        return response
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print("Loading YOLOv9 model...")
    load_model()
    print("Model loaded successfully!")
    app.run(debug=True, host='0.0.0.0', port=5000)
