import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import torch
import torch.serialization
import threading
import time

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
import requests
from io import BytesIO
from PIL import Image

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
# Damage/building detection model (trained custom model - best.pt)
damage_model = None
DAMAGE_MODEL_PATH = os.environ.get('DAMAGE_MODEL_PATH', 'best.pt')

# Live streaming variables
streaming_active = False
streaming_thread = None
stream_cap = None
current_stream_url = None
frame_buffer = None
frame_lock = threading.Lock()

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

def load_damage_model():
    """Load custom YOLO model for damaged_building detection (best.pt)."""
    global damage_model
    if damage_model is None and os.path.isfile(DAMAGE_MODEL_PATH):
        try:
            damage_model = YOLO(DAMAGE_MODEL_PATH)
            print(f"Damage model loaded from {DAMAGE_MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not load damage model from {DAMAGE_MODEL_PATH}: {e}")
    return damage_model

def run_detection_and_draw(frame, conf_threshold=0.25):
    """Run person + damage detection and draw boxes on frame. Returns annotated frame."""
    person_model = load_model()
    annotated = frame.copy()

    # 1) Person detection (class 0 = person in COCO)
    person_results = person_model(frame, verbose=False)
    for result in person_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                color = (0, 255, 0)  # green
                label = f'Person {conf:.2f}'
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 2) Damage detection (custom model: class 0 = damaged_building)
    dm = load_damage_model()
    if dm is not None:
        damage_results = dm(frame, verbose=False, conf=conf_threshold)
        for result in damage_results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                color = (0, 0, 255)  # red for damage
                label = f'Damaged building {conf:.2f}'
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated

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
    """Process video to detect humans and damaged buildings, draw bounding boxes"""
    load_model()
    load_damage_model()

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
    print("[Detection] Starting video processing — detecting humans & damaged buildings...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = run_detection_and_draw(frame, conf_threshold=0.25)
            out.write(annotated_frame)
            frame_count += 1
            if frame_count % 30 == 0 or frame_count == 1:
                print(f"  [Detection] Frame {frame_count}: detecting...")

    finally:
        cap.release()
        out.release()
    
    print(f"[Detection] Finished — {frame_count} frames processed.")

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

def process_frame_for_streaming(frame):
    """Process a single frame for live streaming (person + damaged building)."""
    return run_detection_and_draw(frame, conf_threshold=0.25)

def generate_frames():
    """Generator function for MJPEG streaming"""
    global frame_buffer, frame_lock
    
    while streaming_active:
        with frame_lock:
            if frame_buffer is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame_buffer, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

def streaming_worker_mjpeg(url):
    """Worker thread for MJPEG streaming using requests library"""
    global streaming_active, frame_buffer, frame_lock, current_stream_url
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            print(f"HTTP error: {response.status_code}")
            return False
        
        bytes_data = b''
        mjpeg_frame_count = 0
        print("[Detection] Live stream (MJPEG): detecting humans & damaged buildings...")
        for chunk in response.iter_content(chunk_size=8192):
            if not streaming_active:
                break
            
            bytes_data += chunk
            
            # Look for JPEG frame markers
            while b'\xff\xd8' in bytes_data:
                start = bytes_data.find(b'\xff\xd8')
                # Find the end of this JPEG frame
                end_marker = bytes_data.find(b'\xff\xd9', start)
                
                if end_marker != -1:
                    # Complete JPEG frame found
                    jpeg_data = bytes_data[start:end_marker + 2]
                    bytes_data = bytes_data[end_marker + 2:]  # Remove processed data
                    
                    try:
                        # Convert JPEG bytes to OpenCV image
                        img = Image.open(BytesIO(jpeg_data))
                        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        
                        # Process frame with YOLOv9
                        processed_frame = process_frame_for_streaming(frame)
                        mjpeg_frame_count += 1
                        if mjpeg_frame_count % 90 == 0 or mjpeg_frame_count == 1:
                            print(f"  [Detection] Live stream frame {mjpeg_frame_count}: detecting...")
                        
                        # Update frame buffer
                        with frame_lock:
                            frame_buffer = processed_frame
                    except Exception as e:
                        print(f"Error processing MJPEG frame: {e}")
                        continue
                else:
                    # Incomplete frame, wait for more data
                    break
                    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return False
    except Exception as e:
        print(f"Error in MJPEG stream: {e}")
        return False
    
    return True

def streaming_worker(ip_address, port):
    """Worker thread for processing live stream"""
    global streaming_active, stream_cap, frame_buffer, frame_lock, current_stream_url
    
    # Try different URL formats for DroidCam
    # DroidCam supports multiple formats - try /video first as it's the MJPEG endpoint
    url_formats = [
        f"http://{ip_address}:{port}/video",   # DroidCam / Pi stream (MJPEG)
        f"http://{ip_address}:{port}/stream", # Pi stream alternative path
        f"http://{ip_address}:{port}/",       # Root URL (may serve MJPEG)
        f"http://{ip_address}:{port}/mjpegfeed?640x480",
        f"http://{ip_address}:{port}/mjpegfeed",
    ]
    
    # Try OpenCV VideoCapture first
    cap = None
    connected_url = None
    use_requests = False
    
    print("Attempting connection with OpenCV VideoCapture...")
    for url in url_formats:
        try:
            print(f"Trying OpenCV: {url}")
            # Try different backends
            for backend in [cv2.CAP_ANY, cv2.CAP_FFMPEG]:
                try:
                    cap = cv2.VideoCapture(url, backend)
                    if cap.isOpened():
                        # Give it a moment to connect
                        time.sleep(0.5)
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_FPS, 30)
                            print(f"✓ Successfully connected with OpenCV: {url}")
                            connected_url = url
                            current_stream_url = url
                            break
                        else:
                            cap.release()
                            cap = None
                except:
                    if cap:
                        cap.release()
                        cap = None
                    continue
            
            if cap and cap.isOpened():
                break
        except Exception as e:
            print(f"✗ OpenCV failed for {url}: {e}")
            if cap:
                try:
                    cap.release()
                except:
                    pass
                cap = None
            continue
    
    # If OpenCV failed, try requests library for MJPEG
    if cap is None or not cap.isOpened():
        print("\nOpenCV failed, trying requests library for MJPEG stream...")
        for url in url_formats:
            print(f"Trying requests MJPEG: {url}")
            # Test if URL is accessible
            try:
                test_response = requests.get(url, stream=True, timeout=3)
                if test_response.status_code == 200:
                    print(f"✓ URL is accessible, starting MJPEG stream: {url}")
                    connected_url = url
                    current_stream_url = url
                    use_requests = True
                    break
            except Exception as e:
                print(f"✗ Cannot access {url}: {e}")
                continue
    
    if not use_requests and (cap is None or not cap.isOpened()):
        print("\n✗ Failed to connect to stream. Please check:")
        print("1. DroidCam is running on your phone")
        print("2. Both devices are on the same WiFi network")
        print("3. IP address and port are correct")
        print("4. Try accessing the URL in a browser: http://" + ip_address + ":" + port + "/video")
        streaming_active = False
        return
    
    stream_cap = cap if not use_requests else "requests"
    print("[Detection] Live stream: detecting humans & damaged buildings...")

    try:
        if use_requests:
            # Use requests library for MJPEG streaming
            streaming_worker_mjpeg(connected_url)
        else:
            # Use OpenCV VideoCapture
            stream_frame_count = 0
            while streaming_active:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from stream")
                    time.sleep(0.1)
                    continue
                
                # Process frame with YOLOv9
                processed_frame = process_frame_for_streaming(frame)
                stream_frame_count += 1
                if stream_frame_count % 90 == 0 or stream_frame_count == 1:
                    print(f"  [Detection] Live stream frame {stream_frame_count}: detecting...")
                
                # Update frame buffer
                with frame_lock:
                    frame_buffer = processed_frame
                
                time.sleep(0.033)  # ~30 FPS
    except Exception as e:
        print(f"Error in streaming worker: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap and not use_requests:
            cap.release()
        streaming_active = False
        stream_cap = None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/workspace')
def workspace():
    mode = request.args.get('mode', 'upload')
    if mode not in ('upload', 'mobile', 'pi'):
        mode = 'upload'
    return render_template('workspace.html', mode=mode)


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
            print(f"[Detection] Video uploaded: {filename} — starting detection...")
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

@app.route('/stream/start', methods=['POST'])
def start_stream():
    """Start live streaming from IP camera"""
    global streaming_active, streaming_thread, current_stream_url
    
    if streaming_active:
        return jsonify({'error': 'Streaming is already active'}), 400
    
    data = request.get_json()
    if not data or 'ip_address' not in data:
        return jsonify({'error': 'IP address is required'}), 400
    
    ip_address = data['ip_address'].strip()
    port = data.get('port', '4747')  # DroidCam default port
    
    # Remove http:// or https:// if user included it
    if ip_address.startswith('http://') or ip_address.startswith('https://'):
        # Extract IP from URL if user provided full URL
        try:
            from urllib.parse import urlparse
            parsed = urlparse(ip_address)
            ip_address = parsed.hostname or ip_address.replace('http://', '').replace('https://', '').split('/')[0]
            if parsed.port:
                port = str(parsed.port)
        except:
            ip_address = ip_address.replace('http://', '').replace('https://', '').split('/')[0]
    
    # Construct stream URL for display
    stream_url = f"http://{ip_address}:{port}/video"
    current_stream_url = stream_url
    streaming_active = True
    
    # Start streaming thread with IP and port separately
    streaming_thread = threading.Thread(target=streaming_worker, args=(ip_address, port), daemon=True)
    streaming_thread.start()
    
    # Wait a bit to see if connection succeeds
    time.sleep(2)
    
    if not streaming_active:
        return jsonify({'error': 'Failed to connect to camera. Check IP address and ensure DroidCam is running.'}), 500
    
    return jsonify({
        'success': True,
        'message': 'Streaming started',
        'stream_url': stream_url
    })

@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    """Stop live streaming"""
    global streaming_active, stream_cap
    
    if not streaming_active:
        return jsonify({'error': 'No active stream'}), 400
    
    streaming_active = False
    
    if stream_cap:
        stream_cap.release()
        stream_cap = None
    
    return jsonify({'success': True, 'message': 'Streaming stopped'})

@app.route('/stream/status', methods=['GET'])
def stream_status():
    """Get streaming status"""
    return jsonify({
        'active': streaming_active,
        'stream_url': current_stream_url if streaming_active else None
    })

@app.route('/video_feed')
def video_feed():
    """MJPEG streaming endpoint"""
    if not streaming_active:
        return jsonify({'error': 'Streaming is not active'}), 400
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Loading YOLO models...")
    load_model()
    load_damage_model()
    print("Models loaded successfully!")
    app.run(debug=True, host='0.0.0.0', port=5000)
