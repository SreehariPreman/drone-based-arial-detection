import os
import sys
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import torch

# Debug logging for Pi/stream connection (set to True for verbose stream debug)
STREAM_DEBUG = True
def _stream_log(msg):
    if STREAM_DEBUG:
        import time
        ts = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{ts}] [Stream] {msg}", flush=True)
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
# Higher = fewer false positives for "damaged building". Tune if 100-epoch model over-detects.
DAMAGE_CONF_THRESHOLD = '0.25'

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
            print(f"Damage model loaded from {DAMAGE_MODEL_PATH} (conf threshold={DAMAGE_CONF_THRESHOLD})")
        except Exception as e:
            print(f"Warning: Could not load damage model from {DAMAGE_MODEL_PATH}: {e}")
    return damage_model

def _draw_box(frame, x1, y1, x2, y2, color, label):
    """Draw a bounding box with a filled label background on frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
    text_color = (0, 0, 0) if color == (0, 255, 0) else (255, 255, 255)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


def run_detection_and_draw(frame, conf_threshold=0.25, damage_conf_threshold=None):
    """Run person + damage detection (no tracking) for live streaming.
    Returns (annotated_frame, person_count, building_count) where counts are per-frame."""
    if damage_conf_threshold is None:
        damage_conf_threshold = DAMAGE_CONF_THRESHOLD
    person_model = load_model()
    annotated = frame.copy()
    person_count = 0
    building_count = 0

    # 1) Person detection (class 0 = person in COCO)
    for result in person_model(frame, verbose=False):
        for box in result.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) >= conf_threshold:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                _draw_box(annotated, x1, y1, x2, y2, (0, 255, 0), f'Person {float(box.conf[0]):.2f}')

    # 2) Damage detection (custom model: class 0 = damaged_building)
    dm = load_damage_model()
    if dm is not None:
        for result in dm(frame, verbose=False, conf=damage_conf_threshold):
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= damage_conf_threshold:
                    building_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    _draw_box(annotated, x1, y1, x2, y2, (0, 0, 255), f'Damaged building {conf:.2f}')

    return annotated, person_count, building_count


def run_detection_and_draw_tracked(frame, person_model, dm, conf_threshold=0.25, damage_conf_threshold=None):
    """Run person + damage detection with object tracking for video processing.
    Returns (annotated_frame, person_track_ids_this_frame, building_track_ids_this_frame).
    Track IDs are unique per object across the whole video — collect into sets to count uniques."""
    if damage_conf_threshold is None:
        damage_conf_threshold = DAMAGE_CONF_THRESHOLD
    annotated = frame.copy()
    person_ids = set()
    building_ids = set()

    # 1) Person detection with tracking (persist=True keeps tracker state across frames)
    for result in person_model.track(frame, verbose=False, persist=True, classes=[0]):
        if result.boxes is None:
            continue
        for box in result.boxes:
            if int(box.cls[0]) != 0 or float(box.conf[0]) < conf_threshold:
                continue
            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is not None:
                person_ids.add(track_id)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            id_tag = f' #{track_id}' if track_id is not None else ''
            _draw_box(annotated, x1, y1, x2, y2, (0, 255, 0),
                      f'Person {float(box.conf[0]):.2f}{id_tag}')

    # 2) Building detection with tracking
    if dm is not None:
        for result in dm.track(frame, verbose=False, persist=True, conf=damage_conf_threshold):
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < damage_conf_threshold:
                    continue
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is not None:
                    building_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                id_tag = f' #{track_id}' if track_id is not None else ''
                _draw_box(annotated, x1, y1, x2, y2, (0, 0, 255),
                          f'Damaged building {conf:.2f}{id_tag}')

    return annotated, person_ids, building_ids

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
    # Use sets to count unique tracked objects across the entire video,
    # so the same person/building seen in multiple frames is only counted once.
    unique_person_ids = set()
    unique_building_ids = set()
    print("[Upload] Starting video processing — detecting humans & damaged buildings...")

    # Reset tracker state from any previous run
    person_model = load_model()
    dm = load_damage_model()
    if hasattr(person_model, 'predictor') and person_model.predictor is not None:
        person_model.predictor = None
    if dm is not None and hasattr(dm, 'predictor') and dm.predictor is not None:
        dm.predictor = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, frame_person_ids, frame_building_ids = run_detection_and_draw_tracked(
                frame, person_model, dm, conf_threshold=0.25
            )
            unique_person_ids.update(frame_person_ids)
            unique_building_ids.update(frame_building_ids)

            # Ensure contiguous uint8 BGR for VideoWriter (avoids OpenCV "Bad argument" on some systems)
            if annotated_frame.dtype != np.uint8:
                annotated_frame = np.clip(annotated_frame, 0, 255).astype(np.uint8)
            if len(annotated_frame.shape) == 2:
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2BGR)
            out.write(np.ascontiguousarray(annotated_frame))
            frame_count += 1
            if frame_count % 30 == 0 or frame_count == 1:
                print(f"  [Upload] Frame {frame_count} — persons so far: {len(unique_person_ids)}, buildings so far: {len(unique_building_ids)}")

    finally:
        cap.release()
        out.release()

    total_persons = len(unique_person_ids)
    total_buildings = len(unique_building_ids)
    print(f"[Upload] Finished — {frame_count} frames | total persons: {total_persons} | total buildings: {total_buildings}")

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
    
    return frame_count, total_persons, total_buildings

def process_frame_for_streaming(frame):
    """Process a single frame for live streaming (person + damaged building). Returns annotated frame only."""
    annotated, _, _ = run_detection_and_draw(frame, conf_threshold=0.25)
    return annotated

def generate_frames():
    """Generator function for MJPEG streaming"""
    global frame_buffer, frame_lock
    _stream_log("generate_frames: started (streaming_active=%s)" % streaming_active)
    yielded = 0
    no_buffer_count = 0
    while streaming_active:
        with frame_lock:
            if frame_buffer is not None:
                no_buffer_count = 0
                ret, buffer = cv2.imencode('.jpg', frame_buffer, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yielded += 1
                    if yielded <= 3 or yielded % 100 == 0:
                        _stream_log("generate_frames: yielding frame #%s (%s bytes)" % (yielded, len(frame_bytes)))
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                no_buffer_count += 1
                if no_buffer_count <= 5 or no_buffer_count % 50 == 0:
                    _stream_log("generate_frames: no frame_buffer yet (count=%s)" % no_buffer_count)
        time.sleep(0.033)  # ~30 FPS
    _stream_log("generate_frames: ended (streaming_active=False, yielded %s frames)" % yielded)

def streaming_worker_mjpeg(url):
    """Worker thread for MJPEG streaming using requests library"""
    global streaming_active, frame_buffer, frame_lock, current_stream_url

    _stream_log("streaming_worker_mjpeg: connecting to %s" % url)
    try:
        response = requests.get(url, stream=True, timeout=10)
        _stream_log("streaming_worker_mjpeg: response status=%s headers=%s" % (
            response.status_code, dict(response.headers)))
        if response.status_code != 200:
            _stream_log("HTTP error: %s" % response.status_code)
            return False

        bytes_data = b''
        mjpeg_frame_count = 0
        chunk_count = 0
        stream_total_persons = 0
        stream_total_buildings = 0
        _stream_log("[Live stream] MJPEG: detection started — showing live counts (persons / buildings)...")
        for chunk in response.iter_content(chunk_size=8192):
            if not streaming_active:
                _stream_log("streaming_worker_mjpeg: streaming_active=False, exiting")
                break
            chunk_count += 1
            if chunk_count <= 5 or chunk_count % 200 == 0:
                _stream_log("Chunk #%s received, size=%s bytes, total buffer=%s" % (
                    chunk_count, len(chunk) if chunk else 0, len(bytes_data) + (len(chunk) if chunk else 0)))
            if chunk:
                bytes_data += chunk

            # Look for JPEG frame markers
            while b'\xff\xd8' in bytes_data:
                start = bytes_data.find(b'\xff\xd8')
                end_marker = bytes_data.find(b'\xff\xd9', start)

                if end_marker != -1:
                    jpeg_data = bytes_data[start:end_marker + 2]
                    bytes_data = bytes_data[end_marker + 2:]
                    if mjpeg_frame_count < 3:
                        _stream_log("Complete JPEG frame #%s size=%s bytes" % (mjpeg_frame_count + 1, len(jpeg_data)))
                    try:
                        img = Image.open(BytesIO(jpeg_data))
                        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        processed_frame, person_count, building_count = run_detection_and_draw(frame, conf_threshold=0.25)
                        stream_total_persons += person_count
                        stream_total_buildings += building_count
                        mjpeg_frame_count += 1
                        if mjpeg_frame_count <= 3 or mjpeg_frame_count % 90 == 0:
                            _stream_log("MJPEG frame %s processed, updating frame_buffer" % mjpeg_frame_count)
                        if mjpeg_frame_count == 1 or mjpeg_frame_count % 30 == 0:
                            ts = time.strftime("%H:%M:%S", time.localtime())
                            print(f"  [{ts}] [Live stream] Frame {mjpeg_frame_count} — this frame: persons={person_count}, buildings={building_count} | session total: persons={stream_total_persons}, buildings={stream_total_buildings}", flush=True)
                        with frame_lock:
                            frame_buffer = processed_frame
                    except Exception as e:
                        _stream_log("Error processing MJPEG frame: %s" % e)
                        continue
                else:
                    break

        _stream_log("streaming_worker_mjpeg: loop ended (chunks=%s, frames=%s)" % (chunk_count, mjpeg_frame_count))

    except requests.exceptions.RequestException as e:
        _stream_log("Request error: %s" % e)
        return False
    except Exception as e:
        _stream_log("Error in MJPEG stream: %s" % e)
        import traceback
        traceback.print_exc()
        return False

    return True

def streaming_worker(ip_address, port):
    """Worker thread for processing live stream"""
    global streaming_active, stream_cap, frame_buffer, frame_lock, current_stream_url

    _stream_log("streaming_worker: ip=%s port=%s" % (ip_address, port))
    url_formats = [
        "http://%s:%s/video" % (ip_address, port),
        "http://%s:%s/stream" % (ip_address, port),
        "http://%s:%s/" % (ip_address, port),
        "http://%s:%s/mjpegfeed?640x480" % (ip_address, port),
        "http://%s:%s/mjpegfeed" % (ip_address, port),
    ]
    _stream_log("URLs to try: %s" % url_formats)

    cap = None
    connected_url = None
    use_requests = False

    _stream_log("Attempting connection with OpenCV VideoCapture...")
    for url in url_formats:
        try:
            _stream_log("Trying OpenCV: %s" % url)
            for backend in [cv2.CAP_ANY, cv2.CAP_FFMPEG]:
                try:
                    cap = cv2.VideoCapture(url, backend)
                    if cap.isOpened():
                        time.sleep(0.5)
                        ret, test_frame = cap.read()
                        _stream_log("OpenCV read() ret=%s frame=%s" % (ret, "ok" if (test_frame is not None and test_frame.size > 0) else "empty"))
                        if ret and test_frame is not None and test_frame.size > 0:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_FPS, 30)
                            _stream_log("Successfully connected with OpenCV: %s" % url)
                            connected_url = url
                            current_stream_url = url
                            break
                        else:
                            cap.release()
                            cap = None
                except Exception as be:
                    _stream_log("OpenCV backend error: %s" % be)
                    if cap:
                        cap.release()
                        cap = None
                    continue

            if cap and cap.isOpened():
                break
        except Exception as e:
            _stream_log("OpenCV failed for %s: %s" % (url, e))
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
            continue

    if cap is None or not cap.isOpened():
        _stream_log("OpenCV failed, trying requests library for MJPEG stream...")
        for url in url_formats:
            _stream_log("Trying requests MJPEG: %s" % url)
            try:
                test_response = requests.get(url, stream=True, timeout=3)
                _stream_log("requests.get %s -> status=%s Content-Type=%s" % (
                    url, test_response.status_code, test_response.headers.get("Content-Type")))
                if test_response.status_code == 200:
                    _stream_log("URL is accessible (200), starting MJPEG stream: %s" % url)
                    connected_url = url
                    current_stream_url = url
                    use_requests = True
                    break
            except Exception as e:
                _stream_log("Cannot access %s: %s" % (url, e))
                continue

    if not use_requests and (cap is None or not cap.isOpened()):
        _stream_log("Failed to connect to stream. Check IP, port, and that Pi stream is running.")
        _stream_log("Try in browser: http://%s:%s/video" % (ip_address, port))
        streaming_active = False
        return

    stream_cap = cap if not use_requests else "requests"
    _stream_log("Using %s for stream; connected_url=%s" % ("requests (MJPEG)" if use_requests else "OpenCV", connected_url))

    try:
        if use_requests:
            # Use requests library for MJPEG streaming
            streaming_worker_mjpeg(connected_url)
        else:
            # Use OpenCV VideoCapture
            stream_frame_count = 0
            stream_total_persons = 0
            stream_total_buildings = 0
            print("[Live stream] Detection started — showing live counts (persons / buildings)...")
            while streaming_active:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from stream")
                    time.sleep(0.1)
                    continue

                annotated_frame, person_count, building_count = run_detection_and_draw(frame, conf_threshold=0.25)
                stream_total_persons += person_count
                stream_total_buildings += building_count

                with frame_lock:
                    frame_buffer = annotated_frame

                stream_frame_count += 1
                if stream_frame_count == 1 or stream_frame_count % 30 == 0:
                    print(f"  [Live stream] Frame {stream_frame_count} — this frame: persons={person_count}, buildings={building_count} | session total: persons={stream_total_persons}, buildings={stream_total_buildings}")

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
            print(f"[Upload] Video uploaded: {filename} — starting detection...")
            # Process video
            frame_count, total_persons, total_buildings = process_video(input_path, output_path)
            
            return jsonify({
                'success': True,
                'output_file': output_filename,
                'frames_processed': frame_count,
                'total_persons': total_persons,
                'total_buildings': total_buildings
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

    _stream_log("start_stream: called")
    if streaming_active:
        _stream_log("start_stream: already active, rejecting")
        return jsonify({'error': 'Streaming is already active'}), 400

    data = request.get_json()
    if not data or 'ip_address' not in data:
        _stream_log("start_stream: missing ip_address")
        return jsonify({'error': 'IP address is required'}), 400

    ip_address = data['ip_address'].strip()
    port = str(data.get('port', '4747'))
    _stream_log("start_stream: ip_address=%s port=%s" % (ip_address, port))

    if ip_address.startswith('http://') or ip_address.startswith('https://'):
        try:
            from urllib.parse import urlparse
            parsed = urlparse(ip_address)
            ip_address = parsed.hostname or ip_address.replace('http://', '').replace('https://', '').split('/')[0]
            if parsed.port:
                port = str(parsed.port)
        except Exception:
            ip_address = ip_address.replace('http://', '').replace('https://', '').split('/')[0]

    stream_url = "http://%s:%s/video" % (ip_address, port)
    current_stream_url = stream_url
    streaming_active = True
    _stream_log("start_stream: starting streaming_worker thread for %s:%s" % (ip_address, port))

    streaming_thread = threading.Thread(target=streaming_worker, args=(ip_address, port), daemon=True)
    streaming_thread.start()

    time.sleep(2)
    _stream_log("start_stream: after 2s wait, streaming_active=%s" % streaming_active)

    if not streaming_active:
        _stream_log("start_stream: connection failed (streaming_active=False)")
        return jsonify({'error': 'Failed to connect to camera. Check IP address and ensure DroidCam is running.'}), 500

    _stream_log("start_stream: success, stream_url=%s" % stream_url)
    return jsonify({
        'success': True,
        'message': 'Streaming started',
        'stream_url': stream_url
    })

@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    """Stop live streaming"""
    global streaming_active, stream_cap

    _stream_log("stop_stream: called (streaming_active=%s)" % streaming_active)
    if not streaming_active:
        return jsonify({'error': 'No active stream'}), 400

    streaming_active = False
    if stream_cap and hasattr(stream_cap, 'release'):
        try:
            stream_cap.release()
        except Exception as e:
            _stream_log("stop_stream: release error: %s" % e)
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
    _stream_log("video_feed: request (streaming_active=%s)" % streaming_active)
    if not streaming_active:
        _stream_log("video_feed: rejected (not active)")
        return jsonify({'error': 'Streaming is not active'}), 400
    _stream_log("video_feed: returning Response(generate_frames())")
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Loading YOLO models...")
    load_model()
    load_damage_model()
    print("Models loaded successfully!")
    app.run(debug=True, host='0.0.0.0', port=5000)
