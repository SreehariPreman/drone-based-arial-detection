#!/usr/bin/env python3
"""
Raspberry Pi MJPEG stream server for aerial detection app.
Streams camera video over WiFi so the laptop app can run human & damaged-building detection.

Usage on Pi:
  python3 stream_server.py [--port 8080] [--width 640] [--height 480]

Then on your laptop: open the detection app, go to "Live Stream from Raspberry Pi",
enter the Pi's IP and port (default 8080), click "Start Pi Stream".
"""

import argparse
import io
import sys
import time

# Debug logging: set to True for verbose output
DEBUG = True
def _log(msg):
    if DEBUG:
        ts = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{ts}] [Pi Stream] {msg}", flush=True)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from picamera2 import Picamera2
    HAS_PICAM2 = True
except ImportError:
    Picamera2 = None
    HAS_PICAM2 = False

# Minimal HTTP server (no Flask required on Pi for lighter footprint)
from http.server import HTTPServer, BaseHTTPRequestHandler


def get_camera(width=640, height=480, fps=20, device=0):
    """Open camera: prefer Picamera2 (Pi Camera Module), fallback to OpenCV (USB)."""
    _log("get_camera: opening camera (width=%s, height=%s, fps=%s, device=%s)" % (width, height, fps, device))
    if HAS_PICAM2 and Picamera2:
        try:
            _log("Trying Picamera2...")
            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls={"FrameRate": fps},
            )
            picam2.configure(config)
            picam2.start()
            _log("Picamera2 opened successfully")
            return ("picamera2", picam2)
        except Exception as e:
            _log("Picamera2 failed: %s, trying OpenCV..." % e)
    if HAS_CV2:
        _log("Trying OpenCV VideoCapture(%s)..." % device)
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            _log("OpenCV camera (device=%s) opened successfully" % device)
            return ("opencv", cap)
        else:
            _log("OpenCV VideoCapture(%s) failed to open" % device)
    raise RuntimeError("No camera found. Install opencv-python and/or picamera2.")


def read_frame(camera_type, camera):
    """Read one frame as JPEG bytes."""
    try:
        if camera_type == "picamera2":
            frame = camera.capture_array()
            if frame is None or frame.size == 0:
                _log("read_frame (picamera2): capture_array() returned empty/None")
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                _log("read_frame (picamera2): imencode failed")
                return None
            return jpeg.tobytes()
        else:
            # OpenCV
            ret, frame = camera.read()
            if not ret:
                _log("read_frame (opencv): camera.read() returned ret=False (no frame)")
                return None
            if frame is None or frame.size == 0:
                _log("read_frame (opencv): frame is None or empty after read()")
                return None
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                _log("read_frame (opencv): imencode failed")
                return None
            return jpeg.tobytes()
    except Exception as e:
        _log("read_frame error: %s" % e)
        import traceback
        traceback.print_exc()
        return None


def release_camera(camera_type, camera):
    """Release camera resource."""
    if camera_type == "picamera2":
        try:
            camera.stop()
        except Exception:
            pass
    else:
        camera.release()


class MJPEGHandler(BaseHTTPRequestHandler):
    """Serves MJPEG stream at /video and / (and /stream for compatibility)."""

    def do_GET(self):
        client_addr = self.client_address[0] if self.client_address else "?"
        _log("GET %s from %s" % (self.path, client_addr))
        if self.path not in ("/", "/video", "/stream"):
            _log("404 path not allowed: %s" % self.path)
            self.send_error(404)
            return
        _log("Sending 200 + multipart headers to %s" % client_addr)
        self.send_response(200)
        self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        frame_count = 0
        none_count = 0
        try:
            while True:
                jpeg = read_frame(self.server.camera_type, self.server.camera)
                if jpeg is None:
                    none_count += 1
                    if none_count <= 3 or none_count % 100 == 0:
                        _log("read_frame returned None (count=%s)" % none_count)
                    time.sleep(0.05)
                    continue
                none_count = 0
                frame_count += 1
                if frame_count <= 3 or frame_count % 100 == 0:
                    _log("Sending frame %s size=%s bytes to %s" % (frame_count, len(jpeg), client_addr))
                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-type", "image/jpeg")
                self.send_header("Content-length", len(jpeg))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                try:
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    raise
        except (BrokenPipeError, ConnectionResetError) as e:
            _log("Client disconnected: %s (sent %s frames)" % (type(e).__name__, frame_count))
        except Exception as e:
            _log("Stream error: %s (sent %s frames)" % (e, frame_count))
            import traceback
            traceback.print_exc()

    def log_message(self, format, *args):
        _log("HTTP %s" % (args[0],))


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi MJPEG stream server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--device", type=int, default=0, help="OpenCV camera device index (default: 0, try 1 if no frame)")
    args = parser.parse_args()

    if not HAS_CV2:
        print("opencv-python is required. Install: pip install opencv-python-headless")
        return 1

    _log("Opening camera...")
    camera_type, camera = get_camera(args.width, args.height, device=args.device)
    _log("Camera type: %s" % camera_type)

    # Warmup: verify we can read at least one frame before serving
    _log("Warmup: reading a few frames...")
    for i in range(5):
        jpeg = read_frame(camera_type, camera)
        if jpeg is not None:
            _log("Warmup: frame %s OK (%s bytes)" % (i + 1, len(jpeg)))
            break
        _log("Warmup: frame %s failed (see read_frame message above)" % (i + 1))
        time.sleep(0.2)
    else:
        _log("WARNING: Could not read any frame during warmup. Stream may show no video.")

    server = HTTPServer((args.host, args.port), MJPEGHandler)
    server.camera_type = camera_type
    server.camera = camera

    try:
        _log("Stream server listening at http://<this-pi-ip>:%s/video (host=%s)" % (args.port, args.host))
        _log("On your laptop: enter Pi IP and port in the detection app, then Start Pi Stream.")
        _log("Press Ctrl+C to stop.")
        server.serve_forever()
    except KeyboardInterrupt:
        _log("Stopping (Ctrl+C)...")
    finally:
        _log("Releasing camera and closing server")
        release_camera(camera_type, camera)
        server.server_close()
    return 0


if __name__ == "__main__":
    exit(main())
