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
import time

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


def get_camera(width=640, height=480, fps=20):
    """Open camera: prefer Picamera2 (Pi Camera Module), fallback to OpenCV (USB)."""
    if HAS_PICAM2 and Picamera2:
        try:
            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls={"FrameRate": fps},
            )
            picam2.configure(config)
            picam2.start()
            return ("picamera2", picam2)
        except Exception as e:
            print(f"Picamera2 failed: {e}, trying OpenCV...")
    if HAS_CV2:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            return ("opencv", cap)
    raise RuntimeError("No camera found. Install opencv-python and/or picamera2.")


def read_frame(camera_type, camera):
    """Read one frame as JPEG bytes."""
    if camera_type == "picamera2":
        frame = camera.capture_array()
        # Picamera2 returns RGB; OpenCV expects BGR for imencode
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode(".jpg", frame)
    else:
        ret, frame = camera.read()
        if not ret:
            return None
        ret, jpeg = cv2.imencode(".jpg", frame)
    if not ret:
        return None
    return jpeg.tobytes()


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
        if self.path not in ("/", "/video", "/stream"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                jpeg = read_frame(self.server.camera_type, self.server.camera)
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-type", "image/jpeg")
                self.send_header("Content-length", len(jpeg))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"Stream error: {e}")

    def log_message(self, format, *args):
        print(f"[Stream] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi MJPEG stream server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    args = parser.parse_args()

    if not HAS_CV2:
        print("opencv-python is required. Install: pip install opencv-python-headless")
        return 1

    print("Opening camera...")
    camera_type, camera = get_camera(args.width, args.height)
    print(f"Camera: {camera_type}")

    server = HTTPServer((args.host, args.port), MJPEGHandler)
    server.camera_type = camera_type
    server.camera = camera

    try:
        print(f"Stream server at http://<this-pi-ip>:{args.port}/video")
        print("On your laptop: enter Pi IP and port in the detection app, then Start Pi Stream.")
        print("Press Ctrl+C to stop.")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        release_camera(camera_type, camera)
        server.server_close()
    return 0


if __name__ == "__main__":
    exit(main())
