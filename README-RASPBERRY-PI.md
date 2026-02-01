# Raspberry Pi Stream Setup

This guide explains how to run a video stream on your **Raspberry Pi** over WiFi so your **laptop** can connect and run human & damaged-building detection (same behavior as video upload and mobile streaming).

## What you need

- **Raspberry Pi** (3, 4, or 5) with Raspberry Pi OS (Bullseye or later)
- **Same WiFi** for Pi and laptop
- **Camera**: USB webcam **or** Raspberry Pi Camera Module (v2, v3, or HQ)

---

## 1. Prepare the Pi

### 1.1 Enable the camera (if using Pi Camera Module)

```bash
sudo raspi-config
```

- **Interface Options** → **Camera** → **Enable** → Finish → Reboot.

### 1.2 Install Python and pip (if not already)

```bash
sudo apt update
sudo apt install -y python3 python3-pip
```

### 1.3 (Optional) Pi Camera Module — install picamera2

For **Pi Camera Module** on Bullseye or later:

```bash
sudo apt install -y python3-picamera2
```

For a **USB webcam** you can skip this; the script will use OpenCV only.

---

## 2. Copy the stream code to the Pi

Copy the `raspberry_pi` folder to your Pi (from your laptop):

```bash
# From your laptop (replace pi@192.168.1.50 with your Pi user and IP)
scp -r raspberry_pi pi@192.168.1.50:~/
```

Or clone/copy the repo on the Pi and use the `raspberry_pi` folder there.

---

## 3. Install dependencies on the Pi

SSH into the Pi (or use a monitor/keyboard):

```bash
ssh pi@192.168.1.50
```

Then:

```bash
cd ~/raspberry_pi
pip install --user -r requirements.txt
# or: python3 -m pip install --user -r requirements.txt
```

If you use a **virtual environment**:

```bash
cd ~/raspberry_pi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Find the Pi’s IP address

On the Pi:

```bash
hostname -I
```

Use the first address (e.g. `192.168.1.50`). Your laptop will use this IP to connect.

---

## 5. Run the stream server on the Pi

```bash
cd ~/raspberry_pi
python3 stream_server.py
```

Default port is **8080**. You should see something like:

```
Opening camera...
Camera: opencv   (or: picamera2)
Stream server at http://<this-pi-ip>:8080/video
On your laptop: enter Pi IP and port in the detection app, then Start Pi Stream.
Press Ctrl+C to stop.
```

**Options:**

```bash
python3 stream_server.py --port 8080 --width 640 --height 480
python3 stream_server.py --port 9000   # use a different port
```

Leave this terminal running while you use the stream from the laptop.

---

## 6. Connect from your laptop

1. On the **laptop**, start the detection app (e.g. `python app.py` in the main project).
2. Open the app in the browser (e.g. `http://localhost:5000`).
3. Scroll to **“Live Stream from Raspberry Pi”**.
4. Enter the **Pi’s IP** (e.g. `192.168.1.50`) and **port** (default `8080`).
5. Click **“Start Pi Stream”**.

The same detection (humans & damaged buildings) runs on this stream as on uploaded videos and mobile streaming. Use **“Stop Stream”** when done.

---

## 7. Run the stream automatically on boot (optional)

### Using systemd

Create a service file on the Pi:

```bash
sudo nano /etc/systemd/system/pi-stream.service
```

Paste (adjust paths and user if needed):

```ini
[Unit]
Description=Raspberry Pi MJPEG stream for aerial detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/raspberry_pi
ExecStart=/usr/bin/python3 /home/pi/raspberry_pi/stream_server.py --port 8080
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable pi-stream
sudo systemctl start pi-stream
sudo systemctl status pi-stream
```

Stream URL: `http://<pi-ip>:8080/video`.

---

## Troubleshooting

| Issue | What to try |
|--------|--------------|
| “No camera found” | USB: run `ls /dev/video*`. Pi Camera: enable in `raspi-config` and install `python3-picamera2`. |
| “Failed to connect” on laptop | Pi and laptop on same WiFi; firewall on Pi allows port 8080: `sudo ufw allow 8080` (if using ufw). |
| Port in use | Use another port: `python3 stream_server.py --port 9000` and enter that port in the app. |
| Slow or choppy | Lower resolution: `--width 320 --height 240`. |

---

## Summary

- **On the Pi:** install deps, run `python3 stream_server.py` (port 8080 by default).
- **On the laptop:** open the detection app → “Live Stream from Raspberry Pi” → enter Pi IP and 8080 → “Start Pi Stream”.

Detection (humans & damaged buildings) is the same as for uploaded videos and mobile streaming.
