# beepBoopSR / yolov8nLink

Part of [beepBoopSR](https://github.com/beepBoopSR/beepBoopSR) - Team beepBoop

A yolov8n prototype face detection systemm connected to an ESP32.
Built on the [LytheESP32Framework](https://github.com/Arnvvch/LytheESP32Framework) by Arnvvch.

---

## yolov8nLink

Real-time object detection server using YOLOv8n and FastAPI. Detects objects via webcam, streams annotated video to browser, and triggers ESP32 LED when target class (configurable) is detected with confidence ≥ 80%.

### Stack

- **Backend:** FastAPI, OpenCV, Ultralytics YOLO
- **Firmware:** ESP32 using [Lythe ESP32 Framework](https://github.com/Arnvvch/LytheESP32-Framework) by Arnvvch
- **Frontend:** Vanilla HTML/JS, canvas rendering, Server-Sent Events-style polling

### ESP32 Integration

Connected to ESP32 using [Lythe ESP32 Framework](https://github.com/Arnvvch/LytheESP32-Framework) by Arnvvch:

- Endpoint: `http://<ESP32-IP>/api/module/led`
- Method: `PUT` or `POST`
- Payload: `{"state": "on"}` or `{"state": "off"}`
- State is case-insensitive (`on`/`off`, `true`/`false`, `1`/`0`)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ESP32_IP` | `192.168.18.114` | ESP32 IP address |
| `CONFIDENCE_THRESHOLD` | `0.8` | Detection confidence threshold |
| `TRIGGER_CLASS` | `person` | Class name to trigger LED |
| `MODEL_PATH` | `yolov8n.pt` | YOLO model file |
| `CAMERA_SOURCE` | `0` | Camera index or URL |
| `REQUESTS_TIMEOUT` | `2` | ESP32 request timeout (seconds) |
| `LED_QUEUE_MAX` | `10` | Max queued LED updates |

**ESP32 IP:** Set `ESP32_IP` environment variable before running. Edit the default in `main.py` line 23.

```bash
# Install deps
pip install -r requirements.txt

# Run server
python main.py

# Open browser
http://localhost:8000
```

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/detect` | GET | JSON: latest detection (detections, image, count, esp32 errors) |
| `/stream` | GET | Multipart video stream |
| `/camera/set` | POST | Switch camera source (`?source=0` or URL) |
| `/config/trigger` | GET | Change trigger class (`?cls=person`) |
| `/led/config` | GET | Update ESP32 IP (`?ip=192.168.18.114`) |
| `/led/on` / `/led/off` | GET | Manual LED control |
| `/health` | GET | Service health + ESP32 connectivity |

### Hardware

- ESP32 with LED on GPIO pin 2
- WiFi configured in `config.h` (Lythe framework)
- ESP32 firmware: [Lythe ESP32 Framework](https://github.com/Arnvvch/LytheESP32-Framework) by Arnvvch

---

## Architecture

```
CameraManager thread
   ↓ get_frame() [lock read]
DetectionManager thread
   ↓ set_led() [lock enqueue]
ESP32Controller queue
   ↓ led_worker() (asyncio task)
   → requests.Session.put() [non-blocking]
```

- Camera writes frame → detection reads frame
- Detection → enqueues LED state
- Worker serializes ESP32 HTTP calls

---

## Known Limitations

- YOLO model loads at startup (blocks ~1–3 seconds)
- Frame copy overhead (buffer copy each poll)
- No frame buffering/dropping under high load
- ESP32 errors only logged, no UI notification
- `get_frame()` copy could be eliminated with zero-copy buffer
- No HTTPS (LAN only)

---

## File List

```
.
├── .gitignore           # Git ignore rules
├── LICENSE              # MIT License
├── README.md            # Project overview
├── main.py              # FastAPI server
├── requirements.txt     # Python deps
└── templates/
    └── index.html       # Web UI
```

## License

```
MIT License

Copyright (c) 2026 beepBoopSR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
