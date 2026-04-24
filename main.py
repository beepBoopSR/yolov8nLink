"""
FastAPI server with YOLOv8 webcam object detection.
"""

import asyncio
import base64
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

# Configuration
ESP32_IP = os.getenv("ESP32_IP", "192.168.18.114")
PERSON_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))
TRIGGER_CLASS = os.getenv("TRIGGER_CLASS", "person")
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
REQUESTS_TIMEOUT = float(os.getenv("REQUESTS_TIMEOUT", "2"))
LED_QUEUE_MAX = int(os.getenv("LED_QUEUE_MAX", "10"))

class ESP32Controller:
    def __init__(self, ip):
        self.ip = ip
        self.url = f"http://{ip}/api/module/led"
        self.last_state = None
        self.lock = threading.Lock()
        # Use session with keep-alive and retry
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.1))
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        # Async queue for LED updates
        self.queue = None
        self.loop = None
        self.worker_task = None
        self.error_count = 0
        self.last_error = None

    def set_led(self, state: bool):
        """Queue LED state update asynchronously."""
        with self.lock:
            if state == self.last_state:
                return
            self.last_state = state
        state_str = "on" if state else "off"
        # Non-blocking queue put; drop if full
        try:
            asyncio.run_coroutine_threadsafe(self.queue.put(state_str), self.loop)
        except Exception:
            pass  # Queue full or event loop closed

    async def start_worker(self, loop):
        """Start the LED worker task. Call once at startup."""
        self.loop = loop
        self.queue = asyncio.Queue(maxsize=LED_QUEUE_MAX)
        self.worker_task = asyncio.create_task(self.led_worker())

    async def stop_worker(self):
        """Stop the LED worker task."""
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None

    async def led_worker(self):
        """Background task: serializes all ESP32 requests."""
        while True:
            state_str = await self.queue.get()
            payload = {"state": state_str}
            try:
                print(f"Triggering ESP32: {self.url} -> {payload}")
                response = self.session.put(self.url, json=payload, timeout=REQUESTS_TIMEOUT)
                if response.status_code != 200:
                    self.error_count += 1
                    self.last_error = f"HTTP {response.status_code}"
                    print(f"ESP32 returned error: {response.status_code}")
                else:
                    print(f"ESP32 LED turned {state_str}")
            except requests.exceptions.ConnectTimeout:
                self.error_count += 1
                self.last_error = "timeout"
                print(f"ESP32 timeout - device unreachable")
            except requests.exceptions.ConnectionError:
                self.error_count += 1
                self.last_error = "unreachable"
                print(f"ESP32 unreachable - check network")
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                print(f"ESP32 error: {type(e).__name__}: {e}")
            finally:
                self.queue.task_done()

class CameraManager:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.source = "0"
        self.pending_source = None

    def set_source(self, source: str):
        with self.lock:
            self.pending_source = source
            self.frame = None
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._update, daemon=True)
                self.thread.start()

    def _update(self):
        while self.running:
            new_source = None
            with self.lock:
                if self.pending_source is not None:
                    new_source = self.pending_source
                    self.pending_source = None
            
            if new_source is not None:
                if self.cap:
                    self.cap.release()


                if new_source.isdigit():
                    # CAP_DSHOW is often faster on Windows
                    self.cap = cv2.VideoCapture(int(new_source), cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(new_source)

                if self.cap.isOpened():
                    self.source = new_source
                else:
                    print(f"Failed to open: {new_source}")

            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        with self.lock:
                            self.frame = frame
                    else:
                        time.sleep(0.01)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()

class DetectionManager:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        self.model = YOLO(model_path)
        self.latest_result = {"detections": [], "image": None, "count": 0}
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_inference, daemon=True)
        self.thread.start()

    def _run_inference(self):
        while self.running:
            try:
                frame = camera.get_frame()
                if frame is not None:
                    results = self.model(frame, verbose=False)
                    
                    detections = []
                    person_detected = False
                    
                    if len(results) > 0:
                        result = results[0]
                        boxes = result.boxes
                        if boxes is not None:
                            for i in range(len(boxes)):
                                box = boxes[i]
                                xyxy = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                class_name = result.names[cls]

                                if class_name == TRIGGER_CLASS and conf >= PERSON_CONFIDENCE_THRESHOLD:
                                    person_detected = True

                                detections.append({
                                    "class": class_name,
                                    "confidence": round(conf, 4),
                                    "bbox": {
                                        "x1": float(xyxy[0]),
                                        "y1": float(xyxy[1]),
                                        "x2": float(xyxy[2]),
                                        "y2": float(xyxy[3]),
                                    },
                                })

                        annotated_frame = result.plot()
                        _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        image_base64 = base64.b64encode(buffer).decode("utf-8")
                        
                        with self.lock:
                            self.latest_result = {
                                "detections": detections,
                                "count": len(detections),
                                "image": f"data:image/jpeg;base64,{image_base64}"
                            }
                    
                    esp32.set_led(person_detected)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Inference error: {e}")
                time.sleep(0.1)

    def get_latest(self):
        with self.lock:
            return dict(self.latest_result)

    def clear_latest(self):
        with self.lock:
            self.latest_result = {"detections": [], "image": None, "count": 0}

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

camera = CameraManager()
esp32 = ESP32Controller(ESP32_IP)
detector = DetectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await esp32.start_worker(loop)
    camera.set_source(CAMERA_SOURCE)
    detector.start()
    yield
    detector.stop()
    camera.stop()
    await esp32.stop_worker()

app = FastAPI(title="YOLOv8 Webcam Server", lifespan=lifespan)

# Set up templates directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/health")
async def health_check():
    # Check ESP32 liveness
    try:
        response = esp32.session.get(f"http://{esp32.ip}/api", timeout=1)
        esp32_ok = response.status_code == 200
    except Exception:
        esp32_ok = False
    return {
        "status": "healthy" if esp32_ok else "degraded",
        "esp32_connected": esp32_ok,
        "esp32_ip": esp32.ip,
        "esp32_errors": esp32.error_count,
        "last_error": esp32.last_error,
    }

@app.post("/camera/set")
async def set_camera_source(source: str):
    source = source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Camera source cannot be empty")

    # Basic validation: numeric index or non-empty URL
    if not (source.isdigit() or source.startswith(("http://", "https://", "rtsp://"))):
        raise HTTPException(status_code=400, detail="Invalid camera source format")

    detector.clear_latest()
    camera.set_source(source)
    return {"status": "ok", "source": source}

@app.get("/config/trigger")
async def set_trigger_class(cls: str):
    global TRIGGER_CLASS
    TRIGGER_CLASS = cls
    return {"status": "ok", "trigger_class": TRIGGER_CLASS}

@app.get("/led/config")
async def update_esp_config(ip: str):
    """Update the ESP32 IP address."""
    ip = ip.strip()
    # Basic IPv4 validation (xxx.xxx.xxx.xxx)
    parts = ip.split(".")
    if len(parts) != 4 or not all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        raise HTTPException(status_code=400, detail="Invalid IP address format")

    esp32.ip = ip
    esp32.url = f"http://{ip}/api/module/led"
    with esp32.lock:
        esp32.last_state = None
    return {"status": "updated", "new_ip": ip}

@app.get("/led/{state}")
async def manual_led_control(state: str, brightness: int = 255):
    """Manual endpoint to test ESP32 connection."""
    if state not in ["on", "off"]:
        raise HTTPException(status_code=400, detail="Use 'on' or 'off'")
    
    is_on = state == "on"
    esp32.set_led(is_on)
    return {"status": "request_sent", "state": state, "target_ip": esp32.ip}

@app.get("/detect")
async def detect_objects():
    """Return the latest cached detection result."""
    result = detector.get_latest()
    if result["image"] is None:
        raise HTTPException(
            status_code=503, detail="Camera frame not available yet"
        )
    # Include ESP32 status for UI feedback
    result["esp32_errors"] = esp32.error_count
    result["esp32_last_error"] = esp32.last_error
    return JSONResponse(content=result)

@app.get("/stream")
async def stream_video():
    """Streaming endpoint using the latest detection frame."""

    async def frame_generator():
        while True:
            result = detector.get_latest()
            if result["image"] is not None:
                # Extract bytes from base64
                header, data = result["image"].split(",", 1)
                buffer = base64.b64decode(data)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer + b"\r\n"
                )
            await asyncio.sleep(0.03)

    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
