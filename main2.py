import cv2
import os
import time
import glob
import sqlite3
import shutil
import subprocess
import platform
import threading
from datetime import datetime
from ultralytics import YOLO

# =========================
# CONFIG & SETUP (LOW-LATENCY)
# =========================

LOG_DIR = "logs"
SNAPSHOT_DIR = "outputs/snapshots"
MODEL_PATH = "yolov8n.pt"

# Camera host
CAMERA_HOST = "192.168.137.33:8080"

# Add demonstration fallback to the uploaded file (for testing).
# Developer note: using the local uploaded file path as fallback URL.
UPLOADED_FILE = "/mnt/data/abc4ff0d-30e9-47b0-989f-74029eaa4288.png"

CAMERA_ENDPOINTS = [
    f"http://{CAMERA_HOST}/video",     # typical MJPEG
    f"http://{CAMERA_HOST}/stream",
    f"http://{CAMERA_HOST}/shot.jpg",
    f"file://{UPLOADED_FILE}",         # fallback/test
]

# Lower resolution -> less bandwidth & faster decode. Set to 640x360 by default.
CAP_WIDTH = 640
CAP_HEIGHT = 360

# Flip Mode:
CAMERA_FLIP_MODE = 1

SNAPSHOT_RETENTION_DAYS = 7
LATEST_FRAME_PATH = "latest.jpg"

# Sound alert settings
ALERT_COOLDOWN_SEC = 5

# Threaded capture toggle (recommended True for lowest latency)
USE_THREADED_CAPTURE = True

# Limit how often we write latest.jpg (avoid disk IO stalls)
LATEST_FRAME_SAVE_MIN_INTERVAL = 1.0  # seconds

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# -------------------------
# Threaded VideoCapture (keeps only latest frame)
# -------------------------
class VideoCaptureAsync:
    """Background thread continuously grabbing frames and storing only the latest frame."""
    def __init__(self, src, width=CAP_WIDTH, height=CAP_HEIGHT, backend_preference=None):
        # If backend_preference supplied, pass it to cv2.VideoCapture
        if backend_preference is not None:
            try:
                self.cap = cv2.VideoCapture(src, backend_preference)
            except Exception:
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src)

        # Try to set a minimal buffer and desired size
        try:
            # very important: small buffer
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_read_ok = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        # Prefer grab/retrieve to minimize blocking overhead
        while self.running:
            try:
                # use grab() to quickly drop frames and only decode retrieve() for the final one
                grabbed = self.cap.grab()
                if not grabbed:
                    self.last_read_ok = False
                    time.sleep(0.01)
                    continue
                ret, frame = self.cap.read()  # use read to get decoded frame; some backends need this
                # Some backends need retrieve(); keeping read() for compatibility
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.last_read_ok = True
                else:
                    self.last_read_ok = False
                # tiny sleep to yield cpu; keep small to reduce latency
                time.sleep(0.001)
            except Exception:
                self.last_read_ok = False
                time.sleep(0.05)

    def read(self):
        with self.lock:
            f = self.latest_frame.copy() if (self.latest_frame is not None) else (None)
        return self.last_read_ok, f

    def release(self):
        self.running = False
        try:
            if self.thread:
                self.thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass


# -------------------------
# Utility + Model + DB (unchanged)
# -------------------------
def load_model(model_path, target_class_name="dog"):
    model = YOLO(model_path)
    class_id = None
    for idx, name in model.names.items():
        if name == target_class_name:
            class_id = idx
            break
    if class_id is None:
        raise ValueError(f"'{target_class_name}' not found in model.names: {model.names}")
    print(f"[INFO] Using model '{model_path}' with '{target_class_name}' class id = {class_id}")
    return model, class_id


def init_db(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TEXT,
            speed REAL,
            snapshot_path TEXT
        )
    """)
    conn.commit()
    return conn, c


def normalize_frame(frame):
    if CAMERA_FLIP_MODE is None:
        return frame
    return cv2.flip(frame, CAMERA_FLIP_MODE)


def cleanup_old_snapshots(snapshot_dir: str, keep_days: int):
    if keep_days <= 0:
        return
    cutoff = time.time() - (keep_days * 86400)
    for f in glob.glob(os.path.join(snapshot_dir, "*")):
        try:
            if os.path.getmtime(f) < cutoff:
                os.remove(f)
        except Exception:
            pass


def safe_write_image(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as e:
        print(f"[WARN] Failed to write image {path}: {e}")


# Sound player (asynchronous) - keep as in your prior script (call play_alert_sound_async)
import threading as _threading
def _play_sound_worker(path=None):
    # small beep fallback to avoid blocking
    try:
        if platform.system() == "Windows":
            try:
                import winsound
                winsound.Beep(2000, 700)
                return
            except Exception:
                pass
        if shutil.which("play"):
            subprocess.Popen(["play", "-n", "synth", "0.5", "sine", "1500"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    except Exception:
        pass
    print("\a", end="", flush=True)

def play_alert_sound_async():
    _threading.Thread(target=_play_sound_worker, daemon=True).start()


def should_alert(track_id, last_alert_times, cooldown=ALERT_COOLDOWN_SEC):
    now = time.time()
    if track_id is None:
        return True
    last = last_alert_times.get(track_id)
    if last is None or (now - last) >= cooldown:
        last_alert_times[track_id] = now
        return True
    return False


# -------------------------
# Capture selection (try endpoints, using threaded capture for low latency)
# -------------------------
def open_low_latency_capture(endpoint):
    """
    Try to open capture using FFMPEG backend if available for lower latency.
    Returns VideoCaptureAsync instance (if USE_THREADED_CAPTURE True) or cv2.VideoCapture-like object.
    """
    # Try cv2.CAP_FFMPEG constant if present, else None
    backend = None
    try:
        backend = cv2.CAP_FFMPEG
    except Exception:
        backend = None

    # Create threaded capture for lowest latency
    if USE_THREADED_CAPTURE:
        cap = VideoCaptureAsync(endpoint, width=CAP_WIDTH, height=CAP_HEIGHT, backend_preference=backend)
        cap.start()
        return cap
    else:
        # fallback to normal capture with small buffer
        if backend is not None:
            cap = cv2.VideoCapture(endpoint, backend)
        else:
            cap = cv2.VideoCapture(endpoint)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        return cap


def find_working_capture(endpoints, attempts=2, wait=0.3):
    """Try endpoints until one returns a frame quickly."""
    for ep in endpoints:
        try:
            cap = open_low_latency_capture(ep)
            # quick warm-up: wait small time then try to read latest frame
            t0 = time.time()
            timeout = 3.0
            while time.time() - t0 < timeout:
                time.sleep(0.05)
                if USE_THREADED_CAPTURE:
                    ok, frame = cap.read()
                else:
                    ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"[INFO] Connected low-latency to: {ep}")
                    return cap, ep
            # if not working, release and try next
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            pass
        print(f"[WARN] Endpoint did not respond or returned no frames quickly: {ep}")
    return None, None


# -------------------------
# Main loop (mostly identical but using threaded capture & rate-limited latest.jpg writes)
# -------------------------
def main():
    cap, used_endpoint = find_working_capture(CAMERA_ENDPOINTS)
    if cap is None:
        print("[ERROR] Could not access any camera endpoints. Exiting.")
        print("Tried endpoints:")
        for e in CAMERA_ENDPOINTS:
            print("  -", e)
        return

    # small stabilization
    time.sleep(0.3)

    # Get an initial frame
    if USE_THREADED_CAPTURE:
        ret, frame = cap.read()
    else:
        ret, frame = cap.read()

    if not ret or frame is None:
        print("[WARN] Initial read failed — continuing and letting reconnect logic handle it.")

    frame = normalize_frame(frame) if frame is not None else None
    fps = 30

    # Load model & DB
    model, DOG_CLASS_ID = load_model(MODEL_PATH)
    conn, c = init_db(os.path.join(LOG_DIR, "detections.db"))

    object_tracker = {}
    snapshot_counter = 0
    last_alert_times = {}
    last_latest_save = 0.0

    print("[INFO] Dog detection ACTIVE (low-latency). Using:", used_endpoint)
    try:
        while True:
            # Read latest frame (non-blocking thanks to threaded capture)
            if USE_THREADED_CAPTURE:
                ret, frame = cap.read()
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                # quick reconnect attempt
                time.sleep(0.05)
                continue

            frame = normalize_frame(frame)

            # Run detection
            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[DOG_CLASS_ID],
                    conf=0.35,
                    verbose=False,
                )
            except Exception as e:
                print(f"[WARN] model.track() error: {e}")
                continue

            annotated_frame = results[0].plot() if (results and results[0].boxes is not None) else frame.copy()

            # Rate-limit writing latest.jpg (avoid disk IO causing jitter)
            now_ts = time.time()
            if now_ts - last_latest_save >= LATEST_FRAME_SAVE_MIN_INTERVAL:
                safe_write_image(LATEST_FRAME_PATH, annotated_frame)
                last_latest_save = now_ts

            # Handle detections (no change except using play_alert_sound_async)
            if results and results[0].boxes is not None:
                for r in results:
                    if getattr(r.boxes, "id", None) is None:
                        continue
                    for box, cls_id, track_id in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.id):
                        if int(cls_id) != DOG_CLASS_ID:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        try:
                            track_id_val = int(track_id.item())
                        except Exception:
                            try:
                                track_id_val = int(track_id)
                            except Exception:
                                track_id_val = None
                        now = datetime.now()
                        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                        # speed calc (same)
                        speed = 0.0
                        if track_id_val is not None and track_id_val in object_tracker:
                            prev_cx, prev_cy, prev_time = object_tracker[track_id_val]
                            dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
                            elapsed = (now - prev_time).total_seconds() or 1e-6
                            speed = (dist / elapsed) * (fps / 100.0)
                        if track_id_val is not None:
                            object_tracker[track_id_val] = (cx, cy, now)
                        # Snapshot
                        snapshot_counter += 1
                        snap_filename = f"dog_{snapshot_counter}.jpg"
                        snap_path = os.path.join(SNAPSHOT_DIR, snap_filename)
                        h, w = frame.shape[:2]
                        y1c, y2c = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
                        x1c, x2c = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
                        cropped = frame[y1c:y2c, x1c:x2c]
                        if cropped is not None and cropped.size > 0:
                            safe_write_image(snap_path, cropped)
                        else:
                            safe_write_image(snap_path, annotated_frame)
                        try:
                            c.execute(
                                "INSERT INTO logs (timestamp, speed, snapshot_path) VALUES (?, ?, ?)",
                                (now_str, round(speed, 2), snap_path)
                            )
                            conn.commit()
                        except Exception as e:
                            print(f"[WARN] Failed to log to DB: {e}")
                        print(f"[INFO] Dog detected | Speed={round(speed,2)} | Snapshot saved: {snap_path}")
                        if should_alert(track_id_val, last_alert_times, cooldown=ALERT_COOLDOWN_SEC):
                            play_alert_sound_async()

            # Display quickly; keep a very small wait to avoid blocking too long
            cv2.imshow("Dog Detection (Low Latency)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            cleanup_old_snapshots(SNAPSHOT_DIR, SNAPSHOT_RETENTION_DAYS)

    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            conn.close()
        except Exception:
            pass
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()
