#!/usr/bin/env python3
"""
detect_tiger_lion.py

Detect (attempt) tiger & lion in a camera stream. Uses ultralytics YOLO model (default yolov8n.pt).
NOTE: The default COCO weights do not include 'tiger' or 'lion'. This script will:
 - map "tiger"/"lion" -> proxy "cat" if available in the loaded model (best-effort, not reliable),
 - warn you and fall back to detecting ALL classes if proxies aren't available.
To reliably detect tigers/lions you must train or supply a model that has those classes.
"""

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
# CONFIG & SETUP (MULTI-CLASS + LOW-LATENCY)
# =========================

LOG_DIR = "logs"
SNAPSHOT_DIR = "outputs/snapshots"

# Change this to your custom weights if you train a tiger/lion detector.
MODEL_PATH = "yolov8n.pt"

# Request detection of tiger and lion
DETECT_CLASS_NAMES = ["tiger", "lion"]

# Camera host (phone IP webcam)
CAMERA_HOST = "192.168.137.33:8080"

# Use the uploaded local file as a fallback (developer-provided path)
UPLOADED_FILE = "/mnt/data/abc4ff0d-30e9-47b0-989f-74029eaa4288.png"

CAMERA_ENDPOINTS = [
    f"http://{CAMERA_HOST}/video",
    f"http://{CAMERA_HOST}/stream",
    f"http://{CAMERA_HOST}/shot.jpg",
    f"file://{UPLOADED_FILE}",  # local fallback for testing
]

# Capture size (reduce for faster processing)
CAP_WIDTH = 640
CAP_HEIGHT = 360
CAMERA_FLIP_MODE = 1  # 1 = mirror; set None to disable

SNAPSHOT_RETENTION_DAYS = 7
LATEST_FRAME_PATH = "latest.jpg"

ALERT_COOLDOWN_SEC = 5   # seconds between alerts per tracked object
USE_THREADED_CAPTURE = True
LATEST_FRAME_SAVE_MIN_INTERVAL = 1.0  # seconds

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# -------------------------
# Threaded VideoCapture (keeps only latest frame)
# -------------------------
class VideoCaptureAsync:
    """Background thread continuously grabbing frames and storing only the latest frame."""
    def __init__(self, src, width=CAP_WIDTH, height=CAP_HEIGHT, backend_preference=None):
        if backend_preference is not None:
            try:
                self.cap = cv2.VideoCapture(src, backend_preference)
            except Exception:
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src)

        try:
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
        while self.running:
            try:
                grabbed = self.cap.grab()
                if not grabbed:
                    self.last_read_ok = False
                    time.sleep(0.01)
                    continue
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.last_read_ok = True
                else:
                    self.last_read_ok = False
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
# Model loading (with fallback mapping for tiger/lion)
# -------------------------
def load_model_and_class_map(model_path, requested_class_names):
    """
    Load YOLO model and build a mapping {requested_name: class_id}. If requested names are not present,
    attempt a proxy mapping (e.g., tiger->cat). If none of the requested classes can be mapped,
    return empty class_map which signals "detect all classes".
    """
    model = YOLO(model_path)
    id_to_name = model.names  # dict: id -> name
    name_to_id = {name.lower(): idx for idx, name in id_to_name.items()}

    # Proxy suggestions: approximate substitutes when exact label missing.
    # WARNING: proxies are approximate and not reliable for real tiger/lion detection.
    proxy_map = {
        "tiger": ["cat"],
        "lion": ["cat"],
    }

    class_map = {}
    if requested_class_names:
        for cname in requested_class_names:
            key = cname.lower()
            if key in name_to_id:
                class_map[cname] = name_to_id[key]
            else:
                # try proxies
                proxies = proxy_map.get(key, [])
                found = False
                for p in proxies:
                    if p in name_to_id:
                        print(f"[WARN] '{cname}' not present in model.names — using proxy '{p}'.")
                        class_map[cname] = name_to_id[p]
                        found = True
                        break
                if not found:
                    print(f"[WARN] '{cname}' not found in model.names and no proxy available; skipping '{cname}'.")
    else:
        class_map = {}

    if class_map:
        print(f"[INFO] Model loaded. Mapped requested classes: {class_map}")
    else:
        print("[INFO] Model loaded. No requested classes mapped — falling back to detecting ALL classes.")

    return model, class_map


# -------------------------
# DB (stores class info + bbox)
# -------------------------
def init_db(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TEXT,
            class_name TEXT,
            class_id INTEGER,
            track_id INTEGER,
            speed REAL,
            bbox TEXT,
            snapshot_path TEXT
        )
    """)
    conn.commit()
    return conn, c


# -------------------------
# Utilities
# -------------------------
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


# -------------------------
# Sound player (async) - best-effort
# -------------------------
def _play_sound_worker(path=None):
    try:
        if platform.system() == "Windows":
            try:
                import winsound
                winsound.Beep(2000, 700)
                return
            except Exception:
                pass
        if shutil.which("play"):
            subprocess.Popen(["play", "-n", "synth", "0.5", "sine", "1500"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        if shutil.which("paplay"):
            # try paplay
            subprocess.Popen(["paplay", path] if path and os.path.exists(path) else ["paplay"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    except Exception:
        pass
    # Fallback ASCII bell
    print("\a", end="", flush=True)


def play_alert_sound_async():
    threading.Thread(target=_play_sound_worker, daemon=True).start()


def should_alert(key, last_alert_times, cooldown=ALERT_COOLDOWN_SEC):
    """
    key: combined key (track_id, class_id) or None
    """
    now = time.time()
    if key is None:
        return True
    last = last_alert_times.get(key)
    if last is None or (now - last) >= cooldown:
        last_alert_times[key] = now
        return True
    return False


# -------------------------
# Capture utilities
# -------------------------
def open_low_latency_capture(endpoint):
    backend = None
    try:
        backend = cv2.CAP_FFMPEG
    except Exception:
        backend = None

    if USE_THREADED_CAPTURE:
        cap = VideoCaptureAsync(endpoint, width=CAP_WIDTH, height=CAP_HEIGHT, backend_preference=backend)
        cap.start()
        return cap
    else:
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
    for ep in endpoints:
        try:
            cap = open_low_latency_capture(ep)
            t0 = time.time()
            timeout = 3.0
            while time.time() - t0 < timeout:
                time.sleep(0.05)
                ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"[INFO] Connected low-latency to: {ep}")
                    return cap, ep
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            pass
        print(f"[WARN] Endpoint did not respond or returned no frames quickly: {ep}")
    return None, None


# -------------------------
# MAIN
# -------------------------
def main():
    cap, used_endpoint = find_working_capture(CAMERA_ENDPOINTS)
    if cap is None:
        print("[ERROR] Could not access any camera endpoints. Exiting.")
        print("Tried endpoints:")
        for e in CAMERA_ENDPOINTS:
            print("  -", e)
        return

    time.sleep(0.3)
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[WARN] Initial read failed — continuing and letting reconnect logic handle it.")

    frame = normalize_frame(frame) if frame is not None else None
    fps = 30

    # Load model and class mapping based on DETECT_CLASS_NAMES
    model, class_map = load_model_and_class_map(MODEL_PATH, DETECT_CLASS_NAMES)

    # Prepare numeric class IDs for model.track: if class_map empty -> None (detect all)
    if class_map:
        classes_list = list(class_map.values())
    else:
        classes_list = None

    conn, c = init_db(os.path.join(LOG_DIR, "detections.db"))

    object_tracker = {}  # keys: (track_id, class_id) -> (cx, cy, datetime)
    snapshot_counter = 0
    last_alert_times = {}
    last_latest_save = 0.0

    print("[INFO] Multi-class detection ACTIVE. Using:", used_endpoint)
    if classes_list:
        print("[INFO] Detecting mapped class IDs:", classes_list)
    else:
        print("[INFO] Detecting ALL classes from model (DETECT_CLASS_NAMES unmapped).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame = normalize_frame(frame)

            # Run detection (pass classes_list or None)
            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=classes_list,
                    conf=0.35,
                    verbose=False,
                )
            except Exception as e:
                print(f"[WARN] model.track() error: {e}")
                continue

            annotated_frame = results[0].plot() if (results and results[0].boxes is not None) else frame.copy()

            now_ts = time.time()
            if now_ts - last_latest_save >= LATEST_FRAME_SAVE_MIN_INTERVAL:
                safe_write_image(LATEST_FRAME_PATH, annotated_frame)
                last_latest_save = now_ts

            # Handle detections (multiple)
            if results and results[0].boxes is not None:
                for r in results:
                    if getattr(r.boxes, "id", None) is None:
                        continue

                    for box, cls_id_raw, track_id in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.id):
                        try:
                            # cls_id may be tensor-like; convert to int
                            cls_id = int(cls_id_raw) if not hasattr(cls_id_raw, "item") else int(cls_id_raw.item())
                        except Exception:
                            cls_id = int(cls_id_raw)

                        # If user requested specific classes and this class isn't one of the requested mapped ids -> skip
                        if class_map and cls_id not in list(class_map.values()):
                            continue

                        cls_name = model.names.get(cls_id, str(cls_id))
                        try:
                            track_id_val = int(track_id.item())
                        except Exception:
                            try:
                                track_id_val = int(track_id)
                            except Exception:
                                track_id_val = None

                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        now = datetime.now()
                        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

                        # Speed calculation keyed by (track_id, class_id)
                        speed = 0.0
                        tracker_key = (track_id_val, cls_id) if track_id_val is not None else None
                        if tracker_key is not None and tracker_key in object_tracker:
                            prev_cx, prev_cy, prev_time = object_tracker[tracker_key]
                            dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                            elapsed = (now - prev_time).total_seconds() or 1e-6
                            speed = (dist / elapsed) * (fps / 100.0)

                        if tracker_key is not None:
                            object_tracker[tracker_key] = (cx, cy, now)

                        # Snapshot naming includes class and track
                        snapshot_counter += 1
                        tid = track_id_val if track_id_val is not None else "na"
                        safe_class_name = cls_name.replace(" ", "_")
                        snap_filename = f"{safe_class_name}_{tid}_{snapshot_counter}.jpg"
                        snap_path = os.path.join(SNAPSHOT_DIR, snap_filename)

                        h, w = frame.shape[:2]
                        y1c, y2c = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
                        x1c, x2c = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
                        cropped = frame[y1c:y2c, x1c:x2c]
                        if cropped is not None and cropped.size > 0:
                            safe_write_image(snap_path, cropped)
                        else:
                            safe_write_image(snap_path, annotated_frame)

                        bbox_str = f"{x1},{y1},{x2},{y2}"
                        try:
                            c.execute(
                                "INSERT INTO logs (timestamp, class_name, class_id, track_id, speed, bbox, snapshot_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (now_str, cls_name, cls_id, track_id_val, round(speed, 2), bbox_str, snap_path)
                            )
                            conn.commit()
                        except Exception as e:
                            print(f"[WARN] Failed to log to DB: {e}")

                        print(f"[INFO] Detected {cls_name} (id={cls_id}) | track={track_id_val} | Speed={round(speed,2)} | Snapshot: {snap_path}")

                        # Alert per (track_id, class_id)
                        alert_key = tracker_key
                        if should_alert(alert_key, last_alert_times, cooldown=ALERT_COOLDOWN_SEC):
                            play_alert_sound_async()

            # Display quickly
            cv2.imshow("Tiger/Lion Detection (Low Latency)", annotated_frame)
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
