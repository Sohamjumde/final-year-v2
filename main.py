import cv2
import os
import time
import glob
import sqlite3
import json
import shutil
import subprocess
import platform
from datetime import datetime, timedelta
from ultralytics import YOLO

# =========================
# CONFIG & SETUP
# =========================

LOG_DIR = "logs"
SNAPSHOT_DIR = "outputs/snapshots"
MODEL_PATH = "yolov8n.pt"

# Base IP/host shown in your screenshot
CAMERA_HOST = "192.168.137.33:8080"

# Try a list of likely endpoints for the phone IP Webcam app.
CAMERA_ENDPOINTS = [
    f"http://{CAMERA_HOST}/video",
    f"http://{CAMERA_HOST}/stream",
    f"http://{CAMERA_HOST}/shot.jpg",
]

# Flip Mode:
# None = no flip
# 0 = vertical
# 1 = mirror (most common)
# -1 = both axes
CAMERA_FLIP_MODE = 1

# How many days to keep snapshots
SNAPSHOT_RETENTION_DAYS = 7

# Path to write the latest annotated frame (for a simple web UI)
LATEST_FRAME_PATH = "latest.jpg"

# Sound alert settings
ALERT_COOLDOWN_SEC = 5   # minimum seconds between alerts per track_id

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


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
    """Delete snapshot files older than keep_days."""
    if keep_days <= 0:
        return
    cutoff = time.time() - (keep_days * 86400)
    for f in glob.glob(os.path.join(snapshot_dir, "*")):
        try:
            if os.path.getmtime(f) < cutoff:
                os.remove(f)
        except Exception:
            pass


def open_capture(url, width=640, height=480):
    """Open a cv2 capture and set desired size. Returns VideoCapture object."""
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def safe_write_image(path, img):
    """Write image safely (avoid exceptions affecting main loop)."""
    try:
        cv2.imwrite(path, img)
    except Exception as e:
        print(f"[WARN] Failed to write image {path}: {e}")


def find_working_capture(endpoints, attempts=3, wait=0.5):
    """Try endpoints until one gives a valid frame. Returns (cap, used_endpoint) or (None, None)."""
    for ep in endpoints:
        for attempt in range(attempts):
            cap = open_capture(ep)
            time.sleep(wait)
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] Connected to camera endpoint: {ep}")
                return cap, ep
            try:
                cap.release()
            except Exception:
                pass
        print(f"[WARN] Endpoint did not respond or returned no frames: {ep}")
    return None, None


# -------------------------
# Sound alert utilities
# -------------------------
def play_alert_sound():
    """Play a short alert sound. Uses winsound on Windows, tries sox 'play' on other OSes, else bell."""
    try:
        system = platform.system()
        if system == "Windows":
            try:
                import winsound
                winsound.Beep(1000, 500)  # freq=1000Hz, duration=500ms
                return
            except Exception:
                pass

        # Try SoX 'play' (common on Linux/macOS if user installed sox)
        if shutil.which("play"):
            try:
                subprocess.Popen(["play", "-n", "synth", "0.5", "sine", "1000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass

        # Fallback to ASCII bell
        print("\a", end="", flush=True)
    except Exception as e:
        print(f"[WARN] Could not play sound: {e}")


def should_alert(track_id, last_alert_times, cooldown=ALERT_COOLDOWN_SEC):
    """Return True if we should alert for given track_id based on cooldown dictionary."""
    now = time.time()
    if track_id is None:
        # fallback: always alert if no track_id
        return True
    last = last_alert_times.get(track_id)
    if last is None or (now - last) >= cooldown:
        last_alert_times[track_id] = now
        return True
    return False


def main():
    # -------------------------
    # 1) Camera Init (try endpoints)
    # -------------------------
    cap, used_endpoint = find_working_capture(CAMERA_ENDPOINTS)
    if cap is None:
        print("[ERROR] Could not access any camera endpoints. Exiting.")
        print("Tried endpoints:")
        for e in CAMERA_ENDPOINTS:
            print("  -", e)
        return

    # wait briefly for the stream to stabilize
    time.sleep(0.5)
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[WARN] Initial read failed after connecting, attempting reconnect loop...")
        reconnect_tries = 5
        connected = False
        for _ in range(reconnect_tries):
            try:
                cap.release()
            except Exception:
                pass
            cap, used_endpoint = find_working_capture(CAMERA_ENDPOINTS, attempts=2, wait=0.5)
            if cap:
                ret, frame = cap.read()
                if ret and frame is not None:
                    connected = True
                    break
        if not connected:
            print("[ERROR] Could not access camera stream after reconnect attempts. Exiting.")
            return

    frame = normalize_frame(frame)
    frame_h, frame_w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # -------------------------
    # 2) Load YOLO + DB
    # -------------------------
    model, DOG_CLASS_ID = load_model(MODEL_PATH)
    conn, c = init_db(os.path.join(LOG_DIR, "detections.db"))

    object_tracker = {}
    snapshot_counter = 0
    last_alert_times = {}  # track_id -> last alert epoch

    print("[INFO] Press Q to exit.")
    print("[INFO] Dog detection ACTIVE (No Zones). Using endpoint:", used_endpoint)

    try:
        while True:
            ret, frame = cap.read()

            # Handle dropped frames / reconnect
            if not ret or frame is None:
                print("[WARN] Frame read failed — attempting reconnect to endpoints...")
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(1.0)
                cap, used_endpoint = find_working_capture(CAMERA_ENDPOINTS, attempts=2, wait=0.5)
                if cap is None:
                    time.sleep(2.0)
                    continue
                time.sleep(0.5)
                continue

            # Normalize once and use everywhere
            frame = normalize_frame(frame)

            # -------------------------
            # 3) YOLO Tracking
            # -------------------------
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

            if results and results[0].boxes is not None:
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame.copy()

            # Save latest annotated frame for web UI / remote viewing
            safe_write_image(LATEST_FRAME_PATH, annotated_frame)

            # -------------------------
            # 4) Handle detections (with sound alerts)
            # -------------------------
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

                        # Speed Calculation
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

                        # Ensure coordinates inside image
                        h, w = frame.shape[:2]
                        y1c, y2c = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
                        x1c, x2c = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))

                        cropped = frame[y1c:y2c, x1c:x2c]
                        if cropped is not None and cropped.size > 0:
                            safe_write_image(snap_path, cropped)
                        else:
                            safe_write_image(snap_path, annotated_frame)

                        # Log DB
                        try:
                            c.execute(
                                "INSERT INTO logs (timestamp, speed, snapshot_path) VALUES (?, ?, ?)",
                                (now_str, round(speed, 2), snap_path)
                            )
                            conn.commit()
                        except Exception as e:
                            print(f"[WARN] Failed to log to DB: {e}")

                        print(f"[INFO] Dog detected | Speed={round(speed,2)} | Snapshot saved: {snap_path}")

                        # Play alert sound (debounced per track)
                        if should_alert(track_id_val, last_alert_times, cooldown=ALERT_COOLDOWN_SEC):
                            play_alert_sound()

            # -------------------------
            # 5) Display
            # -------------------------
            cv2.imshow("Dog Detection (No Zones)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Periodically cleanup old snapshots
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
