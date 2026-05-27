"""
=============================================================================
  INDUSTRY-LEVEL AI FIRE & SMOKE DETECTION SYSTEM
  Built with YOLOv8 + OpenCV + PyTorch
  Author  : AI Computer Vision Engineer
  Version : 2.0
  Features: Real-time detection | Alarms | SMS | Email | Logging | Screenshots
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  STANDARD LIBRARY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import time
import logging
import smtplib
import threading
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text     import MIMEText
from email.mime.base     import MIMEBase
from email              import encoders
from pathlib            import Path

# ─────────────────────────────────────────────────────────────────────────────
#  THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import pyttsx3

# Optional – pygame for alarm audio
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not installed – alarm sound disabled. "
          "Install with: pip install pygame")

# Optional – twilio for SMS
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("[WARNING] twilio not installed – SMS alerts disabled. "
          "Install with: pip install twilio")


# =============================================================================
#  CONFIGURATION  –  Edit this section before running
# =============================================================================
class Config:
    # ── Model ────────────────────────────────────────────────────────────────
    MODEL_PATH      = r"D:\Downloads\Fire -smoke Detection.v1i.yolov8\runs\detect\train4\weights\best.pt"          # Path to your custom YOLOv8 model
    CONF_THRESHOLD  = 0.45               # Minimum confidence to trigger alert
    IOU_THRESHOLD   = 0.50               # NMS IoU threshold
    IMG_SIZE        = 640                # Inference image size

    # ── Video source  (uncomment ONE) ────────────────────────────────────────
    #VIDEO_SOURCE = 0                                         # Webcam
    VIDEO_SOURCE = r"D:\Downloads\Fire -smoke Detection.v1i.yolov8\videoplayback.mp4"                                # Video file
    #VIDEO_SOURCE = "rtsp://user:pass@192.168.1.100:554/stream"  # RTSP/IP cam

    # ── Directories ──────────────────────────────────────────────────────────
    SCREENSHOT_DIR  = "screenshots"
    LOG_DIR         = "logs"
    LOG_FILE        = "logs/detection_log.txt"
    ALARM_AUDIO     = r"D:\Downloads\Fire -smoke Detection.v1i.yolov8\discrete.mp3"        # Place alarm.mp3 in working directory

    # ── Alert cooldowns (seconds) ─────────────────────────────────────────────
    SMS_COOLDOWN    = 60                 # Minimum gap between SMS alerts
    EMAIL_COOLDOWN  = 60                 # Minimum gap between email alerts
    SCREENSHOT_COOLDOWN = 5             # Minimum gap between screenshots

    # ── Twilio SMS ────────────────────────────────────────────────────────────
    TWILIO_SID      = "YOUR_TWILIO_ACCOUNT_SID"
    TWILIO_TOKEN    = "YOUR_TWILIO_AUTH_TOKEN"
    TWILIO_FROM     = "+1XXXXXXXXXX"     # Your Twilio phone number
    TWILIO_TO       = "+91XXXXXXXXXX"    # Recipient phone number

    # ── Email (SMTP) ──────────────────────────────────────────────────────────
    EMAIL_SENDER    = "r3629296@gmail.com"
    EMAIL_PASSWORD  = "ftwa qeeu rpzh baey"  # Use App Password for Gmail
    EMAIL_RECEIVER  = "rrahul.aiml2024@citchennai.net"
    SMTP_HOST       = "smtp.gmail.com"
    SMTP_PORT       = 587

    # ── Display ───────────────────────────────────────────────────────────────
    WINDOW_NAME     = "🔥 Fire & Smoke Detection System"
    DISPLAY_WIDTH   = 1280
    DISPLAY_HEIGHT  = 720

    # ── Class colours  (BGR) ─────────────────────────────────────────────────
    CLASS_COLORS = {
        "fire":  (0,   50,  255),   # Red-orange
        "smoke": (180, 180, 180),   # Grey
    }
    DEFAULT_COLOR = (0, 255, 255)   # Cyan for unknown classes


# =============================================================================
#  LOGGING SETUP
# =============================================================================
def setup_logging() -> logging.Logger:
    """Initialise Python logger to both console and file."""
    Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("FireSmokeDetection")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logging()

# =============================================================================
#  VOICE ALERT MANAGER
# =============================================================================
class VoiceAlertManager:

    def __init__(self):

        self.engine = pyttsx3.init()

        # Voice speed
        self.engine.setProperty('rate', 150)

        # Volume
        self.engine.setProperty('volume', 1.0)

        self.last_voice_time = 0

        # Cooldown seconds
        self.voice_cooldown = 15

    def speak_alert(self, label):

        current_time = time.time()

        # Prevent repeated speaking
        if current_time - self.last_voice_time < self.voice_cooldown:
            return

        self.last_voice_time = current_time

        try:

            message = f"Emergency Alert. {label} detected. Please take immediate action."

            logger.info("🔊 Voice alert triggered.")

            self.engine.say(message)

            self.engine.runAndWait()

        except Exception as exc:

            logger.error("Voice alert error: %s", exc)


# =============================================================================
#  ALARM MANAGER
# =============================================================================
class AlarmManager:
    """
    Plays alarm.mp3 in a background thread while detections are active.
    Thread-safe start / stop.
    """

    def __init__(self):
        self._playing  = False
        self._lock     = threading.Lock()
        self._thread   = None

        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                if os.path.exists(Config.ALARM_AUDIO):
                    pygame.mixer.music.load(Config.ALARM_AUDIO)
                    logger.info("Alarm audio loaded: %s", Config.ALARM_AUDIO)
                else:
                    logger.warning("Alarm file not found: %s", Config.ALARM_AUDIO)
            except Exception as exc:
                logger.error("pygame init error: %s", exc)

    def start(self):
        """Start playing alarm (idempotent)."""
        with self._lock:
            if self._playing:
                return
            self._playing = True

        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.play(loops=-1)   # loop indefinitely
                logger.info("🔔 Alarm STARTED")
            except Exception as exc:
                logger.error("Alarm start error: %s", exc)

    def stop(self):
        """Stop alarm (idempotent)."""
        with self._lock:
            if not self._playing:
                return
            self._playing = False

        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
                logger.info("🔕 Alarm STOPPED")
            except Exception as exc:
                logger.error("Alarm stop error: %s", exc)

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing


# =============================================================================
#  ALERT MANAGER  (SMS + Email)
# =============================================================================
class AlertManager:
    """
    Handles Twilio SMS and SMTP email alerts with cooldown guards so
    the system doesn't spam alerts every frame.
    """

    def __init__(self):
        self._last_sms   : float = 0.0
        self._last_email : float = 0.0
        self._sms_lock   = threading.Lock()
        self._email_lock = threading.Lock()

        # Twilio client
        if TWILIO_AVAILABLE:
            try:
                self._twilio = TwilioClient(
                    Config.TWILIO_SID, Config.TWILIO_TOKEN
                )
                logger.info("Twilio client initialised.")
            except Exception as exc:
                self._twilio = None
                logger.error("Twilio init error: %s", exc)
        else:
            self._twilio = None

    # ── SMS ──────────────────────────────────────────────────────────────────
    def send_sms(self, label: str, confidence: float, timestamp: str):
        """Send SMS alert (threaded, with cooldown)."""
        with self._sms_lock:
            if time.time() - self._last_sms < Config.SMS_COOLDOWN:
                return
            self._last_sms = time.time()

        thread = threading.Thread(
            target=self._sms_worker,
            args=(label, confidence, timestamp),
            daemon=True
        )
        thread.start()

    def _sms_worker(self, label: str, confidence: float, timestamp: str):
        if not self._twilio:
            logger.warning("SMS skipped – Twilio not configured.")
            return
        try:
            body = (
                f"🚨 FIRE ALERT!\n"
                f"Detection : {label.upper()}\n"
                f"Confidence: {confidence:.1%}\n"
                f"Time      : {timestamp}\n"
                f"Source    : {Config.VIDEO_SOURCE}"
            )
            self._twilio.messages.create(
                body=body,
                from_=Config.TWILIO_FROM,
                to=Config.TWILIO_TO
            )
            logger.info("📱 SMS alert sent for %s (%.1f%%)", label, confidence * 100)
        except Exception as exc:
            logger.error("SMS send error: %s", exc)

    # ── Email ─────────────────────────────────────────────────────────────────
    def send_email(self, label: str, confidence: float,
                   timestamp: str, screenshot_path: str | None = None):
        """Send email alert (threaded, with cooldown)."""
        with self._email_lock:
            if time.time() - self._last_email < Config.EMAIL_COOLDOWN:
                return
            self._last_email = time.time()

        thread = threading.Thread(
            target=self._email_worker,
            args=(label, confidence, timestamp, screenshot_path),
            daemon=True
        )
        thread.start()

    def _email_worker(self, label: str, confidence: float,
                      timestamp: str, screenshot_path: str | None):
        try:
            msg = MIMEMultipart()
            msg["From"]    = Config.EMAIL_SENDER
            msg["To"]      = Config.EMAIL_RECEIVER
            msg["Subject"] = f"🚨 FIRE/SMOKE ALERT – {label.upper()} Detected!"

            body = (
                f"<h2 style='color:red'>🔥 Emergency Fire/Smoke Alert</h2>"
                f"<table>"
                f"<tr><td><b>Detection</b></td><td>{label.upper()}</td></tr>"
                f"<tr><td><b>Confidence</b></td><td>{confidence:.1%}</td></tr>"
                f"<tr><td><b>Timestamp</b></td><td>{timestamp}</td></tr>"
                f"<tr><td><b>Source</b></td><td>{Config.VIDEO_SOURCE}</td></tr>"
                f"</table>"
                f"<p>Please take immediate action.</p>"
            )
            msg.attach(MIMEText(body, "html"))

            # Attach screenshot if available
            if screenshot_path and os.path.exists(screenshot_path):
                with open(screenshot_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{os.path.basename(screenshot_path)}"'
                )
                msg.attach(part)

            with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT) as server:
                server.starttls()
                server.login(Config.EMAIL_SENDER, Config.EMAIL_PASSWORD)
                server.sendmail(
                    Config.EMAIL_SENDER,
                    Config.EMAIL_RECEIVER,
                    msg.as_string()
                )
            logger.info("📧 Email alert sent for %s (%.1f%%)", label, confidence * 100)
        except Exception as exc:
            logger.error("Email send error: %s", exc)


# =============================================================================
#  SCREENSHOT MANAGER
# =============================================================================
class ScreenshotManager:
    """Saves annotated frames with cooldown guard."""

    def __init__(self):
        Path(Config.SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)
        self._last_save: float = 0.0
        self._lock = threading.Lock()

    def save(self, frame: np.ndarray, label: str) -> str | None:
        """Save frame; returns saved path or None if on cooldown."""
        with self._lock:
            if time.time() - self._last_save < Config.SCREENSHOT_COOLDOWN:
                return None
            self._last_save = time.time()

        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(Config.SCREENSHOT_DIR, f"{label}_{ts}.jpg")
        try:
            cv2.imwrite(path, frame)
            logger.info("📸 Screenshot saved: %s", path)
            return path
        except Exception as exc:
            logger.error("Screenshot save error: %s", exc)
            return None


# =============================================================================
#  DETECTION LOGGER  (file-based)
# =============================================================================
class DetectionLogger:
    """Appends structured detection records to detection_log.txt."""

    def __init__(self):
        Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)

    def log(self, label: str, confidence: float,
            source: str, bbox: tuple):
        """Write one detection record."""
        ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = (
            f"[{ts}] "
            f"TYPE={label.upper():<10} "
            f"CONF={confidence:.4f} "
            f"SOURCE={source} "
            f"BBOX={bbox}\n"
        )
        try:
            with open(Config.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(row)
        except Exception as exc:
            logger.error("Log write error: %s", exc)


# =============================================================================
#  FPS COUNTER
# =============================================================================
class FPSCounter:
    """Exponential moving-average FPS counter."""

    def __init__(self, alpha: float = 0.1):
        self._alpha   = alpha
        self._fps     : float = 0.0
        self._prev_t  : float = time.time()

    def tick(self) -> float:
        now        = time.time()
        inst_fps   = 1.0 / max(now - self._prev_t, 1e-6)
        self._fps  = self._alpha * inst_fps + (1 - self._alpha) * self._fps
        self._prev_t = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# =============================================================================
#  DRAWING UTILITIES
# =============================================================================
def draw_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              label: str, confidence: float, color: tuple) -> None:
    """Draw bounding box, label background, and confidence text."""
    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    text    = f"{label.upper()} {confidence:.0%}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = max(y1 - 10, th + 5)
    cv2.rectangle(
        frame,
        (x1, label_y - th - baseline - 4),
        (x1 + tw + 4, label_y + baseline - 2),
        color, cv2.FILLED
    )

    # Text
    cv2.putText(
        frame, text,
        (x1 + 2, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2, cv2.LINE_AA
    )


def draw_overlay(frame: np.ndarray, fps: float,
                 alert_active: bool, source: str) -> None:
    """Draw HUD overlay: FPS, source, alert banner."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 100), 2, cv2.LINE_AA)

    src_text = f"SRC: {str(source)[:35]}"
    cv2.putText(frame, src_text,
                (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)

    # Alert banner
    if alert_active:
        cv2.rectangle(frame, (0, h - 48), (w, h), (0, 0, 200), cv2.FILLED)
        cv2.putText(frame, "⚠  FIRE / SMOKE DETECTED  ⚠",
                    (w // 2 - 220, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)


# =============================================================================
#  MODEL LOADER
# =============================================================================
def load_model(model_path: str, device: torch.device) -> YOLO:
    """Load YOLOv8 model and move to target device."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Place best.pt in the working directory."
        )
    logger.info("Loading YOLOv8 model: %s  →  device: %s", model_path, device)
    model = YOLO(model_path)
    model.to(device)
    logger.info("Model loaded. Classes: %s", model.names)
    return model


# =============================================================================
#  VIDEO SOURCE FACTORY
# =============================================================================
def open_video_source(source) -> cv2.VideoCapture:
    """
    Opens a video source (webcam int, file path str, or RTSP URL str).
    Applies backend-specific optimisations where possible.
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise IOError(
            f"Cannot open video source: {source}\n"
            "Check the path / URL / device index."
        )

    # For files / webcam, push preferred resolution
    if isinstance(source, int) or (
        isinstance(source, str) and not source.startswith("rtsp")
    ):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.DISPLAY_HEIGHT)

    # Reduce internal buffer to minimise latency on streams
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    logger.info(
        "Video source opened: %s  [%dx%d @ %.0f fps]",
        source,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cap.get(cv2.CAP_PROP_FPS)
    )
    return cap
def trigger_phone_call():

    try:

        trigger_path = r"C:\Users\R RAHUL\Downloads\call_trigger.txt"

        with open(trigger_path, "w") as f:
            f.write(str(time.time()))

        logger.info("📞 Phone call trigger created.")

    except Exception as exc:

        logger.error("Call trigger error: %s", exc)

# =============================================================================
#  MAIN DETECTION LOOP
# =============================================================================
def run_detection():
    """
    Core pipeline:
      read frame  →  YOLOv8 inference  →  parse detections
      →  draw overlays  →  trigger alarms / alerts / logs  →  display
    """

    # ── Device selection ─────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("🖥  GPU detected: %s – using CUDA acceleration.", gpu_name)
        # Optimise for Ampere / Ada GPUs (RTX 30/40 series)
        torch.backends.cudnn.benchmark = True
    else:
        logger.info("🖥  No GPU found – running on CPU.")

    # ── Subsystem init ───────────────────────────────────────────────────────
    model      = load_model(Config.MODEL_PATH, device)
    alarm      = AlarmManager()
    alerter    = AlertManager()
    voice_alert = VoiceAlertManager()
    screenshotter = ScreenshotManager()
    det_logger = DetectionLogger()
    fps_counter = FPSCounter()

    cap = open_video_source(Config.VIDEO_SOURCE)

    logger.info("Starting detection loop. Press  Q  to quit.")

    # ── State ────────────────────────────────────────────────────────────────
    alert_active = False
    no_det_since : float = 0.0   # timestamp when detections last cleared

    # ── Loop ─────────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                logger.warning("Frame read failed – end of stream or lost connection.")
                # For live streams retry; for files, exit
                if isinstance(Config.VIDEO_SOURCE, str) and \
                   not Config.VIDEO_SOURCE.startswith("rtsp"):
                    break
                time.sleep(0.05)
                continue

            # ── Resize for display ────────────────────────────────────────
            display_frame = cv2.resize(
                frame, (Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)
            )

            # ── YOLOv8 inference ──────────────────────────────────────────
            results = model.predict(
                source      = display_frame,
                conf        = Config.CONF_THRESHOLD,
                iou         = Config.IOU_THRESHOLD,
                imgsz       = Config.IMG_SIZE,
                device      = device,
                verbose     = False
            )

            # ── Parse detections ──────────────────────────────────────────
            detections_this_frame = []

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    conf  = float(box.conf[0])
                    cls   = int(box.cls[0])
                    label = model.names.get(cls, f"class_{cls}").lower()
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    detections_this_frame.append({
                        "label": label,
                        "conf" : conf,
                        "bbox" : (x1, y1, x2, y2)
                    })

                    # Draw bounding box
                    color = Config.CLASS_COLORS.get(label, Config.DEFAULT_COLOR)
                    draw_bbox(display_frame, x1, y1, x2, y2, label, conf, color)

                    # Log to file
                    det_logger.log(
                        label, conf,
                        str(Config.VIDEO_SOURCE),
                        (x1, y1, x2, y2)
                    )

            # ── Alert / alarm logic ───────────────────────────────────────
            if detections_this_frame:
                alert_active  = True
                no_det_since  = 0.0

                # Alarm sound
                if not alarm.is_playing:
                    alarm.start()

                # Pick highest confidence detection for alerts
                top = max(detections_this_frame, key=lambda d: d["conf"])
                ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Screenshot
                shot_path = screenshotter.save(display_frame, top["label"])

                # SMS & email (non-blocking)
                alerter.send_sms(top["label"],  top["conf"], ts)
                alerter.send_email(
                    top["label"], top["conf"], ts, shot_path
                )
                voice_alert.speak_alert(top["label"])
                trigger_phone_call()
                

            else:
                # Grace period: keep alarm on for 3 s after last detection
                if alert_active:
                    if no_det_since == 0.0:
                        no_det_since = time.time()
                    elif time.time() - no_det_since > 3.0:
                        alert_active = False
                        no_det_since = 0.0
                        alarm.stop()

            # ── HUD overlay ───────────────────────────────────────────────
            fps = fps_counter.tick()
            draw_overlay(
                display_frame, fps, alert_active, Config.VIDEO_SOURCE
            )

            # ── Show frame ────────────────────────────────────────────────
            cv2.imshow(Config.WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # Q or ESC to quit
                logger.info("User requested exit.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt – shutting down.")

    finally:
        # ── Clean-up ──────────────────────────────────────────────────────
        alarm.stop()
        cap.release()
        cv2.destroyAllWindows()
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()
        logger.info("Detection system stopped cleanly.")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  🔥  AI Fire & Smoke Detection System  |  YOLOv8 + OpenCV")
    print("=" * 65)
    print(f"  Model     : {Config.MODEL_PATH}")
    print(f"  Source    : {Config.VIDEO_SOURCE}")
    print(f"  Threshold : {Config.CONF_THRESHOLD}")
    print(f"  CUDA      : {torch.cuda.is_available()}")
    print("  Press  Q  to quit.\n")

    run_detection()