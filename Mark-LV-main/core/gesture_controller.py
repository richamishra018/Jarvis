from __future__ import annotations

import math
import shutil
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path


_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model_asset() -> Path:
    model_path = Path.home() / ".mark-lv" / "models" / "hand_landmarker.task"
    if model_path.is_file() and model_path.stat().st_size > 100_000:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".task.download")
    try:
        with urllib.request.urlopen(_MODEL_URL, timeout=45) as response:
            with temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        if temporary_path.stat().st_size <= 100_000:
            raise RuntimeError("Downloaded hand model is incomplete")
        temporary_path.replace(model_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return model_path


def classify_landmarks(points: Sequence[tuple[float, float]]) -> str:
    """Classify a normalized 21-point MediaPipe hand skeleton."""
    if len(points) != 21:
        return "NO_HAND"

    thumb_tip, index_tip = points[4], points[8]
    if math.dist(thumb_tip, index_tip) < 0.055:
        return "PINCH"

    extended = [points[tip][1] < points[pip][1] - 0.025
                for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]
    if all(extended):
        return "OPEN_PALM"
    if extended[0] and not any(extended[1:]):
        return "POINT"
    return "NEUTRAL"


class HandGestureController:
    """Optional local webcam hand tracking; callbacks run on its worker thread."""

    def __init__(self, on_update: Callable[[str, float, float], None],
                 camera_index: int = 0):
        self._on_update = on_update
        self._camera_index = camera_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="JarvisHandTracking")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _emit(self, gesture: str, x: float = -1.0, y: float = -1.0) -> None:
        try:
            self._on_update(gesture, x, y)
        except Exception:
            pass

    def _run(self) -> None:
        capture = None
        try:
            import cv2
            import mediapipe as mp

            model_path = _ensure_model_asset()
            vision = mp.tasks.vision
            options = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )

            capture = cv2.VideoCapture(self._camera_index)
            if not capture.isOpened():
                raise RuntimeError("Could not open the selected camera")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            with vision.HandLandmarker.create_from_options(options) as landmarker:
                self._emit("READY")
                swipe_origin_x: float | None = None
                swipe_origin_time = 0.0
                last_swipe = 0.0
                last_gesture = ""
                smooth_x = 0.5
                smooth_y = 0.5

                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        self._stop.wait(0.05)
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = int(time.monotonic() * 1000)
                    result = landmarker.detect_for_video(image, timestamp_ms)
                    if not result.hand_landmarks:
                        if last_gesture != "NO_HAND":
                            self._emit("NO_HAND")
                            last_gesture = "NO_HAND"
                        swipe_origin_x = None
                        continue

                    points = [(point.x, point.y)
                              for point in result.hand_landmarks[0]]
                    x, y = points[8]
                    smooth_x = smooth_x * 0.65 + x * 0.35
                    smooth_y = smooth_y * 0.65 + y * 0.35
                    now = time.monotonic()
                    gesture = classify_landmarks(points)

                    if gesture in ("POINT", "OPEN_PALM"):
                        if (swipe_origin_x is None
                                or now - swipe_origin_time > 0.7):
                            swipe_origin_x = x
                            swipe_origin_time = now
                        elif (abs(x - swipe_origin_x) > 0.24
                              and now - last_swipe > 0.9):
                            delta = x - swipe_origin_x
                            self._emit("SWIPE_RIGHT" if delta > 0 else "SWIPE_LEFT",
                                       smooth_x, smooth_y)
                            last_swipe = now
                            swipe_origin_x = x
                            swipe_origin_time = now
                    else:
                        swipe_origin_x = None

                    if gesture != last_gesture:
                        self._emit(gesture, smooth_x, smooth_y)
                        last_gesture = gesture
                    else:
                        self._emit("TRACKING", smooth_x, smooth_y)

        except Exception as exc:
            if not self._stop.is_set():
                self._emit("ERROR", -1.0, -1.0)
                print(f"[Gestures] {exc}")
        finally:
            if capture is not None:
                capture.release()
            self._emit("STOPPED")