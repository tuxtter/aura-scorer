"""Aura Pose Game - turn pose openness into a live aura score."""

import json
import math
import os
import select
import signal
import subprocess
import sys
import threading
import time

try:
    import termios
    import tty
except ImportError:  # pragma: no cover
    termios = None
    tty = None

import cv2
import gi
import hailo
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.common.parser import get_pipeline_parser
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    QUEUE,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
)
from hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)

logger = get_logger(__name__)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_SCORE_PATH = os.path.join(APP_DIR, ".aura_pose_best_score.json")


def _load_best_score():
    """Load the saved best score from a local JSON file."""
    try:
        if os.path.exists(BEST_SCORE_PATH):
            with open(BEST_SCORE_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, (int, float)):
                    return float(data)
    except Exception:
        logger.debug("Could not load saved best score; defaulting to zero")
    return 0.0


def _save_best_score(score):
    """Persist the best score to disk so it survives app restarts."""
    try:
        with open(BEST_SCORE_PATH, "w", encoding="utf-8") as handle:
            json.dump(float(score), handle)
    except Exception:
        logger.debug("Could not save best score")


def _play_sound_blocking(sound_name):
    """Play a quick system sound without requiring a library dependency."""
    try:
        if sys.platform == "darwin":
            sound_file = "Glass.aiff" if sound_name == "success" else "Pop.aiff"
            sound_path = f"/System/Library/Sounds/{sound_file}"
            if os.path.exists(sound_path):
                subprocess.run(["afplay", sound_path], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=False)
                return

        if sys.platform == "linux":
            if os.system("printf '\a' >/dev/console 2>/dev/null") == 0:
                return
            if os.system("paplay /usr/share/sounds/freedesktop/stereo/complete.oga >/dev/null 2>&1") == 0:
                return
            if os.system("play -q /usr/share/sounds/alsa/Front_Center.wav >/dev/null 2>&1") == 0:
                return

        print("\a", end="", flush=True)
    except Exception:
        logger.debug("Sound playback unavailable on this system")


def _play_sound(sound_name):
    """Play sounds on a daemon thread so the GStreamer callback never blocks."""
    threading.Thread(target=_play_sound_blocking, args=(sound_name,), daemon=True).start()


def _save_best_score_async(score):
    """Write the best score off the callback thread."""
    threading.Thread(target=_save_best_score, args=(score,), daemon=True).start()


LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
POSE_CONNECTIONS = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
)


DIFFICULTY_CONFIG = {
    "easy": {"min_movement": 0.025, "full_movement": 0.45},
    "medium": {"min_movement": 0.04, "full_movement": 0.60},
    "hard": {"min_movement": 0.06, "full_movement": 0.75},
}
SCORE_SMOOTHING_SECONDS = 0.75
LANDMARK_SMOOTHING_SECONDS = 0.12
NOISE_CALIBRATION_SECONDS = 3.0
MISSING_POSE_DECAY_SECONDS = 1.25


class AuraGameCallback(app_callback_class):
    """Stores the smoothed aura score and the live game state."""

    def __init__(self):
        super().__init__()
        self.use_frame = True
        self.aura_score = 0.0
        self.best_score = _load_best_score()
        self.last_pose = None
        self.last_pose_norm = None
        self.last_update = time.monotonic()
        self.movement_noise_floor = 0.0
        self.activity_breakdown = {"left_arm": 0.0, "right_arm": 0.0, "torso": 0.0}
        self.tracking_confidence = 0.0
        self.difficulty = "medium"
        self.status_message = "Dance to build aura"
        self.game_state = "playing"
        self.final_score = 0.0

    def cycle_difficulty(self):
        """Rotate through difficulty presets while the app is running."""
        order = ["easy", "medium", "hard"]
        current_index = order.index(self.difficulty)
        self.difficulty = order[(current_index + 1) % len(order)]
        self.status_message = f"Difficulty: {self.difficulty.upper()}"
        logger.info("Switched aura difficulty to %s", self.difficulty)

    def restart_game(self):
        """Reset the continuous session."""
        self.aura_score = 0.0
        self.status_message = "Dance to build aura"
        self.game_state = "playing"
        self.final_score = 0.0
        self.last_pose = None
        self.last_pose_norm = None
        self.last_update = time.monotonic()
        self.movement_noise_floor = 0.0
        self.activity_breakdown = {"left_arm": 0.0, "right_arm": 0.0, "torso": 0.0}
        self.tracking_confidence = 0.0
        logger.info("Restarted aura pose game")

    def trigger_game_over(self, reason="Session ended"):
        """End the session and show its final score."""
        self.game_state = "game_over"
        self.final_score = self.aura_score
        self.best_score = max(self.best_score, self.final_score)
        _save_best_score_async(self.best_score)
        self.status_message = reason
        _play_sound("game_over")

    def set_frame(self, frame):
        """Keep the display queue focused on the newest rendered frame."""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break
        try:
            self.frame_queue.put_nowait(frame)
        except Exception:
            logger.debug("Display queue was full; dropping rendered frame")


def _normalized_point(point, bbox):
    """Convert a landmark's bbox-relative coordinate into frame coordinates."""
    return (
        point.x() * bbox.width() + bbox.xmin(),
        point.y() * bbox.height() + bbox.ymin(),
    )


def _pixel_point(point, bbox, width, height):
    """Convert a landmark's bbox-relative coordinate into frame pixels."""
    normalized = _normalized_point(point, bbox)
    return (
        int(normalized[0] * width),
        int(normalized[1] * height),
    )


def _smooth_pose_points(prev_points, points, frame_delta):
    """Low-pass filter landmarks to reduce frame-to-frame detector jitter."""
    if prev_points is None or frame_delta <= 0.0:
        return points
    alpha = 1.0 - math.exp(-frame_delta / LANDMARK_SMOOTHING_SECONDS)
    return [
        (
            previous[0] + alpha * (current[0] - previous[0]),
            previous[1] + alpha * (current[1] - previous[1]),
        )
        for previous, current in zip(prev_points, points)
    ]


def _dance_motion_score(prev_points, points, bbox, frame_delta, difficulty, noise_floor):
    """Score normalized movement and return per-region activity and noise."""
    if prev_points is None or frame_delta <= 0.0:
        return 0.0, {"left_arm": 0.0, "right_arm": 0.0, "torso": 0.0}, noise_floor

    regions = (
        ("left_arm", (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)),
        ("right_arm", (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)),
        ("torso", (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)),
    )
    body_scale = max((bbox.width() + bbox.height()) / 2.0, 1e-6)
    raw_speeds = {}
    for name, region in regions:
        motion = sum(
            math.hypot(
                points[index][0] - prev_points[index][0],
                points[index][1] - prev_points[index][1],
            )
            for index in region
        ) / len(region)
        raw_speeds[name] = motion / body_scale / frame_delta

    config = DIFFICULTY_CONFIG[difficulty]
    movement = sum(raw_speeds.values()) / len(raw_speeds)
    if movement < config["min_movement"] * 1.5:
        noise_alpha = 1.0 - math.exp(-frame_delta / NOISE_CALIBRATION_SECONDS)
        noise_floor += noise_alpha * (movement - noise_floor)

    movement_threshold = max(config["min_movement"], noise_floor * 2.5)
    full_movement = max(config["full_movement"], movement_threshold + config["min_movement"])
    dead_zone = noise_floor * 2.0
    region_speeds = {
        name: max(0.0, speed - dead_zone)
        for name, speed in raw_speeds.items()
    }
    movement = sum(region_speeds.values()) / len(region_speeds)
    activity_score = 100.0 * np.clip(
        (movement - movement_threshold) / (full_movement - movement_threshold),
        0.0,
        1.0,
    )
    active_regions = sum(speed >= movement_threshold for speed in region_speeds.values())
    variety_factor = 0.7 + 0.3 * active_regions / len(regions)
    region_scores = {
        name: float(100.0 * np.clip(
            (speed - movement_threshold) / (full_movement - movement_threshold),
            0.0,
            1.0,
        ))
        for name, speed in region_speeds.items()
    }
    return float(activity_score * variety_factor), region_scores, noise_floor


def _pose_confidence(detection, points, bbox):
    """Estimate confidence from detection quality and landmark visibility."""
    det_conf = float(detection.get_confidence())
    valid_points = sum(
        1
        for point in points
        if bbox.xmin() <= point[0] <= bbox.xmax() and bbox.ymin() <= point[1] <= bbox.ymax()
    )
    visibility = valid_points / max(len(points), 1)
    return float(np.clip(0.7 * det_conf + 0.3 * visibility, 0.0, 1.0))


def _aura_colour(score):
    """Map a score to a warm-to-cool aura colour."""
    hue = int(135 - (min(score, 100.0) / 100.0) * 135)
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def _draw_hud(image, score, best_score, width, height, status_message, difficulty,
              activity_breakdown, tracking_confidence):
    colour = _aura_colour(score)
    cv2.rectangle(image, (20, 20), (390, 270), (18, 22, 42), -1)
    cv2.rectangle(image, (20, 20), (390, 270), colour, 2)
    cv2.putText(image, "AURA METER", (42, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(image, f"{score:05.1f} / 100", (42, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2)
    cv2.rectangle(image, (42, 122), (365, 143), (65, 65, 85), -1)
    cv2.rectangle(image, (42, 122), (42 + int(323 * score / 100), 143), colour, -1)
    cv2.putText(image, f"BEST {best_score:05.1f}", (42, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 230), 1)
    for index, (name, label) in enumerate(
        (("left_arm", "L ARM"), ("right_arm", "R ARM"), ("torso", "TORSO"))
    ):
        y = 193 + index * 23
        region_score = activity_breakdown[name]
        cv2.putText(image, f"{label} {region_score:3.0f}", (42, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 230), 1)
        cv2.rectangle(image, (145, y - 11), (365, y - 3), (65, 65, 85), -1)
        cv2.rectangle(image, (145, y - 11), (145 + int(220 * region_score / 100), y - 3),
                      colour, -1)
    label = "MAXIMUM AURA" if score >= 85 else "STRONG AURA" if score >= 65 else "BUILD AURA"
    cv2.putText(image, "DANCE AURA", (width - 260, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    cv2.putText(image, label, (width - 220, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 250), 2)
    cv2.putText(image, difficulty.upper(), (width - 200, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 210, 255), 1)
    cv2.putText(image, f"TRACK {tracking_confidence * 100:02.0f}%", (width - 200, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 210, 255), 1)
    cv2.putText(image, status_message, (width - 290, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(image, "Keep moving to build your aura!", (width - 390, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)


def _draw_game_over_screen(image, width, height, score, best_score, reason):
    """Render the final summary when the session ends."""
    overlay = image.copy()
    cv2.rectangle(overlay, (80, 80), (width - 80, height - 80), (12, 14, 20), -1)
    cv2.rectangle(overlay, (80, 80), (width - 80, height - 80), (255, 120, 120), 2)
    cv2.putText(overlay, "SESSION ENDED", (110, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(overlay, f"FINAL AURA: {score:05.1f}", (110, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 90), 2)
    cv2.putText(overlay, f"BEST SCORE: {best_score:05.1f}", (110, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2)
    cv2.putText(overlay, reason, (110, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(overlay, "Press R to restart.", (110, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return overlay


def app_callback(element, buffer, user_data):
    """Extract pose landmarks, update the dance score, and render the HUD."""
    pad = element.get_static_pad("src")
    fmt, width, height = get_caps_from_pad(pad)
    if not user_data.use_frame or not fmt or not width or not height:
        return Gst.FlowReturn.OK

    frame = get_numpy_from_buffer(buffer, fmt, width, height)
    if frame is None:
        return Gst.FlowReturn.OK

    now = time.monotonic()
    frame_delta = max(0.0, now - user_data.last_update)
    user_data.last_update = now

    roi = hailo.get_roi_from_buffer(buffer)
    detections = [
        detection for detection in roi.get_objects_typed(hailo.HAILO_DETECTION)
        if detection.get_label() == "person"
    ]
    pose = None
    if detections:
        detection = max(detections, key=lambda item: item.get_confidence())
        landmark_sets = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if landmark_sets and len(landmark_sets[0].get_points()) > RIGHT_HIP:
            bbox = detection.get_bbox()
            raw_points = [
                _normalized_point(point, bbox)
                for point in landmark_sets[0].get_points()
            ]
            normalized_points = _smooth_pose_points(
                user_data.last_pose_norm,
                raw_points,
                frame_delta,
            )
            points = [
                _pixel_point(point, bbox, width, height)
                for point in landmark_sets[0].get_points()
            ]
            if user_data.game_state == "playing":
                dance_score, activity_breakdown, user_data.movement_noise_floor = _dance_motion_score(
                    user_data.last_pose_norm,
                    normalized_points,
                    bbox,
                    frame_delta,
                    user_data.difficulty,
                    user_data.movement_noise_floor,
                )
                confidence = _pose_confidence(detection, normalized_points, bbox)
                user_data.activity_breakdown = activity_breakdown
                user_data.tracking_confidence = confidence
                confidence_factor = float(np.clip(confidence / 0.5, 0.0, 1.0))
                dynamic_score = confidence_factor * dance_score
                smoothing = 1.0 - math.exp(-frame_delta / SCORE_SMOOTHING_SECONDS)
                user_data.aura_score = (1.0 - smoothing) * user_data.aura_score + smoothing * dynamic_score
                user_data.status_message = "Keep dancing!" if dance_score >= 35.0 else "Move to build aura"
                user_data.best_score = max(user_data.best_score, user_data.aura_score)
            pose = points
            user_data.last_pose_norm = normalized_points

    if pose is not None:
        user_data.last_pose = pose
    else:
        user_data.last_pose = None
        user_data.last_pose_norm = None
        user_data.activity_breakdown = {"left_arm": 0.0, "right_arm": 0.0, "torso": 0.0}
        user_data.tracking_confidence = 0.0
        if user_data.game_state == "playing":
            user_data.aura_score *= math.exp(-frame_delta / MISSING_POSE_DECAY_SECONDS)
            user_data.status_message = "Pose not detected"

    output = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if user_data.game_state == "playing" and user_data.last_pose is not None:
        colour = _aura_colour(user_data.aura_score)
        for start, end in POSE_CONNECTIONS:
            cv2.line(output, user_data.last_pose[start], user_data.last_pose[end], colour, 3, cv2.LINE_AA)
        for point in user_data.last_pose:
            cv2.circle(output, point, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, point, 8, colour, 2, cv2.LINE_AA)

    if user_data.game_state == "playing":
        _draw_hud(
            output,
            user_data.aura_score,
            user_data.best_score,
            width,
            height,
            user_data.status_message,
            user_data.difficulty,
            user_data.activity_breakdown,
            user_data.tracking_confidence,
        )
    else:
        output = _draw_game_over_screen(
            output,
            width,
            height,
            user_data.aura_score,
            user_data.best_score,
            user_data.status_message,
        )

    user_data.set_frame(output)
    return Gst.FlowReturn.OK


class AuraPoseGame(GStreamerPoseEstimationApp):
    """Pose-estimation pipeline with a live aura score overlay."""

    def __init__(self, app_callback, user_data, parser=None):
        super().__init__(app_callback, user_data, parser)
        self.options_menu.use_frame = True
        user_data.use_frame = True

    def get_pipeline_string(self):
        """Pose pipeline without the GStreamer display branch.

        The game renders its own frames in a separate OpenCV process, so the
        overlay/videoconvert/video sink would only duplicate work.
        """
        infer_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_process_function,
            batch_size=self.batch_size,
        )
        return (
            f"{self.get_source_pipeline()} ! "
            f"{INFERENCE_PIPELINE_WRAPPER(infer_pipeline)} ! "
            f"{TRACKER_PIPELINE(class_id=0)} ! "
            f"{USER_CALLBACK_PIPELINE()} ! "
            f"{QUEUE(name='aura_sink_q', max_size_buffers=1, leaky='downstream')} ! "
            "fakesink name=hailo_display sync=false async=false"
        )


class KeyboardController(threading.Thread):
    """Listen for local keyboard controls while the game runs."""

    def __init__(self, user_data):
        super().__init__(daemon=True)
        self.user_data = user_data
        self.running = True

    def _handle_key(self, key):
        if key in ("d", "D"):
            self.user_data.cycle_difficulty()
        elif key in ("r", "R"):
            self.user_data.restart_game()
        elif key in ("q", "Q"):
            self.user_data.trigger_game_over("Quit requested")

    def run(self):
        if hasattr(sys.stdin, "isatty") and not sys.stdin.isatty():
            return

        if os.name == "nt":
            import msvcrt

            while self.running:
                if msvcrt.kbhit():
                    self._handle_key(msvcrt.getwch().lower())
                time.sleep(0.05)
            return

        if termios is None or tty is None:
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self.running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    key = sys.stdin.read(1)
                    if key:
                        self._handle_key(key.lower())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    """Start the aura pose game."""
    parser = get_pipeline_parser()
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Set the dance-score difficulty.",
    )
    args = parser.parse_args()

    user_data = AuraGameCallback()
    user_data.difficulty = args.difficulty
    keyboard = KeyboardController(user_data)
    keyboard.start()
    app = AuraPoseGame(app_callback, user_data, parser)

    def handle_sigint(_signal_number, _frame):
        logger.info("Stopping aura pose game")
        keyboard.running = False
        app.stop()

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        app.run()
    finally:
        keyboard.running = False
        keyboard.join(timeout=1.0)
        _save_best_score(user_data.best_score)


if __name__ == "__main__":
    main()
