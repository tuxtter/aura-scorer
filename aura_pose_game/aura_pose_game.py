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
from functools import lru_cache

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


CHALLENGE_TARGETS = (
    {"name": "STAR", "arm_span": 0.82, "shoulder_width": 0.30, "shoulder_to_hip": 0.48, "balance": 0.09},
    {"name": "WARRIOR", "arm_span": 0.68, "shoulder_width": 0.25, "shoulder_to_hip": 0.42, "balance": 0.08},
    {"name": "BALANCE", "arm_span": 0.60, "shoulder_width": 0.22, "shoulder_to_hip": 0.40, "balance": 0.05},
)
DIFFICULTY_CONFIG = {
    "easy": {"threshold": 70.0},
    "medium": {"threshold": 78.0},
    "hard": {"threshold": 84.0},
}


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
        self.difficulty = "medium"
        self.combo = 0
        self.challenge_index = 0
        self.challenge_armed = True
        self.status_message = "Challenge live"
        self.challenge_name = CHALLENGE_TARGETS[0]["name"]
        self.level = DIFFICULTY_CONFIG[self.difficulty]
        self.game_state = "playing"
        self.no_person_time = 0.0
        self.final_score = 0.0

    def cycle_difficulty(self):
        """Rotate through difficulty presets while the app is running."""
        order = ["easy", "medium", "hard"]
        current_index = order.index(self.difficulty)
        self.difficulty = order[(current_index + 1) % len(order)]
        self.level = DIFFICULTY_CONFIG[self.difficulty]
        self.status_message = f"Difficulty: {self.difficulty.upper()}"
        logger.info("Switched aura difficulty to %s", self.difficulty)

    def restart_game(self):
        """Reset the continuous session."""
        self.aura_score = 0.0
        self.combo = 0
        self.challenge_index = 0
        self.challenge_armed = True
        self.status_message = "Challenge live"
        self.challenge_name = CHALLENGE_TARGETS[0]["name"]
        self.level = DIFFICULTY_CONFIG[self.difficulty]
        self.game_state = "playing"
        self.no_person_time = 0.0
        self.final_score = 0.0
        self.last_pose = None
        self.last_pose_norm = None
        self.last_update = time.monotonic()
        logger.info("Restarted aura pose game")

    def trigger_game_over(self, reason="Stage failed"):
        """End the session and show its final score."""
        self.game_state = "game_over"
        self.final_score = self.aura_score
        self.best_score = max(self.best_score, self.final_score)
        _save_best_score_async(self.best_score)
        self.status_message = reason
        _play_sound("game_over")

    def advance_challenge(self):
        """Move to the next challenge and keep the combo alive."""
        self.challenge_index = (self.challenge_index + 1) % len(CHALLENGE_TARGETS)
        self.combo += 1
        self.best_score = max(self.best_score, self.aura_score)
        self.challenge_name = CHALLENGE_TARGETS[self.challenge_index]["name"]
        self.challenge_armed = False
        self.status_message = f"{self.challenge_name} ready"
        _play_sound("success")

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


def _pose_metrics(points, bbox):
    """Compute normalized pose metrics for scoring and target matching."""
    def distance(a, b):
        return math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1])

    box_width = max(1e-6, bbox.width())
    box_height = max(1e-6, bbox.height())
    shoulder_width = distance(LEFT_SHOULDER, RIGHT_SHOULDER) / box_width
    arm_span = distance(LEFT_WRIST, RIGHT_WRIST) / box_width
    shoulder_to_hip = (
        distance(LEFT_SHOULDER, LEFT_HIP) + distance(RIGHT_SHOULDER, RIGHT_HIP)
    ) / (2.0 * box_height)

    shoulder_mid_x = (points[LEFT_SHOULDER][0] + points[RIGHT_SHOULDER][0]) / 2
    wrist_mid_x = (points[LEFT_WRIST][0] + points[RIGHT_WRIST][0]) / 2
    center_offset = abs(shoulder_mid_x - wrist_mid_x) / box_width
    wrist_height_difference = abs(points[LEFT_WRIST][1] - points[RIGHT_WRIST][1]) / box_height

    return {
        "box_width": box_width,
        "box_height": box_height,
        "shoulder_width": shoulder_width,
        "arm_span": arm_span,
        "shoulder_to_hip": shoulder_to_hip,
        "center_offset": center_offset,
        "wrist_height_difference": wrist_height_difference,
    }


def _score_pose(points, bbox, metrics=None):
    """Return a 0-100 score from openness, arm span, and symmetry."""
    metrics = metrics or _pose_metrics(points, bbox)
    openness = 100.0 * (
        0.45 * np.clip((metrics["arm_span"] - 0.25) / 0.85, 0.0, 1.0)
        + 0.30 * np.clip((metrics["shoulder_width"] - 0.12) / 0.35, 0.0, 1.0)
        + 0.25 * np.clip((metrics["shoulder_to_hip"] - 0.18) / 0.35, 0.0, 1.0)
    )
    symmetry = 100.0 * (
        1.0
        - 0.55 * np.clip(metrics["center_offset"] / 0.35, 0.0, 1.0)
        - 0.45 * np.clip(metrics["wrist_height_difference"] / 0.45, 0.0, 1.0)
    )
    return float(np.clip(0.7 * openness + 0.3 * symmetry, 0.0, 100.0))


def _target_match_score(points, bbox, target, metrics=None):
    """Compare the current pose against a target challenge pose."""
    metrics = metrics or _pose_metrics(points, bbox)
    arm_score = 100.0 * np.clip(1.0 - abs(metrics["arm_span"] - target["arm_span"]) / 0.55, 0.0, 1.0)
    shoulder_score = 100.0 * np.clip(
        1.0 - abs(metrics["shoulder_width"] - target["shoulder_width"]) / 0.32, 0.0, 1.0
    )
    posture_score = 100.0 * np.clip(
        1.0 - abs(metrics["shoulder_to_hip"] - target["shoulder_to_hip"]) / 0.35, 0.0, 1.0
    )
    balance_score = 100.0 * np.clip(1.0 - metrics["center_offset"] / max(target["balance"], 0.05), 0.0, 1.0)
    return float(np.clip(0.45 * arm_score + 0.25 * shoulder_score + 0.20 * posture_score + 0.10 * balance_score, 0.0, 100.0))


def _pose_motion(prev_points, points, bbox):
    """Measure how much the pose is changing frame-to-frame in image space."""
    if prev_points is None:
        return 0.0

    total_motion = 0.0
    for prev_point, point in zip(prev_points, points):
        total_motion += math.hypot(point[0] - prev_point[0], point[1] - prev_point[1])

    avg_motion = total_motion / max(len(points), 1)
    box_width = max(1e-6, bbox.width())
    relative_motion = avg_motion / max(box_width * 0.25, 1e-6)
    return float(np.clip(relative_motion, 0.0, 1.0) * 100.0)


def _pose_confidence(detection, points, bbox, prev_points=None):
    """Estimate confidence from detection quality and temporal stability."""
    det_conf = float(detection.get_confidence())
    valid_points = sum(
        1
        for point in points
        if bbox.xmin() <= point[0] <= bbox.xmax() and bbox.ymin() <= point[1] <= bbox.ymax()
    )
    visibility = valid_points / max(len(points), 1)

    stability = 1.0
    if prev_points is not None:
        jitter = 0.0
        for prev_point, point in zip(prev_points, points):
            jitter += math.hypot(point[0] - prev_point[0], point[1] - prev_point[1])
        stability = 1.0 - min(1.0, (jitter / max(len(points), 1)) / max(1.0, bbox.width() * 0.25))

    return float(np.clip(0.5 * det_conf + 0.3 * visibility + 0.2 * stability, 0.0, 1.0))


@lru_cache(maxsize=2)
def _draw_stage_background(width, height):
    """Create (once per resolution) a stylized stage backdrop behind the live pose output."""
    stage = np.full((height, width, 3), (18, 22, 42), dtype=np.uint8)
    center_x = width // 2
    center_y = int(height * 0.32)

    xs = np.arange(width, dtype=np.float32) / max(width - 1, 1)
    stage[:, :, 0] = np.clip(18 + (xs * 55).astype(np.int32), 0, 255).astype(np.uint8)[None, :]
    stage[:, :, 2] = np.clip(
        42 + ((1.0 - np.arange(width, dtype=np.float32) / max(width, 1)) * 18).astype(np.int32), 0, 255
    ).astype(np.uint8)[None, :]

    cv2.ellipse(stage, (center_x, center_y), (int(width * 0.28), int(height * 0.19)), 0, 0, 360,
                (72, 82, 140), 2)
    cv2.ellipse(stage, (center_x, center_y), (int(width * 0.19), int(height * 0.12)), 0, 0, 360,
                (255, 175, 90), 2)
    cv2.ellipse(stage, (center_x, int(height * 0.82)), (int(width * 0.42), int(height * 0.22)),
                0, 0, 360, (30, 34, 58), -1)
    cv2.ellipse(stage, (center_x, int(height * 0.82)), (int(width * 0.34), int(height * 0.16)),
                0, 0, 360, (72, 78, 120), 2)

    for ring in range(3):
        radius = int(width * (0.14 + ring * 0.10))
        cv2.circle(stage, (center_x, center_y), radius, (80 + ring * 25, 80 + ring * 15, 110), 1)

    stage.setflags(write=False)
    return stage


def _aura_colour(score):
    """Map a score to a warm-to-cool aura colour."""
    hue = int(135 - (min(score, 100.0) / 100.0) * 135)
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def _draw_target_pose(image, challenge_name, width, height):
    """Draw a dim silhouette of the required stage challenge to guide the player."""
    cx = width // 2
    cy = int(height * 0.34)
    target = CHALLENGE_TARGETS[CHALLENGE_TARGETS.index(next(item for item in CHALLENGE_TARGETS if item["name"] == challenge_name))]
    scale = 1.0
    if target["name"] == "STAR":
        scale = 1.15
    elif target["name"] == "BALANCE":
        scale = 0.9

    points = {
        LEFT_SHOULDER: (int(cx - 80 * scale), int(cy - 25)),
        RIGHT_SHOULDER: (int(cx + 80 * scale), int(cy - 25)),
        LEFT_ELBOW: (int(cx - 155 * scale), int(cy + 80 * scale)),
        RIGHT_ELBOW: (int(cx + 155 * scale), int(cy + 80 * scale)),
        LEFT_WRIST: (int(cx - 195 * scale), int(cy + 175 * scale)),
        RIGHT_WRIST: (int(cx + 195 * scale), int(cy + 175 * scale)),
        LEFT_HIP: (int(cx - 55 * scale), int(cy + 120 * scale)),
        RIGHT_HIP: (int(cx + 55 * scale), int(cy + 120 * scale)),
    }
    colour = (110, 130, 200)
    for start, end in POSE_CONNECTIONS:
        cv2.line(image, points[start], points[end], colour, 2, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(image, point, 5, colour, -1)
    cv2.putText(image, f"TARGET: {challenge_name}", (width - 230, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)


def _draw_hud(image, score, best_score, width, height, challenge_name, combo, status_message, difficulty):
    colour = _aura_colour(score)
    cv2.rectangle(image, (20, 20), (390, 210), (18, 22, 42), -1)
    cv2.rectangle(image, (20, 20), (390, 210), colour, 2)
    cv2.putText(image, "AURA METER", (42, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(image, f"{score:05.1f} / 100", (42, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2)
    cv2.rectangle(image, (42, 122), (365, 143), (65, 65, 85), -1)
    cv2.rectangle(image, (42, 122), (42 + int(323 * score / 100), 143), colour, -1)
    cv2.putText(image, f"BEST {best_score:05.1f}", (42, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 230), 1)
    cv2.putText(image, f"COMBO x{combo}", (42, 196),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 210, 120), 1)

    label = "MAXIMUM AURA" if score >= 85 else "STRONG AURA" if score >= 65 else "BUILD AURA"
    cv2.putText(image, label, (width - 260, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    cv2.putText(image, challenge_name, (width - 220, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 250), 2)
    cv2.putText(image, difficulty.upper(), (width - 200, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 210, 255), 1)
    cv2.putText(image, status_message, (width - 290, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(image, "Spread out and balance your pose!", (width - 390, height - 25),
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
    """Extract pose landmarks, update the score, and render the challenge HUD."""
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
            normalized_points = [
                _normalized_point(point, bbox)
                for point in landmark_sets[0].get_points()
            ]
            points = [
                _pixel_point(point, bbox, width, height)
                for point in landmark_sets[0].get_points()
            ]
            if user_data.game_state == "playing":
                metrics = _pose_metrics(normalized_points, bbox)
                pose_score = _score_pose(normalized_points, bbox, metrics)
                motion_score = _pose_motion(user_data.last_pose_norm, normalized_points, bbox)
                confidence = _pose_confidence(detection, normalized_points, bbox, user_data.last_pose_norm)
                challenge_target = CHALLENGE_TARGETS[user_data.challenge_index]
                match_score = _target_match_score(normalized_points, bbox, challenge_target, metrics)
                dynamic_score = confidence * (0.45 * pose_score + 0.25 * motion_score + 0.30 * match_score)
                smoothing = np.clip(0.12 + 0.28 * confidence, 0.12, 0.42)
                user_data.aura_score = (1.0 - smoothing) * user_data.aura_score + smoothing * dynamic_score
                if match_score >= user_data.level["threshold"]:
                    if user_data.challenge_armed:
                        user_data.status_message = f"Perfect {challenge_target['name']}!"
                        user_data.aura_score = min(100.0, user_data.aura_score + 8.0)
                        user_data.advance_challenge()
                else:
                    user_data.challenge_armed = True

                user_data.best_score = max(user_data.best_score, user_data.aura_score)
            pose = points
            user_data.last_pose_norm = normalized_points
    else:
        if user_data.game_state == "playing":
            user_data.no_person_time += frame_delta
            if user_data.no_person_time > 2.5:
                user_data.trigger_game_over("No pose detected")
        else:
            user_data.no_person_time = 0.0

    if pose is not None:
        user_data.last_pose = pose
        user_data.no_person_time = 0.0

    output = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    output = cv2.addWeighted(_draw_stage_background(width, height), 0.55, output, 0.45, 0)

    if user_data.game_state == "playing" and user_data.last_pose is not None:
        colour = _aura_colour(user_data.aura_score)
        for start, end in POSE_CONNECTIONS:
            cv2.line(output, user_data.last_pose[start], user_data.last_pose[end], colour, 3, cv2.LINE_AA)
        for point in user_data.last_pose:
            cv2.circle(output, point, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, point, 8, colour, 2, cv2.LINE_AA)

    if user_data.game_state == "playing":
        _draw_target_pose(output, user_data.challenge_name, width, height)
        _draw_hud(
            output,
            user_data.aura_score,
            user_data.best_score,
            width,
            height,
            user_data.challenge_name,
            user_data.combo,
            user_data.status_message,
            user_data.difficulty,
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
        help="Set the aura challenge difficulty.",
    )
    # 640x480 keeps per-frame CPU work low enough for Raspberry Pi 5 + Hailo-8L.
    parser.set_defaults(width=640, height=480)
    args = parser.parse_args()

    user_data = AuraGameCallback()
    user_data.difficulty = args.difficulty
    user_data.level = DIFFICULTY_CONFIG[user_data.difficulty]
    keyboard = KeyboardController(user_data)
    keyboard.start()
    app = AuraPoseGame(app_callback, user_data, parser)

    def handle_sigint(_signal_number, _frame):
        logger.info("Stopping aura pose game")
        keyboard.running = False
        app.stop()

    signal.signal(signal.SIGINT, handle_sigint)
    app.run()


if __name__ == "__main__":
    main()
