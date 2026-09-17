"""Aura Pose Game - turn pose openness into a live aura score."""

import math
import signal
import time

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
from hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)

logger = get_logger(__name__)

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


class AuraGameCallback(app_callback_class):
    """Stores the smoothed aura score and the latest pose."""

    def __init__(self):
        super().__init__()
        self.use_frame = True
        self.aura_score = 0.0
        self.best_score = 0.0
        self.last_pose = None
        self.last_pose_norm = None
        self.last_update = time.monotonic()

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


def _score_pose(points, bbox):
    """Return a 0-100 score from openness, arm span, and symmetry."""
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

    openness = 100.0 * (
        0.45 * np.clip((arm_span - 0.25) / 0.85, 0.0, 1.0)
        + 0.30 * np.clip((shoulder_width - 0.12) / 0.35, 0.0, 1.0)
        + 0.25 * np.clip((shoulder_to_hip - 0.18) / 0.35, 0.0, 1.0)
    )
    symmetry = 100.0 * (
        1.0
        - 0.55 * np.clip(center_offset / 0.35, 0.0, 1.0)
        - 0.45 * np.clip(wrist_height_difference / 0.45, 0.0, 1.0)
    )
    return float(np.clip(0.7 * openness + 0.3 * symmetry, 0.0, 100.0))


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


def _aura_colour(score):
    """Map a score to a warm-to-cool aura colour."""
    hue = int(135 - (min(score, 100.0) / 100.0) * 135)
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def _draw_hud(image, score, best_score, width, height):
    colour = _aura_colour(score)
    cv2.rectangle(image, (20, 20), (390, 178), (18, 22, 42), -1)
    cv2.rectangle(image, (20, 20), (390, 178), colour, 2)
    cv2.putText(image, "AURA METER", (42, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(image, f"{score:05.1f} / 100", (42, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2)
    cv2.rectangle(image, (42, 122), (365, 143), (65, 65, 85), -1)
    cv2.rectangle(image, (42, 122), (42 + int(323 * score / 100), 143), colour, -1)
    cv2.putText(image, f"BEST {best_score:05.1f}", (42, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 230), 1)
    label = "MAXIMUM AURA" if score >= 85 else "STRONG AURA" if score >= 65 else "BUILD AURA"
    cv2.putText(image, label, (width - 260, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    cv2.putText(image, "Spread out and balance your pose!", (width - 390, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)


def app_callback(element, buffer, user_data):
    """Extract pose landmarks, update the score, and render the game overlay."""
    pad = element.get_static_pad("src")
    fmt, width, height = get_caps_from_pad(pad)
    if not user_data.use_frame or not fmt or not width or not height:
        return Gst.FlowReturn.OK

    frame = get_numpy_from_buffer(buffer, fmt, width, height)
    if frame is None:
        return Gst.FlowReturn.OK

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
            pose_score = _score_pose(normalized_points, bbox)
            motion_score = _pose_motion(user_data.last_pose_norm, normalized_points, bbox)
            confidence = _pose_confidence(detection, normalized_points, bbox, user_data.last_pose_norm)
            dynamic_score = confidence * (0.65 * pose_score + 0.35 * motion_score)
            smoothing = np.clip(0.10 + 0.30 * confidence, 0.10, 0.40)
            user_data.aura_score = (1.0 - smoothing) * user_data.aura_score + smoothing * dynamic_score
            user_data.best_score = max(user_data.best_score, user_data.aura_score)
            pose = points
            user_data.last_pose_norm = normalized_points

    if pose is not None:
        user_data.last_pose = pose

    output = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if user_data.last_pose is not None:
        colour = _aura_colour(user_data.aura_score)
        for start, end in POSE_CONNECTIONS:
            cv2.line(output, user_data.last_pose[start], user_data.last_pose[end], colour, 3, cv2.LINE_AA)
        for point in user_data.last_pose:
            cv2.circle(output, point, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, point, 8, colour, 2, cv2.LINE_AA)

    _draw_hud(output, user_data.aura_score, user_data.best_score, width, height)
    user_data.set_frame(output)
    return Gst.FlowReturn.OK


class AuraPoseGame(GStreamerPoseEstimationApp):
    """Pose-estimation pipeline with a live aura score overlay."""

    def __init__(self, app_callback, user_data, parser=None):
        super().__init__(app_callback, user_data, parser)
        self.options_menu.use_frame = True
        user_data.use_frame = True


def main():
    """Start the aura pose game."""
    parser = get_pipeline_parser()
    user_data = AuraGameCallback()
    app = AuraPoseGame(app_callback, user_data, parser)

    def handle_sigint(_signal_number, _frame):
        logger.info("Stopping aura pose game")
        app.stop()

    signal.signal(signal.SIGINT, handle_sigint)
    app.run()


if __name__ == "__main__":
    main()
