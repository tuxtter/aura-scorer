"""Aura Pose Game - turn pose openness into a live aura score."""

import json
import math
import os
import random
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
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
HEAD_POINTS = (0, 1, 2, 3, 4)
# (name, keypoints, weight, optional). Optional regions only count when visible.
REGIONS = (
    ("left_arm", (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 1.0, False),
    ("right_arm", (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), 1.0, False),
    ("torso", (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP), 1.0, False),
    ("head", HEAD_POINTS, 0.5, True),
    ("left_leg", (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 0.75, True),
    ("right_leg", (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 0.75, True),
)
KEYPOINT_MIN_CONFIDENCE = 0.4


def _empty_breakdown():
    return {name: 0.0 for name, _, _, _ in REGIONS}
POSE_CONNECTIONS = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
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
DUEL_DURATION_SECONDS = 90.0
SPRINT_CHALLENGE_SECONDS = 30.0
HOLD_CHALLENGE_SECONDS = 20.0
HOLD_CHALLENGE_LIMIT_SECONDS = 60.0
COMBO_SCORE_THRESHOLD = 35.0
COMBO_REGION_THRESHOLD = 35.0
DUEL_TARGET_MIN = 20
DUEL_TARGET_MAX = 85
DUEL_TARGET_ICONS = ("pan_de_muerto", "calaverita", "ghost", "pumpkin")


def _random_duel_target(previous=None):
    targets = [
        target for target in range(DUEL_TARGET_MIN, DUEL_TARGET_MAX + 1)
        if target != previous
    ]
    return random.choice(targets)


def _random_target_icon():
    return random.choice(DUEL_TARGET_ICONS)


def _update_duel_target(player):
    if player.target_armed and player.aura_score >= player.target_score:
        player.points += 1
        player.target_score = _random_duel_target(player.target_score)
        player.target_icon = _random_target_icon()
        player.target_armed = False
        player.achievement_message = "TARGET HIT! +1 POINT"
        player.achievement_until = time.monotonic() + 1.5
        _play_sound("success")
    elif not player.target_armed and player.aura_score < player.target_score:
        player.target_armed = True


def _victory_quote(winner, loser):
    """Pick a Street Fighter-style victory line aimed at the loser's aura."""
    margin = winner.points - loser.points
    rival = loser.name
    if margin >= 3:
        lines = (
            f"{rival}, your aura is so faint I can barely see it!",
            f"Is that all, {rival}? Your aura peaked at only {loser.best_score:.0f}!",
            f"You are not ready, {rival}. Come back when your aura glows!",
        )
    elif margin == 1:
        lines = (
            f"Close one, {rival}! But my aura burned just a little brighter!",
            f"Your aura reached {loser.best_score:.0f}... not enough to stop me!",
            f"Nice try, {rival}. Train your aura and challenge me again!",
        )
    else:
        lines = (
            f"Your aura reached {loser.best_score:.0f}, {rival}. Mine reached {winner.best_score:.0f}!",
            f"{rival}, feel the power of my aura!",
            f"You fought well, {rival}, but my aura was stronger!",
        )
    return random.choice(lines)


def _reset_combo_state(player):
    player.combo_seconds = 0.0
    player.best_combo_seconds = 0.0
    player.combo_tier = 0
    player.achievements = set()
    player.achievement_message = ""
    player.achievement_until = 0.0


def _award_achievement(player, key, message):
    if key in player.achievements:
        return
    player.achievements.add(key)
    player.achievement_message = message
    player.achievement_until = time.monotonic() + 2.5
    _play_sound("success")


def _update_combo(player, dance_score, frame_delta):
    active_regions = sum(
        score is not None and score >= COMBO_REGION_THRESHOLD
        for score in player.activity_breakdown.values()
    )
    if dance_score >= COMBO_SCORE_THRESHOLD and active_regions >= 2:
        player.combo_seconds += min(frame_delta, 0.5)
        player.best_combo_seconds = max(player.best_combo_seconds, player.combo_seconds)
    else:
        player.combo_seconds = 0.0

    tier = int(player.combo_seconds // 5.0)
    if tier > player.combo_tier:
        player.combo_tier = tier
        if tier == 1:
            _award_achievement(player, "combo_5", "5-SECOND COMBO!")
    elif tier == 0:
        player.combo_tier = 0

    if active_regions >= 4:
        _award_achievement(player, "full_body", "FULL BODY!")


class AuraGameCallback(app_callback_class):
    """Stores the smoothed aura score and the live game state."""

    def __init__(self, mode="solo", challenge="none"):
        super().__init__()
        self.use_frame = True
        self.mode = mode
        self.challenge = challenge
        self.aura_score = 0.0
        self.best_score = _load_best_score()
        self.last_pose = None
        self.last_pose_norm = None
        self.last_update = time.monotonic()
        self.movement_noise_floor = 0.0
        self.activity_breakdown = _empty_breakdown()
        self.tracking_confidence = 0.0
        self.difficulty = "medium"
        self.status_message = "Dance to build aura"
        self.game_state = "playing"
        self.final_score = 0.0
        self.final_duel_result = None
        self.duel_elapsed = 0.0
        self.challenge_elapsed = 0.0
        self.challenge_hold_elapsed = 0.0
        self.challenge_start_best_score = self.best_score
        self.challenge_peak_score = 0.0
        self.challenge_summary = ""
        _reset_combo_state(self)
        self.duel_players = [DuelPlayerState("PLAYER 1"), DuelPlayerState("PLAYER 2")]

    def cycle_difficulty(self):
        """Rotate through difficulty presets while the app is running."""
        order = ["easy", "medium", "hard"]
        current_index = order.index(self.difficulty)
        self.difficulty = order[(current_index + 1) % len(order)]
        self.status_message = f"Difficulty: {self.difficulty.upper()}"
        logger.info("Switched aura difficulty to %s", self.difficulty)

    def cycle_challenge(self):
        """Rotate solo challenge presets and start a fresh run."""
        order = ["none", "sprint", "hold"]
        current_index = order.index(self.challenge)
        self.challenge = order[(current_index + 1) % len(order)]
        self.restart_game()
        label = "OFF" if self.challenge == "none" else self.challenge.upper()
        self.status_message = f"Challenge: {label}"
        logger.info("Switched aura challenge to %s", self.challenge)

    def toggle_mode(self, player_names=None):
        """Switch between solo and duel, starting a fresh run in the new mode."""
        self.mode = "duel" if self.mode == "solo" else "solo"
        if player_names is not None:
            for player, name in zip(self.duel_players, player_names):
                player.name = name
        self.restart_game()
        self.status_message = f"Mode: {self.mode.upper()}"
        logger.info("Switched aura game mode to %s", self.mode)

    def restart_game(self):
        """Reset the continuous session."""
        self.aura_score = 0.0
        self.status_message = "Dance to build aura"
        self.game_state = "playing"
        self.final_score = 0.0
        self.final_duel_result = None
        self.duel_elapsed = 0.0
        self.challenge_elapsed = 0.0
        self.challenge_hold_elapsed = 0.0
        self.challenge_start_best_score = self.best_score
        self.challenge_peak_score = 0.0
        self.challenge_summary = ""
        _reset_combo_state(self)
        self.last_pose = None
        self.last_pose_norm = None
        self.last_update = time.monotonic()
        self.movement_noise_floor = 0.0
        self.activity_breakdown = _empty_breakdown()
        self.tracking_confidence = 0.0
        for player in self.duel_players:
            player.reset()
        logger.info("Restarted aura pose game")

    def log_duel_winner(self):
        """Report the duel result using points earned by reaching aura targets."""
        player_one, player_two = self.duel_players
        if player_one.points == player_two.points:
            logger.info("1-vs-1 duel ended in a tie: both players scored %d points",
                        player_one.points)
            return (
                "TIE GAME",
                f"POINTS {player_one.points} - {player_two.points}  |  "
                f"BEST AURA {player_one.best_score:.1f} - {player_two.best_score:.1f}",
                None,
            )
        else:
            winner = player_one if player_one.points > player_two.points else player_two
            opponent = player_two if winner is player_one else player_one
            logger.info(
                "1-vs-1 duel winner: %s with %d points (opponent: %d)",
                winner.name,
                winner.points,
                opponent.points,
            )
            return (
                f"{winner.name} WINS!",
                f"POINTS {winner.points} - {opponent.points}  |  "
                f"BEST AURA {winner.best_score:.1f} - {opponent.best_score:.1f}",
                (winner.name, _victory_quote(winner, opponent)),
            )

    def trigger_game_over(self, reason="Session ended"):
        """End the session and show its final score."""
        self.game_state = "game_over"
        self.final_score = self.aura_score
        if self.mode == "duel":
            self.final_duel_result = self.log_duel_winner()
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


class DuelPlayerState:
    """Per-player dance score and pose history for the two-player mode."""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.aura_score = 0.0
        self.best_score = 0.0
        self.last_pose = None
        self.last_pose_norm = None
        self.center_x = None
        self.movement_noise_floor = 0.0
        self.activity_breakdown = _empty_breakdown()
        self.tracking_confidence = 0.0
        _reset_combo_state(self)
        self.points = 0
        self.target_score = _random_duel_target()
        self.target_icon = _random_target_icon()
        self.target_armed = True


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


def _dance_motion_score(prev_points, points, bbox, frame_delta, difficulty, noise_floor,
                        visible=None):
    """Score normalized movement; return per-region activity (None if unseen) and noise."""
    if prev_points is None or frame_delta <= 0.0:
        return 0.0, _empty_breakdown(), noise_floor

    if visible is None:
        visible = [True] * len(points)
    body_scale = max((bbox.width() + bbox.height()) / 2.0, 1e-6)
    raw_speeds = {}
    weights = {}
    for name, region, weight, optional in REGIONS:
        if optional and not all(visible[index] for index in region):
            continue
        motion = sum(
            math.hypot(
                points[index][0] - prev_points[index][0],
                points[index][1] - prev_points[index][1],
            )
            for index in region
        ) / len(region)
        raw_speeds[name] = motion / body_scale / frame_delta
        weights[name] = weight
    total_weight = sum(weights.values())

    def weighted(values):
        return sum(values[name] * weights[name] for name in values) / total_weight

    config = DIFFICULTY_CONFIG[difficulty]
    movement = weighted(raw_speeds)
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
    movement = weighted(region_speeds)
    activity_score = 100.0 * np.clip(
        (movement - movement_threshold) / (full_movement - movement_threshold),
        0.0,
        1.0,
    )
    active_weight = sum(
        weights[name] for name, speed in region_speeds.items() if speed >= movement_threshold
    )
    variety_factor = 0.7 + 0.3 * active_weight / total_weight
    region_scores = {name: None for name, _, _, _ in REGIONS}
    for name, speed in region_speeds.items():
        region_scores[name] = float(100.0 * np.clip(
            (speed - movement_threshold) / (full_movement - movement_threshold),
            0.0,
            1.0,
        ))
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


def _update_player_score(player, detection, frame_delta, difficulty, width, height):
    bbox = detection.get_bbox()
    player.center_x = (bbox.xmin() + bbox.xmax()) / 2.0
    landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)[0].get_points()
    raw_points = [
        _normalized_point(point, bbox)
        for point in landmarks
    ]
    visible = [
        point.confidence() >= KEYPOINT_MIN_CONFIDENCE
        and 0.0 <= point.x() <= 1.0 and 0.0 <= point.y() <= 1.0
        for point in landmarks
    ]
    normalized_points = _smooth_pose_points(player.last_pose_norm, raw_points, frame_delta)
    dance_score, activity, player.movement_noise_floor = _dance_motion_score(
        player.last_pose_norm,
        normalized_points,
        bbox,
        frame_delta,
        difficulty,
        player.movement_noise_floor,
        visible,
    )
    confidence = _pose_confidence(detection, normalized_points, bbox)
    confidence_factor = float(np.clip(confidence / 0.5, 0.0, 1.0))
    player.activity_breakdown = activity
    _update_combo(player, dance_score, frame_delta)
    combo_bonus = min(player.combo_seconds, 10.0) * 0.01
    dynamic_score = min(100.0, confidence_factor * dance_score * (1.0 + combo_bonus))
    smoothing = 1.0 - math.exp(-frame_delta / SCORE_SMOOTHING_SECONDS)
    player.aura_score = (
        (1.0 - smoothing) * player.aura_score + smoothing * dynamic_score
    )
    player.best_score = max(player.best_score, player.aura_score)
    if player.aura_score >= 85.0:
        _award_achievement(player, "maximum_aura", "MAXIMUM AURA!")
    player.tracking_confidence = confidence
    player.last_pose_norm = normalized_points
    player.last_pose = [
        _pixel_point(point, bbox, width, height)
        for point in landmarks
    ]
    return player.last_pose, dance_score


def _decay_player_score(player, frame_delta):
    player.last_pose = None
    player.last_pose_norm = None
    player.activity_breakdown = _empty_breakdown()
    player.tracking_confidence = 0.0
    player.combo_seconds = 0.0
    player.combo_tier = 0
    player.aura_score *= math.exp(-frame_delta / MISSING_POSE_DECAY_SECONDS)


def _assign_duel_detections(players, candidates):
    """Assign detections to screen-side players, preferring previous positions."""
    assigned = [None, None]
    if len(candidates) >= 2:
        candidates = sorted(
            candidates,
            key=lambda item: item[0].get_confidence(),
            reverse=True,
        )[:2]
        candidates.sort(key=lambda item: item[2])
        if all(player.center_x is not None for player in players):
            direct_cost = sum(
                abs(players[index].center_x - candidates[index][2])
                for index in range(2)
            )
            crossed_cost = (
                abs(players[0].center_x - candidates[1][2])
                + abs(players[1].center_x - candidates[0][2])
            )
            if crossed_cost < direct_cost:
                candidates.reverse()
        return candidates

    if not candidates:
        return assigned
    candidate = candidates[0]
    known_players = [
        index for index, player in enumerate(players) if player.center_x is not None
    ]
    if known_players:
        known_index = min(
            known_players,
            key=lambda player_index: abs(players[player_index].center_x - candidate[2]),
        )
        known_cost = abs(players[known_index].center_x - candidate[2])
        unknown_players = [index for index in range(2) if index not in known_players]
        if unknown_players:
            unknown_index = min(
                unknown_players,
                key=lambda player_index: abs((0.25 + 0.5 * player_index) - candidate[2]),
            )
            unknown_cost = abs((0.25 + 0.5 * unknown_index) - candidate[2])
            index = unknown_index if unknown_cost <= known_cost else known_index
        else:
            index = known_index
    else:
        index = 0 if candidate[2] < 0.5 else 1
    assigned[index] = candidate
    return assigned


def _aura_colour(score):
    """Map a score to a warm-to-cool aura colour."""
    hue = int(135 - (min(score, 100.0) / 100.0) * 135)
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def _draw_pan_de_muerto_marker(image, center_x, center_y):
    """Draw a small pan de muerto icon as the duel target marker."""
    center = (center_x, center_y)
    cv2.ellipse(image, center, (12, 10), 0, 0, 360, (36, 104, 178), -1, cv2.LINE_AA)
    cv2.ellipse(image, center, (12, 10), 0, 0, 360, (255, 225, 175), 1, cv2.LINE_AA)
    bone_color = (245, 220, 185)
    for start, end in (
        ((center_x - 8, center_y - 7), (center_x + 8, center_y + 7)),
        ((center_x + 8, center_y - 7), (center_x - 8, center_y + 7)),
    ):
        cv2.line(image, start, end, bone_color, 3, cv2.LINE_AA)
        for endpoint in (start, end):
            cv2.circle(image, endpoint, 2, bone_color, -1, cv2.LINE_AA)
    cv2.circle(image, center, 2, (255, 245, 220), -1, cv2.LINE_AA)


def _draw_calaverita_marker(image, center_x, center_y):
    """Draw a small decorated sugar skull (calaverita de azucar) as the target marker."""
    skull = (245, 245, 250)
    outline = (40, 40, 60)
    cv2.circle(image, (center_x, center_y - 2), 11, skull, -1, cv2.LINE_AA)
    cv2.rectangle(image, (center_x - 6, center_y + 6), (center_x + 6, center_y + 12),
                  skull, -1, cv2.LINE_AA)
    cv2.circle(image, (center_x, center_y - 2), 11, outline, 1, cv2.LINE_AA)
    for dx in (-5, 5):
        cv2.circle(image, (center_x + dx, center_y - 3), 3, outline, -1, cv2.LINE_AA)
        cv2.circle(image, (center_x + dx, center_y - 3), 4, (200, 90, 220), 1, cv2.LINE_AA)
    cv2.circle(image, (center_x, center_y + 2), 1, outline, -1, cv2.LINE_AA)
    for dx in (-3, 0, 3):
        cv2.line(image, (center_x + dx, center_y + 8), (center_x + dx, center_y + 11),
                 outline, 1, cv2.LINE_AA)
    cv2.circle(image, (center_x, center_y - 11), 2, (60, 200, 255), -1, cv2.LINE_AA)


def _draw_ghost_marker(image, center_x, center_y):
    """Draw a small cartoon ghost as the target marker."""
    body = np.array([
        (center_x - 10, center_y + 12), (center_x - 10, center_y - 2),
        (center_x - 6, center_y - 9), (center_x, center_y - 12),
        (center_x + 6, center_y - 9), (center_x + 10, center_y - 2),
        (center_x + 10, center_y + 12), (center_x + 6, center_y + 8),
        (center_x + 3, center_y + 12), (center_x, center_y + 8),
        (center_x - 3, center_y + 12), (center_x - 6, center_y + 8),
    ], dtype=np.int32)
    cv2.fillPoly(image, [body], (250, 250, 255), cv2.LINE_AA)
    cv2.polylines(image, [body], True, (200, 190, 180), 1, cv2.LINE_AA)
    for dx in (-4, 4):
        cv2.ellipse(image, (center_x + dx, center_y - 3), (2, 3), 0, 0, 360,
                    (40, 30, 30), -1, cv2.LINE_AA)
    cv2.ellipse(image, (center_x, center_y + 3), (2, 3), 0, 0, 360,
                (40, 30, 30), -1, cv2.LINE_AA)


def _draw_pumpkin_marker(image, center_x, center_y):
    """Draw a small jack-o'-lantern as the target marker."""
    orange = (20, 130, 250)
    dark_orange = (10, 85, 190)
    cv2.ellipse(image, (center_x - 6, center_y + 1), (6, 10), 0, 0, 360, orange, -1, cv2.LINE_AA)
    cv2.ellipse(image, (center_x + 6, center_y + 1), (6, 10), 0, 0, 360, orange, -1, cv2.LINE_AA)
    cv2.ellipse(image, (center_x, center_y + 1), (6, 11), 0, 0, 360, orange, -1, cv2.LINE_AA)
    cv2.ellipse(image, (center_x, center_y + 1), (12, 11), 0, 0, 360, dark_orange, 1, cv2.LINE_AA)
    cv2.rectangle(image, (center_x - 1, center_y - 15), (center_x + 2, center_y - 9),
                  (40, 140, 50), -1, cv2.LINE_AA)
    glow = (60, 235, 255)
    for dx in (-5, 5):
        eye = np.array([(center_x + dx - 3, center_y + 1), (center_x + dx + 3, center_y + 1),
                        (center_x + dx, center_y - 4)], dtype=np.int32)
        cv2.fillPoly(image, [eye], glow, cv2.LINE_AA)
    cv2.ellipse(image, (center_x, center_y + 5), (6, 4), 0, 0, 180, glow, 2, cv2.LINE_AA)


def _draw_target_marker(image, icon, center_x, center_y):
    if icon == "calaverita":
        _draw_calaverita_marker(image, center_x, center_y)
    elif icon == "ghost":
        _draw_ghost_marker(image, center_x, center_y)
    elif icon == "pumpkin":
        _draw_pumpkin_marker(image, center_x, center_y)
    else:
        _draw_pan_de_muerto_marker(image, center_x, center_y)


def _draw_hud(image, score, best_score, width, height, status_message, difficulty,
              activity_breakdown, tracking_confidence, challenge, challenge_elapsed,
              challenge_hold_elapsed, challenge_start_best_score, combo_seconds,
              achievement_message, achievement_until):
    colour = _aura_colour(score)
    cv2.rectangle(image, (20, 20), (390, 340), (18, 22, 42), -1)
    cv2.rectangle(image, (20, 20), (390, 340), colour, 2)
    cv2.putText(image, "AURA METER", (42, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(image, f"{score:05.1f} / 100", (42, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2)
    cv2.rectangle(image, (42, 122), (365, 143), (65, 65, 85), -1)
    cv2.rectangle(image, (42, 122), (42 + int(323 * score / 100), 143), colour, -1)
    cv2.putText(image, f"BEST {best_score:05.1f}", (42, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 230), 1)
    labels = (("left_arm", "L ARM"), ("right_arm", "R ARM"), ("torso", "TORSO"),
              ("head", "HEAD"), ("left_leg", "L LEG"), ("right_leg", "R LEG"))
    for index, (name, label) in enumerate(labels):
        y = 193 + index * 23
        region_score = activity_breakdown.get(name)
        text = "--" if region_score is None else f"{region_score:3.0f}"
        cv2.putText(image, f"{label} {text}", (42, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 230), 1)
        cv2.rectangle(image, (145, y - 11), (365, y - 3), (65, 65, 85), -1)
        if region_score is not None:
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
    cv2.putText(image, "M: DUEL  C: CHALLENGE", (width - 260, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 230), 1)
    if challenge == "sprint":
        time_left = max(0, int(SPRINT_CHALLENGE_SECONDS - challenge_elapsed))
        challenge_text = (
            f"SPRINT  0:{time_left:02d}  BEAT BEST {challenge_start_best_score:.1f}"
        )
        cv2.putText(image, challenge_text, (width - 370, 198),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 210, 120), 1)
    elif challenge == "hold":
        time_left = max(0, int(HOLD_CHALLENGE_LIMIT_SECONDS - challenge_elapsed))
        clock = f"{time_left // 60}:{time_left % 60:02d}"
        challenge_text = (
            f"HOLD  {challenge_hold_elapsed:.0f}/{HOLD_CHALLENGE_SECONDS:.0f}s"
            f"  {clock}"
        )
        cv2.putText(image, challenge_text, (width - 300, 198),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 210, 120), 1)
    if combo_seconds >= 5.0:
        cv2.putText(image, f"COMBO x{int(combo_seconds // 5)}  +{min(combo_seconds, 10.0):.0f}%",
                    (width - 245, height - 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (100, 235, 255), 2)
    if achievement_message and time.monotonic() < achievement_until:
        cv2.putText(image, achievement_message, (width - 300, height - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (120, 255, 150), 2)
    cv2.putText(image, "Keep moving to build your aura!", (width - 390, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)


def _draw_duel_hud(image, players, width, height, time_left=DUEL_DURATION_SECONDS):
    """Street Fighter-style bars: each fills from its screen edge toward a centre timer."""
    margin = 12
    box_w, box_h = 84, 52
    bar_h = 26
    bar_y = margin
    centre = width // 2
    bar_w = centre - box_w // 2 - margin - 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    colours = ((255, 170, 70), (90, 220, 255))

    for index, player in enumerate(players):
        x0 = margin if index == 0 else centre + box_w // 2 + 6
        x1 = x0 + bar_w
        fill = max(0, min(bar_w, int(bar_w * player.aura_score / 100)))
        cv2.rectangle(image, (x0 - 3, bar_y - 3), (x1 + 3, bar_y + bar_h + 3), (15, 15, 15), -1)
        cv2.rectangle(image, (x0, bar_y), (x1, bar_y + bar_h), (45, 40, 40), -1)
        if fill:
            fx0, fx1 = (x0, x0 + fill) if index == 0 else (x1 - fill, x1)
            colour = colours[index]
            cv2.rectangle(image, (fx0, bar_y), (fx1, bar_y + bar_h), colour, -1)
            shine = image[bar_y:bar_y + bar_h // 3, fx0:fx1]
            cv2.addWeighted(shine, 0.65, np.full_like(shine, 255), 0.35, 0, shine)
        target_x = (
            x0 + int(bar_w * player.target_score / 100)
            if index == 0
            else x1 - int(bar_w * player.target_score / 100)
        )
        _draw_target_marker(image, player.target_icon, target_x, bar_y + bar_h // 2)
        cv2.rectangle(image, (x0 - 3, bar_y - 3), (x1 + 3, bar_y + bar_h + 3), (240, 240, 240), 2)

        label = f"{player.name}  {player.aura_score:04.1f}"
        (label_w, _), _ = cv2.getTextSize(label, font, 0.55, 2)
        label_x = x0 if index == 0 else x1 - label_w
        cv2.putText(image, label, (label_x, bar_y + bar_h + 24), font, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, label, (label_x, bar_y + bar_h + 24), font, 0.55, colours[index], 2, cv2.LINE_AA)
        points_label = f"POINTS {player.points}   TARGET {player.target_score}"
        (points_w, _), _ = cv2.getTextSize(points_label, font, 0.45, 1)
        points_x = x0 if index == 0 else x1 - points_w
        cv2.putText(image, points_label, (points_x, bar_y + bar_h + 44),
                    font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        best = f"BEST {player.best_score:04.1f}"
        (best_w, _), _ = cv2.getTextSize(best, font, 0.45, 1)
        best_x = x0 if index == 0 else x1 - best_w
        cv2.putText(image, best, (best_x, bar_y + bar_h + 62), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, best, (best_x, bar_y + bar_h + 62), font, 0.45, (235, 235, 235), 2, cv2.LINE_AA)
        if player.combo_seconds >= 5.0:
            combo = f"COMBO x{int(player.combo_seconds // 5)}"
            combo_x = x0 if index == 0 else x1 - cv2.getTextSize(combo, font, 0.45, 1)[0][0]
            cv2.putText(image, combo, (combo_x, bar_y + bar_h + 80),
                        font, 0.45, colours[index], 1, cv2.LINE_AA)
        if player.achievement_message and time.monotonic() < player.achievement_until:
            achievement_x = x0 if index == 0 else x1 - cv2.getTextSize(
                player.achievement_message, font, 0.45, 1
            )[0][0]
            cv2.putText(image, player.achievement_message, (achievement_x, height - 35),
                        font, 0.45, colours[index], 1, cv2.LINE_AA)

    bx0 = centre - box_w // 2
    cv2.rectangle(image, (bx0, margin - 3), (bx0 + box_w, margin - 3 + box_h), (15, 15, 15), -1)
    cv2.rectangle(image, (bx0, margin - 3), (bx0 + box_w, margin - 3 + box_h), (240, 240, 240), 2)
    timer_text = f"{int(time_left) // 60}:{int(time_left) % 60:02d}"
    (text_w, text_h), _ = cv2.getTextSize(timer_text, font, 1.0, 2)
    cv2.putText(image, timer_text, (centre - text_w // 2, margin - 3 + (box_h + text_h) // 2),
                font, 1.0, (90, 90, 255) if time_left <= 10 else (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(image, "Q: end", (width - 60, height - 10), font, 0.4, (240, 240, 240), 1)
    cv2.putText(image, "M: switch to SOLO", (margin, height - 10), font, 0.4, (240, 240, 240), 1)
    cv2.putText(image, "REACH THE MARKER: +1 POINT", (centre - 130, height - 10),
                font, 0.4, (255, 255, 255), 1)


def _wrap_text(text, max_width, font, scale):
    """Split text into lines that fit within max_width pixels."""
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and cv2.getTextSize(candidate, font, scale, 1)[0][0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_game_over_screen(image, width, height, score, best_score, reason,
                           duel_result=None, challenge_summary=""):
    """Render the final summary when the session ends."""
    overlay = image.copy()
    cv2.rectangle(overlay, (80, 80), (width - 80, height - 40), (12, 14, 20), -1)
    cv2.rectangle(overlay, (80, 80), (width - 80, height - 40), (255, 120, 120), 2)
    cv2.putText(overlay, "SESSION ENDED", (110, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    if duel_result:
        cv2.putText(overlay, duel_result[0], (110, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 90), 2)
        cv2.putText(overlay, duel_result[1], (110, 255),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
        quote = duel_result[2] if len(duel_result) > 2 else None
        y = 295
        if quote:
            speaker, text = quote
            cv2.putText(overlay, f"{speaker}:", (110, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 90), 2)
            y += 28
            for line in _wrap_text(f'"{text}"', width - 220, cv2.FONT_HERSHEY_SIMPLEX, 0.6):
                cv2.putText(overlay, line, (110, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                y += 26
        reason_y = y + 14
        cv2.putText(overlay, reason, (110, reason_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        restart_y = reason_y + 34
    else:
        cv2.putText(overlay, f"FINAL AURA: {score:05.1f}", (110, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 90), 2)
        cv2.putText(overlay, f"BEST SCORE: {best_score:05.1f}", (110, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2)
        cv2.putText(overlay, reason, (110, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        restart_y = 390
    if challenge_summary:
        cv2.putText(overlay, challenge_summary, (110, restart_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 210, 120), 1)
        restart_y += 40
    cv2.putText(overlay, "Press R to restart.", (110, restart_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return overlay


def _prompt_player_names():
    """Read duel player names, using the standard labels for blank entries."""
    names = []
    for default_name in ("PLAYER 1", "PLAYER 2"):
        while True:
            try:
                name = input(f"{default_name} name (max 20 characters): ").strip()
            except EOFError:
                logger.warning("Could not read player names; using default names")
                return ["PLAYER 1", "PLAYER 2"]
            if not name:
                names.append(default_name)
                break
            if len(name) <= 20:
                names.append(name)
                break
            print("Names must be 20 characters or fewer.")
    return names


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
    candidates = []
    for detection in roi.get_objects_typed(hailo.HAILO_DETECTION):
        if detection.get_label() != "person":
            continue
        landmark_sets = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmark_sets or len(landmark_sets[0].get_points()) <= RIGHT_HIP:
            continue
        bbox = detection.get_bbox()
        center_x = (bbox.xmin() + bbox.xmax()) / 2.0
        candidates.append((detection, landmark_sets[0], center_x))

    pose_draw_data = []
    if user_data.mode == "duel":
        assignments = _assign_duel_detections(user_data.duel_players, candidates)
        for player, candidate in zip(user_data.duel_players, assignments):
            if candidate is None:
                _decay_player_score(player, frame_delta)
            else:
                pose, _ = _update_player_score(
                    player, candidate[0], frame_delta, user_data.difficulty, width, height
                )
                pose_draw_data.append((pose, (255, 170, 70) if player is user_data.duel_players[0] else (90, 220, 255)))
                user_data.best_score = max(user_data.best_score, player.best_score)
            _update_duel_target(player)
    else:
        user_data.last_pose = None
        if candidates:
            detection = max(candidates, key=lambda item: item[0].get_confidence())[0]
            pose, dance_score = _update_player_score(
                user_data, detection, frame_delta, user_data.difficulty, width, height
            )
            user_data.status_message = "Keep dancing!" if dance_score >= 35.0 else "Move to build aura"
            user_data.best_score = max(user_data.best_score, user_data.aura_score)
            pose_draw_data.append((pose, _aura_colour(user_data.aura_score)))
        else:
            _decay_player_score(user_data, frame_delta)
            user_data.status_message = "Pose not detected"

    output = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if user_data.game_state == "playing":
        for pose, colour in pose_draw_data:
            for start, end in POSE_CONNECTIONS:
                cv2.line(output, pose[start], pose[end], colour, 3, cv2.LINE_AA)
            for point in pose:
                cv2.circle(output, point, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(output, point, 8, colour, 2, cv2.LINE_AA)

    if user_data.game_state == "playing" and user_data.mode == "duel":
        user_data.duel_elapsed += min(frame_delta, 0.5)
        time_left = max(0.0, DUEL_DURATION_SECONDS - user_data.duel_elapsed)
        _draw_duel_hud(output, user_data.duel_players, width, height, time_left)
        if time_left <= 0.0:
            user_data.trigger_game_over("Time's up!")
            output = _draw_game_over_screen(
                output, width, height, user_data.final_score, user_data.best_score,
                user_data.status_message, user_data.final_duel_result,
            )
    elif user_data.game_state == "playing":
        if user_data.challenge != "none":
            user_data.challenge_elapsed += min(frame_delta, 0.5)
            user_data.challenge_peak_score = max(
                user_data.challenge_peak_score, user_data.aura_score
            )
            if user_data.challenge == "sprint" and (
                user_data.challenge_elapsed >= SPRINT_CHALLENGE_SECONDS
            ):
                if user_data.challenge_peak_score > user_data.challenge_start_best_score + 0.1:
                    user_data.challenge_summary = "NEW PERSONAL BEST - CHALLENGE COMPLETE!"
                    reason = "Challenge complete!"
                else:
                    user_data.challenge_summary = "BEST NOT BEAT - TRY AGAIN!"
                    reason = "Sprint challenge ended."
                user_data.trigger_game_over(reason)
            elif user_data.challenge == "hold":
                if user_data.aura_score >= 70.0:
                    user_data.challenge_hold_elapsed += min(frame_delta, 0.5)
                else:
                    user_data.challenge_hold_elapsed = 0.0
                if user_data.challenge_hold_elapsed >= HOLD_CHALLENGE_SECONDS:
                    user_data.challenge_summary = "20-SECOND AURA HOLD COMPLETE!"
                    user_data.trigger_game_over("Challenge complete!")
                elif user_data.challenge_elapsed >= HOLD_CHALLENGE_LIMIT_SECONDS:
                    user_data.challenge_summary = "HOLD NOT COMPLETED - TRY AGAIN!"
                    user_data.trigger_game_over("Hold challenge ended.")

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
                user_data.challenge,
                user_data.challenge_elapsed,
                user_data.challenge_hold_elapsed,
                user_data.challenge_start_best_score,
                user_data.combo_seconds,
                user_data.achievement_message,
                user_data.achievement_until,
            )
        else:
            output = _draw_game_over_screen(
                output, width, height, user_data.final_score, user_data.best_score,
                user_data.status_message, challenge_summary=user_data.challenge_summary,
            )
    else:
        output = _draw_game_over_screen(
            output,
            width,
            height,
            user_data.final_score,
            user_data.best_score,
            user_data.status_message,
            user_data.final_duel_result,
            user_data.challenge_summary,
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

    def __init__(self, user_data, stop_app):
        super().__init__(daemon=True)
        self.user_data = user_data
        self.stop_app = stop_app
        self.running = True
        self.terminal_fd = None
        self.original_terminal_settings = None

    def _prompt_duel_names(self):
        if self.original_terminal_settings is None:
            return _prompt_player_names()
        termios.tcsetattr(
            self.terminal_fd, termios.TCSADRAIN, self.original_terminal_settings
        )
        try:
            return _prompt_player_names()
        finally:
            if self.running:
                tty.setcbreak(self.terminal_fd)

    def _handle_key(self, key):
        if key in ("d", "D"):
            self.user_data.cycle_difficulty()
        elif key in ("m", "M"):
            player_names = (
                self._prompt_duel_names()
                if self.user_data.mode == "solo"
                else None
            )
            self.user_data.toggle_mode(player_names)
        elif key in ("c", "C"):
            if self.user_data.mode == "solo":
                self.user_data.cycle_challenge()
        elif key in ("r", "R"):
            self.user_data.restart_game()
        elif key in ("q", "Q"):
            if self.user_data.game_state == "playing":
                self.user_data.trigger_game_over("Session ended")
            else:
                self.running = False
                self.stop_app()

    def run(self):
        if hasattr(sys.stdin, "isatty") and not sys.stdin.isatty():
            logger.warning("Keyboard controls unavailable: run the game from an interactive terminal")
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
        self.terminal_fd = fd
        self.original_terminal_settings = old_settings
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
    parser.add_argument(
        "--mode",
        choices=["solo", "duel"],
        default="solo",
        help="Play solo or start a two-player best-score duel.",
    )
    parser.add_argument(
        "--challenge",
        choices=["none", "sprint", "hold"],
        default="none",
        help="Start a solo challenge: beat your best in 30 seconds or hold 70 aura for 20 seconds.",
    )
    args = parser.parse_args()
    if args.mode == "duel" and args.challenge != "none":
        parser.error("--challenge can only be used with --mode solo.")

    user_data = AuraGameCallback(mode=args.mode, challenge=args.challenge)
    if args.mode == "duel" and sys.stdin.isatty():
        for player, name in zip(user_data.duel_players, _prompt_player_names()):
            player.name = name
    user_data.difficulty = args.difficulty
    app = AuraPoseGame(app_callback, user_data, parser)
    keyboard = KeyboardController(user_data, app.shutdown)
    keyboard.start()

    def handle_sigint(_signal_number, _frame):
        logger.info("Stopping aura pose game")
        keyboard.running = False
        app.shutdown()

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        app.run()
    finally:
        keyboard.running = False
        keyboard.join(timeout=1.0)
        _save_best_score(user_data.best_score)


if __name__ == "__main__":
    main()
