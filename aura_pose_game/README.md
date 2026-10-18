# Aura Pose Game

## Description

Aura Pose Game uses a pose-estimation pipeline to track a dancer and turn
movement into a live score from 0 to 100. It keeps the camera feed visible and
overlays the detected skeleton and aura meter. There are no named target poses.

The score rewards movement intensity and involving multiple body regions
(left arm, right arm, torso, plus head and legs at lower weight). Head and
leg regions only count while their keypoints are confidently visible, so
upper-body-only framing is not penalized (the HUD shows `--` for them). It is smoothed over time, so sustained,
varied dancing builds aura while standing still lets the score fade. Detection
confidence and landmark visibility also affect the score. Difficulty changes
how much movement is needed to reach a high score. The HUD shows per-region
activity and tracking confidence. Landmark smoothing and a short adaptive
noise calibration reduce score changes caused by tracking jitter; movement is
normalized to the detected person size.

The game is one continuous session with no countdowns or time limits. Losing
pose detection does not end the session; the score fades until movement is
detected again. The best score is retained while the app runs and saved for
future launches, including on normal application shutdown.

Run a two-player best-score challenge with `--mode duel`. The two players are
assigned by screen position (left and right); their best aura scores are
compared when the match is ended with `Q`.

## Requirements

Use a Hailo device with the pose-estimation model resources installed and a
working camera or video source.

## Usage

```bash
python3 aura_pose_game.py --input usb
```

Start a 1-vs-1 match:

```bash
python3 aura_pose_game.py --input usb --mode duel
```

The standard pipeline options are also available, including `--input`,
`--hef-path`, `--show-fps`, and `--disable-sync`.

## Controls

Stand in view of the camera and keep dancing to build your aura. The meter
changes from "BUILD AURA" to "STRONG AURA" and finally "MAXIMUM AURA".

- `D`: cycle difficulty
- `R`: restart the session
- `Q`: end the session (in duel mode, logs the winner or a tie)
