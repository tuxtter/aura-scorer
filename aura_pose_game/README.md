# Aura Pose Game

## Description

Aura Pose Game turns a person's pose into a live score from 0 to 100. It is
based on the pose-estimation pipeline used by `easter_game.py`, but keeps the
camera feed visible and overlays a pose skeleton and aura meter.

The score combines:

- Arm span and shoulder width for pose openness
- Torso openness between the shoulders and hips
- Left/right balance for symmetry

Scores are smoothed to avoid flickering and the best score is retained while
the app runs.

## Requirements

Use a Hailo device with the pose-estimation model resources installed and a
working camera or video source.

## Usage

```bash
python3 aura_pose_game.py --input usb
```

The standard pipeline options are also available, including `--input`,
`--hef-path`, `--show-fps`, and `--disable-sync`.

## Controls

Stand in view of the camera, spread your arms, and hold a balanced pose. The
meter changes from "BUILD AURA" to "STRONG AURA" and finally "MAXIMUM AURA".
