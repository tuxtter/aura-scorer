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

Standard solo mode, when challenges are off, is one continuous session with no
countdowns or time limits. Losing pose detection does not end the session; the
score fades until movement is detected again. The best score is retained while
the app runs and saved for future launches, including on normal application
shutdown.

Run a two-player target-scoring challenge with `--mode duel`. The two players
are assigned by screen position (left and right). A pan de muerto, calaverita
de azucar, ghost or pumpkin icon (chosen at random) marks a random target on each player's aura
bar; reaching it awards one point and sets
a new target. The live point totals and current targets are shown below the
bars.
The player with the most points wins when the 90-second match timer reaches
zero (or earlier if ended with `Q`). The countdown is shown at the top of the
screen and turns red for the last 10 seconds. Enter player names when starting
in duel mode or switching to duel with `M`; leave a name blank to use its
default. The end screen shows the point totals and each player's best aura. The winner also delivers a
Street Fighter-style victory line to the rival that references their aura.

Start a timed solo challenge:

```bash
python3 aura_pose_game.py --input usb --challenge sprint
python3 aura_pose_game.py --input usb --challenge hold
```

Sprint challenges give you 30 seconds to beat your saved personal best. Hold
challenges give you 60 seconds to keep your aura score at 70 or higher for 20
continuous seconds. Press `C` during solo play to cycle through challenges or
turn challenges off; changing the challenge starts a fresh run.

Sustained movement across at least two body regions builds a combo. Combo tiers
increase every five seconds, and combos add a 1%-per-second aura-score bonus
capped at 10%. Moving four or more regions unlocks a one-time "FULL BODY"
achievement for the session; the first five-second combo and reaching 85 aura
also trigger achievements.

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
- `C`: cycle solo challenge (off, sprint, hold)
- `M`: switch between solo and duel mode (starts a fresh run)
- `R`: restart the session (resets the 90-second duel timer)
- `Q`: end the session and show the result; press `Q` again to exit
