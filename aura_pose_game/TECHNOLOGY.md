# The Technology Behind Aura Pose Game

Aura Pose Game watches you dance through a camera and gives you a score based
on how much you move. This page explains, in plain language, the three pieces
of technology that make that possible.

## The big picture

1. A **camera** films you.
2. **Pose estimation** (a kind of artificial intelligence) finds your body in
   each picture, including your arms, legs and head.
3. The **Hailo-8L** chip does that heavy thinking very quickly.
4. The **Raspberry Pi 5** is the small computer that runs everything, turns the
   movement into a score and draws the game on the screen.

All of this happens many times every second, so the game reacts as you move.

## Pose estimation: teaching a computer to see a body

Pose estimation is a computer skill. Given a picture of a person, it works out
where their body parts are: shoulders, elbows, wrists, hips, knees, ankles, and
so on.

Think of it like the stick figure you might draw of a person. The computer
places a dot on each joint and connects the dots with lines. That is the
colourful skeleton you see drawn over yourself in the game.

The computer learned to do this by studying a huge number of example photos of
people in many positions, until it became good at spotting bodies it had never
seen before. This is what people mean by "artificial intelligence" or "machine
learning".

**What the game does with it:** the game never judges how you look. It only
checks how much those dots move from one moment to the next. Big, lively
movement with several body parts (both arms, your torso, and your legs when
they are visible) earns more aura. Standing still lets your score slowly fade.

**Good to know:**
- The game follows body *positions*, not your face or identity.
- Two people can play at once in duel mode. The game tells them apart by who
  stands on the left and who stands on the right.

## Hailo-8L: a chip built for AI

Running pose estimation needs a lot of calculations, and doing them for every
camera picture is hard work for an ordinary computer chip.

The Hailo-8L is a small add-on chip designed to do just this kind of AI work.
You can think of it as a specialist:

- A normal computer chip (the CPU) is like a general helper who can do any job,
  but not super fast at any single one.
- The Hailo-8L is like a person who only ever does one job, spotting bodies in
  pictures, so they do it extremely quickly and without breaking a sweat.

Because the Hailo-8L handles the AI part, the rest of the computer is free to
run the game, play sounds and draw the screen smoothly. It also uses very
little power, so it stays cool and needs no big cooling system.

## Raspberry Pi 5: the small computer that runs it all

The Raspberry Pi 5 is a tiny, low-cost computer about the size of a credit
card. It is popular with students, hobbyists and makers.

In this project it acts as the "brain" of the setup:

- It receives the video from the camera.
- It sends each picture to the Hailo-8L to find the body.
- It takes the results and works out your aura score.
- It draws the score bars, the timer and the skeleton on the screen.
- It plays the sound effects and remembers your best score.

The Hailo-8L chip connects to the Raspberry Pi 5 through a special slot on the
board. Together they make a small, affordable, low-power system that can do
something that used to need a much bigger computer.

## Why this combination?

| Piece | Its job | Everyday comparison |
|-------|---------|---------------------|
| Camera | Captures what is happening | The eyes |
| Pose estimation | Finds the body in the picture | Recognising a person |
| Hailo-8L | Does the AI work fast | A specialist who only does one job |
| Raspberry Pi 5 | Runs the game and the screen | The coordinator |

Everything runs on the device itself. The game does not need the internet to
work, and the camera pictures are not sent anywhere.

## Want to try it?

See the [README](./README.md) for how to start the game.
