# Project Vista

## Offline Environmental Description System

Project Vista is an offline environmental awareness system designed to describe a user's surroundings using a camera and a lightweight vision-language model.

The current prototype uses:

- Raspberry Pi 4B 4GB as the target device
- Raspberry Pi Camera Module 3 for the final hardware
- Python 3.11
- SmolVLM-500M-Instruct
- llama.cpp / llama CLI for local model inference
- OpenCV for video and camera processing
- pyttsx3 for offline text-to-speech

The system can analyze either a video file or a live camera feed.

---

# Features

Vista can:

- Analyze prerecorded video files
- Analyze a live camera
- Sample multiple frames from a 20-second period
- Describe objects, people, animals and visible surroundings
- Remove repeated observations
- Combine observations into a short environmental summary
- Read the summary aloud using offline text-to-speech
- Run locally without sending images to an online AI service

---

# Current Architecture

```text
Camera / Video
      |
      v
   OpenCV
      |
      v
16 sampled frames
      |
      v
   SmolVLM
      |
      v
Individual observations
      |
      v
Duplicate removal
      |
      v
Accumulated summary
      |
      v
    pyttsx3
      |
      v
   Spoken output
