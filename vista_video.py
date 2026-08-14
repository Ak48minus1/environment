import argparse
import base64
import glob
import os
import subprocess
import sys
import time

import cv2
import pyttsx3
import requests


# ============================================================
# VISTA CONFIGURATION
# ============================================================

LLAMA_BIN = os.path.expanduser(
    "~/.llama-app/llama"
)

MODEL_REPO = (
    "ggml-org/SmolVLM-500M-Instruct-GGUF"
)

MODEL_NAME = (
    "SmolVLM-500M-Instruct-Q8_0.gguf"
)

MMPROJ_NAME = (
    "mmproj-SmolVLM-500M-Instruct-f16.gguf"
)

MODEL_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    MODEL_NAME
)

MMPROJ_PATH = os.path.join(
    MODEL_DIR,
    MMPROJ_NAME
)

LLAMA_URL = (
    "http://127.0.0.1:8080"
)

VIDEO_LENGTH = 20

NUMBER_OF_FRAMES = 16

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

MAX_DESCRIPTION_LENGTH = 180

MAX_SUMMARY_WORDS = 70


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Vista - Offline Environmental "
            "Description System"
        )
    )

    source_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    source_group.add_argument(
        "--camera",
        type=int,
        metavar="INDEX",
        help=(
            "Use a camera. "
            "Example: --camera 0"
        )
    )

    source_group.add_argument(
        "--video",
        type=str,
        metavar="FILE",
        help=(
            "Analyze a video file. "
            "Example: --video test.mp4"
        )
    )

    return parser.parse_args()


# ============================================================
# DOWNLOAD MODELS
# ============================================================

def download_models():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_exists = os.path.exists(
        MODEL_PATH
    )

    mmproj_exists = os.path.exists(
        MMPROJ_PATH
    )

    if model_exists and mmproj_exists:

        print(
            "Models already downloaded."
        )

        return

    print()
    print("=" * 60)
    print("VISTA MODEL SETUP")
    print("=" * 60)

    if not os.path.exists(
        LLAMA_BIN
    ):

        print()
        print(
            "ERROR: Llama executable "
            "was not found."
        )

        print()
        print("Expected:")
        print(LLAMA_BIN)

        print()
        print(
            "Install llama first."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    if not model_exists:

        print()
        print(
            "Downloading SmolVLM model..."
        )

        print(
            "This is approximately 417 MB."
        )

        print()

        command = [
            LLAMA_BIN,
            "download",
            "-hf",
            MODEL_REPO,
            "-hff",
            MODEL_NAME,
        ]

        result = subprocess.run(
            command
        )

        if result.returncode != 0:

            print(
                "ERROR: Model download failed."
            )

            sys.exit(1)

        cache_pattern = os.path.expanduser(
            "~/.cache/huggingface/hub/"
            "models--ggml-org--SmolVLM-500M-Instruct-GGUF/"
            "snapshots/*/"
            + MODEL_NAME
        )

        matches = glob.glob(
            cache_pattern
        )

        if not matches:

            print(
                "ERROR: Model downloaded "
                "but could not be located."
            )

            sys.exit(1)

        subprocess.run(
            [
                "cp",
                matches[0],
                MODEL_PATH
            ]
        )

    # --------------------------------------------------------
    # MULTIMODAL PROJECTOR
    # --------------------------------------------------------

    if not mmproj_exists:

        print()
        print(
            "Downloading multimodal projector..."
        )

        print(
            "This is approximately 191 MB."
        )

        print()

        command = [
            LLAMA_BIN,
            "download",
            "-hf",
            MODEL_REPO,
            "-hff",
            MMPROJ_NAME,
        ]

        result = subprocess.run(
            command
        )

        if result.returncode != 0:

            print(
                "ERROR: Projector download failed."
            )

            sys.exit(1)

        cache_pattern = os.path.expanduser(
            "~/.cache/huggingface/hub/"
            "models--ggml-org--SmolVLM-500M-Instruct-GGUF/"
            "snapshots/*/"
            + MMPROJ_NAME
        )

        matches = glob.glob(
            cache_pattern
        )

        if not matches:

            print(
                "ERROR: Projector downloaded "
                "but could not be located."
            )

            sys.exit(1)

        subprocess.run(
            [
                "cp",
                matches[0],
                MMPROJ_PATH
            ]
        )

    print()
    print(
        "Models downloaded successfully."
    )


# ============================================================
# CHECK LLAMA SERVER
# ============================================================

def llama_running():

    try:

        response = requests.get(
            LLAMA_URL + "/health",
            timeout=2
        )

        return (
            response.status_code == 200
        )

    except requests.RequestException:

        return False


# ============================================================
# START LLAMA SERVER
# ============================================================

def start_llama():

    if llama_running():

        print(
            "Llama server already running."
        )

        return None

    print()
    print("=" * 60)
    print("STARTING LOCAL AI MODEL")
    print("=" * 60)

    command = [
        LLAMA_BIN,
        "serve",
        "-m",
        MODEL_PATH,
        "--mmproj",
        MMPROJ_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print(
        "Starting SmolVLM..."
    )

    for _ in range(60):

        if llama_running():

            print(
                "SmolVLM is ready."
            )

            return process

        time.sleep(1)

    print(
        "ERROR: Llama server did not start."
    )

    process.terminate()

    sys.exit(1)


# ============================================================
# RESIZE FRAME WITHOUT DISTORTION
# ============================================================

def resize_frame(
    frame
):

    height, width = frame.shape[:2]

    if width <= 0 or height <= 0:

        return frame

    scale = min(
        CAMERA_WIDTH / width,
        CAMERA_HEIGHT / height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    resized = cv2.resize(
        frame,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Place the resized frame on a black canvas.
    # This preserves the original aspect ratio.
    # --------------------------------------------------------

    canvas = (
        __import__("numpy")
        .zeros(
            (
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                3
            ),
            dtype="uint8"
        )
    )

    x = (
        CAMERA_WIDTH -
        new_width
    ) // 2

    y = (
        CAMERA_HEIGHT -
        new_height
    ) // 2

    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized

    return canvas


# ============================================================
# CAMERA CAPTURE
# ============================================================

def capture_camera(
    camera_index
):

    print()
    print("=" * 60)
    print(
        "CAPTURING 20 SECOND CAMERA SCAN"
    )
    print("=" * 60)

    cap = cv2.VideoCapture(
        camera_index
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return []

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    frames = []

    frame_interval = (
        VIDEO_LENGTH /
        NUMBER_OF_FRAMES
    )

    start_time = time.time()

    next_capture = 0

    print()
    print(
        "Recording..."
    )

    while (
        time.time() - start_time
        < VIDEO_LENGTH
    ):

        ret, frame = cap.read()

        if not ret:

            continue

        elapsed = (
            time.time() -
            start_time
        )

        if (
            elapsed >= next_capture
            and
            len(frames) <
            NUMBER_OF_FRAMES
        ):

            frame = resize_frame(
                frame
            )

            frames.append(
                frame.copy()
            )

            print(
                f"Captured frame "
                f"{len(frames)}/"
                f"{NUMBER_OF_FRAMES}"
            )

            next_capture += (
                frame_interval
            )

    cap.release()

    print()
    print(
        "Capture complete."
    )

    return frames


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_information(
    video_path
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        return None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    width = cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )

    height = cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )

    cap.release()

    if fps <= 0:

        return None

    duration = (
        frame_count / fps
    )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(
    seconds
):

    seconds = int(
        max(
            0,
            seconds
        )
    )

    minutes = (
        seconds // 60
    )

    seconds = (
        seconds % 60
    )

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# CAPTURE VIDEO SEGMENT
# ============================================================

def capture_video_segment(
    video_path,
    start_time,
    segment_length=VIDEO_LENGTH
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return []

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0

    total_frames = cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    total_duration = (
        total_frames / fps
    )

    end_time = min(
        start_time +
        segment_length,
        total_duration
    )

    actual_duration = (
        end_time -
        start_time
    )

    if actual_duration <= 0:

        cap.release()

        return []

    print()
    print(
        f"Processing video from "
        f"{format_time(start_time)} "
        f"to "
        f"{format_time(end_time)}"
    )

    # --------------------------------------------------------
    # Calculate frame timestamps.
    #
    # We deliberately do NOT request the exact final
    # timestamp because OpenCV can fail at the end of
    # some video files.
    # --------------------------------------------------------

    frame_times = []

    for i in range(
        NUMBER_OF_FRAMES
    ):

        timestamp = (
            start_time
            +
            (
                i *
                actual_duration /
                NUMBER_OF_FRAMES
            )
        )

        timestamp = min(
            timestamp,
            max(
                start_time,
                end_time - 0.05
            )
        )

        frame_times.append(
            timestamp
        )

    frames = []

    print()
    print(
        "Sampling frames..."
    )

    for index, timestamp in enumerate(
        frame_times
    ):

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        ret, frame = cap.read()

        if not ret:

            print(
                f"Could not read frame "
                f"{index + 1}/"
                f"{NUMBER_OF_FRAMES}"
            )

            continue

        frame = resize_frame(
            frame
        )

        frames.append(
            frame.copy()
        )

        print(
            f"Captured frame "
            f"{index + 1}/"
            f"{NUMBER_OF_FRAMES}"
            f"  "
            f"({format_time(timestamp)})"
        )

    cap.release()

    print()
    print(
        "Video segment captured."
    )

    return frames


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(
    frame
):

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            75
        ]
    )

    if not success:

        return None

    return base64.b64encode(
        encoded.tobytes()
    ).decode(
        "utf-8"
    )


# ============================================================
# ANALYZE ONE FRAME
# ============================================================

def analyze_frame(
    frame
):

    image_data = (
        image_to_base64(
            frame
        )
    )

    if image_data is None:

        return None

    prompt = """
You are an environmental observation assistant
for a visually impaired person.

Look carefully at this camera frame.

Report ONLY clearly visible information.

Focus on:

- people
- animals
- objects
- obstacles
- clearly readable signs or text
- doors
- furniture
- vehicles
- roads
- buildings
- surroundings
- visible actions

IMPORTANT:

Do not identify people by name.

Do not guess age.

Do not guess emotions.

Do not guess location.

Do not guess intentions.

Do not guess professions.

Do not guess an object if it is unclear.

Only describe things that are clearly visible.

Use ONE short factual sentence.

Maximum approximately 25 words.

Do not say "the image shows".

Do not say "I can see".

Return ONLY the observation.
"""

    payload = {

        "model": MODEL_NAME,

        "messages": [

            {
                "role": "user",

                "content": [

                    {
                        "type": "text",
                        "text": prompt
                    },

                    {
                        "type": "image_url",

                        "image_url": {

                            "url":
                            (
                                "data:image/jpeg;base64,"
                                +
                                image_data
                            )

                        }

                    }

                ]

            }

        ],

        "max_tokens": 60,

        "temperature": 0.05,

        "stream": False
    }

    try:

        response = requests.post(
            LLAMA_URL +
            "/v1/chat/completions",
            json=payload,
            timeout=60
        )

        if response.status_code != 200:

            print(
                "LLAMA ERROR:",
                response.status_code
            )

            return None

        data = response.json()

        text = data[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]

        text = text.strip()

        if (
            len(text)
            >
            MAX_DESCRIPTION_LENGTH
        ):

            text = text[
                :MAX_DESCRIPTION_LENGTH
            ]

        return text

    except Exception as e:

        print(
            "Frame analysis error:",
            e
        )

        return None


# ============================================================
# CLEAN OBSERVATIONS
# ============================================================

def clean_observations(
    observations
):

    cleaned = []

    bad_phrases = [

        "i cannot",
        "i can't",
        "as an ai",
        "the image shows",
        "this image shows",
        "i am unable",
        "i don't know"

    ]

    for observation in observations:

        if not observation:

            continue

        observation = (
            observation.strip()
        )

        if not observation:

            continue

        lower = (
            observation.lower()
        )

        if any(
            phrase in lower
            for phrase in bad_phrases
        ):

            continue

        duplicate = False

        current_words = set(
            lower.split()
        )

        for previous in cleaned:

            previous_words = set(
                previous.lower().split()
            )

            if (
                lower ==
                previous.lower()
            ):

                duplicate = True

                break

            if (
                len(current_words) >= 5
                and
                len(previous_words) >= 5
            ):

                intersection = (
                    current_words &
                    previous_words
                )

                similarity = (
                    len(intersection)
                    /
                    min(
                        len(current_words),
                        len(previous_words)
                    )
                )

                if similarity >= 0.70:

                    duplicate = True

                    break

        if not duplicate:

            cleaned.append(
                observation
            )

    return cleaned


# ============================================================
# CREATE FALLBACK SUMMARY
# ============================================================

def create_fallback_summary(
    observations
):

    if not observations:

        return (
            "I could not identify any "
            "clear objects or events."
        )

    text = " ".join(
        observations
    ).lower()

    # --------------------------------------------------------
    # Common subjects.
    # --------------------------------------------------------

    subjects = [

        "cat",
        "dog",
        "person",
        "man",
        "woman",
        "child",
        "bird",
        "car",
        "vehicle",
        "bicycle",
        "motorcycle",
        "chair",
        "table"

    ]

    main_subject = None

    for subject in subjects:

        if subject in text:

            main_subject = subject

            break

    # --------------------------------------------------------
    # Actions.
    # --------------------------------------------------------

    actions = [

        "lying",
        "sitting",
        "standing",
        "walking",
        "running",
        "moving",
        "driving",
        "holding",
        "talking",
        "looking"

    ]

    action = None

    for candidate in actions:

        if candidate in text:

            action = candidate

            break

    # --------------------------------------------------------
    # Locations / environments.
    # --------------------------------------------------------

    environments = [

        "tiled floor",
        "tile floor",
        "floor",
        "road",
        "street",
        "sidewalk",
        "grass",
        "room",
        "office",
        "desk",
        "chair"

    ]

    environment = None

    for candidate in environments:

        if candidate in text:

            environment = candidate

            break

    # --------------------------------------------------------
    # Accessories.
    # --------------------------------------------------------

    accessories = [

        "collar",
        "tag",
        "bell",
        "glasses",
        "backpack"

    ]

    accessory = None

    for candidate in accessories:

        if candidate in text:

            accessory = candidate

            break

    # --------------------------------------------------------
    # Build concise fallback.
    # --------------------------------------------------------

    if (
        main_subject
        and action
        and environment
        and accessory
    ):

        return (
            f"A {main_subject} is "
            f"{action} on a "
            f"{environment}, with a "
            f"{accessory}. The scene "
            f"remains mostly unchanged."
        )

    if (
        main_subject
        and action
        and environment
    ):

        return (
            f"A {main_subject} is "
            f"{action} on a "
            f"{environment}. The scene "
            f"remains mostly unchanged."
        )

    if (
        main_subject
        and action
    ):

        return (
            f"A {main_subject} is "
            f"{action}. The scene "
            f"remains mostly unchanged."
        )

    return observations[0]


# ============================================================
# CREATE ACCUMULATIVE SUMMARY
# ============================================================

def create_summary(
    observations
):

    if not observations:

        return (
            "I could not identify any "
            "clear objects or events."
        )

    observations_text = "\n".join(
        "- " + observation
        for observation in observations
    )

    prompt = f"""
You are Vista, an environmental assistant
for a visually impaired person.

These are observations from different frames
of THE SAME scene:

{observations_text}

Create ONE short description of the scene.

Think about all observations together.

IMPORTANT:

- Identify the MAIN subject.
- State what the main subject is doing.
- Mention important objects associated with it.
- Mention meaningful changes only.
- Ignore repeated descriptions.
- Do NOT repeat the same subject multiple times.
- Do NOT simply concatenate the sentences.
- Do NOT invent anything.
- Do NOT guess identity, age, location,
  emotion or intention.
- Do NOT mention frame numbers.
- Do NOT mention the AI.
- Do NOT mention the camera.
- Do NOT mention the analysis.
- Use simple language suitable for speech.
- If the scene is mostly unchanged,
  say that briefly.

If the same subject appears in almost every
observation, describe it ONLY ONCE.

Return EXACTLY ONE sentence.

Maximum 35 words.

OBSERVATIONS:

{observations_text}
"""

    payload = {

        "model": MODEL_NAME,

        "messages": [

            {
                "role": "user",
                "content": prompt
            }

        ],

        "max_tokens": 60,

        "temperature": 0.0,

        "stream": False
    }

    try:

        response = requests.post(
            LLAMA_URL +
            "/v1/chat/completions",
            json=payload,
            timeout=60
        )

        if response.status_code != 200:

            print(
                "Summary error:",
                response.status_code
            )

            return create_fallback_summary(
                observations
            )

        data = response.json()

        summary = data[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]

        summary = summary.strip()

        summary = summary.strip(
            "\"'"
        )

        words = summary.split()

        # ----------------------------------------------------
        # Reject unusually long output.
        # ----------------------------------------------------

        if len(words) > 35:

            return create_fallback_summary(
                observations
            )

        # ----------------------------------------------------
        # Detect repeated main subjects.
        # ----------------------------------------------------

        repeated_subjects = [

            "cat",
            "dog",
            "person",
            "man",
            "woman",
            "child",
            "car",
            "vehicle",
            "bird",
            "chair"

        ]

        lower_words = [
            word.lower().strip(
                ".,!?;:"
            )
            for word in words
        ]

        for subject in repeated_subjects:

            if (
                lower_words.count(
                    subject
                )
                >=
                3
            ):

                return create_fallback_summary(
                    observations
                )

        if not summary:

            return create_fallback_summary(
                observations
            )

        return summary

    except Exception as e:

        print(
            "Summary error:",
            e
        )

        return create_fallback_summary(
            observations
        )


# ============================================================
# TEXT TO SPEECH
# ============================================================

def speak(
    text
):

    print()
    print(
        "Speaking..."
    )

    print(
        text
    )

    print()

    try:

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            155
        )

        engine.setProperty(
            "volume",
            1.0
        )

        engine.say(
            text
        )

        engine.runAndWait()

        engine.stop()

    except Exception as e:

        print(
            "TTS ERROR:",
            e
        )


# ============================================================
# ANALYZE FRAMES
# ============================================================

def analyze_frames(
    frames
):

    if not frames:

        return None

    print()
    print("=" * 60)
    print(
        "ANALYZING FRAMES"
    )
    print("=" * 60)

    observations = []

    for index, frame in enumerate(
        frames
    ):

        print()
        print(
            f"Frame "
            f"{index + 1}/"
            f"{len(frames)}:"
        )

        description = analyze_frame(
            frame
        )

        if description:

            print(
                description
            )

            observations.append(
                description
            )

        else:

            print(
                "No reliable description."
            )

    observations = (
        clean_observations(
            observations
        )
    )

    print()
    print("=" * 60)
    print(
        "UNIQUE OBSERVATIONS"
    )
    print("=" * 60)

    if observations:

        for observation in observations:

            print(
                "-",
                observation
            )

    else:

        print(
            "No reliable observations."
        )

    print()
    print("=" * 60)
    print(
        "CREATING ACCUMULATIVE SUMMARY"
    )
    print("=" * 60)

    summary = create_summary(
        observations
    )

    print()
    print("=" * 60)
    print(
        "VISTA SUMMARY"
    )
    print("=" * 60)

    print(
        summary
    )

    speak(
        summary
    )

    return summary


# ============================================================
# CAMERA MODE
# ============================================================

def run_camera_mode(
    camera_index
):

    print()
    print(
        "MODE: CAMERA"
    )

    while True:

        frames = capture_camera(
            camera_index
        )

        if not frames:

            print(
                "No frames captured."
            )

            time.sleep(2)

            continue

        analyze_frames(
            frames
        )

        print()
        print(
            "Starting next "
            "20-second camera scan..."
        )


# ============================================================
# VIDEO MODE
# ============================================================

def run_video_mode(
    video_path
):

    if not os.path.isfile(
        video_path
    ):

        print()
        print(
            "ERROR: Video file "
            "does not exist:"
        )

        print(
            video_path
        )

        sys.exit(1)

    information = (
        get_video_information(
            video_path
        )
    )

    if information is None:

        print()
        print(
            "ERROR: Could not read video."
        )

        sys.exit(1)

    duration = information[
        "duration"
    ]

    print()
    print(
        "MODE: VIDEO FILE"
    )

    print()
    print(
        "Video:",
        video_path
    )

    print(
        "Resolution:",
        f"{int(information['width'])}x"
        f"{int(information['height'])}"
    )

    print(
        "FPS:",
        f"{information['fps']:.2f}"
    )

    print(
        "Duration:",
        format_time(duration)
    )

    print()

    # --------------------------------------------------------
    # Process video in 20-second segments.
    # --------------------------------------------------------

    segment_start = 0.0

    segment_number = 1

    while (
        segment_start < duration
    ):

        segment_end = min(
            segment_start +
            VIDEO_LENGTH,
            duration
        )

        print()
        print("=" * 60)
        print(
            f"VIDEO SEGMENT "
            f"{segment_number}"
        )

        print(
            f"{format_time(segment_start)}"
            " -> "
            f"{format_time(segment_end)}"
        )

        print("=" * 60)

        frames = (
            capture_video_segment(
                video_path,
                segment_start,
                VIDEO_LENGTH
            )
        )

        if frames:

            analyze_frames(
                frames
            )

        else:

            print(
                "No frames captured "
                "from this segment."
            )

        segment_start += (
            VIDEO_LENGTH
        )

        segment_number += 1

        if (
            segment_start >=
            duration
        ):

            break

        print()
        print(
            "Moving to next "
            "20-second video segment..."
        )

    print()
    print("=" * 60)
    print(
        "VIDEO ANALYSIS COMPLETE"
    )
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    print()
    print("=" * 60)
    print(
        "VISTA"
    )
    print(
        "Offline Environmental "
        "Description System"
    )
    print("=" * 60)

    download_models()

    llama_process = (
        start_llama()
    )

    try:

        if args.camera is not None:

            run_camera_mode(
                args.camera
            )

        elif args.video:

            run_video_mode(
                args.video
            )

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Stopping Vista..."
        )

    finally:

        if llama_process:

            print(
                "Stopping SmolVLM..."
            )

            llama_process.terminate()

            try:

                llama_process.wait(
                    timeout=5
                )

            except subprocess.TimeoutExpired:

                llama_process.kill()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()
