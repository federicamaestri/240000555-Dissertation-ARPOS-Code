from pathlib import Path
from datetime import datetime
import re
import pandas as pd

# ==============================
# Project paths
# ==============================
# This gets the main project folder: arpos-dissertation/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# This is where your PIS-* folders are
DATA_ROOT = PROJECT_ROOT / "data"
# This is where the CSV will be saved
OUTPUT_PATH = PROJECT_ROOT / "results" / "dataset_index.csv"

# ==============================
# Dataset structure
# ==============================
STATES = [
    "Resting1Cropped",
    "Resting2Cropped",

    # Different possible spellings in ARPOS
    "AfterExcerciseCropped",
    "AfterExerciseCropped",
    "AfterExcersizeCropped",
    "AfterExcersize",
]

CHANNELS = ["Color", "IR"]

ROIS = [
    "forehead",
    "leftcheek",
    "rightcheek",
    "cheeksCombined",
    "lips",
]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# ==============================
# Helper functions
# ==============================
# Count image files inside an ROI folder
def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0

    total = 0

    for extension in IMAGE_EXTENSIONS:
        total += len(list(folder.glob(f"*{extension}")))
        total += len(list(folder.glob(f"*{extension.upper()}")))

    return total

# Extract Color or IR frame timestamps from ParticipantInformation.txt.
def extract_frame_times(participant_info_path: Path, channel: str):
    if not participant_info_path.exists():
        return []

    text = participant_info_path.read_text(errors="ignore")

    if channel == "Color":
        pattern = r"Color Date\s*:\s*(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})"
    elif channel == "IR":
        pattern = r"IR Date\s*:\s*(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})"
    else:
        raise ValueError("channel must be Color or IR")

    raw_times = re.findall(pattern, text)

    parsed_times = []

    for raw_time in raw_times:
        try:
            parsed_times.append(datetime.strptime(raw_time, "%d/%m/%Y %H:%M:%S"))
        except ValueError:
            continue

    return parsed_times

# Convert a datetime into seconds since midnight.
def seconds_of_day(dt: datetime) -> int:
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def estimate_fps(participant_info_path: Path, channel: str):
    times = extract_frame_times(participant_info_path, channel)

    if len(times) < 2:
        return {
            "estimated_fps": None,
            "start_time": None,
            "end_time": None,
            "metadata_frame_count": len(times),
            "duration_seconds": None,
            "fps_warning": "Fewer than 2 frame timestamps found",
        }

    start_time = min(times)
    end_time = max(times)

    # +1 because timestamps are rounded to full seconds
    duration_seconds = (end_time - start_time).total_seconds() + 1

    fps_warning = ""

    if duration_seconds <= 0:
        return {
            "estimated_fps": None,
            "start_time": start_time,
            "end_time": end_time,
            "metadata_frame_count": len(times),
            "duration_seconds": duration_seconds,
            "fps_warning": "Invalid non-positive duration",
        }

    # Fix metadata cases where the dates make the recording look much too long.
    # A normal ARPOS recording is around one minute, not many hours.
    if duration_seconds > 600:
        start_clock = seconds_of_day(start_time)
        end_clock = seconds_of_day(end_time)

        clock_duration_seconds = end_clock - start_clock + 1

        # If clock time goes past midnight, correct it.
        if clock_duration_seconds <= 0:
            clock_duration_seconds += 24 * 3600

        if 10 <= clock_duration_seconds <= 300:
            fps_warning = (
                f"Used clock-time duration because datetime duration was unrealistic: "
                f"{duration_seconds:.1f}s -> {clock_duration_seconds:.1f}s"
            )
            duration_seconds = clock_duration_seconds
        else:
            fps_warning = (
                f"Unrealistic duration. Datetime duration={duration_seconds:.1f}s, "
                f"clock-time fallback={clock_duration_seconds:.1f}s"
            )

    estimated_fps = len(times) / duration_seconds

    # Warn if FPS is still unrealistic for video.
    if estimated_fps < 5:
        fps_warning = (
            fps_warning + " | " if fps_warning else ""
        ) + f"Estimated FPS is very low: {estimated_fps:.3f}"

    elif estimated_fps > 100:
        fps_warning = (
            fps_warning + " | " if fps_warning else ""
        ) + f"Estimated FPS is very high: {estimated_fps:.3f}"

    return {
        "estimated_fps": estimated_fps,
        "start_time": start_time,
        "end_time": end_time,
        "metadata_frame_count": len(times),
        "duration_seconds": duration_seconds,
        "fps_warning": fps_warning,
    }

# ==============================
# Build dataset index
# ==============================
def build_dataset_index(data_root: Path) -> pd.DataFrame:
    rows = []

    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    participant_folders = sorted([
        folder for folder in data_root.iterdir()
        if folder.is_dir() and folder.name.startswith("PIS-")
    ])

    print(f"Found {len(participant_folders)} participant folders.")

    for participant_folder in participant_folders:
        participant_id = participant_folder.name
        groundtruth_folder = participant_folder / "GroundTruth"

        for state in STATES:
            state_folder = participant_folder / state

            if not state_folder.exists():
                continue

            participant_info_path = state_folder / "ParticipantInformation.txt"

            for channel in CHANNELS:
                channel_folder = state_folder / channel
                completed_file = state_folder / f"{channel}Completed.txt"

                fps_info = estimate_fps(
                    participant_info_path,
                    channel
                )

                for roi in ROIS:
                    roi_folder = channel_folder / roi
                    image_count = count_images(roi_folder)

                    rows.append({
                        "participant": participant_id,
                        "state": state,
                        "channel": channel,
                        "roi": roi,

                        "roi_path": str(roi_folder),
                        "participant_info_path": str(participant_info_path),
                        "groundtruth_path": str(groundtruth_folder),
                        "completed_file": str(completed_file),

                        "state_exists": state_folder.exists(),
                        "channel_exists": channel_folder.exists(),
                        "roi_exists": roi_folder.exists(),
                        "completed_exists": completed_file.exists(),
                        "groundtruth_exists": groundtruth_folder.exists(),

                        "n_roi_images": image_count,
                        "metadata_frame_count": fps_info["metadata_frame_count"],
                        "estimated_fps": fps_info["estimated_fps"],
                        "duration_seconds": fps_info["duration_seconds"],
                        "start_time": fps_info["start_time"],
                        "end_time": fps_info["end_time"],
                        "fps_warning": fps_info["fps_warning"],
                    })

    return pd.DataFrame(rows)

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    index_df = build_dataset_index(DATA_ROOT)

    index_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Saved dataset index to:", OUTPUT_PATH)
    print()

    print("Rows:", len(index_df))
    print("Participants:", index_df["participant"].nunique())
    print("States:", sorted(index_df["state"].unique()))
    print()

    print("Rows with FPS warnings:")
    warnings = index_df[index_df["fps_warning"].fillna("") != ""]

    if len(warnings) == 0:
        print("No FPS warnings.")
    else:
        print(warnings[[
            "participant",
            "state",
            "channel",
            "roi",
            "estimated_fps",
            "duration_seconds",
            "start_time",
            "end_time",
            "fps_warning",
        ]].head(50))

    print()
    print(index_df.head(20))
