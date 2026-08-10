from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy import signal

# ==============================
# Project paths
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "results" / "dataset_index.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "baseline_rppg_results.csv"

# ==============================
# Helper functions
# ==============================
# Sort image filenames 
def natural_sort_key(path: Path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]

# Find all images in a folder
def find_images(folder: Path):
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    files = []
    for extension in image_extensions:
        files.extend(folder.glob(extension))

    return sorted(files, key=natural_sort_key)

# Extract one time-series signal from an ROI folder
# For Color: use mean green channel intensity
# For IR: convert to grayscale and use mean intensity
def extract_roi_signal(roi_path: Path, channel: str):
    image_files = find_images(roi_path)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {roi_path}")

    values = []

    for image_path in image_files:
        if channel == "Color":
            image = Image.open(image_path).convert("RGB")
            array = np.asarray(image)
            mean_value = array[:, :, 1].mean()
        elif channel == "IR":
            image = Image.open(image_path).convert("L")
            array = np.asarray(image)
            mean_value = array.mean()
        else:
            raise ValueError(f"Unknown channel: {channel}")
        values.append(mean_value)

    return np.asarray(values, dtype=float)

# Remove slow linear trend from the raw intensity signal
def detrend_signal(raw_signal: np.ndarray):
    return signal.detrend(raw_signal)

# Bandpass filter the signal to keep only heart-rate frequencies
def bandpass_heart_rate_signal(
    input_signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    order: int = 3
):
    nyquist = fps / 2

    if high_hz >= nyquist:
        high_hz = nyquist * 0.95

    low = low_hz / nyquist
    high = high_hz / nyquist

    b, a = signal.butter(order, [low, high], btype="bandpass")

    return signal.filtfilt(b, a, input_signal)

# Estimate heart rate using FFT
def estimate_hr_fft(
    filtered_signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5
):
    n = len(filtered_signal)

    if n < 10:
        raise ValueError("Signal is too short for FFT.")

    window = np.hanning(n)
    windowed_signal = filtered_signal * window

    frequencies = np.fft.rfftfreq(n, d=1 / fps)
    fft_power = np.abs(np.fft.rfft(windowed_signal)) ** 2

    valid_band = (frequencies >= low_hz) & (frequencies <= high_hz)

    if not np.any(valid_band):
        raise ValueError("No valid frequencies in HR band.")

    valid_frequencies = frequencies[valid_band]
    valid_power = fft_power[valid_band]

    peak_index = np.argmax(valid_power)
    peak_frequency = valid_frequencies[peak_index]

    estimated_hr_bpm = peak_frequency * 60

    return estimated_hr_bpm, peak_frequency

# Process one ROI
def process_one_roi(row):
    roi_path = Path(row["roi_path"])
    channel = row["channel"]
    fps = float(row["estimated_fps"])

    if not roi_path.exists():
        raise FileNotFoundError(f"ROI folder does not exist: {roi_path}")

    if np.isnan(fps) or fps <= 0:
        raise ValueError("Invalid FPS.")

    start_time = time.perf_counter()

    raw_trace = extract_roi_signal(roi_path, channel)
    detrended_trace = detrend_signal(raw_trace)
    filtered_trace = bandpass_heart_rate_signal(detrended_trace, fps)
    estimated_hr, peak_frequency = estimate_hr_fft(filtered_trace, fps)

    runtime_seconds = time.perf_counter() - start_time

    return {
        "participant": row["participant"],
        "state": row["state"],
        "channel": channel,
        "roi": row["roi"],
        "fps": fps,
        "n_images": int(row["n_roi_images"]),
        "metadata_frame_count": int(row["metadata_frame_count"]),
        "estimated_hr_bpm": estimated_hr,
        "peak_frequency_hz": peak_frequency,
        "runtime_seconds": runtime_seconds,
        "roi_path": str(roi_path),
    }

# ==============================
# Main experiment
# ==============================
def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dataset index: {INDEX_PATH}\n"
            "Run src/make_dataset_index.py first."
        )

    index_df = pd.read_csv(INDEX_PATH)

    print("Loaded dataset index:")
    print(f"Rows: {len(index_df)}")
    print(f"Participants: {index_df['participant'].nunique()}")
    print()

    usable_rows = index_df[
        (index_df["roi_exists"] == True) &
        (index_df["n_roi_images"] > 0) &
        (index_df["estimated_fps"].notna())
    ].copy()

    print(f"Usable rows: {len(usable_rows)}")
    print()

    results = []

    for _, row in usable_rows.iterrows():
        label = f"{row['participant']} | {row['state']} | {row['channel']} | {row['roi']}"

        try:
            result = process_one_roi(row)
            results.append(result)

            print(f"{label} -> {result['estimated_hr_bpm']:.2f} BPM")

        except Exception as error:
            print(f"{label} -> ERROR: {error}")

            results.append({
                "participant": row["participant"],
                "state": row["state"],
                "channel": row["channel"],
                "roi": row["roi"],
                "fps": row["estimated_fps"],
                "n_images": row["n_roi_images"],
                "metadata_frame_count": row["metadata_frame_count"],
                "estimated_hr_bpm": np.nan,
                "peak_frequency_hz": np.nan,
                "runtime_seconds": np.nan,
                "roi_path": row["roi_path"],
                "error": str(error),
            })

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Saved results to: {OUTPUT_PATH}")
    print()
    print(results_df.head())

if __name__ == "__main__":
    main()