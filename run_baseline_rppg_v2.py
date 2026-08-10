from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy import signal

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "results" / "dataset_index.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "baseline_rppg_results_v2.csv"

# ============================================================
# Helper functions
# ============================================================
def natural_sort_key(path: Path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]

def find_images(folder: Path):
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for extension in image_extensions:
        files.extend(folder.glob(extension))
    return sorted(files, key=natural_sort_key)

# Extract ROI signals from a folder
def extract_roi_signals(roi_path: Path, channel: str):
    image_files = find_images(roi_path)

    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {roi_path}")

    if channel == "Color":
        red_values = []
        green_values = []
        blue_values = []
        for image_path in image_files:
            image = Image.open(image_path).convert("RGB")
            array = np.asarray(image, dtype=float)
            red_values.append(array[:, :, 0].mean())
            green_values.append(array[:, :, 1].mean())
            blue_values.append(array[:, :, 2].mean())
        red = np.asarray(red_values, dtype=float)
        green = np.asarray(green_values, dtype=float)
        blue = np.asarray(blue_values, dtype=float)

        signals = {
            "red": red,
            "green": green,
            "blue": blue,
            "rgb_mean": (red + green + blue) / 3.0,
            "green_minus_red": green - red,
            "green_minus_blue": green - blue,
        }
    elif channel == "IR":
        ir_values = []
        for image_path in image_files:
            image = Image.open(image_path).convert("L")
            array = np.asarray(image, dtype=float)
            ir_values.append(array.mean())
        signals = {
            "ir_intensity": np.asarray(ir_values, dtype=float)
        }
    else:
        raise ValueError(f"Unknown channel: {channel}")
    return signals

# Detrend the raw signal
def detrend_signal(raw_signal: np.ndarray):
    return signal.detrend(raw_signal)

# Bandpass filter the heart-rate signal
def bandpass_heart_rate_signal(
    input_signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    order: int = 3
):
    if len(input_signal) < 20:
        raise ValueError("Signal is too short for filtering.")
    nyquist = fps / 2.0
    if low_hz <= 0:
        raise ValueError("low_hz must be positive.")
    if high_hz >= nyquist:
        high_hz = nyquist * 0.95
    low = low_hz / nyquist
    high = high_hz / nyquist
    if low >= high:
        raise ValueError(
            f"Invalid bandpass limits: low={low_hz}, high={high_hz}, fps={fps}"
        )

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
    if n < 20:
        raise ValueError("Signal is too short for FFT.")
    if np.allclose(filtered_signal, filtered_signal[0]):
        raise ValueError("Signal is almost constant.")
    window = np.hanning(n)
    windowed_signal = filtered_signal * window
    frequencies = np.fft.rfftfreq(n, d=1.0 / fps)
    spectrum = np.fft.rfft(windowed_signal)
    power = np.abs(spectrum) ** 2
    valid_band = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(valid_band):
        raise ValueError("No valid frequencies found in HR band.")
    valid_frequencies = frequencies[valid_band]
    valid_power = power[valid_band]
    peak_index = np.argmax(valid_power)
    peak_frequency_hz = valid_frequencies[peak_index]
    estimated_hr_bpm = peak_frequency_hz * 60.0
    return estimated_hr_bpm, peak_frequency_hz

# Process one ROI
def process_one_roi(row):
    roi_path = Path(row["roi_path"])
    channel = row["channel"]
    fps = float(row["estimated_fps"])
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI folder does not exist: {roi_path}")
    if np.isnan(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")
    extracted_signals = extract_roi_signals(roi_path, channel)
    results = []

    for signal_method, raw_trace in extracted_signals.items():
        start_time = time.perf_counter()
        detrended_trace = detrend_signal(raw_trace)
        filtered_trace = bandpass_heart_rate_signal(detrended_trace, fps)
        estimated_hr_bpm, peak_frequency_hz = estimate_hr_fft(filtered_trace, fps)
        runtime_seconds = time.perf_counter() - start_time

        results.append({
            "participant": row["participant"],
            "state": row["state"],
            "channel": channel,
            "roi": row["roi"],
            "signal_method": signal_method,
            "fps": fps,
            "n_images": int(row["n_roi_images"]),
            "metadata_frame_count": int(row["metadata_frame_count"]),
            "estimated_hr_bpm": estimated_hr_bpm,
            "peak_frequency_hz": peak_frequency_hz,
            "runtime_seconds": runtime_seconds,
            "roi_path": str(roi_path),
            "error": "",
        })
    return results

# ============================================================
# Main experiment
# ============================================================
def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dataset index: {INDEX_PATH}\n"
            "Run src/make_dataset_index.py first."
        )

    index_df = pd.read_csv(INDEX_PATH)

    print("Loaded dataset index")
    print("--------------------")
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

    all_results = []

    for _, row in usable_rows.iterrows():
        label = (
            f"{row['participant']} | {row['state']} | "
            f"{row['channel']} | {row['roi']}"
        )

        try:
            roi_results = process_one_roi(row)
            all_results.extend(roi_results)

            for result in roi_results:
                print(
                    f"{label} | {result['signal_method']} "
                    f"-> {result['estimated_hr_bpm']:.2f} BPM"
                )

        except Exception as error:
            print(f"{label} -> ERROR: {error}")

            all_results.append({
                "participant": row["participant"],
                "state": row["state"],
                "channel": row["channel"],
                "roi": row["roi"],
                "signal_method": np.nan,
                "fps": row["estimated_fps"],
                "n_images": row["n_roi_images"],
                "metadata_frame_count": row["metadata_frame_count"],
                "estimated_hr_bpm": np.nan,
                "peak_frequency_hz": np.nan,
                "runtime_seconds": np.nan,
                "roi_path": row["roi_path"],
                "error": str(error),
            })

    results_df = pd.DataFrame(all_results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Saved results to:")
    print(OUTPUT_PATH)
    print()

    print("Result summary")
    print("--------------")
    print(f"Rows saved: {len(results_df)}")
    print(f"Participants: {results_df['participant'].nunique()}")
    print(f"Signal methods: {results_df['signal_method'].dropna().unique()}")
    print()
    print(results_df.head(20))

if __name__ == "__main__":
    main()