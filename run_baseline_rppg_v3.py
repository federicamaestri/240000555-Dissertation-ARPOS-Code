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
OUTPUT_PATH = PROJECT_ROOT / "results" / "baseline_rppg_results_v3.csv"

# ============================================================
# Settings
# ============================================================
HR_METHODS = [
    "global_fft",
    "welch",
    "windowed_fft_median",
]

LOW_HZ = 0.7
HIGH_HZ = 3.5

# ============================================================
# Image loading and signal extraction
# ============================================================
def natural_sort_key(path: Path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]

def find_images(folder: Path):
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in extensions:
        files.extend(folder.glob(ext))
    return sorted(files, key=natural_sort_key)


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
            arr = np.asarray(image, dtype=float)

            red_values.append(arr[:, :, 0].mean())
            green_values.append(arr[:, :, 1].mean())
            blue_values.append(arr[:, :, 2].mean())

        red = np.asarray(red_values, dtype=float)
        green = np.asarray(green_values, dtype=float)
        blue = np.asarray(blue_values, dtype=float)

        return {
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
            arr = np.asarray(image, dtype=float)
            ir_values.append(arr.mean())

        return {
            "ir_intensity": np.asarray(ir_values, dtype=float)
        }
    else:
        raise ValueError(f"Unknown channel: {channel}")

# ============================================================
# Preprocessing
# ============================================================
def detrend_signal(raw_signal: np.ndarray):
    return signal.detrend(raw_signal)

def bandpass_heart_rate_signal(
    input_signal: np.ndarray,
    fps: float,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
    order: int = 3,
):
    if len(input_signal) < 20:
        raise ValueError("Signal is too short for filtering.")
    nyquist = fps / 2.0
    if high_hz >= nyquist:
        high_hz = nyquist * 0.95
    low = low_hz / nyquist
    high = high_hz / nyquist
    if low >= high:
        raise ValueError(
            f"Invalid bandpass limits: low={low_hz}, high={high_hz}, fps={fps}"
        )
    b, a = signal.butter(order, [low, high], btype="bandpass")
    filtered = signal.filtfilt(b, a, input_signal)
    return filtered

def check_signal_is_valid(x: np.ndarray):
    if len(x) < 20:
        raise ValueError("Signal is too short.")
    if np.std(x) < 1e-8:
        raise ValueError("Signal is almost constant.")

# ============================================================
# HR estimation methods
# ============================================================
def estimate_hr_global_fft(
    filtered_signal: np.ndarray,
    fps: float,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
):
    check_signal_is_valid(filtered_signal)

    n = len(filtered_signal)

    window = np.hanning(n)
    x = filtered_signal * window

    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(x)) ** 2

    valid = (freqs >= low_hz) & (freqs <= high_hz)

    if not np.any(valid):
        raise ValueError("No valid frequencies in HR band.")

    valid_freqs = freqs[valid]
    valid_power = power[valid]

    peak_idx = np.argmax(valid_power)
    peak_frequency_hz = valid_freqs[peak_idx]
    estimated_hr_bpm = peak_frequency_hz * 60.0

    return {
        "estimated_hr_bpm": estimated_hr_bpm,
        "peak_frequency_hz": peak_frequency_hz,
        "n_windows": 1,
    }

def estimate_hr_welch(
    filtered_signal: np.ndarray,
    fps: float,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
):
    check_signal_is_valid(filtered_signal)
    n = len(filtered_signal)

    # 20s gives around 3 BPM frequency resolution
    nperseg = int(round(20 * fps))
    nperseg = min(nperseg, n)

    if nperseg < 32:
        raise ValueError("Signal is too short for Welch estimation.")

    noverlap = nperseg // 2
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    freqs, power = signal.welch(
        filtered_signal,
        fs=fps,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
    )

    valid = (freqs >= low_hz) & (freqs <= high_hz)

    if not np.any(valid):
        raise ValueError("No valid frequencies in HR band.")

    valid_freqs = freqs[valid]
    valid_power = power[valid]

    peak_idx = np.argmax(valid_power)
    peak_frequency_hz = valid_freqs[peak_idx]
    estimated_hr_bpm = peak_frequency_hz * 60.0

    return {
        "estimated_hr_bpm": estimated_hr_bpm,
        "peak_frequency_hz": peak_frequency_hz,
        "n_windows": np.nan,
    }

def estimate_hr_windowed_fft_median(
    filtered_signal: np.ndarray,
    fps: float,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
    window_seconds: float = 20.0,
    overlap: float = 0.5,
):
    check_signal_is_valid(filtered_signal)

    n = len(filtered_signal)

    window_samples = int(round(window_seconds * fps))
    window_samples = min(window_samples, n)

    if window_samples < 32:
        raise ValueError("Signal is too short for windowed FFT.")

    step_samples = int(round(window_samples * (1.0 - overlap)))
    step_samples = max(1, step_samples)

    starts = list(range(0, n - window_samples + 1, step_samples))

    if len(starts) == 0:
        starts = [0]

    hr_estimates = []

    for start in starts:
        segment = filtered_signal[start:start + window_samples]

        try:
            result = estimate_hr_global_fft(
                segment,
                fps,
                low_hz=low_hz,
                high_hz=high_hz,
            )
            hr_estimates.append(result["estimated_hr_bpm"])
        except Exception:
            continue

    if len(hr_estimates) == 0:
        raise ValueError("No valid windowed FFT estimates.")

    estimated_hr_bpm = float(np.median(hr_estimates))
    peak_frequency_hz = estimated_hr_bpm / 60.0

    return {
        "estimated_hr_bpm": estimated_hr_bpm,
        "peak_frequency_hz": peak_frequency_hz,
        "n_windows": len(hr_estimates),
    }

def estimate_hr(filtered_signal: np.ndarray, fps: float, method: str):
    if method == "global_fft":
        return estimate_hr_global_fft(filtered_signal, fps)

    if method == "welch":
        return estimate_hr_welch(filtered_signal, fps)

    if method == "windowed_fft_median":
        return estimate_hr_windowed_fft_median(filtered_signal, fps)

    raise ValueError(f"Unknown HR estimation method: {method}")

# ============================================================
# Process one ROI row
# ============================================================
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
        try:
            detrended_trace = detrend_signal(raw_trace)
            filtered_trace = bandpass_heart_rate_signal(detrended_trace, fps)

            for hr_method in HR_METHODS:
                start_time = time.perf_counter()

                try:
                    hr_result = estimate_hr(filtered_trace, fps, hr_method)
                    runtime_seconds = time.perf_counter() - start_time

                    results.append({
                        "participant": row["participant"],
                        "state": row["state"],
                        "channel": channel,
                        "roi": row["roi"],
                        "signal_method": signal_method,
                        "hr_estimation_method": hr_method,
                        "fps": fps,
                        "n_images": int(row["n_roi_images"]),
                        "metadata_frame_count": int(row["metadata_frame_count"]),
                        "estimated_hr_bpm": hr_result["estimated_hr_bpm"],
                        "peak_frequency_hz": hr_result["peak_frequency_hz"],
                        "n_windows": hr_result["n_windows"],
                        "runtime_seconds": runtime_seconds,
                        "roi_path": str(roi_path),
                        "error": "",
                    })

                except Exception as error:
                    runtime_seconds = time.perf_counter() - start_time

                    results.append({
                        "participant": row["participant"],
                        "state": row["state"],
                        "channel": channel,
                        "roi": row["roi"],
                        "signal_method": signal_method,
                        "hr_estimation_method": hr_method,
                        "fps": fps,
                        "n_images": int(row["n_roi_images"]),
                        "metadata_frame_count": int(row["metadata_frame_count"]),
                        "estimated_hr_bpm": np.nan,
                        "peak_frequency_hz": np.nan,
                        "n_windows": np.nan,
                        "runtime_seconds": runtime_seconds,
                        "roi_path": str(roi_path),
                        "error": str(error),
                    })

        except Exception as preprocessing_error:
            for hr_method in HR_METHODS:
                results.append({
                    "participant": row["participant"],
                    "state": row["state"],
                    "channel": channel,
                    "roi": row["roi"],
                    "signal_method": signal_method,
                    "hr_estimation_method": hr_method,
                    "fps": fps,
                    "n_images": int(row["n_roi_images"]),
                    "metadata_frame_count": int(row["metadata_frame_count"]),
                    "estimated_hr_bpm": np.nan,
                    "peak_frequency_hz": np.nan,
                    "n_windows": np.nan,
                    "runtime_seconds": np.nan,
                    "roi_path": str(roi_path),
                    "error": str(preprocessing_error),
                })

    return results

# ============================================================
# Main
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
                if pd.isna(result["estimated_hr_bpm"]):
                    print(
                        f"{label} | {result['signal_method']} | "
                        f"{result['hr_estimation_method']} -> ERROR: {result['error']}"
                    )
                else:
                    print(
                        f"{label} | {result['signal_method']} | "
                        f"{result['hr_estimation_method']} "
                        f"-> {result['estimated_hr_bpm']:.2f} BPM"
                    )

        except Exception as error:
            print(f"{label} -> ERROR: {error}")

    results_df = pd.DataFrame(all_results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Saved results to:")
    print(OUTPUT_PATH)
    print()
    print("Rows saved:", len(results_df))
    print("Successful estimates:", results_df["estimated_hr_bpm"].notna().sum())
    print("Failed estimates:", results_df["estimated_hr_bpm"].isna().sum())
    print()
    print(results_df.head(20))

if __name__ == "__main__":
    main()