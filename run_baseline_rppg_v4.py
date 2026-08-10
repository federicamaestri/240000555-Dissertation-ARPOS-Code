from pathlib import Path
import re
import time
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
from scipy import signal
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning
import warnings

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "results" / "dataset_index.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "baseline_rppg_results_v4.csv"

# ============================================================
# v4 focused experiment settings
# ============================================================
CHANNELS_TO_PROCESS = ["Color"]

ROIS_TO_PROCESS = [
    "forehead",
    "leftcheek",
    "rightcheek",
    "cheeksCombined",
    "lips",
]

SOURCE_METHODS = [
    "green_minus_blue",
    "green_minus_red",
    "pca",
    "fastica",
]

WINDOW_SIZES_SECONDS = [5, 10, 15, 20]

OVERLAPS = [0.0, 0.25, 0.5, 0.75]

LOW_HZ = 0.7       # 42 BPM
HIGH_HZ = 3.5      # 210 BPM

HR_ESTIMATION_METHOD = "windowed_fft_snr_component_median"

# ============================================================
# Image loading
# ============================================================
def natural_sort_key(path: Path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]

def find_images(folder: Path):
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
    files = []
    for extension in extensions:
        files.extend(folder.glob(extension))
    return sorted(files, key=natural_sort_key)

# Load RGB traces from a single ROI folder
def load_rgb_roi_traces(roi_path: Path):
    image_files = find_images(roi_path)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {roi_path}")
    red_values = []
    green_values = []
    blue_values = []

    for image_path in image_files:
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image, dtype=float)

        red_values.append(array[:, :, 0].mean())
        green_values.append(array[:, :, 1].mean())
        blue_values.append(array[:, :, 2].mean())

    rgb_traces = np.column_stack([
        np.asarray(red_values, dtype=float),
        np.asarray(green_values, dtype=float),
        np.asarray(blue_values, dtype=float),
    ])
    return rgb_traces, len(image_files)

# ============================================================
# Signal preprocessing
# ============================================================
def safe_zscore(x: np.ndarray):
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        return x - mean

    return (x - mean) / std

# Detrend each column independently
def detrend_columns(matrix: np.ndarray):
    return np.column_stack([
        signal.detrend(matrix[:, i])
        for i in range(matrix.shape[1])
    ])

# Bandpass filter the signal
def bandpass_signal(
    x: np.ndarray,
    fps: float,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
    order: int = 3,
):
    if len(x) < 20:
        raise ValueError("Signal is too short for filtering.")
    nyquist = fps / 2.0
    if nyquist <= low_hz:
        raise ValueError(
            f"FPS too low for HR bandpass filtering. "
            f"fps={fps:.3f}, nyquist={nyquist:.3f}, low_hz={low_hz}"
        )
    if high_hz >= nyquist:
        high_hz = nyquist * 0.95
    low = low_hz / nyquist
    high = high_hz / nyquist
    if low >= high:
        raise ValueError(
            f"Invalid bandpass limits: fps={fps:.3f}, "
            f"Wn=[{low:.3f}, {high:.3f}]"
        )
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return signal.filtfilt(b, a, x)

# Preprocess a 1D signal
def preprocess_1d_signal(x: np.ndarray, fps: float):
    if np.std(x) < 1e-8:
        raise ValueError("Signal is almost constant.")
    x_detrended = signal.detrend(x)
    x_normalised = safe_zscore(x_detrended)
    x_filtered = bandpass_signal(x_normalised, fps)
    return x_filtered

# ============================================================
# Source / signal extraction methods
# ============================================================
# Extract candidate signals from RGB traces
def get_candidate_signals(rgb_traces: np.ndarray, source_method: str, fps: float):
    red = rgb_traces[:, 0]
    green = rgb_traces[:, 1]
    blue = rgb_traces[:, 2]

    candidates = {}

    if source_method == "green_minus_blue":
        x = green - blue
        candidates["green_minus_blue"] = preprocess_1d_signal(x, fps)

    elif source_method == "green_minus_red":
        x = green - red
        candidates["green_minus_red"] = preprocess_1d_signal(x, fps)

    elif source_method == "pca":
        x = detrend_columns(rgb_traces)

        x = np.column_stack([
            safe_zscore(x[:, i])
            for i in range(x.shape[1])
        ])

        pca = PCA(n_components=3)
        components = pca.fit_transform(x)

        for i in range(components.shape[1]):
            component = components[:, i]
            candidates[f"pca_component_{i + 1}"] = bandpass_signal(component, fps)

    elif source_method == "fastica":
        x = detrend_columns(rgb_traces)

        x = np.column_stack([
            safe_zscore(x[:, i])
            for i in range(x.shape[1])
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)

            ica = FastICA(
                n_components=3,
                random_state=42,
                max_iter=1000,
                tol=0.001,
            )

            components = ica.fit_transform(x)

        for i in range(components.shape[1]):
            component = components[:, i]
            candidates[f"fastica_component_{i + 1}"] = bandpass_signal(component, fps)

    else:
        raise ValueError(f"Unknown source method: {source_method}")

    if len(candidates) == 0:
        raise ValueError(f"No candidate signals produced for {source_method}")

    return candidates

# ============================================================
# FFT + SNR estimation
# ============================================================
def estimate_hr_fft_with_snr(
    x: np.ndarray,
    fps: float,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
):
    if len(x) < 20:
        raise ValueError("Window too short for FFT.")

    if np.std(x) < 1e-8:
        raise ValueError("Window signal is almost constant.")

    n = len(x)

    window = np.hanning(n)
    x_windowed = x * window

    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(x_windowed)) ** 2

    valid = (freqs >= low_hz) & (freqs <= high_hz)

    if not np.any(valid):
        raise ValueError("No frequencies in HR band.")

    valid_freqs = freqs[valid]
    valid_power = power[valid]

    if len(valid_power) < 3:
        raise ValueError("Too few frequency bins in HR band.")

    peak_index = int(np.argmax(valid_power))
    peak_frequency_hz = valid_freqs[peak_index]
    estimated_hr_bpm = peak_frequency_hz * 60.0
    peak_power = valid_power[peak_index]

    # Estimate noise as the median power excluding the peak bin and neighbours
    mask_noise = np.ones(len(valid_power), dtype=bool)

    for offset in [-1, 0, 1]:
        idx = peak_index + offset
        if 0 <= idx < len(mask_noise):
            mask_noise[idx] = False

    noise_power_values = valid_power[mask_noise]

    if len(noise_power_values) == 0:
        noise_power = np.median(valid_power)
    else:
        noise_power = np.median(noise_power_values)

    snr = peak_power / (noise_power + 1e-12)
    snr_db = 10.0 * np.log10(snr + 1e-12)

    return {
        "estimated_hr_bpm": float(estimated_hr_bpm),
        "peak_frequency_hz": float(peak_frequency_hz),
        "peak_power": float(peak_power),
        "noise_power": float(noise_power),
        "snr": float(snr),
        "snr_db": float(snr_db),
    }

def estimate_recording_hr_windowed_snr(
    candidate_signals: dict,
    fps: float,
    window_size_seconds: float,
    overlap: float,
):
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be in [0, 1).")

    first_signal = next(iter(candidate_signals.values()))
    n = len(first_signal)

    window_samples = int(round(window_size_seconds * fps))

    if window_samples < 20:
        raise ValueError(
            f"Window too short: {window_size_seconds}s = {window_samples} samples"
        )

    if window_samples > n:
        raise ValueError(
            f"Window longer than signal: window={window_samples}, signal={n}"
        )

    step_samples = int(round(window_samples * (1.0 - overlap)))
    step_samples = max(1, step_samples)

    starts = list(range(0, n - window_samples + 1, step_samples))

    if len(starts) == 0:
        raise ValueError("No valid windows generated.")

    window_estimates = []
    selected_components = []

    for start in starts:
        end = start + window_samples

        best_window_result = None
        best_component_name = None

        for component_name, component_signal in candidate_signals.items():
            segment = component_signal[start:end]

            try:
                result = estimate_hr_fft_with_snr(segment, fps)
            except Exception:
                continue

            if best_window_result is None:
                best_window_result = result
                best_component_name = component_name
            elif result["snr"] > best_window_result["snr"]:
                best_window_result = result
                best_component_name = component_name

        if best_window_result is not None:
            best_window_result["selected_component"] = best_component_name
            best_window_result["window_start_frame"] = start
            best_window_result["window_end_frame"] = end
            best_window_result["window_start_seconds"] = start / fps
            best_window_result["window_end_seconds"] = end / fps

            window_estimates.append(best_window_result)
            selected_components.append(best_component_name)

    if len(window_estimates) == 0:
        raise ValueError("No valid window-level HR estimates.")

    hr_values = np.asarray([
        item["estimated_hr_bpm"]
        for item in window_estimates
    ])

    snr_values = np.asarray([
        item["snr"]
        for item in window_estimates
    ])

    snr_db_values = np.asarray([
        item["snr_db"]
        for item in window_estimates
    ])

    component_counts = Counter(selected_components)

    component_summary = "; ".join([
        f"{component}:{count}"
        for component, count in component_counts.items()
    ])

    return {
        "estimated_hr_bpm": float(np.median(hr_values)),
        "mean_window_hr_bpm": float(np.mean(hr_values)),
        "std_window_hr_bpm": float(np.std(hr_values)),
        "min_window_hr_bpm": float(np.min(hr_values)),
        "max_window_hr_bpm": float(np.max(hr_values)),

        "median_snr": float(np.median(snr_values)),
        "mean_snr": float(np.mean(snr_values)),
        "median_snr_db": float(np.median(snr_db_values)),
        "mean_snr_db": float(np.mean(snr_db_values)),

        "n_windows": int(len(window_estimates)),
        "component_selection_summary": component_summary,
    }

# ============================================================
# Process one ROI row
# ============================================================
def process_one_roi(row):
    roi_path = Path(row["roi_path"])
    fps = float(row["estimated_fps"])

    participant = row["participant"]
    state = row["state"]
    channel = row["channel"]
    roi = row["roi"]

    if channel != "Color":
        raise ValueError("v4 focused script currently supports Color only.")

    if not roi_path.exists():
        raise FileNotFoundError(f"ROI folder does not exist: {roi_path}")

    if np.isnan(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")

    rgb_traces, n_images_loaded = load_rgb_roi_traces(roi_path)

    results = []

    for source_method in SOURCE_METHODS:
        source_start = time.perf_counter()

        try:
            candidate_signals = get_candidate_signals(
                rgb_traces=rgb_traces,
                source_method=source_method,
                fps=fps,
            )

            source_runtime = time.perf_counter() - source_start

            for window_size_seconds in WINDOW_SIZES_SECONDS:
                for overlap in OVERLAPS:
                    estimate_start = time.perf_counter()

                    try:
                        hr_result = estimate_recording_hr_windowed_snr(
                            candidate_signals=candidate_signals,
                            fps=fps,
                            window_size_seconds=window_size_seconds,
                            overlap=overlap,
                        )

                        estimate_runtime = time.perf_counter() - estimate_start

                        results.append({
                            "participant": participant,
                            "state": state,
                            "channel": channel,
                            "roi": roi,

                            "source_method": source_method,
                            "signal_method": source_method,
                            "hr_estimation_method": HR_ESTIMATION_METHOD,

                            "window_size_seconds": window_size_seconds,
                            "overlap": overlap,

                            "fps": fps,
                            "n_images": int(row["n_roi_images"]),
                            "n_images_loaded": n_images_loaded,
                            "metadata_frame_count": int(row["metadata_frame_count"]),

                            "estimated_hr_bpm": hr_result["estimated_hr_bpm"],
                            "mean_window_hr_bpm": hr_result["mean_window_hr_bpm"],
                            "std_window_hr_bpm": hr_result["std_window_hr_bpm"],
                            "min_window_hr_bpm": hr_result["min_window_hr_bpm"],
                            "max_window_hr_bpm": hr_result["max_window_hr_bpm"],

                            "median_snr": hr_result["median_snr"],
                            "mean_snr": hr_result["mean_snr"],
                            "median_snr_db": hr_result["median_snr_db"],
                            "mean_snr_db": hr_result["mean_snr_db"],

                            "n_windows": hr_result["n_windows"],
                            "component_selection_summary": hr_result["component_selection_summary"],

                            "source_runtime_seconds": source_runtime,
                            "estimate_runtime_seconds": estimate_runtime,
                            "total_runtime_seconds": source_runtime + estimate_runtime,

                            "roi_path": str(roi_path),
                            "error": "",
                        })

                    except Exception as estimate_error:
                        estimate_runtime = time.perf_counter() - estimate_start

                        results.append({
                            "participant": participant,
                            "state": state,
                            "channel": channel,
                            "roi": roi,

                            "source_method": source_method,
                            "signal_method": source_method,
                            "hr_estimation_method": HR_ESTIMATION_METHOD,

                            "window_size_seconds": window_size_seconds,
                            "overlap": overlap,

                            "fps": fps,
                            "n_images": int(row["n_roi_images"]),
                            "n_images_loaded": n_images_loaded,
                            "metadata_frame_count": int(row["metadata_frame_count"]),

                            "estimated_hr_bpm": np.nan,
                            "mean_window_hr_bpm": np.nan,
                            "std_window_hr_bpm": np.nan,
                            "min_window_hr_bpm": np.nan,
                            "max_window_hr_bpm": np.nan,

                            "median_snr": np.nan,
                            "mean_snr": np.nan,
                            "median_snr_db": np.nan,
                            "mean_snr_db": np.nan,

                            "n_windows": np.nan,
                            "component_selection_summary": "",

                            "source_runtime_seconds": source_runtime,
                            "estimate_runtime_seconds": estimate_runtime,
                            "total_runtime_seconds": source_runtime + estimate_runtime,

                            "roi_path": str(roi_path),
                            "error": str(estimate_error),
                        })

        except Exception as source_error:
            source_runtime = time.perf_counter() - source_start

            for window_size_seconds in WINDOW_SIZES_SECONDS:
                for overlap in OVERLAPS:
                    results.append({
                        "participant": participant,
                        "state": state,
                        "channel": channel,
                        "roi": roi,

                        "source_method": source_method,
                        "signal_method": source_method,
                        "hr_estimation_method": HR_ESTIMATION_METHOD,

                        "window_size_seconds": window_size_seconds,
                        "overlap": overlap,

                        "fps": fps,
                        "n_images": int(row["n_roi_images"]),
                        "n_images_loaded": n_images_loaded,
                        "metadata_frame_count": int(row["metadata_frame_count"]),

                        "estimated_hr_bpm": np.nan,
                        "mean_window_hr_bpm": np.nan,
                        "std_window_hr_bpm": np.nan,
                        "min_window_hr_bpm": np.nan,
                        "max_window_hr_bpm": np.nan,

                        "median_snr": np.nan,
                        "mean_snr": np.nan,
                        "median_snr_db": np.nan,
                        "mean_snr_db": np.nan,

                        "n_windows": np.nan,
                        "component_selection_summary": "",

                        "source_runtime_seconds": source_runtime,
                        "estimate_runtime_seconds": np.nan,
                        "total_runtime_seconds": source_runtime,

                        "roi_path": str(roi_path),
                        "error": str(source_error),
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
        (index_df["estimated_fps"].notna()) &
        (index_df["channel"].isin(CHANNELS_TO_PROCESS)) &
        (index_df["roi"].isin(ROIS_TO_PROCESS))
    ].copy()

    print("v4 focused settings")
    print("-------------------")
    print("Channels:", CHANNELS_TO_PROCESS)
    print("ROIs:", ROIS_TO_PROCESS)
    print("Source methods:", SOURCE_METHODS)
    print("Window sizes:", WINDOW_SIZES_SECONDS)
    print("Overlaps:", OVERLAPS)
    print()

    print(f"Usable rows for v4: {len(usable_rows)}")
    print()

    all_results = []

    total_rows = len(usable_rows)

    for index, (_, row) in enumerate(usable_rows.iterrows(), start=1):
        label = (
            f"{row['participant']} | {row['state']} | "
            f"{row['channel']} | {row['roi']}"
        )

        print(f"[{index}/{total_rows}] Processing {label}")

        row_start = time.perf_counter()

        try:
            roi_results = process_one_roi(row)
            all_results.extend(roi_results)

            success_count = sum(
                not pd.isna(result["estimated_hr_bpm"])
                for result in roi_results
            )

            fail_count = len(roi_results) - success_count

            row_runtime = time.perf_counter() - row_start

            print(
                f"    saved {len(roi_results)} rows "
                f"({success_count} success, {fail_count} failed) "
                f"in {row_runtime:.2f}s"
            )

        except Exception as error:
            row_runtime = time.perf_counter() - row_start

            print(f"    ERROR after {row_runtime:.2f}s: {error}")

            all_results.append({
                "participant": row["participant"],
                "state": row["state"],
                "channel": row["channel"],
                "roi": row["roi"],
                "source_method": np.nan,
                "signal_method": np.nan,
                "hr_estimation_method": HR_ESTIMATION_METHOD,
                "window_size_seconds": np.nan,
                "overlap": np.nan,
                "fps": row["estimated_fps"],
                "n_images": row["n_roi_images"],
                "n_images_loaded": np.nan,
                "metadata_frame_count": row["metadata_frame_count"],
                "estimated_hr_bpm": np.nan,
                "mean_window_hr_bpm": np.nan,
                "std_window_hr_bpm": np.nan,
                "min_window_hr_bpm": np.nan,
                "max_window_hr_bpm": np.nan,
                "median_snr": np.nan,
                "mean_snr": np.nan,
                "median_snr_db": np.nan,
                "mean_snr_db": np.nan,
                "n_windows": np.nan,
                "component_selection_summary": "",
                "source_runtime_seconds": np.nan,
                "estimate_runtime_seconds": np.nan,
                "total_runtime_seconds": row_runtime,
                "roi_path": row["roi_path"],
                "error": str(error),
            })

    results_df = pd.DataFrame(all_results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Saved v4 results to:")
    print(OUTPUT_PATH)
    print()

    print("v4 result summary")
    print("-----------------")
    print("Rows saved:", len(results_df))
    print("Successful HR estimates:", results_df["estimated_hr_bpm"].notna().sum())
    print("Failed HR estimates:", results_df["estimated_hr_bpm"].isna().sum())
    print()

    if "error" in results_df.columns:
        failures = results_df[results_df["estimated_hr_bpm"].isna()]
        if len(failures) > 0:
            print("Failure reasons:")
            print(failures["error"].value_counts().head(20))
            print()

    print(results_df.head(20))

if __name__ == "__main__":
    main()