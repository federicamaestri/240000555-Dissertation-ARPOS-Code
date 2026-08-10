from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

# ==============================
# Project paths
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Choose one example recording
# ==============================
participant_folder = PROJECT_ROOT / "data" / "PIS-1032"   # change if needed
state = "Resting1Cropped"
channel = "Color"
roi = "forehead"

roi_folder = participant_folder / state / channel / roi
participant_info_path = participant_folder / state / "ParticipantInformation.txt"

# ==============================
# Helper functions
# ==============================
def natural_sort_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]

def find_images(folder):
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for extension in image_extensions:
        files.extend(folder.glob(extension))
    return sorted(files, key=natural_sort_key)

def extract_signal(roi_folder, channel):
    image_files = find_images(roi_folder)

    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {roi_folder}")

    values = []

    for image_path in image_files:
        if channel == "Color":
            image = Image.open(image_path).convert("RGB")
            arr = np.asarray(image)
            mean_value = arr[:, :, 1].mean()   # green channel
        elif channel == "IR":
            image = Image.open(image_path).convert("L")
            arr = np.asarray(image)
            mean_value = arr.mean()
        else:
            raise ValueError("channel must be 'Color' or 'IR'")

        values.append(mean_value)

    return np.asarray(values, dtype=float)

def estimate_fps_from_participant_info(participant_info_path, channel):
    text = participant_info_path.read_text(errors="ignore")

    if channel == "Color":
        pattern = r"Color Date\s*:\s*(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})"
    elif channel == "IR":
        pattern = r"IR Date\s*:\s*(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})"
    else:
        raise ValueError("channel must be 'Color' or 'IR'")

    times = re.findall(pattern, text)

    if len(times) == 0:
        raise ValueError(f"No timestamps found for {channel}")

    parsed = pd.to_datetime(times, format="%d/%m/%Y %H:%M:%S")

    start = parsed.min()
    end = parsed.max()

    duration_seconds = (end - start).total_seconds() + 1
    fps = len(parsed) / duration_seconds

    return fps

def detrend_and_filter(raw_signal, fps, low_hz=0.7, high_hz=3.5, order=3):
    detrended = signal.detrend(raw_signal)

    nyquist = fps / 2
    if high_hz >= nyquist:
        high_hz = nyquist * 0.95

    b, a = signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass")
    filtered = signal.filtfilt(b, a, detrended)

    return detrended, filtered

def estimate_hr_fft(filtered_signal, fps, low_hz=0.7, high_hz=3.5):
    n = len(filtered_signal)
    window = np.hanning(n)
    spectrum = np.fft.rfft(filtered_signal * window)
    freqs = np.fft.rfftfreq(n, d=1 / fps)
    power = np.abs(spectrum) ** 2

    valid = (freqs >= low_hz) & (freqs <= high_hz)

    peak_frequency = freqs[valid][np.argmax(power[valid])]
    estimated_hr = peak_frequency * 60

    return estimated_hr, peak_frequency, freqs, power

# ==============================
# Load signal
# ==============================
fps = estimate_fps_from_participant_info(participant_info_path, channel)
raw_signal = extract_signal(roi_folder, channel)
time_axis = np.arange(len(raw_signal)) / fps

detrended_signal, filtered_signal = detrend_and_filter(raw_signal, fps)
estimated_hr, peak_frequency, freqs, power = estimate_hr_fft(filtered_signal, fps)

# Rolling mean for baseline drift inspection
rolling_window_seconds = 1.0
rolling_window_samples = max(3, int(round(rolling_window_seconds * fps)))
baseline_drift = pd.Series(raw_signal).rolling(
    window=rolling_window_samples,
    center=True,
    min_periods=1
).mean().to_numpy()

# Normalised filtered signal for nicer plotting
filtered_z = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)

# ==============================
# Figure 1: Raw signal
# ==============================
plt.figure(figsize=(10, 4))
plt.plot(time_axis, raw_signal)
plt.xlabel("Time (s)")
plt.ylabel("Mean intensity")
plt.title(f"Raw Optical Signal - {participant_folder.name} / {state} / {channel} / {roi}")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_raw_signal.png", dpi=300)
plt.show()

# ==============================
# Figure 2: Zoomed raw signal
# ==============================
start_sec = 5
end_sec = 10
mask = (time_axis >= start_sec) & (time_axis <= end_sec)

plt.figure(figsize=(10, 4))
plt.plot(time_axis[mask], raw_signal[mask])
plt.xlabel("Time (s)")
plt.ylabel("Mean intensity")
plt.title(f"Raw Signal (Zoomed {start_sec}-{end_sec}s)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_raw_signal_zoomed.png", dpi=300)
plt.show()

# ==============================
# Figure 3: Raw + baseline drift
# ==============================
plt.figure(figsize=(10, 4))
plt.plot(time_axis, raw_signal, label="Raw signal")
plt.plot(time_axis, baseline_drift, label=f"Rolling mean (~{rolling_window_seconds:.1f}s)")
plt.xlabel("Time (s)")
plt.ylabel("Mean intensity")
plt.title("Raw Signal with Estimated Baseline Drift")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_baseline_drift.png", dpi=300)
plt.show()

# ==============================
# Figure 4: Detrended signal
# ==============================
plt.figure(figsize=(10, 4))
plt.plot(time_axis, detrended_signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Detrended Signal")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_detrended_signal.png", dpi=300)
plt.show()

# ==============================
# Figure 5: Filtered signal
# ==============================
plt.figure(figsize=(10, 4))
plt.plot(time_axis, filtered_z)
plt.xlabel("Time (s)")
plt.ylabel("Normalised amplitude")
plt.title("Bandpass-Filtered Signal")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_filtered_signal.png", dpi=300)
plt.show()

# ==============================
# Figure 6: FFT spectrum
# ==============================
plt.figure(figsize=(10, 4))
plt.plot(freqs * 60, power)
plt.axvline(estimated_hr, linestyle="--", label=f"Estimated HR = {estimated_hr:.2f} BPM")
plt.xlim(40, 210)
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Power")
plt.title("Frequency Spectrum (FFT)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_fft_spectrum.png", dpi=300)
plt.show()

print(f"Estimated HR: {estimated_hr:.2f} BPM")
print(f"Figures saved in: {FIGURES_DIR}")