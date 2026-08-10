from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "baseline_with_groundtruth_v4.csv"
BEST_CONFIG_PATHS = [
    PROJECT_ROOT / "results" / "summaries_v4" / "robust_best_full_configuration_v4.csv",
    PROJECT_ROOT / "results" / "summaries_v4" / "best_full_configuration_v4.csv",
]

SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries_v4_reliability"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "v4_reliability"
INTENSITY_CACHE_PATH = SUMMARY_DIR / "roi_intensity_features_v4.csv"
ENRICHED_OUTPUT_PATH = SUMMARY_DIR / "baseline_with_groundtruth_v4_reliability_enriched.csv"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Columns
# ============================================================
CONFIG_COLUMNS = [
    "channel",
    "roi",
    "source_method",
    "hr_estimation_method",
    "window_size_seconds",
    "overlap",
]

RELIABILITY_COLUMNS = [
    "median_snr_db",
    "mean_snr_db",
    "std_window_hr_bpm",
    "mean_rgb_intensity",
    "std_rgb_intensity",
    "noise_proxy_rgb",
    "temporal_diff_noise_proxy",
    "mean_spatial_std_intensity",
    "dark_pixel_fraction",
    "bright_pixel_fraction",
]

# ============================================================
# Basic metrics
# ============================================================
def rmse(errors):
    errors = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(errors ** 2)))

def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def spearman_r(x, y):
    data = pd.DataFrame({
        "x": x,
        "y": y,
    }).dropna()
    if len(data) < 2:
        return np.nan
    return float(data["x"].corr(data["y"], method="spearman"))

def calculate_summary(df, label):
    successful = df.dropna(subset=[
        "estimated_hr_bpm",
        "groundtruth_hr_bpm",
        "absolute_error_bpm",
    ]).copy()

    if len(successful) == 0:
        return {
            "label": label,
            "n": 0,
            "mae_bpm": np.nan,
            "rmse_bpm": np.nan,
            "median_error_bpm": np.nan,
            "max_error_bpm": np.nan,
            "pearson_r": np.nan,
        }

    return {
        "label": label,
        "n": len(successful),
        "mae_bpm": float(successful["absolute_error_bpm"].mean()),
        "rmse_bpm": rmse(successful["absolute_error_bpm"]),
        "median_error_bpm": float(successful["absolute_error_bpm"].median()),
        "max_error_bpm": float(successful["absolute_error_bpm"].max()),
        "pearson_r": pearson_r(
            successful["groundtruth_hr_bpm"],
            successful["estimated_hr_bpm"],
        ),
    }

# ============================================================
# Image/intensity feature extraction
# ===========================================================
def natural_sort_key(path: Path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]

def find_images(folder: Path):
    extensions = [
        "*.jpg", "*.jpeg", "*.png", "*.bmp",
        "*.JPG", "*.JPEG", "*.PNG", "*.BMP",
    ]

    files = []

    for extension in extensions:
        files.extend(folder.glob(extension))

    return sorted(files, key=natural_sort_key)

# Compute ROI intensity features
def compute_roi_intensity_features(roi_path: str):
    roi_path = Path(roi_path)
    image_files = find_images(roi_path)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {roi_path}")

    frame_red = []
    frame_green = []
    frame_blue = []
    frame_mean_rgb = []
    frame_spatial_std = []
    dark_fractions = []
    bright_fractions = []

    for image_path in image_files:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            array = np.asarray(image, dtype=float)

        red = array[:, :, 0]
        green = array[:, :, 1]
        blue = array[:, :, 2]

        mean_red = red.mean()
        mean_green = green.mean()
        mean_blue = blue.mean()

        mean_rgb = (mean_red + mean_green + mean_blue) / 3.0

        frame_red.append(mean_red)
        frame_green.append(mean_green)
        frame_blue.append(mean_blue)
        frame_mean_rgb.append(mean_rgb)

        grayscale = array.mean(axis=2)
        frame_spatial_std.append(grayscale.std())

        dark_fractions.append(float(np.mean(array < 5)))
        bright_fractions.append(float(np.mean(array > 250)))

    frame_red = np.asarray(frame_red, dtype=float)
    frame_green = np.asarray(frame_green, dtype=float)
    frame_blue = np.asarray(frame_blue, dtype=float)
    frame_mean_rgb = np.asarray(frame_mean_rgb, dtype=float)

    eps = 1e-12

    mean_rgb_intensity = float(np.mean(frame_mean_rgb))
    std_rgb_intensity = float(np.std(frame_mean_rgb))
    noise_proxy_rgb = float(std_rgb_intensity / (mean_rgb_intensity + eps))

    if len(frame_mean_rgb) > 1:
        temporal_diff_noise_proxy = float(
            np.mean(np.abs(np.diff(frame_mean_rgb))) / (mean_rgb_intensity + eps)
        )
    else:
        temporal_diff_noise_proxy = np.nan

    return {
        "roi_path": str(roi_path),
        "n_frames_intensity": len(image_files),

        "mean_red_intensity": float(np.mean(frame_red)),
        "mean_green_intensity": float(np.mean(frame_green)),
        "mean_blue_intensity": float(np.mean(frame_blue)),
        "mean_rgb_intensity": mean_rgb_intensity,

        "std_red_intensity": float(np.std(frame_red)),
        "std_green_intensity": float(np.std(frame_green)),
        "std_blue_intensity": float(np.std(frame_blue)),
        "std_rgb_intensity": std_rgb_intensity,

        "noise_proxy_rgb": noise_proxy_rgb,
        "temporal_diff_noise_proxy": temporal_diff_noise_proxy,

        "mean_spatial_std_intensity": float(np.mean(frame_spatial_std)),
        "dark_pixel_fraction": float(np.mean(dark_fractions)),
        "bright_pixel_fraction": float(np.mean(bright_fractions)),
    }

def build_or_load_intensity_cache(df):
    all_roi_paths = sorted(
        df["roi_path"]
        .dropna()
        .astype(str)
        .unique()
    )
    if INTENSITY_CACHE_PATH.exists():
        cache_df = pd.read_csv(INTENSITY_CACHE_PATH)
        existing_paths = set(cache_df["roi_path"].astype(str))

        missing_paths = [
            path for path in all_roi_paths
            if path not in existing_paths
        ]

        if len(missing_paths) == 0:
            print("Loaded existing intensity cache:")
            print(INTENSITY_CACHE_PATH)
            print()
            return cache_df

        print(f"Intensity cache exists, but {len(missing_paths)} ROI folders are missing.")
    else:
        cache_df = pd.DataFrame()
        missing_paths = all_roi_paths

    print("Computing ROI intensity/noise features")
    print("--------------------------------------")
    print(f"ROI folders to process: {len(missing_paths)}")
    print("This may take a few minutes because it reads the ROI images.")
    print()

    rows = []

    for i, roi_path in enumerate(missing_paths, start=1):
        print(f"[{i}/{len(missing_paths)}] {roi_path}")

        try:
            features = compute_roi_intensity_features(roi_path)
            features["intensity_error"] = ""
        except Exception as error:
            features = {
                "roi_path": roi_path,
                "n_frames_intensity": np.nan,
                "mean_red_intensity": np.nan,
                "mean_green_intensity": np.nan,
                "mean_blue_intensity": np.nan,
                "mean_rgb_intensity": np.nan,
                "std_red_intensity": np.nan,
                "std_green_intensity": np.nan,
                "std_blue_intensity": np.nan,
                "std_rgb_intensity": np.nan,
                "noise_proxy_rgb": np.nan,
                "temporal_diff_noise_proxy": np.nan,
                "mean_spatial_std_intensity": np.nan,
                "dark_pixel_fraction": np.nan,
                "bright_pixel_fraction": np.nan,
                "intensity_error": str(error),
            }

        rows.append(features)

    new_cache_rows = pd.DataFrame(rows)

    if len(cache_df) > 0:
        cache_df = pd.concat([cache_df, new_cache_rows], ignore_index=True)
        cache_df = cache_df.drop_duplicates(subset=["roi_path"], keep="last")
    else:
        cache_df = new_cache_rows

    cache_df.to_csv(INTENSITY_CACHE_PATH, index=False)

    print()
    print("Saved intensity cache:")
    print(INTENSITY_CACHE_PATH)
    print()

    return cache_df

# ============================================================
# Best configuration handling
# ============================================================
def load_best_configuration():
    for path in BEST_CONFIG_PATHS:
        if path.exists():
            best_df = pd.read_csv(path)

            if len(best_df) > 0:
                best_row = best_df.iloc[0].to_dict()

                print("Loaded best configuration from:")
                print(path)
                print()

                return best_row

    raise FileNotFoundError(
        "Could not find robust_best_full_configuration_v4.csv "
        "or best_full_configuration_v4.csv. "
        "Run src/find_best_configuration_v4.py first."
    )

def filter_to_configuration(df, config):
    filtered = df.copy()

    for column in CONFIG_COLUMNS:
        if column not in filtered.columns or column not in config:
            continue

        if column in ["window_size_seconds", "overlap"]:
            filtered = filtered[
                np.isclose(
                    pd.to_numeric(filtered[column], errors="coerce"),
                    float(config[column]),
                    equal_nan=False,
                )
            ]
        else:
            filtered = filtered[
                filtered[column].astype(str) == str(config[column])
            ]

    return filtered.copy()


def make_config_label(config):
    parts = []

    for column in CONFIG_COLUMNS:
        if column in config:
            parts.append(str(config[column]))

    return " | ".join(parts)

# ============================================================
# Plot helpers
# ============================================================
def save_estimated_vs_groundtruth(df, filename, title):
    plot_df = df.dropna(subset=[
        "groundtruth_hr_bpm",
        "estimated_hr_bpm",
    ]).copy()

    if len(plot_df) == 0:
        return

    x = plot_df["groundtruth_hr_bpm"].to_numpy(dtype=float)
    y = plot_df["estimated_hr_bpm"].to_numpy(dtype=float)

    min_value = min(np.min(x), np.min(y))
    max_value = max(np.max(x), np.max(y))

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.xlabel("Ground-truth HR (BPM)")
    plt.ylabel("Estimated HR (BPM)")
    plt.title(title)

    mae_value = plot_df["absolute_error_bpm"].mean()
    rmse_value = rmse(plot_df["absolute_error_bpm"])
    r_value = pearson_r(x, y)

    text = (
        f"MAE = {mae_value:.2f} BPM\n"
        f"RMSE = {rmse_value:.2f} BPM\n"
        f"r = {r_value:.3f}"
    )

    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        va="top",
        bbox={"boxstyle": "round", "alpha": 0.2},
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

def save_scatter(df, x_column, y_column, filename, title, xlabel, ylabel):
    plot_df = df.dropna(subset=[x_column, y_column]).copy()

    if len(plot_df) == 0:
        return

    x = plot_df[x_column].to_numpy(dtype=float)
    y = plot_df[y_column].to_numpy(dtype=float)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.6)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    r = pearson_r(x, y)
    rho = spearman_r(x, y)

    text = (
        f"Pearson r = {r:.3f}\n"
        f"Spearman ρ = {rho:.3f}"
    )

    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        va="top",
        bbox={"boxstyle": "round", "alpha": 0.2},
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

def save_binned_mae_plot(df, x_column, filename, title, xlabel, n_bins=5):
    plot_df = df.dropna(subset=[x_column, "absolute_error_bpm"]).copy()

    if len(plot_df) < n_bins:
        return

    try:
        plot_df["bin"] = pd.qcut(
            plot_df[x_column],
            q=n_bins,
            duplicates="drop",
        )
    except ValueError:
        return

    summary = (
        plot_df
        .groupby("bin", observed=True)
        .agg(
            n=("absolute_error_bpm", "size"),
            mean_x=(x_column, "mean"),
            mae_bpm=("absolute_error_bpm", "mean"),
            rmse_input=("absolute_error_bpm", lambda x: rmse(x)),
        )
        .reset_index()
    )

    summary = summary.rename(columns={
        "rmse_input": "rmse_bpm"
    })

    summary.to_csv(
        SUMMARY_DIR / filename.replace(".png", ".csv"),
        index=False
    )

    x_labels = [
        f"{row['mean_x']:.3f}\n(n={int(row['n'])})"
        for _, row in summary.iterrows()
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, summary["mae_bpm"])

    plt.xlabel(xlabel)
    plt.ylabel("MAE (BPM)")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

# ============================================================
# Window trace reconstruction
# ============================================================
def reconstruct_window_estimates(row):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from run_baseline_rppg_v4 import (
        load_rgb_roi_traces,
        get_candidate_signals,
        estimate_hr_fft_with_snr,
    )

    roi_path = Path(row["roi_path"])
    fps = float(row["fps"])
    source_method = str(row["source_method"])
    window_size_seconds = float(row["window_size_seconds"])
    overlap = float(row["overlap"])

    rgb_traces, _ = load_rgb_roi_traces(roi_path)

    candidate_signals = get_candidate_signals(
        rgb_traces=rgb_traces,
        source_method=source_method,
        fps=fps,
    )

    first_signal = next(iter(candidate_signals.values()))
    n = len(first_signal)

    window_samples = int(round(window_size_seconds * fps))
    step_samples = int(round(window_samples * (1.0 - overlap)))
    step_samples = max(1, step_samples)

    starts = list(range(0, n - window_samples + 1, step_samples))

    rows = []

    for start in starts:
        end = start + window_samples

        best_result = None
        best_component = None

        for component_name, component_signal in candidate_signals.items():
            segment = component_signal[start:end]

            try:
                result = estimate_hr_fft_with_snr(segment, fps)
            except Exception:
                continue

            if best_result is None or result["snr"] > best_result["snr"]:
                best_result = result
                best_component = component_name

        if best_result is None:
            continue

        rows.append({
            "window_start_seconds": start / fps,
            "window_mid_seconds": ((start + end) / 2.0) / fps,
            "window_end_seconds": end / fps,
            "estimated_hr_bpm": best_result["estimated_hr_bpm"],
            "snr_db": best_result["snr_db"],
            "selected_component": best_component,
        })

    return pd.DataFrame(rows)

def save_window_trace(row, filename, title):
    try:
        window_df = reconstruct_window_estimates(row)
    except Exception as error:
        print(f"Could not reconstruct window trace for {filename}: {error}")
        return

    if len(window_df) == 0:
        return

    groundtruth_hr = float(row["groundtruth_hr_bpm"])
    aggregate_hr = float(row["estimated_hr_bpm"])

    window_df.to_csv(
        SUMMARY_DIR / filename.replace(".png", ".csv"),
        index=False
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        window_df["window_mid_seconds"],
        window_df["estimated_hr_bpm"],
        marker="o",
    )

    plt.axhline(
        groundtruth_hr,
        linestyle="--",
        label=f"Ground truth = {groundtruth_hr:.1f} BPM",
    )

    plt.axhline(
        aggregate_hr,
        linestyle=":",
        label=f"Aggregate estimate = {aggregate_hr:.1f} BPM",
    )

    plt.xlabel("Time in recording (seconds)")
    plt.ylabel("Window-level estimated HR (BPM)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

# ============================================================
# Main analysis
# ============================================================
def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. "
            "Run src/add_groundtruth_v4.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    print("Loaded v4 results:")
    print(INPUT_PATH)
    print("Rows:", len(df))
    print()

    # --------------------------------------------------------
    # Add intensity/noise features
    # --------------------------------------------------------
    intensity_df = build_or_load_intensity_cache(df)

    df = df.merge(
        intensity_df,
        on="roi_path",
        how="left",
    )

    df.to_csv(ENRICHED_OUTPUT_PATH, index=False)

    print("Saved enriched v4 reliability file:")
    print(ENRICHED_OUTPUT_PATH)
    print()

    # --------------------------------------------------------
    # Select best configuration
    # --------------------------------------------------------
    best_config = load_best_configuration()
    best_label = make_config_label(best_config)

    best_df = filter_to_configuration(df, best_config)

    print("Best v4 configuration:")
    print(best_label)
    print()

    print("Rows for best configuration:", len(best_df))
    print("Successful rows:", best_df["estimated_hr_bpm"].notna().sum())
    print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summaries = pd.DataFrame([
        calculate_summary(df, "all_v4_rows"),
        calculate_summary(best_df, "best_v4_configuration"),
    ])

    summaries.to_csv(
        SUMMARY_DIR / "reliability_main_summary_v4.csv",
        index=False
    )

    print("Reliability main summary:")
    print(summaries)
    print()

    # --------------------------------------------------------
    # Correlations with error
    # --------------------------------------------------------
    successful_best = best_df.dropna(subset=[
        "estimated_hr_bpm",
        "groundtruth_hr_bpm",
        "absolute_error_bpm",
    ]).copy()

    correlation_rows = []

    for column in RELIABILITY_COLUMNS:
        if column not in successful_best.columns:
            continue

        temp = successful_best.dropna(subset=[column, "absolute_error_bpm"])

        if len(temp) < 2:
            continue

        correlation_rows.append({
            "feature": column,
            "n": len(temp),
            "pearson_with_absolute_error": pearson_r(
                temp[column],
                temp["absolute_error_bpm"],
            ),
            "spearman_with_absolute_error": spearman_r(
                temp[column],
                temp["absolute_error_bpm"],
            ),
            "feature_mean": float(temp[column].mean()),
            "feature_median": float(temp[column].median()),
            "absolute_error_mean": float(temp["absolute_error_bpm"].mean()),
        })

    correlation_df = pd.DataFrame(correlation_rows)

    correlation_df.to_csv(
        SUMMARY_DIR / "reliability_correlations_best_config_v4.csv",
        index=False
    )

    print("Reliability correlations with absolute error:")
    print(correlation_df)
    print()

    # --------------------------------------------------------
    # Main plots
    # --------------------------------------------------------
    save_estimated_vs_groundtruth(
        successful_best,
        filename="estimated_vs_groundtruth_best_v4.png",
        title="v4 Best Configuration: Estimated HR vs Ground Truth",
    )

    save_scatter(
        successful_best,
        x_column="median_snr_db",
        y_column="absolute_error_bpm",
        filename="absolute_error_vs_snr_best_v4.png",
        title="Absolute Error vs SNR",
        xlabel="Median SNR (dB)",
        ylabel="Absolute error (BPM)",
    )

    save_scatter(
        successful_best,
        x_column="noise_proxy_rgb",
        y_column="absolute_error_bpm",
        filename="absolute_error_vs_noise_proxy_best_v4.png",
        title="Absolute Error vs Intensity Noise Proxy",
        xlabel="Noise proxy: std RGB intensity / mean RGB intensity",
        ylabel="Absolute error (BPM)",
    )

    save_scatter(
        successful_best,
        x_column="mean_rgb_intensity",
        y_column="absolute_error_bpm",
        filename="absolute_error_vs_mean_intensity_best_v4.png",
        title="Absolute Error vs Mean ROI Intensity",
        xlabel="Mean RGB intensity",
        ylabel="Absolute error (BPM)",
    )

    save_scatter(
        successful_best,
        x_column="std_window_hr_bpm",
        y_column="absolute_error_bpm",
        filename="absolute_error_vs_window_hr_std_best_v4.png",
        title="Absolute Error vs Window HR Variability",
        xlabel="Standard deviation of window HR estimates (BPM)",
        ylabel="Absolute error (BPM)",
    )

    save_scatter(
        successful_best,
        x_column="temporal_diff_noise_proxy",
        y_column="absolute_error_bpm",
        filename="absolute_error_vs_temporal_diff_noise_best_v4.png",
        title="Absolute Error vs Frame-to-Frame Intensity Instability",
        xlabel="Temporal difference noise proxy",
        ylabel="Absolute error (BPM)",
    )

    save_scatter(
        successful_best,
        x_column="median_snr_db",
        y_column="noise_proxy_rgb",
        filename="noise_proxy_vs_snr_best_v4.png",
        title="Intensity Noise Proxy vs SNR",
        xlabel="Median SNR (dB)",
        ylabel="Noise proxy",
    )

    # --------------------------------------------------------
    # Binned plots
    # --------------------------------------------------------
    save_binned_mae_plot(
        successful_best,
        x_column="median_snr_db",
        filename="binned_mae_by_snr_best_v4.png",
        title="MAE by SNR Bin",
        xlabel="Mean SNR per bin (dB)",
    )

    save_binned_mae_plot(
        successful_best,
        x_column="noise_proxy_rgb",
        filename="binned_mae_by_noise_proxy_best_v4.png",
        title="MAE by Intensity Noise Proxy Bin",
        xlabel="Mean noise proxy per bin",
    )

    save_binned_mae_plot(
        successful_best,
        x_column="mean_rgb_intensity",
        filename="binned_mae_by_mean_intensity_best_v4.png",
        title="MAE by Mean ROI Intensity Bin",
        xlabel="Mean intensity per bin",
    )

    save_binned_mae_plot(
        successful_best,
        x_column="std_window_hr_bpm",
        filename="binned_mae_by_window_hr_std_best_v4.png",
        title="MAE by Window HR Variability Bin",
        xlabel="Mean window HR standard deviation per bin",
    )

    # --------------------------------------------------------
    # Example window traces
    # --------------------------------------------------------
    if len(successful_best) > 0:
        typical_row = successful_best.iloc[
            (successful_best["absolute_error_bpm"] - successful_best["absolute_error_bpm"].median())
            .abs()
            .argsort()
            .iloc[0]
        ]

        worst_row = successful_best.sort_values(
            "absolute_error_bpm",
            ascending=False,
        ).iloc[0]

        save_window_trace(
            typical_row,
            filename="window_hr_trace_typical_best_v4.png",
            title=(
                "Typical v4 Window-Level HR Trace\n"
                f"{typical_row['participant']} | {typical_row['state']} | "
                f"error={typical_row['absolute_error_bpm']:.2f} BPM"
            ),
        )

        save_window_trace(
            worst_row,
            filename="window_hr_trace_worst_best_v4.png",
            title=(
                "Worst v4 Window-Level HR Trace\n"
                f"{worst_row['participant']} | {worst_row['state']} | "
                f"error={worst_row['absolute_error_bpm']:.2f} BPM"
            ),
        )

    print("Saved reliability summaries to:")
    print(SUMMARY_DIR)
    print()

    print("Saved reliability figures to:")
    print(FIGURES_DIR)

if __name__ == "__main__":
    main()