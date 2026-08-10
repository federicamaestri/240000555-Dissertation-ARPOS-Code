from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_RESULTS_PATH = PROJECT_ROOT / "results" / "baseline_rppg_results_v4_1.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "baseline_with_groundtruth_v4_1.csv"

# ============================================================
# State mapping
# ============================================================
STATE_TO_GROUNDTRUTH = {
    "Resting1Cropped": "Resting1",
    "Resting2Cropped": "Resting2",

    # Different spellings used in ARPOS
    "AfterExcerciseCropped": "AfterExcersize",
    "AfterExerciseCropped": "AfterExcersize",
    "AfterExcersizeCropped": "AfterExcersize",
    "AfterExcersize": "AfterExcersize",
}

# ============================================================
# Helper functions
# ============================================================
# Extract HR values
def extract_hr_values(hr_file_path: Path) -> np.ndarray:
    if not hr_file_path.exists():
        raise FileNotFoundError(f"Missing HR file: {hr_file_path}")

    lines = hr_file_path.read_text(errors="ignore").splitlines()

    hr_values = []

    for line in lines:
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)

        if not numbers:
            continue

        values = [float(number) for number in numbers]

        plausible_hr_values = [
            value for value in values
            if 30 <= value <= 220
        ]

        if plausible_hr_values:
            hr_values.append(plausible_hr_values[-1])

    if len(hr_values) == 0:
        raise ValueError(f"No plausible HR values found in {hr_file_path}")

    return np.asarray(hr_values, dtype=float)

# Get ground-truth HR statistics
def get_groundtruth_hr(participant: str, state: str):
    if state not in STATE_TO_GROUNDTRUTH:
        raise ValueError(f"No ground-truth mapping for state: {state}")

    groundtruth_state = STATE_TO_GROUNDTRUTH[state]

    hr_file_path = (
        PROJECT_ROOT
        / "data"
        / participant
        / "GroundTruth"
        / groundtruth_state
        / "HR.txt"
    )

    hr_values = extract_hr_values(hr_file_path)

    return {
        "groundtruth_state": groundtruth_state,
        "groundtruth_hr_path": str(hr_file_path),
        "groundtruth_n_samples": len(hr_values),
        "groundtruth_mean_hr_bpm": float(np.mean(hr_values)),
        "groundtruth_median_hr_bpm": float(np.median(hr_values)),
        "groundtruth_min_hr_bpm": float(np.min(hr_values)),
        "groundtruth_max_hr_bpm": float(np.max(hr_values)),
    }

# ============================================================
# Main
# ============================================================
def main():
    if not BASELINE_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find v4.1 baseline results: {BASELINE_RESULTS_PATH}\n"
            "Run src/run_baseline_rppg_v4.py first after changing its OUTPUT_PATH to baseline_rppg_results_v4_1.csv."
        )

    results = pd.read_csv(BASELINE_RESULTS_PATH)

    enriched_rows = []

    for _, row in results.iterrows():
        participant = row["participant"]
        state = row["state"]

        try:
            groundtruth = get_groundtruth_hr(participant, state)

            estimated_hr_bpm = row["estimated_hr_bpm"]
            groundtruth_hr_bpm = groundtruth["groundtruth_median_hr_bpm"]

            if pd.isna(estimated_hr_bpm):
                signed_error_bpm = np.nan
                absolute_error_bpm = np.nan
                squared_error_bpm = np.nan
            else:
                signed_error_bpm = estimated_hr_bpm - groundtruth_hr_bpm
                absolute_error_bpm = abs(signed_error_bpm)
                squared_error_bpm = signed_error_bpm ** 2

            enriched_row = row.to_dict()
            enriched_row.update(groundtruth)
            enriched_row.update({
                "version": "v4_1_zero_padding_parabolic_interpolation",
                "groundtruth_hr_bpm": groundtruth_hr_bpm,
                "signed_error_bpm": signed_error_bpm,
                "absolute_error_bpm": absolute_error_bpm,
                "squared_error_bpm": squared_error_bpm,
                "groundtruth_error": "",
            })

        except Exception as error:
            enriched_row = row.to_dict()
            enriched_row.update({
                "version": "v4_1_zero_padding_parabolic_interpolation",
                "groundtruth_state": np.nan,
                "groundtruth_hr_path": np.nan,
                "groundtruth_n_samples": np.nan,
                "groundtruth_mean_hr_bpm": np.nan,
                "groundtruth_median_hr_bpm": np.nan,
                "groundtruth_min_hr_bpm": np.nan,
                "groundtruth_max_hr_bpm": np.nan,
                "groundtruth_hr_bpm": np.nan,
                "signed_error_bpm": np.nan,
                "absolute_error_bpm": np.nan,
                "squared_error_bpm": np.nan,
                "groundtruth_error": str(error),
            })

        enriched_rows.append(enriched_row)

    enriched_df = pd.DataFrame(enriched_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:")
    print(OUTPUT_PATH)
    print()

    print("Rows:", len(enriched_df))
    print("Successful HR estimates:", enriched_df["estimated_hr_bpm"].notna().sum())
    print("Failed HR estimates:", enriched_df["estimated_hr_bpm"].isna().sum())
    print()

    print("Rows with ground-truth errors:")
    gt_errors = enriched_df[
        enriched_df["groundtruth_error"].fillna("") != ""
    ]

    if len(gt_errors) == 0:
        print("No ground-truth errors found.")
    else:
        print(gt_errors[[
            "participant",
            "state",
            "groundtruth_error",
        ]].drop_duplicates())

    print()

    print("Preview:")
    preview_columns = [
        "participant",
        "state",
        "channel",
        "roi",
        "source_method",
        "hr_estimation_method",
        "window_size_seconds",
        "overlap",
        "estimated_hr_bpm",
        "groundtruth_hr_bpm",
        "absolute_error_bpm",
        "median_snr_db",
        "n_windows",
        "median_raw_frequency_resolution_bpm",
        "median_padded_frequency_resolution_bpm",
        "mean_abs_interpolation_delta_bins",
        "error",
        "groundtruth_error",
    ]

    preview_columns = [
        column for column in preview_columns
        if column in enriched_df.columns
    ]

    print(enriched_df[preview_columns].head(30))

if __name__ == "__main__":
    main()