from pathlib import Path
import re
import numpy as np
import pandas as pd

# ==============================
# Project paths
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_RESULTS_PATH = PROJECT_ROOT / "results" / "baseline_rppg_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "baseline_with_groundtruth.csv"

# ==============================
# State mapping
# ==============================
# Left side = image folder state name
# Right side = GroundTruth folder state name
STATE_TO_GROUNDTRUTH = {
    "Resting1Cropped": "Resting1",
    "Resting2Cropped": "Resting2",

    # Different spellings used in the ARPOS folders
    "AfterExcerciseCropped": "AfterExcersize",
    "AfterExerciseCropped": "AfterExcersize",
    "AfterExcersizeCropped": "AfterExcersize",
    "AfterExcersize": "AfterExcersize",
}

# ==============================
# Helper functions
# ==============================
# Extract plausible HR values from an HR.txt file
def extract_hr_values(hr_file_path: Path) -> np.ndarray:
    if not hr_file_path.exists():
        raise FileNotFoundError(f"Missing HR file: {hr_file_path}")

    lines = hr_file_path.read_text(errors="ignore").splitlines()

    hr_values = []

    for line in lines:
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
        if not numbers:
            continue
        numeric_values = [float(value) for value in numbers]
        plausible_values = [
            value for value in numeric_values
            if 30 <= value <= 220
        ]
        if not plausible_values:
            continue
        # If the file has one value per line, this is that value.
        # If the line has timestamp + HR, the HR is usually the final value.
        hr_values.append(plausible_values[-1])

    if len(hr_values) == 0:
        raise ValueError(f"No plausible HR values found in {hr_file_path}")

    return np.asarray(hr_values, dtype=float)

# Get ground-truth HR information for one participant/state
def get_groundtruth_hr(participant: str, state: str):
    if state not in STATE_TO_GROUNDTRUTH:
        raise ValueError(f"No ground-truth mapping defined for state: {state}")

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

# ==============================
# Main
# ==============================
def main():
    if not BASELINE_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find baseline results: {BASELINE_RESULTS_PATH}\n"
            "Run src/run_baseline_rppg.py first."
        )

    results = pd.read_csv(BASELINE_RESULTS_PATH)

    enriched_rows = []

    for _, row in results.iterrows():
        participant = row["participant"]
        state = row["state"]

        try:
            gt = get_groundtruth_hr(participant, state)

            groundtruth_hr_bpm = gt["groundtruth_median_hr_bpm"]
            estimated_hr_bpm = row["estimated_hr_bpm"]

            absolute_error = abs(estimated_hr_bpm - groundtruth_hr_bpm)
            squared_error = (estimated_hr_bpm - groundtruth_hr_bpm) ** 2

            enriched_row = row.to_dict()
            enriched_row.update(gt)
            enriched_row.update({
                "groundtruth_hr_bpm": groundtruth_hr_bpm,
                "absolute_error_bpm": absolute_error,
                "squared_error_bpm": squared_error,
                "groundtruth_error": "",
            })

        except Exception as error:
            enriched_row = row.to_dict()
            enriched_row.update({
                "groundtruth_state": np.nan,
                "groundtruth_hr_path": np.nan,
                "groundtruth_n_samples": np.nan,
                "groundtruth_mean_hr_bpm": np.nan,
                "groundtruth_median_hr_bpm": np.nan,
                "groundtruth_min_hr_bpm": np.nan,
                "groundtruth_max_hr_bpm": np.nan,
                "groundtruth_hr_bpm": np.nan,
                "absolute_error_bpm": np.nan,
                "squared_error_bpm": np.nan,
                "groundtruth_error": str(error),
            })

        enriched_rows.append(enriched_row)

    enriched_df = pd.DataFrame(enriched_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print()
    print(enriched_df[[
        "participant",
        "state",
        "channel",
        "roi",
        "estimated_hr_bpm",
        "groundtruth_hr_bpm",
        "absolute_error_bpm",
        "groundtruth_n_samples",
        "groundtruth_error",
    ]].head(20))

    print()
    print("Rows with ground-truth errors:")
    print(enriched_df[enriched_df["groundtruth_error"].fillna("") != ""][
        ["participant", "state", "groundtruth_error"]
    ].drop_duplicates())

if __name__ == "__main__":
    main()