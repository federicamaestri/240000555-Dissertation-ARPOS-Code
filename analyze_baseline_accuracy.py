from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Project paths
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "baseline_with_groundtruth.csv"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "accuracy"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Helper functions
# ==============================
def rmse(values):
    return np.sqrt(np.mean(np.square(values)))

def make_summary(df, group_columns):
    summary = (
        df
        .groupby(group_columns)
        .agg(
            n=("absolute_error_bpm", "count"),
            mae_bpm=("absolute_error_bpm", "mean"),
            rmse_bpm=("absolute_error_bpm", rmse),
            mean_estimated_hr_bpm=("estimated_hr_bpm", "mean"),
            mean_groundtruth_hr_bpm=("groundtruth_hr_bpm", "mean"),
        )
        .reset_index()
        .sort_values("mae_bpm")
    )

    return summary

def save_bar_plot(summary_df, x_column, y_column, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 4))
    plt.bar(summary_df[x_column].astype(str), summary_df[y_column])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

# ==============================
# Load data
# ==============================
df = pd.read_csv(INPUT_PATH)

# Keep only successful rows
df = df.dropna(subset=[
    "estimated_hr_bpm",
    "groundtruth_hr_bpm",
    "absolute_error_bpm"
]).copy()

print("Loaded rows:", len(df))
print("Participants:", df["participant"].nunique())
print()

# ==============================
# Overall metrics
# ==============================
overall_mae = df["absolute_error_bpm"].mean()
overall_rmse = rmse(df["absolute_error_bpm"])

overall = pd.DataFrame([{
    "n": len(df),
    "mae_bpm": overall_mae,
    "rmse_bpm": overall_rmse,
}])

overall.to_csv(SUMMARY_DIR / "overall_accuracy.csv", index=False)

print("Overall accuracy:")
print(overall)
print()

# ==============================
# Summaries
# ==============================
by_roi = make_summary(df, ["roi"])
by_channel = make_summary(df, ["channel"])
by_state = make_summary(df, ["state"])
by_roi_channel = make_summary(df, ["roi", "channel"])
by_participant = make_summary(df, ["participant"])
by_participant_roi = make_summary(df, ["participant", "roi"])

by_roi.to_csv(SUMMARY_DIR / "accuracy_by_roi.csv", index=False)
by_channel.to_csv(SUMMARY_DIR / "accuracy_by_channel.csv", index=False)
by_state.to_csv(SUMMARY_DIR / "accuracy_by_state.csv", index=False)
by_roi_channel.to_csv(SUMMARY_DIR / "accuracy_by_roi_channel.csv", index=False)
by_participant.to_csv(SUMMARY_DIR / "accuracy_by_participant.csv", index=False)
by_participant_roi.to_csv(SUMMARY_DIR / "accuracy_by_participant_roi.csv", index=False)


print("Accuracy by ROI:")
print(by_roi)
print()

print("Accuracy by channel:")
print(by_channel)
print()

print("Accuracy by state:")
print(by_state)
print()

# ==============================
# Plots
# ==============================
save_bar_plot(
    by_roi,
    x_column="roi",
    y_column="mae_bpm",
    title="Mean Absolute Error by ROI",
    xlabel="ROI",
    ylabel="MAE (BPM)",
    filename="mae_by_roi.png"
)

save_bar_plot(
    by_channel,
    x_column="channel",
    y_column="mae_bpm",
    title="Mean Absolute Error by Channel",
    xlabel="Channel",
    ylabel="MAE (BPM)",
    filename="mae_by_channel.png"
)

save_bar_plot(
    by_state,
    x_column="state",
    y_column="mae_bpm",
    title="Mean Absolute Error by Recording State",
    xlabel="Recording state",
    ylabel="MAE (BPM)",
    filename="mae_by_state.png"
)

save_bar_plot(
    by_participant,
    x_column="participant",
    y_column="mae_bpm",
    title="Mean Absolute Error by Participant",
    xlabel="Participant",
    ylabel="MAE (BPM)",
    filename="mae_by_participant.png"
)

# ==============================
# ROI and channel grouped plot
# ==============================
pivot = by_roi_channel.pivot(
    index="roi",
    columns="channel",
    values="mae_bpm"
)

pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

plt.figure(figsize=(9, 4))
pivot.plot(kind="bar", ax=plt.gca())
plt.xlabel("ROI")
plt.ylabel("MAE (BPM)")
plt.title("Mean Absolute Error by ROI and Channel")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mae_by_roi_and_channel.png", dpi=300)
plt.close()

# ==============================
# Estimated vs ground-truth scatter
# ==============================
plt.figure(figsize=(5, 5))
plt.scatter(df["groundtruth_hr_bpm"], df["estimated_hr_bpm"])

min_hr = min(df["groundtruth_hr_bpm"].min(), df["estimated_hr_bpm"].min())
max_hr = max(df["groundtruth_hr_bpm"].max(), df["estimated_hr_bpm"].max())

plt.plot([min_hr, max_hr], [min_hr, max_hr], linestyle="--")

plt.xlabel("Ground-truth HR (BPM)")
plt.ylabel("Estimated HR (BPM)")
plt.title("Estimated HR vs Ground-truth HR")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "estimated_vs_groundtruth_hr.png", dpi=300)
plt.close()

# ==============================
# Error distribution
# ==============================
plt.figure(figsize=(8, 4))
plt.hist(df["absolute_error_bpm"], bins=15)
plt.xlabel("Absolute error (BPM)")
plt.ylabel("Frequency")
plt.title("Distribution of Absolute HR Error")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "absolute_error_distribution.png", dpi=300)
plt.close()

print("Saved summary CSVs to:")
print(SUMMARY_DIR)

print()
print("Saved figures to:")
print(FIGURES_DIR)