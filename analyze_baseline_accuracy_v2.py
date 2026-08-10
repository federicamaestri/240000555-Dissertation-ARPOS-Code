from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "baseline_with_groundtruth_v2.csv"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries_v2"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "accuracy_v2"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helper functions
# ============================================================
def rmse_from_errors(errors):
    return np.sqrt(np.mean(np.square(errors)))

def make_summary(df, group_columns):
    summary = (
        df
        .groupby(group_columns)
        .agg(
            n=("absolute_error_bpm", "count"),
            mae_bpm=("absolute_error_bpm", "mean"),
            rmse_bpm=("absolute_error_bpm", rmse_from_errors),
            mean_estimated_hr_bpm=("estimated_hr_bpm", "mean"),
            mean_groundtruth_hr_bpm=("groundtruth_hr_bpm", "mean"),
        )
        .reset_index()
        .sort_values("mae_bpm")
    )
    return summary

def save_bar_plot(summary_df, x_column, y_column, title, xlabel, ylabel, filename):
    plt.figure(figsize=(9, 4))
    plt.bar(summary_df[x_column].astype(str), summary_df[y_column])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(INPUT_PATH)

print("Loaded:")
print(INPUT_PATH)
print()

print("Total rows:", len(df))
print("Rows with successful HR estimate:", df["estimated_hr_bpm"].notna().sum())
print("Rows with failed HR estimate:", df["estimated_hr_bpm"].isna().sum())
print()

failed_rows = df[df["estimated_hr_bpm"].isna()]
if len(failed_rows) > 0:
    print("Failure reasons:")
    print(failed_rows["error"].value_counts())
    print()

# Keep only successful rows
df_success = df.dropna(subset=[
    "estimated_hr_bpm",
    "groundtruth_hr_bpm",
    "absolute_error_bpm"
]).copy()

# ============================================================
# Overall accuracy
# ============================================================
overall = pd.DataFrame([{
    "n": len(df_success),
    "mae_bpm": df_success["absolute_error_bpm"].mean(),
    "rmse_bpm": rmse_from_errors(df_success["absolute_error_bpm"]),
}])

overall.to_csv(SUMMARY_DIR / "overall_accuracy_v2.csv", index=False)

print("Overall accuracy:")
print(overall)
print()

# ============================================================
# Summaries
# ============================================================
by_signal_method = make_summary(df_success, ["signal_method"])
by_roi = make_summary(df_success, ["roi"])
by_channel = make_summary(df_success, ["channel"])
by_state = make_summary(df_success, ["state"])
by_participant = make_summary(df_success, ["participant"])

by_roi_signal_method = make_summary(df_success, ["roi", "signal_method"])
by_channel_signal_method = make_summary(df_success, ["channel", "signal_method"])
by_roi_channel = make_summary(df_success, ["roi", "channel"])
by_state_signal_method = make_summary(df_success, ["state", "signal_method"])

# Save CSVs
by_signal_method.to_csv(SUMMARY_DIR / "accuracy_by_signal_method.csv", index=False)
by_roi.to_csv(SUMMARY_DIR / "accuracy_by_roi.csv", index=False)
by_channel.to_csv(SUMMARY_DIR / "accuracy_by_channel.csv", index=False)
by_state.to_csv(SUMMARY_DIR / "accuracy_by_state.csv", index=False)
by_participant.to_csv(SUMMARY_DIR / "accuracy_by_participant.csv", index=False)

by_roi_signal_method.to_csv(SUMMARY_DIR / "accuracy_by_roi_signal_method.csv", index=False)
by_channel_signal_method.to_csv(SUMMARY_DIR / "accuracy_by_channel_signal_method.csv", index=False)
by_roi_channel.to_csv(SUMMARY_DIR / "accuracy_by_roi_channel.csv", index=False)
by_state_signal_method.to_csv(SUMMARY_DIR / "accuracy_by_state_signal_method.csv", index=False)

print("Accuracy by signal method:")
print(by_signal_method)
print()

print("Accuracy by ROI:")
print(by_roi)
print()

print("Accuracy by channel:")
print(by_channel)
print()

print("Accuracy by state:")
print(by_state)
print()

# ============================================================
# Bar plots
# ============================================================
save_bar_plot(
    by_signal_method,
    x_column="signal_method",
    y_column="mae_bpm",
    title="Mean Absolute Error by Signal Method",
    xlabel="Signal method",
    ylabel="MAE (BPM)",
    filename="mae_by_signal_method.png"
)

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

# ============================================================
# Grouped plot: ROI × signal method
# ============================================================
pivot_roi_signal = by_roi_signal_method.pivot(
    index="roi",
    columns="signal_method",
    values="mae_bpm"
)

pivot_roi_signal = pivot_roi_signal.loc[
    pivot_roi_signal.mean(axis=1).sort_values().index
]

plt.figure(figsize=(12, 5))
pivot_roi_signal.plot(kind="bar", ax=plt.gca())
plt.xlabel("ROI")
plt.ylabel("MAE (BPM)")
plt.title("Mean Absolute Error by ROI and Signal Method")
plt.xticks(rotation=30, ha="right")
plt.legend(title="Signal method", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mae_by_roi_and_signal_method.png", dpi=300)
plt.close()

# ============================================================
# Grouped plot: channel × signal method
# ============================================================
pivot_channel_signal = by_channel_signal_method.pivot(
    index="signal_method",
    columns="channel",
    values="mae_bpm"
)

pivot_channel_signal = pivot_channel_signal.loc[
    pivot_channel_signal.mean(axis=1).sort_values().index
]

plt.figure(figsize=(10, 5))
pivot_channel_signal.plot(kind="bar", ax=plt.gca())
plt.xlabel("Signal method")
plt.ylabel("MAE (BPM)")
plt.title("Mean Absolute Error by Signal Method and Channel")
plt.xticks(rotation=30, ha="right")
plt.legend(title="Channel")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mae_by_signal_method_and_channel.png", dpi=300)
plt.close()

# ============================================================
# Scatter plot: estimated vs ground truth
# ============================================================
plt.figure(figsize=(5, 5))
plt.scatter(df_success["groundtruth_hr_bpm"], df_success["estimated_hr_bpm"])

min_hr = min(
    df_success["groundtruth_hr_bpm"].min(),
    df_success["estimated_hr_bpm"].min()
)

max_hr = max(
    df_success["groundtruth_hr_bpm"].max(),
    df_success["estimated_hr_bpm"].max()
)

plt.plot([min_hr, max_hr], [min_hr, max_hr], linestyle="--")

plt.xlabel("Ground-truth HR (BPM)")
plt.ylabel("Estimated HR (BPM)")
plt.title("Estimated HR vs Ground-truth HR")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "estimated_vs_groundtruth_hr.png", dpi=300)
plt.close()

# ============================================================
# Error distribution
# ============================================================
plt.figure(figsize=(8, 4))
plt.hist(df_success["absolute_error_bpm"], bins=15)
plt.xlabel("Absolute error (BPM)")
plt.ylabel("Frequency")
plt.title("Distribution of Absolute HR Error")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "absolute_error_distribution.png", dpi=300)
plt.close()

print("Saved summaries to:")
print(SUMMARY_DIR)
print()

print("Saved figures to:")
print(FIGURES_DIR)