from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Project paths
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "results" / "baseline_rppg_results.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "baseline_summary"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Load results
# ==============================
results = pd.read_csv(RESULTS_PATH)

print("Loaded results:")
print(results.head())
print()
print("Number of rows:", len(results))
print("Participants:", results["participant"].unique())

# ==============================
# Plot 1: Estimated HR by ROI
# one plot for each participant/state/channel
# ==============================
for (participant, state, channel), group in results.groupby(
    ["participant", "state", "channel"]
):
    group = group.sort_values("roi")

    plt.figure(figsize=(9, 4))
    plt.bar(group["roi"], group["estimated_hr_bpm"])
    plt.xlabel("ROI")
    plt.ylabel("Estimated HR (BPM)")
    plt.title(f"Estimated HR by ROI\n{participant} / {state} / {channel}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    filename = f"{participant}_{state}_{channel}_estimated_hr_by_roi.png"
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

# ==============================
# Plot 2: Mean estimated HR by ROI
# across all current test participants
# ==============================
roi_summary = (
    results
    .groupby("roi")["estimated_hr_bpm"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8, 4))
plt.bar(roi_summary.index, roi_summary.values)
plt.xlabel("ROI")
plt.ylabel("Mean estimated HR (BPM)")
plt.title("Mean Estimated HR by ROI")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mean_estimated_hr_by_roi.png", dpi=300)
plt.close()

# ==============================
# Plot 3: Mean estimated HR by channel
# ==============================
channel_summary = (
    results
    .groupby("channel")["estimated_hr_bpm"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(6, 4))
plt.bar(channel_summary.index, channel_summary.values)
plt.xlabel("Channel")
plt.ylabel("Mean estimated HR (BPM)")
plt.title("Mean Estimated HR by Channel")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mean_estimated_hr_by_channel.png", dpi=300)
plt.close()

# ==============================
# Plot 4: Mean estimated HR by state
# ==============================
state_summary = (
    results
    .groupby("state")["estimated_hr_bpm"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(7, 4))
plt.bar(state_summary.index, state_summary.values)
plt.xlabel("State")
plt.ylabel("Mean estimated HR (BPM)")
plt.title("Mean Estimated HR by Recording State")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mean_estimated_hr_by_state.png", dpi=300)
plt.close()

# ==============================
# Plot 5: Distribution of estimates
# ==============================
plt.figure(figsize=(8, 4))
plt.hist(results["estimated_hr_bpm"].dropna(), bins=15)
plt.xlabel("Estimated HR (BPM)")
plt.ylabel("Frequency")
plt.title("Distribution of Estimated HR Values")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "estimated_hr_distribution.png", dpi=300)
plt.close()

print()
print("Figures saved to:")
print(FIGURES_DIR)