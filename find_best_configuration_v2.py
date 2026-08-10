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
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "best_configuration_v2"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helper functions
# ============================================================
def rmse_from_errors(errors):
    return np.sqrt(np.mean(np.square(errors)))

# Summarise configuration
def summarise_configuration(df, group_columns):
    total_counts = (
        df
        .groupby(group_columns)
        .size()
        .reset_index(name="n_total")
    )

    successful = df.dropna(subset=[
        "estimated_hr_bpm",
        "groundtruth_hr_bpm",
        "absolute_error_bpm"
    ]).copy()

    accuracy = (
        successful
        .groupby(group_columns)
        .agg(
            n_success=("absolute_error_bpm", "count"),
            mae_bpm=("absolute_error_bpm", "mean"),
            rmse_bpm=("absolute_error_bpm", rmse_from_errors),
            median_error_bpm=("absolute_error_bpm", "median"),
            mean_signed_error_bpm=("signed_error_bpm", "mean"),
            mean_estimated_hr_bpm=("estimated_hr_bpm", "mean"),
            mean_groundtruth_hr_bpm=("groundtruth_hr_bpm", "mean"),
        )
        .reset_index()
    )

    summary = total_counts.merge(
        accuracy,
        on=group_columns,
        how="left"
    )

    summary["success_rate"] = summary["n_success"] / summary["n_total"]

    summary = summary.sort_values(
        by=["mae_bpm", "success_rate"],
        ascending=[True, False]
    )

    return summary


def save_top_bar_plot(summary, title, filename, top_n=10):
    plot_df = summary.dropna(subset=["mae_bpm"]).head(top_n).copy()

    plot_df["configuration"] = (
        plot_df["channel"].astype(str)
        + " | "
        + plot_df["roi"].astype(str)
        + " | "
        + plot_df["signal_method"].astype(str)
    )

    plt.figure(figsize=(11, 5))
    plt.bar(plot_df["configuration"], plot_df["mae_bpm"])
    plt.xlabel("Configuration")
    plt.ylabel("MAE (BPM)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

# ============================================================
# Main
# ============================================================
def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find input file: {INPUT_PATH}\n"
            "Run src/add_groundtruth_v2.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    print("Loaded:")
    print(INPUT_PATH)
    print()
    print("Total rows:", len(df))
    print("Successful rows:", df["estimated_hr_bpm"].notna().sum())
    print("Failed rows:", df["estimated_hr_bpm"].isna().sum())
    print()

    # ========================================================
    # 1. Best full configuration:
    # channel + ROI + signal_method
    # ========================================================
    full_config = summarise_configuration(
        df,
        ["channel", "roi", "signal_method"]
    )

    full_config.to_csv(
        SUMMARY_DIR / "best_full_configuration.csv",
        index=False
    )

    print("=" * 80)
    print("Top full configurations: channel + ROI + signal_method")
    print("=" * 80)
    print(full_config.head(20))
    print()

    # ========================================================
    # 2. Best full configuration per state:
    # state + channel + ROI + signal_method
    # ========================================================
    config_by_state = summarise_configuration(
        df,
        ["state", "channel", "roi", "signal_method"]
    )

    config_by_state.to_csv(
        SUMMARY_DIR / "best_configuration_by_state.csv",
        index=False
    )

    print("=" * 80)
    print("Best configurations by state")
    print("=" * 80)

    for state, group in config_by_state.groupby("state"):
        print()
        print(f"State: {state}")
        print(group.head(10))

    # ========================================================
    # 3. Best configuration per ROI
    # ========================================================
    best_per_roi = (
        full_config
        .dropna(subset=["mae_bpm"])
        .sort_values("mae_bpm")
        .groupby("roi")
        .head(1)
        .sort_values("mae_bpm")
    )

    best_per_roi.to_csv(
        SUMMARY_DIR / "best_configuration_per_roi.csv",
        index=False
    )

    print()
    print("=" * 80)
    print("Best configuration per ROI")
    print("=" * 80)
    print(best_per_roi)
    print()

    # ========================================================
    # 4. Best configuration per channel
    # ========================================================
    best_per_channel = (
        full_config
        .dropna(subset=["mae_bpm"])
        .sort_values("mae_bpm")
        .groupby("channel")
        .head(1)
        .sort_values("mae_bpm")
    )

    best_per_channel.to_csv(
        SUMMARY_DIR / "best_configuration_per_channel.csv",
        index=False
    )

    print("=" * 80)
    print("Best configuration per channel")
    print("=" * 80)
    print(best_per_channel)
    print()

    # ========================================================
    # 5. Robust best configuration
    # ========================================================
    # This avoids picking a configuration that looks good only
    # because it succeeded on very few rows
    robust_candidates = full_config[
        (full_config["n_success"] >= 6) &
        (full_config["success_rate"] >= 0.80)
    ].copy()

    robust_candidates = robust_candidates.sort_values("mae_bpm")

    robust_candidates.to_csv(
        SUMMARY_DIR / "robust_best_full_configuration.csv",
        index=False
    )

    print("=" * 80)
    print("Robust best configurations")
    print("Criteria: n_success >= 6 and success_rate >= 0.80")
    print("=" * 80)

    if len(robust_candidates) == 0:
        print("No configurations passed the robustness filter.")
    else:
        print(robust_candidates.head(20))

    # ========================================================
    # 6. Plots
    # ========================================================
    save_top_bar_plot(
        full_config,
        title="Top 10 Full Configurations by MAE",
        filename="top_10_full_configurations_by_mae.png",
        top_n=10
    )

    if len(robust_candidates) > 0:
        save_top_bar_plot(
            robust_candidates,
            title="Top 10 Robust Full Configurations by MAE",
            filename="top_10_robust_configurations_by_mae.png",
            top_n=10
        )

    # Plot success rate of top configurations
    top_success = full_config.dropna(subset=["mae_bpm"]).head(15).copy()
    top_success["configuration"] = (
        top_success["channel"].astype(str)
        + " | "
        + top_success["roi"].astype(str)
        + " | "
        + top_success["signal_method"].astype(str)
    )

    plt.figure(figsize=(11, 5))
    plt.bar(top_success["configuration"], top_success["success_rate"])
    plt.xlabel("Configuration")
    plt.ylabel("Success rate")
    plt.title("Success Rate of Top 15 Configurations")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "success_rate_top_15_configurations.png", dpi=300)
    plt.close()

    print()
    print("Saved summaries to:")
    print(SUMMARY_DIR)
    print()
    print("Saved figures to:")
    print(FIGURES_DIR)

if __name__ == "__main__":
    main()