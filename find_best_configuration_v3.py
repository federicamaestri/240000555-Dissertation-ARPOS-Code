from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "baseline_with_groundtruth_v3.csv"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries_v3"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "best_configuration_v3"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helper functions
# ============================================================
def rmse_from_errors(errors):
    return np.sqrt(np.mean(np.square(errors)))

# Summarise the configuration
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
            max_error_bpm=("absolute_error_bpm", "max"),
            mean_signed_error_bpm=("signed_error_bpm", "mean"),
            mean_estimated_hr_bpm=("estimated_hr_bpm", "mean"),
            mean_groundtruth_hr_bpm=("groundtruth_hr_bpm", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
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
        by=["mae_bpm", "rmse_bpm", "success_rate"],
        ascending=[True, True, False]
    )

    return summary


def save_top_configuration_plot(summary_df, filename, title, top_n=10):
    plot_df = summary_df.dropna(subset=["mae_bpm"]).head(top_n).copy()

    plot_df["configuration"] = (
        plot_df["channel"].astype(str)
        + " | "
        + plot_df["roi"].astype(str)
        + " | "
        + plot_df["signal_method"].astype(str)
        + " | "
        + plot_df["hr_estimation_method"].astype(str)
    )

    plt.figure(figsize=(13, 5))
    plt.bar(plot_df["configuration"], plot_df["mae_bpm"])
    plt.xlabel("Configuration")
    plt.ylabel("MAE (BPM)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def save_metric_by_category(summary_df, x_column, y_column, filename, title, xlabel, ylabel):
    plot_df = summary_df.dropna(subset=[y_column]).copy()

    plt.figure(figsize=(9, 4))
    plt.bar(plot_df[x_column].astype(str), plot_df[y_column])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
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
            "Run src/add_groundtruth_v3.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    print("Loaded:")
    print(INPUT_PATH)
    print()

    print("Total rows:", len(df))
    print("Successful rows:", df["estimated_hr_bpm"].notna().sum())
    print("Failed rows:", df["estimated_hr_bpm"].isna().sum())
    print()

    if "error" in df.columns:
        failed = df[df["estimated_hr_bpm"].isna()]
        if len(failed) > 0:
            print("Failure reasons:")
            print(failed["error"].value_counts())
            print()

    # ========================================================
    # 1. Best full configuration:
    # channel + ROI + signal_method + HR estimation method
    # ========================================================
    full_config = summarise_configuration(
        df,
        ["channel", "roi", "signal_method", "hr_estimation_method"]
    )

    full_config.to_csv(
        SUMMARY_DIR / "best_full_configuration_v3.csv",
        index=False
    )

    print("=" * 100)
    print("Top full configurations")
    print("channel + ROI + signal_method + HR estimation method")
    print("=" * 100)
    print(full_config.head(20))
    print()

    # ========================================================
    # 2. Best HR estimation method overall
    # ========================================================
    by_hr_method = summarise_configuration(
        df,
        ["hr_estimation_method"]
    )

    by_hr_method.to_csv(
        SUMMARY_DIR / "accuracy_by_hr_estimation_method.csv",
        index=False
    )

    print("=" * 100)
    print("Accuracy by HR estimation method")
    print("=" * 100)
    print(by_hr_method)
    print()

    # ========================================================
    # 3. Best signal method overall
    # ========================================================
    by_signal_method = summarise_configuration(
        df,
        ["signal_method"]
    )

    by_signal_method.to_csv(
        SUMMARY_DIR / "accuracy_by_signal_method_v3.csv",
        index=False
    )

    print("=" * 100)
    print("Accuracy by signal method")
    print("=" * 100)
    print(by_signal_method)
    print()

    # ========================================================
    # 4. Best ROI overall
    # ========================================================
    by_roi = summarise_configuration(
        df,
        ["roi"]
    )

    by_roi.to_csv(
        SUMMARY_DIR / "accuracy_by_roi_v3.csv",
        index=False
    )

    print("=" * 100)
    print("Accuracy by ROI")
    print("=" * 100)
    print(by_roi)
    print()

    # ========================================================
    # 5. Best channel overall
    # ========================================================
    by_channel = summarise_configuration(
        df,
        ["channel"]
    )

    by_channel.to_csv(
        SUMMARY_DIR / "accuracy_by_channel_v3.csv",
        index=False
    )

    print("=" * 100)
    print("Accuracy by channel")
    print("=" * 100)
    print(by_channel)
    print()

    # ========================================================
    # 6. Best state overall
    # ========================================================
    by_state = summarise_configuration(
        df,
        ["state"]
    )

    by_state.to_csv(
        SUMMARY_DIR / "accuracy_by_state_v3.csv",
        index=False
    )

    print("=" * 100)
    print("Accuracy by state")
    print("=" * 100)
    print(by_state)
    print()

    # ========================================================
    # 7. Best configuration by state
    # ========================================================
    config_by_state = summarise_configuration(
        df,
        ["state", "channel", "roi", "signal_method", "hr_estimation_method"]
    )

    config_by_state.to_csv(
        SUMMARY_DIR / "best_configuration_by_state_v3.csv",
        index=False
    )

    print("=" * 100)
    print("Best configurations by state")
    print("=" * 100)

    for state, group in config_by_state.groupby("state"):
        print()
        print(f"State: {state}")
        print(group.head(10))

    print()

    # ========================================================
    # 8. Best configuration per HR estimation method
    # ========================================================
    best_per_hr_method = (
        full_config
        .dropna(subset=["mae_bpm"])
        .sort_values("mae_bpm")
        .groupby("hr_estimation_method")
        .head(1)
        .sort_values("mae_bpm")
    )

    best_per_hr_method.to_csv(
        SUMMARY_DIR / "best_configuration_per_hr_method_v3.csv",
        index=False
    )

    print("=" * 100)
    print("Best configuration per HR estimation method")
    print("=" * 100)
    print(best_per_hr_method)
    print()

    # ========================================================
    # 9. Robust best configurations
    # ========================================================
    robust_candidates = full_config[
        (full_config["n_success"] >= 6) &
        (full_config["success_rate"] >= 0.80)
    ].copy()

    robust_candidates = robust_candidates.sort_values(
        by=["mae_bpm", "rmse_bpm", "max_error_bpm"],
        ascending=[True, True, True]
    )

    robust_candidates.to_csv(
        SUMMARY_DIR / "robust_best_full_configuration_v3.csv",
        index=False
    )

    print("=" * 100)
    print("Robust best configurations")
    print("Criteria: n_success >= 6 and success_rate >= 0.80")
    print("=" * 100)

    if len(robust_candidates) == 0:
        print("No configurations passed the robustness filter.")
    else:
        print(robust_candidates.head(20))

    print()

    # ========================================================
    # 10. Compare global FFT vs Welch vs windowed FFT
    #     for the best v2-style configuration
    # ========================================================
    best_v2_like = df[
        (df["channel"] == "Color") &
        (df["roi"] == "cheeksCombined") &
        (df["signal_method"] == "green_minus_blue")
    ].copy()

    best_v2_like_summary = summarise_configuration(
        best_v2_like,
        ["hr_estimation_method"]
    )

    best_v2_like_summary.to_csv(
        SUMMARY_DIR / "cheeks_combined_green_minus_blue_by_hr_method.csv",
        index=False
    )

    print("=" * 100)
    print("Color + cheeksCombined + green_minus_blue by HR estimation method")
    print("=" * 100)
    print(best_v2_like_summary)
    print()

    # ========================================================
    # 11. Individual rows for best overall configuration
    # ========================================================
    best_overall = full_config.dropna(subset=["mae_bpm"]).iloc[0]

    best_rows = df[
        (df["channel"] == best_overall["channel"]) &
        (df["roi"] == best_overall["roi"]) &
        (df["signal_method"] == best_overall["signal_method"]) &
        (df["hr_estimation_method"] == best_overall["hr_estimation_method"])
    ].copy()

    best_rows = best_rows.sort_values(["participant", "state"])

    best_rows.to_csv(
        SUMMARY_DIR / "best_overall_configuration_individual_rows_v3.csv",
        index=False
    )

    print("=" * 100)
    print("Best overall configuration individual rows")
    print("=" * 100)
    print(
        best_rows[[
            "participant",
            "state",
            "channel",
            "roi",
            "signal_method",
            "hr_estimation_method",
            "estimated_hr_bpm",
            "groundtruth_hr_bpm",
            "signed_error_bpm",
            "absolute_error_bpm",
            "error",
        ]]
    )
    print()

    # ========================================================
    # 12. Figures
    # ========================================================
    save_top_configuration_plot(
        full_config,
        filename="top_10_full_configurations_v3_by_mae.png",
        title="Top 10 Full Configurations by MAE",
        top_n=10
    )

    save_top_configuration_plot(
        robust_candidates,
        filename="top_10_robust_configurations_v3_by_mae.png",
        title="Top 10 Robust Full Configurations by MAE",
        top_n=10
    )

    save_metric_by_category(
        by_hr_method,
        x_column="hr_estimation_method",
        y_column="mae_bpm",
        filename="mae_by_hr_estimation_method.png",
        title="Mean Absolute Error by HR Estimation Method",
        xlabel="HR estimation method",
        ylabel="MAE (BPM)"
    )

    save_metric_by_category(
        by_signal_method,
        x_column="signal_method",
        y_column="mae_bpm",
        filename="mae_by_signal_method_v3.png",
        title="Mean Absolute Error by Signal Method",
        xlabel="Signal method",
        ylabel="MAE (BPM)"
    )

    save_metric_by_category(
        by_roi,
        x_column="roi",
        y_column="mae_bpm",
        filename="mae_by_roi_v3.png",
        title="Mean Absolute Error by ROI",
        xlabel="ROI",
        ylabel="MAE (BPM)"
    )

    save_metric_by_category(
        by_channel,
        x_column="channel",
        y_column="mae_bpm",
        filename="mae_by_channel_v3.png",
        title="Mean Absolute Error by Channel",
        xlabel="Channel",
        ylabel="MAE (BPM)"
    )

    save_metric_by_category(
        by_state,
        x_column="state",
        y_column="mae_bpm",
        filename="mae_by_state_v3.png",
        title="Mean Absolute Error by Recording State",
        xlabel="Recording state",
        ylabel="MAE (BPM)"
    )

    save_metric_by_category(
        best_v2_like_summary,
        x_column="hr_estimation_method",
        y_column="mae_bpm",
        filename="cheeks_combined_green_minus_blue_by_hr_method.png",
        title="Color + cheeksCombined + green-minus-blue by HR Method",
        xlabel="HR estimation method",
        ylabel="MAE (BPM)"
    )

    # Success rate of top 15 configurations
    top_success = full_config.dropna(subset=["mae_bpm"]).head(15).copy()

    top_success["configuration"] = (
        top_success["channel"].astype(str)
        + " | "
        + top_success["roi"].astype(str)
        + " | "
        + top_success["signal_method"].astype(str)
        + " | "
        + top_success["hr_estimation_method"].astype(str)
    )

    plt.figure(figsize=(13, 5))
    plt.bar(top_success["configuration"], top_success["success_rate"])
    plt.xlabel("Configuration")
    plt.ylabel("Success rate")
    plt.title("Success Rate of Top 15 Configurations")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "success_rate_top_15_configurations_v3.png", dpi=300)
    plt.close()

    print("Saved summaries to:")
    print(SUMMARY_DIR)
    print()
    print("Saved figures to:")
    print(FIGURES_DIR)

if __name__ == "__main__":
    main()