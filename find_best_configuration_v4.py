from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Project paths
# ===========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "baseline_with_groundtruth_v4.csv"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries_v4"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "best_configuration_v4"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helper metrics
# ============================================================
def rmse_from_errors(errors):
    errors = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(np.square(errors))))

def pearson_r(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def r2_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-8:
        return np.nan
    return float(1 - (ss_res / ss_tot))

def make_config_label(row, columns):
    return " | ".join(str(row[column]) for column in columns)

# ============================================================
# Summary function
# ============================================================
# Summarise performance for any grouping
def summarise_configuration(df, group_columns):
    total_counts = (
        df
        .groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="n_total")
    )

    successful = df.dropna(subset=[
        "estimated_hr_bpm",
        "groundtruth_hr_bpm",
        "absolute_error_bpm",
    ]).copy()

    metric_rows = []

    for key, group in successful.groupby(group_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = {
            column: value
            for column, value in zip(group_columns, key)
        }

        y_true = group["groundtruth_hr_bpm"].to_numpy(dtype=float)
        y_pred = group["estimated_hr_bpm"].to_numpy(dtype=float)
        errors = group["absolute_error_bpm"].to_numpy(dtype=float)

        row.update({
            "n_success": len(group),
            "mae_bpm": float(np.mean(errors)),
            "rmse_bpm": rmse_from_errors(errors),
            "median_error_bpm": float(np.median(errors)),
            "max_error_bpm": float(np.max(errors)),
            "mean_signed_error_bpm": float(group["signed_error_bpm"].mean()),
            "mean_estimated_hr_bpm": float(group["estimated_hr_bpm"].mean()),
            "mean_groundtruth_hr_bpm": float(group["groundtruth_hr_bpm"].mean()),
            "r2": r2_score_manual(y_true, y_pred),
            "pearson_r": pearson_r(y_true, y_pred),
        })

        if "median_snr_db" in group.columns:
            row["mean_median_snr_db"] = float(group["median_snr_db"].mean())
            row["median_median_snr_db"] = float(group["median_snr_db"].median())

        if "mean_snr_db" in group.columns:
            row["mean_snr_db"] = float(group["mean_snr_db"].mean())

        if "n_windows" in group.columns:
            row["mean_n_windows"] = float(group["n_windows"].mean())

        if "total_runtime_seconds" in group.columns:
            row["mean_runtime_seconds"] = float(group["total_runtime_seconds"].mean())

        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)

    summary = total_counts.merge(
        metrics,
        on=group_columns,
        how="left"
    )

    summary["success_rate"] = summary["n_success"] / summary["n_total"]

    summary = summary.sort_values(
        by=["mae_bpm", "rmse_bpm", "max_error_bpm", "success_rate"],
        ascending=[True, True, True, False],
        na_position="last"
    )

    return summary

# ============================================================
# Plotting helpers
# ============================================================
def save_bar_plot(summary_df, x_column, y_column, filename, title, xlabel, ylabel, top_n=None):
    plot_df = summary_df.dropna(subset=[y_column]).copy()

    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plt.figure(figsize=(10, 4))
    plt.bar(plot_df[x_column].astype(str), plot_df[y_column])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

def save_top_configuration_plot(summary_df, filename, title, config_columns, top_n=10):
    plot_df = summary_df.dropna(subset=["mae_bpm"]).head(top_n).copy()

    plot_df["configuration"] = plot_df.apply(
        lambda row: make_config_label(row, config_columns),
        axis=1
    )

    plt.figure(figsize=(14, 5))
    plt.bar(plot_df["configuration"], plot_df["mae_bpm"])
    plt.xlabel("Configuration")
    plt.ylabel("MAE (BPM)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

# Save a heatmap of MAE by window size and overlap
def save_window_heatmap(summary_df, filename, title):
    plot_df = summary_df.dropna(subset=["mae_bpm"]).copy()

    pivot = plot_df.pivot(
        index="window_size_seconds",
        columns="overlap",
        values="mae_bpm"
    )

    pivot = pivot.sort_index()

    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, aspect="auto")

    plt.xticks(
        ticks=np.arange(len(pivot.columns)),
        labels=[str(col) for col in pivot.columns]
    )

    plt.yticks(
        ticks=np.arange(len(pivot.index)),
        labels=[str(idx) for idx in pivot.index]
    )

    plt.xlabel("Overlap")
    plt.ylabel("Window size (s)")
    plt.title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if not np.isnan(value):
                plt.text(j, i, f"{value:.1f}", ha="center", va="center")

    plt.colorbar(label="MAE (BPM)")
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
            "Run src/add_groundtruth_v4.py first."
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
            print(failed["error"].value_counts().head(20))
            print()

    # ========================================================
    # Full v4 configuration
    # ========================================================
    full_config_columns = [
        "channel",
        "roi",
        "source_method",
        "hr_estimation_method",
        "window_size_seconds",
        "overlap",
    ]

    full_config = summarise_configuration(df, full_config_columns)

    full_config.to_csv(
        SUMMARY_DIR / "best_full_configuration_v4.csv",
        index=False
    )

    print("=" * 120)
    print("Top full v4 configurations")
    print("=" * 120)
    print(full_config.head(20))
    print()

    # ========================================================
    # Robust best configurations
    # ========================================================
    robust_candidates = full_config[
        (full_config["n_success"] >= 80) &
        (full_config["success_rate"] >= 0.80)
    ].copy()

    robust_candidates = robust_candidates.sort_values(
        by=["mae_bpm", "rmse_bpm", "max_error_bpm"],
        ascending=[True, True, True]
    )

    robust_candidates.to_csv(
        SUMMARY_DIR / "robust_best_full_configuration_v4.csv",
        index=False
    )

    print("=" * 120)
    print("Top robust v4 configurations")
    print("Criteria: n_success >= 80 and success_rate >= 0.80")
    print("=" * 120)
    print(robust_candidates.head(20))
    print()

    # ========================================================
    # Main factor summaries
    # ========================================================
    by_source_method = summarise_configuration(df, ["source_method"])
    by_roi = summarise_configuration(df, ["roi"])
    by_window_size = summarise_configuration(df, ["window_size_seconds"])
    by_overlap = summarise_configuration(df, ["overlap"])
    by_state = summarise_configuration(df, ["state"])
    by_source_window = summarise_configuration(df, ["source_method", "window_size_seconds"])
    by_source_overlap = summarise_configuration(df, ["source_method", "overlap"])
    by_window_overlap = summarise_configuration(df, ["window_size_seconds", "overlap"])
    by_roi_source = summarise_configuration(df, ["roi", "source_method"])
    by_state_source = summarise_configuration(df, ["state", "source_method"])
    by_state_window_overlap = summarise_configuration(df, ["state", "window_size_seconds", "overlap"])

    by_source_method.to_csv(SUMMARY_DIR / "accuracy_by_source_method_v4.csv", index=False)
    by_roi.to_csv(SUMMARY_DIR / "accuracy_by_roi_v4.csv", index=False)
    by_window_size.to_csv(SUMMARY_DIR / "accuracy_by_window_size_v4.csv", index=False)
    by_overlap.to_csv(SUMMARY_DIR / "accuracy_by_overlap_v4.csv", index=False)
    by_state.to_csv(SUMMARY_DIR / "accuracy_by_state_v4.csv", index=False)
    by_source_window.to_csv(SUMMARY_DIR / "accuracy_by_source_method_window_size_v4.csv", index=False)
    by_source_overlap.to_csv(SUMMARY_DIR / "accuracy_by_source_method_overlap_v4.csv", index=False)
    by_window_overlap.to_csv(SUMMARY_DIR / "accuracy_by_window_size_overlap_v4.csv", index=False)
    by_roi_source.to_csv(SUMMARY_DIR / "accuracy_by_roi_source_method_v4.csv", index=False)
    by_state_source.to_csv(SUMMARY_DIR / "accuracy_by_state_source_method_v4.csv", index=False)
    by_state_window_overlap.to_csv(SUMMARY_DIR / "accuracy_by_state_window_overlap_v4.csv", index=False)

    print("=" * 120)
    print("Accuracy by source method")
    print("=" * 120)
    print(by_source_method)
    print()

    print("=" * 120)
    print("Accuracy by ROI")
    print("=" * 120)
    print(by_roi)
    print()

    print("=" * 120)
    print("Accuracy by window size")
    print("=" * 120)
    print(by_window_size)
    print()

    print("=" * 120)
    print("Accuracy by overlap")
    print("=" * 120)
    print(by_overlap)
    print()

    print("=" * 120)
    print("Accuracy by state")
    print("=" * 120)
    print(by_state)
    print()

    # ========================================================
    # Best configuration per category
    # ========================================================
    best_per_source = (
        full_config
        .dropna(subset=["mae_bpm"])
        .sort_values(["mae_bpm", "rmse_bpm"])
        .groupby("source_method")
        .head(1)
        .sort_values("mae_bpm")
    )

    best_per_roi = (
        full_config
        .dropna(subset=["mae_bpm"])
        .sort_values(["mae_bpm", "rmse_bpm"])
        .groupby("roi")
        .head(1)
        .sort_values("mae_bpm")
    )

    best_per_state = (
        summarise_configuration(df, ["state"] + full_config_columns)
        .dropna(subset=["mae_bpm"])
        .sort_values(["mae_bpm", "rmse_bpm"])
        .groupby("state")
        .head(1)
        .sort_values("state")
    )

    best_per_source.to_csv(SUMMARY_DIR / "best_configuration_per_source_method_v4.csv", index=False)
    best_per_roi.to_csv(SUMMARY_DIR / "best_configuration_per_roi_v4.csv", index=False)
    best_per_state.to_csv(SUMMARY_DIR / "best_configuration_per_state_v4.csv", index=False)

    print("=" * 120)
    print("Best configuration per source method")
    print("=" * 120)
    print(best_per_source)
    print()

    print("=" * 120)
    print("Best configuration per ROI")
    print("=" * 120)
    print(best_per_roi)
    print()

    print("=" * 120)
    print("Best configuration per state")
    print("=" * 120)
    print(best_per_state)
    print()

    # ========================================================
    # Best overall individual rows
    # ========================================================
    best_overall = full_config.dropna(subset=["mae_bpm"]).iloc[0]

    best_rows = df.copy()

    for column in full_config_columns:
        best_rows = best_rows[best_rows[column] == best_overall[column]]

    best_rows = best_rows.sort_values(["participant", "state"])

    best_rows.to_csv(
        SUMMARY_DIR / "best_overall_configuration_individual_rows_v4.csv",
        index=False
    )

    print("=" * 120)
    print("Best overall configuration individual rows")
    print("=" * 120)
    print("Best configuration:")
    print(make_config_label(best_overall, full_config_columns))
    print()
    print(best_rows[[
        "participant",
        "state",
        "channel",
        "roi",
        "source_method",
        "window_size_seconds",
        "overlap",
        "estimated_hr_bpm",
        "groundtruth_hr_bpm",
        "signed_error_bpm",
        "absolute_error_bpm",
        "median_snr_db",
        "n_windows",
        "error",
    ]].head(40))
    print()

    # ========================================================
    # SNR threshold analysis
    # ========================================================
    successful = df.dropna(subset=[
        "estimated_hr_bpm",
        "groundtruth_hr_bpm",
        "absolute_error_bpm",
        "median_snr_db",
    ]).copy()

    snr_threshold_rows = []

    for threshold in [-5, 0, 3, 6, 9, 12, 15]:
        subset = successful[successful["median_snr_db"] >= threshold]

        if len(subset) == 0:
            snr_threshold_rows.append({
                "snr_threshold_db": threshold,
                "n_success": 0,
                "coverage": 0,
                "mae_bpm": np.nan,
                "rmse_bpm": np.nan,
                "max_error_bpm": np.nan,
            })
            continue

        errors = subset["absolute_error_bpm"].to_numpy(dtype=float)

        snr_threshold_rows.append({
            "snr_threshold_db": threshold,
            "n_success": len(subset),
            "coverage": len(subset) / len(successful),
            "mae_bpm": float(np.mean(errors)),
            "rmse_bpm": rmse_from_errors(errors),
            "max_error_bpm": float(np.max(errors)),
        })

    snr_threshold_summary = pd.DataFrame(snr_threshold_rows)

    snr_threshold_summary.to_csv(
        SUMMARY_DIR / "snr_threshold_analysis_v4.csv",
        index=False
    )

    print("=" * 120)
    print("SNR threshold analysis")
    print("=" * 120)
    print(snr_threshold_summary)
    print()

    # ========================================================
    # Figures
    # ========================================================
    save_top_configuration_plot(
        full_config,
        filename="top_10_full_configurations_v4_by_mae.png",
        title="Top 10 v4 Configurations by MAE",
        config_columns=full_config_columns,
        top_n=10
    )

    save_top_configuration_plot(
        robust_candidates,
        filename="top_10_robust_configurations_v4_by_mae.png",
        title="Top 10 Robust v4 Configurations by MAE",
        config_columns=full_config_columns,
        top_n=10
    )

    save_bar_plot(
        by_source_method,
        x_column="source_method",
        y_column="mae_bpm",
        filename="mae_by_source_method_v4.png",
        title="Mean Absolute Error by Source Method",
        xlabel="Source method",
        ylabel="MAE (BPM)"
    )

    save_bar_plot(
        by_roi,
        x_column="roi",
        y_column="mae_bpm",
        filename="mae_by_roi_v4.png",
        title="Mean Absolute Error by ROI",
        xlabel="ROI",
        ylabel="MAE (BPM)"
    )

    save_bar_plot(
        by_window_size,
        x_column="window_size_seconds",
        y_column="mae_bpm",
        filename="mae_by_window_size_v4.png",
        title="Mean Absolute Error by Window Size",
        xlabel="Window size (s)",
        ylabel="MAE (BPM)"
    )

    save_bar_plot(
        by_overlap,
        x_column="overlap",
        y_column="mae_bpm",
        filename="mae_by_overlap_v4.png",
        title="Mean Absolute Error by Window Overlap",
        xlabel="Overlap",
        ylabel="MAE (BPM)"
    )

    save_bar_plot(
        by_state,
        x_column="state",
        y_column="mae_bpm",
        filename="mae_by_state_v4.png",
        title="Mean Absolute Error by Recording State",
        xlabel="State",
        ylabel="MAE (BPM)"
    )

    save_window_heatmap(
        by_window_overlap,
        filename="mae_heatmap_window_size_overlap_v4.png",
        title="MAE by Window Size and Overlap"
    )

    save_bar_plot(
        snr_threshold_summary,
        x_column="snr_threshold_db",
        y_column="mae_bpm",
        filename="mae_by_snr_threshold_v4.png",
        title="MAE After Applying SNR Threshold",
        xlabel="SNR threshold (dB)",
        ylabel="MAE (BPM)"
    )

    save_bar_plot(
        snr_threshold_summary,
        x_column="snr_threshold_db",
        y_column="coverage",
        filename="coverage_by_snr_threshold_v4.png",
        title="Coverage After Applying SNR Threshold",
        xlabel="SNR threshold (dB)",
        ylabel="Coverage"
    )

    print("Saved summaries to:")
    print(SUMMARY_DIR)
    print()

    print("Saved figures to:")
    print(FIGURES_DIR)

if __name__ == "__main__":
    main()