from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "summaries_version_comparison_with_v4_1"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "version_comparison_with_v4_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Version files
# ============================================================
VERSION_FILES = [
    {
        "version": "v1_green_fft",
        "path": PROJECT_ROOT / "results" / "baseline_with_groundtruth.csv",
        "config_columns": ["channel", "roi"],
    },
    {
        "version": "v2_colour_signals_fft",
        "path": PROJECT_ROOT / "results" / "baseline_with_groundtruth_v2.csv",
        "config_columns": ["channel", "roi", "signal_method"],
    },
    {
        "version": "v3_colour_signals_hr_methods",
        "path": PROJECT_ROOT / "results" / "baseline_with_groundtruth_v3.csv",
        "config_columns": ["channel", "roi", "signal_method", "hr_estimation_method"],
    },
    {
        "version": "v4_fastica_window_snr",
        "path": PROJECT_ROOT / "results" / "baseline_with_groundtruth_v4.csv",
        "config_columns": [
            "channel",
            "roi",
            "source_method",
            "hr_estimation_method",
            "window_size_seconds",
            "overlap",
        ],
    },
    {
        "version": "v4_1_interp",
        "path": PROJECT_ROOT / "results" / "baseline_with_groundtruth_v4_1.csv",
        "config_columns": [
            "channel",
            "roi",
            "source_method",
            "hr_estimation_method",
            "window_size_seconds",
            "overlap",
        ],
    },
]

# ============================================================
# Column handling
# ============================================================
ESTIMATED_HR_COLUMNS = [
    "estimated_hr_bpm",
    "estimated_hr",
    "estimated_hr_BPM",
]

GROUNDTRUTH_HR_COLUMNS = [
    "groundtruth_hr_bpm",
    "groundtruth",
    "groundtruth_median_hr_bpm",
    "groundtruth_hr",
]

ABSOLUTE_ERROR_COLUMNS = [
    "absolute_error_bpm",
    "absolute_error",
    "error",
]

def find_first_existing_column(df, candidate_columns):
    for column in candidate_columns:
        if column in df.columns:
            return column
    return None

# Standardise column names
def standardise_columns(df):
    df = df.copy()
    estimated_col = find_first_existing_column(df, ESTIMATED_HR_COLUMNS)
    groundtruth_col = find_first_existing_column(df, GROUNDTRUTH_HR_COLUMNS)
    abs_error_col = find_first_existing_column(df, ABSOLUTE_ERROR_COLUMNS)

    if estimated_col is None:
        raise ValueError(
            "Could not find estimated HR column. "
            f"Tried: {ESTIMATED_HR_COLUMNS}"
        )

    if groundtruth_col is None:
        raise ValueError(
            "Could not find ground-truth HR column. "
            f"Tried: {GROUNDTRUTH_HR_COLUMNS}"
        )

    df["estimated_hr_bpm"] = pd.to_numeric(df[estimated_col], errors="coerce")
    df["groundtruth_hr_bpm"] = pd.to_numeric(df[groundtruth_col], errors="coerce")

    calculated_signed_error = df["estimated_hr_bpm"] - df["groundtruth_hr_bpm"]
    calculated_absolute_error = calculated_signed_error.abs()

    if abs_error_col is not None:
        df["absolute_error_bpm"] = pd.to_numeric(df[abs_error_col], errors="coerce")
        df["absolute_error_bpm"] = df["absolute_error_bpm"].where(
            df["absolute_error_bpm"].notna(),
            calculated_absolute_error,
        )
    else:
        df["absolute_error_bpm"] = calculated_absolute_error

    df["signed_error_bpm"] = calculated_signed_error
    df["squared_error_bpm"] = df["signed_error_bpm"] ** 2

    return df

# ============================================================
# Metrics
# ============================================================
def rmse(errors):
    errors = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(errors ** 2)))

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
    return float(1 - ss_res / ss_tot)

def calculate_metrics(df):
    successful = df.dropna(subset=[
        "estimated_hr_bpm",
        "groundtruth_hr_bpm",
        "absolute_error_bpm",
    ]).copy()

    n_total = len(df)
    n_success = len(successful)
    n_failed = n_total - n_success

    if n_success == 0:
        return {
            "n_total": n_total,
            "n_success": 0,
            "n_failed": n_failed,
            "success_rate": 0,
            "mae_bpm": np.nan,
            "rmse_bpm": np.nan,
            "median_error_bpm": np.nan,
            "max_error_bpm": np.nan,
            "mean_signed_error_bpm": np.nan,
            "pearson_r": np.nan,
            "r2": np.nan,
        }

    y_true = successful["groundtruth_hr_bpm"].to_numpy(dtype=float)
    y_pred = successful["estimated_hr_bpm"].to_numpy(dtype=float)
    abs_errors = successful["absolute_error_bpm"].to_numpy(dtype=float)

    return {
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": n_success / n_total,
        "mae_bpm": float(np.mean(abs_errors)),
        "rmse_bpm": rmse(abs_errors),
        "median_error_bpm": float(np.median(abs_errors)),
        "max_error_bpm": float(np.max(abs_errors)),
        "mean_signed_error_bpm": float(successful["signed_error_bpm"].mean()),
        "pearson_r": pearson_r(y_true, y_pred),
        "r2": r2_score_manual(y_true, y_pred),
    }

def make_config_label(row, config_columns):
    parts = []

    for column in config_columns:
        if column in row.index:
            parts.append(str(row[column]))

    return " | ".join(parts)

def summarise_by_configuration(df, config_columns):
    available_config_columns = [
        column for column in config_columns
        if column in df.columns
    ]

    if len(available_config_columns) == 0:
        raise ValueError("No configuration columns found.")

    rows = []

    for key, group in df.groupby(available_config_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = {
            column: value
            for column, value in zip(available_config_columns, key)
        }

        metrics = calculate_metrics(group)
        row.update(metrics)
        row["configuration"] = make_config_label(pd.Series(row), available_config_columns)

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=["mae_bpm", "rmse_bpm", "max_error_bpm", "success_rate"],
        ascending=[True, True, True, False],
        na_position="last",
    )

    return summary

# ============================================================
# Plot helpers
# ============================================================
def save_bar_plot(df, x_column, y_column, filename, title, ylabel):
    plot_df = df.dropna(subset=[y_column]).copy()

    plt.figure(figsize=(11, 5))
    plt.bar(plot_df[x_column].astype(str), plot_df[y_column])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

def save_best_metric_plot(df, y_column, filename, title, ylabel):
    plot_df = df.dropna(subset=[y_column]).copy()

    plt.figure(figsize=(11, 5))
    plt.bar(plot_df["version"], plot_df[y_column])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Version")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

def save_best_mae_rmse_combined(df):
    plot_df = df.dropna(subset=[
        "best_config_mae_bpm",
        "best_config_rmse_bpm",
    ]).copy()

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, plot_df["best_config_mae_bpm"], width, label="MAE")
    plt.bar(x + width / 2, plot_df["best_config_rmse_bpm"], width, label="RMSE")

    plt.xticks(x, plot_df["version"], rotation=25, ha="right")
    plt.ylabel("Error (BPM)")
    plt.title("Best Configuration MAE and RMSE by Version")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "best_configuration_mae_rmse_by_version.png", dpi=300)
    plt.close()

def save_overall_mae_rmse_combined(df):
    plot_df = df.dropna(subset=[
        "overall_mae_bpm",
        "overall_rmse_bpm",
    ]).copy()

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, plot_df["overall_mae_bpm"], width, label="MAE")
    plt.bar(x + width / 2, plot_df["overall_rmse_bpm"], width, label="RMSE")

    plt.xticks(x, plot_df["version"], rotation=25, ha="right")
    plt.ylabel("Error (BPM)")
    plt.title("Overall MAE and RMSE by Version")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "overall_mae_rmse_by_version.png", dpi=300)
    plt.close()

# ============================================================
# Main
# ============================================================
def main():
    version_summary_rows = []
    all_config_summaries = []

    for version_info in VERSION_FILES:
        version_name = version_info["version"]
        path = version_info["path"]
        config_columns = version_info["config_columns"]

        print("=" * 100)
        print(f"Processing {version_name}")
        print("=" * 100)

        if not path.exists():
            print(f"Missing file, skipping: {path}")
            print()
            continue

        df = pd.read_csv(path)
        df = standardise_columns(df)

        overall_metrics = calculate_metrics(df)

        config_summary = summarise_by_configuration(df, config_columns)
        config_summary.insert(0, "version", version_name)

        all_config_summaries.append(config_summary)

        valid_configs = config_summary.dropna(subset=["mae_bpm"])

        if len(valid_configs) == 0:
            print("No valid configurations found, skipping best configuration.")
            print()
            continue

        best_config = valid_configs.iloc[0]

        summary_row = {
            "version": version_name,
            "file": str(path),

            "n_total": overall_metrics["n_total"],
            "n_success": overall_metrics["n_success"],
            "n_failed": overall_metrics["n_failed"],
            "success_rate": overall_metrics["success_rate"],

            "overall_mae_bpm": overall_metrics["mae_bpm"],
            "overall_rmse_bpm": overall_metrics["rmse_bpm"],
            "overall_median_error_bpm": overall_metrics["median_error_bpm"],
            "overall_max_error_bpm": overall_metrics["max_error_bpm"],
            "overall_mean_signed_error_bpm": overall_metrics["mean_signed_error_bpm"],
            "overall_pearson_r": overall_metrics["pearson_r"],
            "overall_r2": overall_metrics["r2"],

            "best_configuration": best_config["configuration"],
            "best_config_mae_bpm": best_config["mae_bpm"],
            "best_config_rmse_bpm": best_config["rmse_bpm"],
            "best_config_median_error_bpm": best_config["median_error_bpm"],
            "best_config_max_error_bpm": best_config["max_error_bpm"],
            "best_config_mean_signed_error_bpm": best_config["mean_signed_error_bpm"],
            "best_config_pearson_r": best_config["pearson_r"],
            "best_config_r2": best_config["r2"],
            "best_config_n_total": best_config["n_total"],
            "best_config_n_success": best_config["n_success"],
            "best_config_success_rate": best_config["success_rate"],
        }

        version_summary_rows.append(summary_row)

        print("Overall:")
        print(pd.DataFrame([overall_metrics]))
        print()

        print("Best configuration:")
        print(best_config)
        print()

    if len(version_summary_rows) == 0:
        raise RuntimeError("No version files were found.")

    version_summary = pd.DataFrame(version_summary_rows)

    all_config_summary = pd.concat(
        all_config_summaries,
        ignore_index=True,
    )

    version_summary.to_csv(
        OUTPUT_DIR / "version_comparison_summary_with_v4_1.csv",
        index=False,
    )

    all_config_summary.to_csv(
        OUTPUT_DIR / "all_version_configuration_summaries_with_v4_1.csv",
        index=False,
    )

    print("=" * 100)
    print("Version comparison summary")
    print("=" * 100)
    print(version_summary)
    print()

    print("=" * 100)
    print("Best configurations")
    print("=" * 100)
    print(version_summary[[
        "version",
        "best_configuration",
        "best_config_mae_bpm",
        "best_config_rmse_bpm",
        "best_config_max_error_bpm",
        "best_config_n_success",
        "best_config_success_rate",
        "best_config_pearson_r",
        "best_config_r2",
    ]])
    print()

    # ========================================================
    # Figures
    # ========================================================
    save_bar_plot(
        version_summary,
        x_column="version",
        y_column="overall_mae_bpm",
        filename="overall_mae_by_version.png",
        title="Overall MAE by Version",
        ylabel="MAE (BPM)",
    )

    save_bar_plot(
        version_summary,
        x_column="version",
        y_column="overall_rmse_bpm",
        filename="overall_rmse_by_version.png",
        title="Overall RMSE by Version",
        ylabel="RMSE (BPM)",
    )

    save_best_metric_plot(
        version_summary,
        y_column="best_config_mae_bpm",
        filename="best_configuration_mae_by_version.png",
        title="Best Configuration MAE by Version",
        ylabel="MAE (BPM)",
    )

    save_best_metric_plot(
        version_summary,
        y_column="best_config_rmse_bpm",
        filename="best_configuration_rmse_by_version.png",
        title="Best Configuration RMSE by Version",
        ylabel="RMSE (BPM)",
    )

    save_best_metric_plot(
        version_summary,
        y_column="best_config_max_error_bpm",
        filename="best_configuration_max_error_by_version.png",
        title="Best Configuration Maximum Error by Version",
        ylabel="Maximum error (BPM)",
    )

    save_bar_plot(
        version_summary,
        x_column="version",
        y_column="success_rate",
        filename="success_rate_by_version.png",
        title="Overall Success Rate by Version",
        ylabel="Success rate",
    )

    save_bar_plot(
        version_summary,
        x_column="version",
        y_column="n_success",
        filename="successful_estimates_by_version.png",
        title="Successful HR Estimates by Version",
        ylabel="Number of successful estimates",
    )

    save_best_metric_plot(
        version_summary,
        y_column="best_config_pearson_r",
        filename="best_configuration_pearson_r_by_version.png",
        title="Best Configuration Pearson r by Version",
        ylabel="Pearson r",
    )

    save_best_metric_plot(
        version_summary,
        y_column="best_config_r2",
        filename="best_configuration_r2_by_version.png",
        title="Best Configuration R² by Version",
        ylabel="R²",
    )

    save_best_mae_rmse_combined(version_summary)
    save_overall_mae_rmse_combined(version_summary)

    print("Saved comparison summaries to:")
    print(OUTPUT_DIR)
    print()

    print("Saved comparison figures to:")
    print(FIGURES_DIR)

if __name__ == "__main__":
    main()