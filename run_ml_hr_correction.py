from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CANDIDATES = [
    PROJECT_ROOT / "results" / "baseline_with_groundtruth_v4_1.csv",
    PROJECT_ROOT / "results" / "baseline_with_groundtruth_v4.1.csv",
]

OUT_DIR = PROJECT_ROOT / "results" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_MODE = "final_pipeline_all_rois"

RANDOM_STATE = 42
MIN_ROWS_FOR_BEST_CONFIG = 60

# ============================================================
# Utility functions
# ============================================================
def filter_final_pipeline_all_rois(df):
    filtered = df.copy()

    modality_col = pick_column(
        filtered,
        ["channel", "modality", "imaging_modality", "imaging_channel"],
        required=False,
    )
    source_col = pick_column(
        filtered,
        ["source_method", "signal_method"],
        required=False,
    )
    method_col = pick_column(
        filtered,
        ["hr_estimation_method", "method", "estimation_method"],
        required=False,
    )
    window_col = pick_column(
        filtered,
        ["window_size_seconds", "window_seconds", "window_size"],
        required=False,
    )
    overlap_col = pick_column(
        filtered,
        ["overlap", "window_overlap"],
        required=False,
    )

    if modality_col:
        filtered = filtered[
            safe_contains(filtered[modality_col], "color")
            | safe_contains(filtered[modality_col], "colour")
        ]

    if source_col:
        filtered = filtered[safe_contains(filtered[source_col], "fastica")]

    if method_col:
        possible = filtered[safe_contains(filtered[method_col], "interp")]
        if len(possible) > 0:
            filtered = possible

    if window_col:
        filtered[window_col] = pd.to_numeric(filtered[window_col], errors="coerce")
        filtered = filtered[np.isclose(filtered[window_col], 5.0, atol=0.001)]

    if overlap_col:
        filtered[overlap_col] = pd.to_numeric(filtered[overlap_col], errors="coerce")
        filtered = filtered[np.isclose(filtered[overlap_col], 0.75, atol=0.001)]

    return filtered

def find_input_file():
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find v4.1 ground-truth results. Expected one of:\n"
        + "\n".join(str(p) for p in INPUT_CANDIDATES)
        + "\n\nRun the v4.1 pipeline and add_groundtruth script first."
    )

def pick_column(df, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns are:\n{list(df.columns)}"
        )

    return None

def safe_contains(series, text):
    return series.astype(str).str.lower().str.contains(text.lower(), na=False)

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    err = y_pred - y_true

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    me = float(np.mean(err))
    max_abs_error = float(np.max(np.abs(err)))

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = np.nan

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan

    return {
        "n": len(y_true),
        "ME": me,
        "MAE": mae,
        "RMSE": rmse,
        "MaxAbsError": max_abs_error,
        "PearsonR": pearson_r,
        "R2": r2,
    }

def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def filter_known_best_v4_1(df):
    filtered = df.copy()

    modality_col = pick_column(
        filtered,
        ["channel", "modality", "imaging_modality", "imaging_channel"],
        required=False,
    )
    roi_col = pick_column(filtered, ["roi", "ROI"], required=False)
    source_col = pick_column(filtered, ["source_method", "signal_method"], required=False)
    method_col = pick_column(
        filtered,
        ["hr_estimation_method", "method", "estimation_method"],
        required=False,
    )
    window_col = pick_column(
        filtered,
        ["window_size_seconds", "window_seconds", "window_size"],
        required=False,
    )
    overlap_col = pick_column(filtered, ["overlap", "window_overlap"], required=False)

    if modality_col:
        filtered = filtered[
            safe_contains(filtered[modality_col], "color")
            | safe_contains(filtered[modality_col], "colour")
        ]

    if roi_col:
        filtered = filtered[safe_contains(filtered[roi_col], "forehead")]

    if source_col:
        filtered = filtered[safe_contains(filtered[source_col], "fastica")]

    if method_col:
        # Prefer the interpolated v4.1 method if the column exists.
        possible = filtered[safe_contains(filtered[method_col], "interp")]
        if len(possible) > 0:
            filtered = possible

    if window_col:
        filtered[window_col] = pd.to_numeric(filtered[window_col], errors="coerce")
        filtered = filtered[np.isclose(filtered[window_col], 5.0, atol=0.001)]

    if overlap_col:
        filtered[overlap_col] = pd.to_numeric(filtered[overlap_col], errors="coerce")
        filtered = filtered[np.isclose(filtered[overlap_col], 0.75, atol=0.001)]

    return filtered


def choose_best_config_if_needed(df, gt_col, est_col):
    candidate_config_cols = [
        "channel",
        "modality",
        "imaging_modality",
        "roi",
        "source_method",
        "signal_method",
        "hr_estimation_method",
        "method",
        "window_size_seconds",
        "window_seconds",
        "overlap",
    ]

    config_cols = [c for c in candidate_config_cols if c in df.columns]

    if not config_cols:
        print("No configuration columns found. Using all rows.")
        return df.copy(), {}

    temp = df.copy()
    temp[gt_col] = pd.to_numeric(temp[gt_col], errors="coerce")
    temp[est_col] = pd.to_numeric(temp[est_col], errors="coerce")
    temp = temp.dropna(subset=[gt_col, est_col])

    temp["_abs_error_tmp"] = (temp[est_col] - temp[gt_col]).abs()

    grouped = (
        temp.groupby(config_cols, dropna=False)
        .agg(n=("_abs_error_tmp", "size"), mae=("_abs_error_tmp", "mean"))
        .reset_index()
    )

    eligible = grouped[grouped["n"] >= MIN_ROWS_FOR_BEST_CONFIG]

    if len(eligible) == 0:
        eligible = grouped.sort_values(["n", "mae"], ascending=[False, True]).head(1)
    else:
        eligible = eligible.sort_values("mae").head(1)

    best = eligible.iloc[0].to_dict()

    selected = temp.copy()
    for col in config_cols:
        selected = selected[selected[col].astype(str) == str(best[col])]

    selected = selected.drop(columns=["_abs_error_tmp"], errors="ignore")

    return selected, {col: best[col] for col in config_cols}

def prepare_ml_features(df, gt_col, est_col):
    leak_keywords = [
        "ground",
        "gt",
        "truth",
        "error",
        "abs",
        "squared",
        "mae",
        "rmse",
        "pearson",
        "r2",
    ]

    useful_numeric_keywords = [
        "estimated_hr",
        "snr",
        "window",
        "intensity",
        "noise",
        "fps",
        "duration",
        "frame",
        "peak",
        "power",
        "frequency_resolution",
        "interpolation",
        "dark",
        "bright",
        "spatial",
        "temporal",
        "overlap",
    ]

    categorical_candidates = [
        "state",
        "recording_state",
        "roi",
        "channel",
        "modality",
        "imaging_modality",
        "source_method",
        "signal_method",
        "hr_estimation_method",
        "method",
    ]

    numeric_cols = [est_col]

    for col in df.columns:
        col_lower = col.lower()

        if col == est_col:
            continue

        if any(k in col_lower for k in leak_keywords):
            continue

        if any(k in col_lower for k in useful_numeric_keywords):
            numeric_cols.append(col)

    numeric_cols = list(dict.fromkeys([c for c in numeric_cols if c in df.columns]))

    categorical_cols = []
    for col in categorical_candidates:
        if col in df.columns and col not in categorical_cols:
            if df[col].nunique(dropna=True) > 1:
                categorical_cols.append(col)

    keep_cols = [gt_col, est_col] + numeric_cols + categorical_cols
    keep_cols = list(dict.fromkeys(keep_cols))

    ml_df = df[keep_cols].copy()

    ml_df[gt_col] = pd.to_numeric(ml_df[gt_col], errors="coerce")
    ml_df[est_col] = pd.to_numeric(ml_df[est_col], errors="coerce")

    for col in numeric_cols:
        ml_df[col] = pd.to_numeric(ml_df[col], errors="coerce")

    for col in categorical_cols:
        ml_df[col] = ml_df[col].astype(str).fillna("unknown")

    ml_df = ml_df.dropna(subset=[gt_col, est_col])

    X = ml_df[numeric_cols + categorical_cols]
    y_true = ml_df[gt_col].to_numpy(dtype=float)
    y_est = ml_df[est_col].to_numpy(dtype=float)
    y_residual = y_true - y_est

    return ml_df, X, y_true, y_est, y_residual, numeric_cols, categorical_cols

def build_model(model_name, numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers = []

    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    if model_name == "Ridge":
        model = Ridge(alpha=1.0)

    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "GradientBoosting":
        model = GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

def participant_groups(df):
    participant_col = pick_column(
        df,
        ["participant_id", "participant", "participantID", "subject", "subject_id"],
        required=False,
    )

    if participant_col is None:
        return None, None

    return participant_col, df[participant_col].astype(str).to_numpy()


# ============================================================
# Main script
# ============================================================
def main():
    warnings.filterwarnings("ignore")

    input_path = find_input_file()
    print(f"Loading: {input_path}")

    df = pd.read_csv(input_path)

    gt_col = pick_column(
        df,
        [
            "groundtruth_hr_bpm",
            "ground_truth_hr_bpm",
            "gt_hr_bpm",
            "hr_groundtruth",
            "groundtruth_hr",
            "ground_truth_hr",
            "median_groundtruth_hr_bpm",
            "median_gt_hr_bpm",
        ],
        required=True,
    )

    est_col = pick_column(
        df,
        [
            "estimated_hr_bpm",
            "estimated_hr",
            "hr_estimate_bpm",
            "predicted_hr_bpm",
            "estimated_hr_mean_bpm",
        ],
        required=True,
    )

    df[gt_col] = pd.to_numeric(df[gt_col], errors="coerce")
    df[est_col] = pd.to_numeric(df[est_col], errors="coerce")
    df = df.dropna(subset=[gt_col, est_col])

    print(f"Detected ground-truth column: {gt_col}")
    print(f"Detected estimated-HR column: {est_col}")
    print(f"Rows before ML filtering: {len(df)}")

    selected_config_info = {}

    if DATA_MODE == "final_pipeline_all_rois":
        df_selected = filter_final_pipeline_all_rois(df)
        selected_config_info = {
            "mode": "final_pipeline_all_rois",
            "expected": "colour + FastICA + 5s + 0.75 overlap + interpolation, all five ROIs",
        }

    elif DATA_MODE == "best_config":
        df_selected = filter_known_best_v4_1(df)

        if len(df_selected) < MIN_ROWS_FOR_BEST_CONFIG:
            print(
                "Known v4.1 best configuration filter returned too few rows. "
                "Selecting best available configuration from the CSV."
            )
            df_selected, selected_config_info = choose_best_config_if_needed(
                df, gt_col, est_col
            )
        else:
            selected_config_info = {
                "mode": "known_best_v4_1",
                "expected": "colour + forehead + FastICA + 5s + 0.75 overlap + interpolation",
            }

    elif DATA_MODE == "all_configs":
        df_selected = df.copy()
        selected_config_info = {"mode": "all_configs"}

    else:
        raise ValueError(
            "DATA_MODE must be 'final_pipeline_all_rois', 'best_config', or 'all_configs'."
        )
    print(f"Rows used for ML: {len(df_selected)}")

    roi_col = pick_column(df_selected, ["roi", "ROI"], required=False)
    if roi_col:
        print("\nROIs included in ML dataset:")
        print(df_selected[roi_col].value_counts())

    if len(df_selected) < 30:
        raise ValueError(
            "Too few rows for ML. Check whether your v4.1 results and ground-truth merge are correct."
        )

    ml_df, X, y_true, y_est, y_residual, numeric_cols, categorical_cols = prepare_ml_features(
        df_selected, gt_col, est_col
    )

    participant_col, groups = participant_groups(df_selected.loc[ml_df.index])

    print("\nNumeric features:")
    for col in numeric_cols:
        print(f"  - {col}")

    print("\nCategorical features:")
    for col in categorical_cols:
        print(f"  - {col}")

    if groups is not None and len(np.unique(groups)) >= 5:
        cv = LeaveOneGroupOut()
        splits = list(cv.split(X, y_residual, groups=groups))
        cv_name = f"Leave-one-participant-out CV using column '{participant_col}'"
    else:
        n_splits = min(5, len(ml_df))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(X, y_residual))
        cv_name = f"{n_splits}-fold CV without participant grouping"

    print(f"\nCross-validation strategy: {cv_name}")

    predictions = pd.DataFrame(
        {
            "groundtruth_hr_bpm": y_true,
            "baseline_estimated_hr_bpm": y_est,
        }
    )

    for col in [
        "participant_id",
        "participant",
        "state",
        "recording_state",
        "roi",
        "channel",
        "modality",
        "source_method",
        "window_size_seconds",
        "overlap",
    ]:
        if col in df_selected.columns:
            predictions[col] = df_selected.loc[ml_df.index, col].values

    summary_rows = []

    baseline_metrics = calculate_metrics(y_true, y_est)
    baseline_metrics["Model"] = "Baseline_v4_1_uncorrected"
    summary_rows.append(baseline_metrics)

    model_names = ["Ridge", "RandomForest", "GradientBoosting"]

    for model_name in model_names:
        print(f"\nTraining residual-correction model: {model_name}")

        y_pred_corrected = np.full_like(y_true, fill_value=np.nan, dtype=float)

        for train_idx, test_idx in splits:
            model = build_model(model_name, numeric_cols, categorical_cols)

            model.fit(X.iloc[train_idx], y_residual[train_idx])

            predicted_residual = model.predict(X.iloc[test_idx])
            y_pred_corrected[test_idx] = y_est[test_idx] + predicted_residual

        predictions[f"{model_name}_corrected_hr_bpm"] = y_pred_corrected

        metrics = calculate_metrics(y_true, y_pred_corrected)
        metrics["Model"] = f"{model_name}_residual_correction"
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)
    summary = summary[
        ["Model", "n", "ME", "MAE", "RMSE", "MaxAbsError", "PearsonR", "R2"]
    ]

    baseline_mae = summary.loc[
        summary["Model"] == "Baseline_v4_1_uncorrected", "MAE"
    ].iloc[0]

    baseline_rmse = summary.loc[
        summary["Model"] == "Baseline_v4_1_uncorrected", "RMSE"
    ].iloc[0]

    summary["MAE_change_vs_baseline_percent"] = (
        (summary["MAE"] - baseline_mae) / baseline_mae * 100
    )

    summary["RMSE_change_vs_baseline_percent"] = (
        (summary["RMSE"] - baseline_rmse) / baseline_rmse * 100
    )

    summary_path = OUT_DIR / "ml_results_summary.csv"
    predictions_path = OUT_DIR / "ml_predictions.csv"
    dataset_path = OUT_DIR / "ml_dataset_used.csv"
    info_path = OUT_DIR / "ml_experiment_info.txt"

    summary.to_csv(summary_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    ml_df.to_csv(dataset_path, index=False)

    best_model_row = summary[summary["Model"] != "Baseline_v4_1_uncorrected"].sort_values(
        "MAE"
    ).iloc[0]

    best_model_name = best_model_row["Model"].replace("_residual_correction", "")
    best_prediction_col = f"{best_model_name}_corrected_hr_bpm"

    with open(info_path, "w") as f:
        f.write("Machine-learning HR correction experiment\n")
        f.write("=========================================\n\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Data mode: {DATA_MODE}\n")
        f.write(f"Rows used: {len(ml_df)}\n")
        f.write(f"Cross-validation: {cv_name}\n\n")
        f.write("Selected configuration information:\n")
        for key, value in selected_config_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nNumeric features:\n")
        for col in numeric_cols:
            f.write(f"  - {col}\n")
        f.write("\nCategorical features:\n")
        for col in categorical_cols:
            f.write(f"  - {col}\n")
        f.write("\nBest ML model by MAE:\n")
        f.write(str(best_model_row.to_dict()))
        f.write("\n")

    # Plot 1: MAE/RMSE comparison
    plot_summary = summary.copy()
    x = np.arange(len(plot_summary))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, plot_summary["MAE"], width, label="MAE")
    plt.bar(x + width / 2, plot_summary["RMSE"], width, label="RMSE")
    plt.xticks(x, plot_summary["Model"], rotation=25, ha="right")
    plt.ylabel("Error (BPM)")
    plt.title("ML correction compared with uncorrected v4.1 estimates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ml_error_comparison.png", dpi=300)
    plt.close()

    # Plot 2: estimated vs ground truth for best ML correction
    min_hr = min(np.nanmin(y_true), np.nanmin(predictions[best_prediction_col]), np.nanmin(y_est))
    max_hr = max(np.nanmax(y_true), np.nanmax(predictions[best_prediction_col]), np.nanmax(y_est))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_est, alpha=0.55, label="Uncorrected v4.1")
    plt.scatter(y_true, predictions[best_prediction_col], alpha=0.55, label=best_model_name)
    plt.plot([min_hr, max_hr], [min_hr, max_hr], linestyle="--", label="Perfect agreement")
    plt.xlabel("Ground-truth HR (BPM)")
    plt.ylabel("Estimated HR (BPM)")
    plt.title("Ground truth vs estimated HR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ml_scatter_best_model.png", dpi=300)
    plt.close()

    print("\nDone.")
    print(f"Summary saved to: {summary_path}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Dataset used saved to: {dataset_path}")
    print(f"Info saved to: {info_path}")
    print(f"Figures saved to: {OUT_DIR}")

    print("\nResults summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()