from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# ============================================================
# Settings
# ============================================================
SKIP_IF_OUTPUT_EXISTS = False

RUN_V1 = True
RUN_V2 = True
RUN_V3 = True
RUN_V4 = True
RUN_V4_1 = True

# ============================================================
# Helper functions
# ============================================================
def script_path(*parts):
    return PROJECT_ROOT / "src" / Path(*parts)

def output_path(*parts):
    return PROJECT_ROOT / "results" / Path(*parts)

def find_existing_script(*possible_names):
    for name in possible_names:
        path = script_path(name)
        if path.exists():
            return path

    return None

def run_step(name, script, expected_output=None):
    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    if script is None:
        print("SKIPPED: script not found.")
        return

    if not script.exists():
        print(f"SKIPPED: missing script: {script}")
        return

    if expected_output is not None:
        expected_output = Path(expected_output)

        if SKIP_IF_OUTPUT_EXISTS and expected_output.exists():
            print(f"SKIPPED: output already exists: {expected_output}")
            return

    start = time.perf_counter()

    print(f"Running: {script}")
    print()

    result = subprocess.run(
        [PYTHON, str(script)],
        cwd=str(PROJECT_ROOT),
    )

    runtime = time.perf_counter() - start

    if result.returncode != 0:
        print()
        print(f"FAILED after {runtime:.2f}s")
        print(f"Script: {script}")
        sys.exit(result.returncode)

    print()
    print(f"Finished in {runtime:.2f}s")


# ============================================================
# Main pipeline
# ============================================================
def main():
    print("ARPOS dissertation full pipeline")
    print("Project root:", PROJECT_ROOT)
    print("Python:", PYTHON)
    print()

    # --------------------------------------------------------
    # Dataset index
    # --------------------------------------------------------
    run_step(
        name="1. Build dataset index",
        script=script_path("make_dataset_index.py"),
        expected_output=output_path("dataset_index.csv"),
    )

    # --------------------------------------------------------
    # v1
    # --------------------------------------------------------
    if RUN_V1:
        run_step(
            name="2. v1 baseline",
            script=script_path("run_baseline_rppg.py"),
            expected_output=output_path("baseline_rppg_results.csv"),
        )

        run_step(
            name="3. Add ground truth to v1",
            script=script_path("add_groundtruth.py"),
            expected_output=output_path("baseline_with_groundtruth.csv"),
        )

        run_step(
            name="4. Analyse v1 accuracy",
            script=script_path("analyze_baseline_accuracy.py"),
        )

        run_step(
            name="Optional: Plot v1 baseline summary",
            script=script_path("plot_baseline_summary.py"),
        )

    # --------------------------------------------------------
    # v2
    # --------------------------------------------------------
    if RUN_V2:
        run_step(
            name="5. v2 baseline",
            script=script_path("run_baseline_rppg_v2.py"),
            expected_output=output_path("baseline_rppg_results_v2.csv"),
        )

        run_step(
            name="6. Add ground truth to v2",
            script=script_path("add_groundtruth_v2.py"),
            expected_output=output_path("baseline_with_groundtruth_v2.csv"),
        )

        run_step(
            name="7. Analyse v2 accuracy",
            script=script_path("analyze_baseline_accuracy_v2.py"),
        )

        run_step(
            name="8. Find best v2 configuration",
            script=script_path("find_best_configuration_v2.py"),
            expected_output=output_path("summaries_v2", "best_full_configuration_v2.csv"),
        )

    # --------------------------------------------------------
    # v3
    # --------------------------------------------------------
    if RUN_V3:
        run_step(
            name="9. v3 baseline",
            script=script_path("run_baseline_rppg_v3.py"),
            expected_output=output_path("baseline_rppg_results_v3.csv"),
        )

        run_step(
            name="10. Add ground truth to v3",
            script=script_path("add_groundtruth_v3.py"),
            expected_output=output_path("baseline_with_groundtruth_v3.csv"),
        )

        run_step(
            name="11. Find best v3 configuration",
            script=script_path("find_best_configuration_v3.py"),
            expected_output=output_path("summaries_v3", "best_full_configuration_v3.csv"),
        )

    # --------------------------------------------------------
    # v4 original
    # --------------------------------------------------------
    if RUN_V4:
        run_step(
            name="12. v4 original baseline",
            script=script_path("run_baseline_rppg_v4.py"),
            expected_output=output_path("baseline_rppg_results_v4.csv"),
        )

        run_step(
            name="13. Add ground truth to v4",
            script=script_path("add_groundtruth_v4.py"),
            expected_output=output_path("baseline_with_groundtruth_v4.csv"),
        )

        run_step(
            name="14. Find best v4 configuration",
            script=script_path("find_best_configuration_v4.py"),
            expected_output=output_path("summaries_v4", "best_full_configuration_v4.csv"),
        )

        run_step(
            name="15. Analyse v4 reliability",
            script=script_path("analyze_v4_reliability.py"),
            expected_output=output_path("summaries_v4_reliability", "reliability_main_summary_v4.csv"),
        )

    # --------------------------------------------------------
    # v4.1 final
    # --------------------------------------------------------
    if RUN_V4_1:
        v4_1_run_script = find_existing_script(
            "run_baseline_rppg_v4_1.py",
            "run_baseline_rppg_v4.1.py",
        )

        v4_1_gt_script = find_existing_script(
            "add_groundtruth_v4_1.py",
            "add_groundtruth_v4.1.py",
        )

        v4_1_find_script = find_existing_script(
            "find_best_configuration_v4_1.py",
            "find_best_configuration_v4.1.py",
        )

        v4_1_reliability_script = find_existing_script(
            "analyze_v4_1_reliability.py",
            "analyze_v4.1_reliability.py",
        )

        run_step(
            name="16. v4.1 final baseline",
            script=v4_1_run_script,
            expected_output=output_path("baseline_rppg_results_v4_1.csv"),
        )

        run_step(
            name="17. Add ground truth to v4.1",
            script=v4_1_gt_script,
            expected_output=output_path("baseline_with_groundtruth_v4_1.csv"),
        )

        run_step(
            name="18. Find best v4.1 configuration",
            script=v4_1_find_script,
            expected_output=output_path("summaries_v4_1", "best_full_configuration_v4_1.csv"),
        )

        run_step(
            name="19. Analyse v4.1 reliability",
            script=v4_1_reliability_script,
            expected_output=output_path("summaries_v4_1_reliability", "reliability_main_summary_v4_1.csv"),
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------
    run_step(
        name="20. Final version comparison",
        script=script_path("compare_versions.py"),
        expected_output=output_path("summaries_version_comparison", "version_comparison_summary.csv"),
    )

    print()
    print("=" * 100)
    print("Pipeline complete")
    print("=" * 100)
    print()
    print("Main final outputs to check:")
    print(output_path("summaries_version_comparison", "version_comparison_summary.csv"))
    print(output_path("figures", "version_comparison", "best_configuration_mae_by_version.png"))
    print(output_path("figures", "v4_1_reliability", "estimated_vs_groundtruth_best_v4_1.png"))

if __name__ == "__main__":
    main()