# 240000555-Dissertation-ARPOS-Code

This repository contains the code used for an MSc dissertation on remote photoplethysmography (rPPG)-based heart-rate estimation using the ARPOS dataset.

The project compares several HR-estimation pipelines, starting from a simple FFT-based baseline and progressing to a final v4.1 pipeline using colour-based source extraction, SNR-based window selection, zero-padding, and parabolic peak interpolation. An additional exploratory machine-learning correction experiment is also included.

## Project Structure
The repository should be organised as follows:

```text
arpos-dissertation/
│
├── run_all.py
│
├── data/
│   ├── PIS-...
│   ├── PIS-...
│   └── ...
│
├── src/
│   ├── make_dataset_index.py
│   ├── run_baseline_rppg.py
│   ├── add_groundtruth.py
│   ├── analyze_baseline_accuracy.py
│   ├── run_baseline_rppg_v2.py
│   ├── add_groundtruth_v2.py
│   ├── analyze_baseline_accuracy_v2.py
│   ├── find_best_configuration_v2.py
│   ├── run_baseline_rppg_v3.py
│   ├── add_groundtruth_v3.py
│   ├── find_best_configuration_v3.py
│   ├── run_baseline_rppg_v4.py
│   ├── add_groundtruth_v4.py
│   ├── find_best_configuration_v4.py
│   ├── analyze_v4_reliability.py
│   ├── run_baseline_rppg_v4_1.py
│   ├── add_groundtruth_v4_1.py
│   ├── find_best_configuration_v4_1.py
│   ├── analyze_v4_1_reliability.py
│   ├── compare_versions.py
│   └── run_ml_hr_correction.py
│
└── results/
```

`run_all.py` must stay in the main project folder.

All other Python files should be placed inside the `src/` folder. This is important because `run_all.py` looks for the analysis scripts inside:

```text
PROJECT_ROOT / "src"
```

The `results/` folder is created automatically by the scripts when outputs are saved.

## Requirements

The scripts use the following main Python packages:

```bash
pip install numpy pandas matplotlib pillow scipy scikit-learn
```

## Dataset Placement

The ARPOS participant folders should be placed inside the `data/` folder.

For example:

```text
data/
├── PIS-XXXX/
│   ├── Resting1Cropped/
│   ├── Resting2Cropped/
│   ├── AfterExcersizeCropped/
│   └── GroundTruth/
├── PIS-XXXX/
└── ...
```

The scripts expect the ground-truth HR files to be inside each participant folder under:

```text
GroundTruth/Resting1/HR.txt
GroundTruth/Resting2/HR.txt
GroundTruth/AfterExcersize/HR.txt
```

## How to Run the Full Pipeline

From the main project folder, run:

```bash
python run_all.py
```

This runs the main dissertation analysis pipeline in the correct order.

It performs:

1. Dataset indexing
2. v1 baseline HR estimation
3. v1 ground-truth alignment
4. v1 accuracy analysis
5. v2 HR estimation
6. v2 ground-truth alignment
7. v2 accuracy analysis
8. v2 best-configuration search
9. v3 HR estimation
10. v3 ground-truth alignment
11. v3 best-configuration search
12. v4 HR estimation
13. v4 ground-truth alignment
14. v4 best-configuration search
15. v4 reliability analysis
16. v4.1 HR estimation
17. v4.1 ground-truth alignment
18. v4.1 best-configuration search
19. v4.1 reliability analysis
20. Final version comparison

After `run_all.py` has finished, run the machine-learning correction experiment separately:

```bash
python src/run_ml_hr_correction.py
```

The ML script should be run after `run_all.py` because it uses the final v4.1 ground-truth-aligned results.

## Manual Script Order

If you do not want to use `run_all.py`, run the scripts manually in the following order from the main project folder:

```bash
python src/make_dataset_index.py

python src/run_baseline_rppg.py
python src/add_groundtruth.py
python src/analyze_baseline_accuracy.py

python src/run_baseline_rppg_v2.py
python src/add_groundtruth_v2.py
python src/analyze_baseline_accuracy_v2.py
python src/find_best_configuration_v2.py

python src/run_baseline_rppg_v3.py
python src/add_groundtruth_v3.py
python src/find_best_configuration_v3.py

python src/run_baseline_rppg_v4.py
python src/add_groundtruth_v4.py
python src/find_best_configuration_v4.py
python src/analyze_v4_reliability.py

python src/run_baseline_rppg_v4_1.py
python src/add_groundtruth_v4_1.py
python src/find_best_configuration_v4_1.py
python src/analyze_v4_1_reliability.py

python src/compare_versions.py

python src/run_ml_hr_correction.py
```
