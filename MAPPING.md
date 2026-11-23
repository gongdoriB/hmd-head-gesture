# MAPPING: script -> reported number

## A. IWCS 2026 poster (release version) — `iwcs_loto_loso/`
Reported: 1,440 windows (8 participants x 3 trials x 6 gestures x 10 reps), 150 x 6 sequences, Macro F1 mean of seeds 42/43/44.

| Model | LOTO | LOSO |
|---|---|---|
| CNN1D | 0.958 | 0.970 |
| RF (48 features) | 0.942 | 0.960 |
| LSTM | 0.857 | 0.943 |

Verified against `results/iwcs_150/summary_metrics.csv` (macro_f1_mean: cnn1d loto 0.9581 / loso 0.9700; rf 0.9419 / 0.9602; lstm 0.8575 / 0.9433). Per-fold values: `results/iwcs_150/fold_metrics.csv`; per-seed: `seed_aggregate_metrics.csv`; settings of the run: `run_config.json`.

Producing script: `iwcs_loto_loso/unified_loto_loso_analysis.py`. Its built-in default sequence length is 128; the poster run overrode it to 150 (per the poster Q&A note and `run_config.json`, `sequence_length: 150`). Command (the exact original command line was not found as a log; this is reconstructed from the argparse options and run_config.json, so 확인 필요 for the original invocation):

    python iwcs_loto_loso/unified_loto_loso_analysis.py --datasets dataset_A --sequence-length 150 \
        --seeds 42 43 44 --epochs 60 --output 04_results/dataset_A_loto_loso_150

(protocols and models default to all: loto+loso, rf+cnn1d+lstm; 288 fold runs.) Run from the repo root.
Input: see DATA_NEEDED.md (dataset A). Pre-processing: label/timestamp-restart windowing, first-5-frame circular-mean baseline for Euler angles, linear resampling to 150, RF = 8 stats x 6 channels = 48 features (400 trees), CNN1D (Conv32k5-Conv64k5-Conv64k3-GAP), LSTM(64), per-fold train-only standardisation, 16.7% stratified validation, early stopping patience 8.

`iwcs_loto_loso/create_150_results_figure.py`: draws the LOTO/LOSO bar figure from `04_results/dataset_A_loto_loso_150/results/summary_metrics.csv` (copy of that file is in `results/iwcs_150/`; place it at that path or edit SOURCE).
`figures/iwcs_macro_f1_loto_loso_150.png`, `figures/iwcs_cnn1d_loso_confusion_matrix.png`: aggregate figures from the same run.

### Supplementary (poster text mentions these analyses) — `iwcs_loto_loso/supplementary/`
- `nested_loso_tuning.py`: subject-grouped nested LOSO tuning. Result: CNN1D 0.977, LSTM 0.957, RF 0.957 (dataset A) in `results/supplementary/tuned_summary_metrics.csv`. CAVEAT: its baseline_vs_tuned.csv uses a baseline of CNN1D 0.9697 (from the older 128-frame `final_loto_loso` run, not the 150-frame run), and the script reads that folder by default. Sequence length used for the tuning run is not recorded in the files I read (확인 필요).
- `dataset_a_lstm_training_size_diagnostic.py`: LSTM LOSO training-size diagnostic; reads `04_results/final_loto_loso/runs` (128-frame run). No result file copied.
The poster abstract text quotes CNN1D LOTO 0.959 (128-frame figure?) while the 150-frame summary says 0.958; the release uses 0.958.

## B. KDCS (earlier version, stratified shuffled 5-fold) — `kdcs_5fold/`
Not the IWCS numbers. `08_compare_rf_cnn_lstm.py` -> `results/kdcs_5fold/rf_cnn_lstm_5fold_summary.csv` (Macro F1: RF 0.982, CNN1D 0.986, LSTM 0.970; StratifiedKFold(5, shuffle, random_state=42); 1,152 train / 288 test per fold). Numbers read from the CSV, not re-run.
Other scripts: 01 feature table (needs raw CSV; writes results/feature_table.csv, which 08 reads), 02 rule-based vs RF, 03 subject generalisation, 04 conservative subject split, 05 RF hyperparameter sensitivity, 06 k-fold/GroupKFold, 07 train-all-trials-1,2 / test-trial-3. Results not copied except 08's two CSVs. These scripts use ROOT = repo root: put raw CSVs under `data/raw/<any subfolder>/pXX_trial_N_{train,test}.csv`; outputs go to `results/`.

## C. Unity data collection — `unity/`
- `HMDGestureExperiment.cs`: runs the protocol (5 s startup; 3 trials; 6 gestures x 10 reps shuffled per trial; 3 s per gesture, 2 s rest; images shown per gesture). Samples the head Transform every `sampleInterval = 0.02 s` (nominally 50 Hz; actual rate depends on frame rate, since sampling happens inside the frame loop) and writes `<subjectId>_trial_<n>_<train|test>.csv` (trials 1-2 "train", trial 3 "test") to `Application.persistentDataPath`. Columns: subject_id, trial, label, timestamp (s from gesture start), head_px/py/pz, head_rx/ry/rz/rw (quaternion), head_ex/ey/ez (Euler, normalised), head_avx/avy/avz (angular velocity deg/s from Euler differences).
- `HeadTracking.cs`: separate real-time rule-based nod/shake (YES/NO) detector (pitch/yaw range thresholds in a 0.6 s window); it is a demo, not part of the reported results.
Unity 6000.3.14f1. Scene, prefabs, sprites are not included.
