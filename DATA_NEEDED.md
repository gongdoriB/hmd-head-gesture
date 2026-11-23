# Data needed (not included)

Raw per-session CSVs, one per (participant, trial), named `p01_trial_1_train.csv`, `p01_trial_2_train.csv`, `p01_trial_3_test.csv`, ... for p01-p08 (24 files). Produced by `unity/HMDGestureExperiment.cs`.

Columns (header row): `subject_id,trial,label,timestamp,head_px,head_py,head_pz,head_rx,head_ry,head_rz,head_rw,head_ex,head_ey,head_ez,head_avx,head_avy,head_avz`
- `label` in {Rotating_Left, Rotating_Right, Tilting_Up, Tilting_Down, Leaning_Left, Leaning_Right}.
- `timestamp` = seconds since the start of each 3 s gesture (restarts at each gesture; the scripts use label change or timestamp restart as the window boundary).
- ~150 rows per gesture window (50 Hz x 3 s); 60 windows per trial -> 1,440 windows in total (240 per class).
- Signals actually used: head_ex, head_ey, head_ez, head_avx, head_avy, head_avz.

Where each pipeline looks:
- IWCS (`iwcs_loto_loso/unified_loto_loso_analysis.py`): `<repo>/02_data/dataset_A_jeju/raw/{p01,p02,p03,hmd_p04_data,hmd_head_gesture_data}/pXX_trial_N_(train|test).csv` (hard-coded in `dataset_a_files()`; exactly 24 unique (subject, trial) files required; edit that function to use your own folder).
- KDCS (`kdcs_5fold/`): `<repo>/data/raw/<subfolder>/pXX_trial_N_(train|test).csv`.
Other output folders are created next to the repo root (`04_results/...`, `results/`).
