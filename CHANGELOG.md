# Changelog

All notable changes to this project will be documented in this file.

---

## [0.6.0] - 2025-09-23

### Added

* `preprocess/preprocessing_workflow.py` now scans env subfolders (`*/*.txt`, `*/*.csv`) and preserves `<env>/<node>` in outputs.

### Changed

* **Layout:**

  * raw → `data/<env>/<node>.txt,csv`
  * frames → `data/preprocessed_data/frames_<Tx>_<Rx>_<env>.npy`
* `DataCleaner`: outputs **2 cols** (`rssi,lqi`); passes through valid CSVs.

### Breaking

* Removed `timestamp` from cleaned CSVs.

---

## [0.5.0] - 2025-09-20

### Added

* Introduced `RunHelper` class to centralize helpers for tagging, run directories, checkpoints, epoch tracking, and confusion matrix plotting.
* New `training/frame_loader.py` for dataset handling (`FramesDataset`, parsing, loading).
* Dedicated `training/models.py` with `ModelFactory` for model selection.
* `training/trainer.py` encapsulates train/test loop logic.

### Changed

* `WSNPipeline` in `training/pipeline.py` simplified: now delegates helper logic to `RunHelper`.
* Clearer project structure: split responsibilities across `helper.py`, `models.py`, `trainer.py`, `frame_loader.py`.
* `pipeline.py` now acts as a lightweight entry point.

---

## [0.4.4] - 2025-9-19

### Added

- Confusion matrix plots now include run info in their titles for easier traceability.

### Changed

- Final confusion matrix selection improved by initializing global best accuracy from existing `meta_run*.json` files and updating after each run.
- Removed redundant candidate_resume_path variable
- Directly assign resume_path and reset to None if checkpoint not found

---

## [0.4.3] - 2025-09-17

### Added

- Special case for single-run training (`RESUME_TRAINING=False`, `N_TRAIN_RUNS=1`): saves as `model_single.pt`, `meta_single.json`, `ConfMat_single.png`
- Updated `get_tag()` to append `_epoch({EPOCHS})` to output directory name for clearer separation across epoch settings

### Changed

- Multi-run continues with run-indexed naming for model, metadata, and confusion matrix files
- Centralized `run_name` determination in `run()` to avoid duplication
- Improved logging: single-run mode logs "=== Single training run ===" instead of run counters
- Simplified confusion matrix saving by removing duplicate `plt.savefig`

---

## [0.4.2] - 2025-09-17

### Changed

- Simplified `run_multiple` API by defaulting to config values for `n_runs` and `resume_latest`.
- Top-level `run_multiple` now only exposes `n_runs`, with resume behavior always read from config.
- Centralized `prev_total_epochs` handling inside `run()`, removing calculation and passing from `run_multiple`.
- `run()` now reads `prev_total_epochs` directly from metadata using `resume_path` and `get_total_epochs_by_index`.

---

## [0.4.1] - 2025-09-16

### Added

- Extended `meta_run.json` to store richer training information (config, data split, results).

### Changed

- Confusion matrix output file naming now uses run index instead of tag for clarity.
- Refactored `train_model.py`:
  - Introduced helper function `get_split_name()` to generate consistent split naming (`Seen(...)` / `Unseen(...)`).
  - Introduced helper function `get_tag()` to construct run tags from scenario, split, model type, and input channel.
  - Removed duplicate inline definitions of `split_name` and `tag` in `run_multiple()` and `run()`.
- Simplified workflow by removing `PREPROCESSING_ACTIVE` flag; preprocessing now always runs before training.

---

## [0.4.0] - 2025-09-16

### Added

- Implemented persistent cumulative epoch tracking across multiple training runs, even when resuming from checkpoints.
- Added logic to always read the previous run's metadata (`meta_run{N-1}.json`) to correctly accumulate total epochs for each model.
- Updated output folder and result file naming to include number of runs and total epochs for better experiment tracking.
- Improved training summary logging to report the correct cumulative epoch count after each run.
- Added helper function to extract `total_epochs` from metadata JSON for robust resuming and reporting.

### Changed

- Refactored `run_multiple` and `run` methods to use previous run index and metadata for accurate epoch accumulation.
- Ensured that all training artifacts (`.pt`, `.json`, `.png`) are consistently indexed and saved in the same output directory for a given configuration.
- Enhanced documentation and code comments to clarify the new cumulative epoch tracking and result organization.

---

## [0.3.0] - 2025-09-15

### Added

- Added workflow control constants: `RESUME_TRAINING`, `PREPROCESSING_ACTIVE`, and `N_TRAIN_RUNS` in `config/constants.py` for flexible pipeline execution.
- Enhanced `WSNPipeline` and training logic to support resuming training from the latest checkpoint and running multiple training cycles automatically.
- Added utility to dynamically locate and load the latest model checkpoint for resuming training.
- Updated `MLWorkflow` and workflow entrypoint to use the new constants for controlling preprocessing, checkpoint resuming, and number of training runs.
- Improved documentation and usage examples in workflow script to reflect new workflow control options.

### Changed

- Refactored workflow and training scripts to use the new constants instead of hardcoded arguments for pipeline control.

---

## [0.2.0] - 2025-09-14

### Added

- Introduced `workflow/workflow.py` for end-to-end ML pipeline orchestration (preprocessing + training)
- Added `MLWorkflow` class to manage full pipeline execution from raw data to trained model
- Added conditional CSV saving via `SAVE_FRAME_CSV` constant in `constants.py`
- Updated README and directory structure to document new workflow module and usage
- Added support for passing preloaded frames directly to training pipeline

### Changed

- Refactored training pipeline to use a `Config` dataclass for configuration management
- Updated `WSNPipeline` and related classes to use attribute access for config values
- Improved error handling and logging throughout pipeline modules
- Enhanced modularity and clarity of pipeline entry points

---

## [0.1.0] - 2025-09-12

### Added

- Add `train_model.py` supporting CNN/ResNet training for Scenario I & II
- Add logging support via project logger into training pipeline
- Save output artifacts (`.pt`, `.json`, `.png`) under `train_result/` with auto-tagged folders

### Changed

- Update `.gitignore` to exclude training results and data artifacts
- Update `constants.py` with training configs and adjust `RAW_DATA_DIR` to point to `.txt` files compatible with preprocessing/training
- Update `README.md` with training instructions, usage flow, and file structure comments
