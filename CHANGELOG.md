# Changelog

All notable changes to this project will be documented in this file.

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

## [0.1.0] - 2025-09-12

### Added
- Add `train_model.py` supporting CNN/ResNet training for Scenario I & II
- Add logging support via project logger into training pipeline
- Save output artifacts (`.pt`, `.json`, `.png`) under `train_result/` with auto-tagged folders

### Changed
- Update `.gitignore` to exclude training results and data artifacts
- Update `constants.py` with training configs and adjust `RAW_DATA_DIR` to point to `.txt` files compatible with preprocessing/training
- Update `README.md` with training instructions, usage flow, and file structure comments