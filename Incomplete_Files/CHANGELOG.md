# Changelog

All notable changes to this project will be documented in this file.

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