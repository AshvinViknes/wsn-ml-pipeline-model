# Quick Start Guide

Follow these steps to set up and run the pipeline from scratch.

- Clone the repository:
	```sh
	git clone <repo-url>
	```
- Install prerequisites:
	- Check Python version (3.9+ recommended; install if not already installed):
		```sh
		python3 --version
		```
	- Install pip (if not already installed):
		```sh
		python3 -m ensurepip --upgrade
		```
	- Install virtualenv:
		```sh
		pip install virtualenv
		```
- Open a shell at the project root:
	```sh
	cd path/to/wsn-ml-pipeline-model
	```
- Create and activate a virtual environment:
	```sh
	python3 -m venv .venv
	source .venv/bin/activate   # Linux/Mac
	.venv\Scripts\activate      # Windows (PowerShell)
	python -m pip install -U pip
	```
- Install requirements:
	```sh
	python -m pip install -r requirements.txt
	```
- Run the full workflow (this includes preprocessing + training):
	```sh
	python -m wsn_ml_pipeline_model.workflow.workflow
	```
- After running, you’ll find results in the output directory defined in WSN_ML_PIPELINE_MODEL/CONFIG/CONSTANTS.PY. Default directories:
	```diff
	+ RAW_DATA_DIR = 'wsn_ml_pipeline_model/data/raw'
	- CLEANED_DATA_DIR = 'wsn_ml_pipeline_model/data/cleaned'
	- PREPROCESSED_DATA_DIR = 'wsn_ml_pipeline_model/data/preprocessed_data/frames_'
	+ LOG_FILE_PATH = 'wsn_ml_pipeline_model/logs'
	- TRAIN_INPUT_DIR  = "wsn_ml_pipeline_model/data/preprocessed_data"
	+ TRAIN_OUTPUT_DIR = "wsn_ml_pipeline_model/training/train_result" 
	```

# wsn-ml-pipeline-model
IoT Data Processing Pipeline automates cleaning, normalizing, and segmenting raw sensor data into frames ready for machine learning. It removes outliers, scales data, exports in CSV/NumPy, and includes model training components. Modular and extensible for end-to-end IoT ML workflows.

---

## Features

- **Automated Data Cleaning:** Remove outliers and handle missing values from raw sensor data.
- **Normalization & Scaling:** Standardize sensor data for ML readiness.
- **Segmentation:** Split continuous data into frames/windows for model input.
- **Export:** Save processed data as CSV or NumPy arrays.
- **Logging:** Track pipeline operations and errors.
- **Modular Design:** Easily extend or customize each pipeline stage.
- **Model Training Support:** Train CNN/ResNet models with configurable parameters and automated evaluation.
- **Scenario Configuration:** Easily switch between Scenario I (environment classification) and Scenario II (node classification), including Seen/Unseen settings.

---

## Directory Structure

```
wsn_ml_pipeline_model/
├── config/                # Configuration, constants, and logging setup
│   ├── constants.py
│   └── logger.py
├── data/                  # Data storage
│   ├── raw/               # Raw sensor data
│   ├── cleaned/           # Cleaned data output
│   └── preprocessed_data/ # Preprocessed/segmented data
├── data_cleaner/          # Data cleaning
│   └── clean_data.py      
├── logs/                  # Log files
│   └── app.log
├── preprocess/            # preprocessing scripts
│   ├── preprocessing.py
│   └── preprocessing_workflow.py
├── training/              # Model training scripts
│   ├── train_result/      # Output directory for training runs (plots, reports, models)
│   └── train_model.py     # Entry point for training models
├── workflow/              # End-to-end ML workflow orchestration
│   └── workflow.py        # Entry point for full pipeline (preprocessing + training)
├── utils/                 # Utility functions (e.g., saving data)
│   └── save_utils.py
├── requirements.txt       # Python dependencies
├── .gitignore
├── CHANGELOG.md           
├── LICENSE                # License (GPL v3)
└── README.md              # This file
```

---
## Prerequisites

- **Python 3.8+** (recommended)
- **pip** (Python package installer)
- **virtualenv** (optional, for isolated environments)

To check your Python version:
```sh
python3 --version
```

To install pip (if not already installed):
```sh
python3 -m ensurepip --upgrade
```

To install virtualenv (optional but recommended):
```sh
pip install virtualenv
```

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd wsn_ml_pipeline_model
   ```

2. **Create a virtual env:**
   ```sh
    python3 -m venv venv
    source venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## Usage

1. **Prepare your raw sensor data** in `data/raw/` as `.txt` files.

2. **Configure pipeline parameters** in `config/constants.py`.

3. **Run the full ML workflow (preprocessing + training):**
   ```sh
   python -m wsn_ml_pipeline_model.workflow.workflow
   ```
   - This will clean, normalize, and segment your data, then train a CNN or ResNet model based on the settings in `config/constants.py`. Results (cleaned data, preprocessed frames, plots, reports, model weights) are saved to `data/cleaned/`, `data/preprocessed_data/`, and `training/train_result/`.

4. **(Advanced) Run only the preprocessing workflow:**
   ```sh
   python -m wsn_ml_pipeline_model.preprocess.preprocessing_workflow
   ```
   - This will only clean, normalize, and segment your data, saving results in `data/cleaned/` as `.csv` files and `data/preprocessed_data/` as `.npy` files.

5. **(Advanced) Train the ML model using preprocessed data:**
   ```sh
   python -m wsn_ml_pipeline_model.training.train_model
   ```
   - This will train a CNN or ResNet model using existing preprocessed data, and save results (plots, reports, model weights) to `training/train_result/`.

6. **Check logs** in `logs/app.log` for pipeline status and errors.
---

## Modules Overview

- `config/constants.py`: Pipeline configuration and constants.
- `config/logger.py`: Logging setup.
- `data_cleaner/clean_data.py`: Data cleaning logic.
- `preprocess/preprocessing.py`: Preprocessing functions (normalization, segmentation).
- `preprocess/preprocessing_workflow.py`: Orchestrates the full preprocessing pipeline.
- `utils/save_utils.py`: Utility functions for saving data.
- `training/train_model.py`: Train and evaluate ML models using the preprocessed data.
- `workflow/workflow.py`: Orchestrates the end-to-end ML workflow (preprocessing + training).

---

## Requirements

See `requirements.txt`:

- numpy
- pandas
- scikit-learn
- matplotlib
- torch (only the light CPU version is sufficient)

---

## How to Use constants.py

All settings for training are in wsn_ml_pipeline_model/config/constants.py.
Edit these values before running the workflow to change how the model trains.

- Main options:
	- Data paths:
		TRAIN_INPUT_DIR: where your .npy data lives
		TRAIN_OUTPUT_DIR: where results (models, logs, confusion matrices) are saved
	- Experiment setup:
		SCENARIO = "I" → classify environment
		SCENARIO = "II" → classify receiving node
		SEEN_SPLIT = True → random train/test split
		SEEN_SPLIT = False → hold out a node (HELD_OUT_NODE) or environment (HELD_OUT_ENV)
	- Training settings:
		BATCH_SIZE, EPOCHS, LR, SEED
		INPUT_CHANNEL = "rssi" | "lqi" | "both"
		MODEL_TYPE = "cnn" | "resnet"
		N_TRAIN_RUNS: how many independent runs to perform
		RESUME_TRAINING = True → continue from the last checkpoint
- Outputs: After training, you will find:
	- model_runX.pt → trained model
	- meta_runX.json → training metadata
	- ConfMat_runX.png → confusion matrix for each run
	- Final_Confusion_Matrix.png → best confusion matrix across runs

---

## License

This project is licensed under the GNU GPL v3.

---

## Contributing

Contributions are welcome!
