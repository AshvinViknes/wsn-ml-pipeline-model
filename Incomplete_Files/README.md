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
├── utils/                 # Utility functions (e.g., saving data)
│   └── save_utils.py
├── requirements.txt       # Python dependencies
├── .gitignore
├── CHANGELOG.md           # 
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

1. **Prepare your raw sensor data** in `data/raw/` as CSV files.

2. **Configure pipeline parameters** in `config/constants.py`.
> ⚠️ Make sure you are located **one level above** the `wsn_ml_pipeline_model/` directory (e.g., in the `Incomplete_Files/` folder) when running the following commands.

3. **Run the preprocessing workflow:**
   ```sh
   python -m wsn_ml_pipeline_model.preprocess.preprocessing_workflow
   ```
   - This will clean, normalize, and segment your data, saving results in `data/cleaned/` as `.csv` files and `data/preprocessed_data/` as `.npy` files.
   
4. **Train the ML model using preprocessed data:**
   ```sh
   python -m wsn_ml_pipeline_model.training.train_model
   ```
   - This will train a CNN or ResNet model based on the settings in `config/constants.py`, and save results (plots, reports, model weights) to `training/train_result/`.

5. **Check logs** in `logs/app.log` for pipeline status and errors.

---

## Modules Overview

- `config/constants.py`: Pipeline configuration and constants.
- `config/logger.py`: Logging setup.
- `data_cleaner/clean_data.py`: Data cleaning logic.
- `preprocess/preprocessing.py`: Preprocessing functions (normalization, segmentation).
- `preprocess/preprocessing_workflow.py`: Orchestrates the full preprocessing pipeline.
- `utils/save_utils.py`: Utility functions for saving data.
- `training/train_model.py`: Train and evaluate ML models using the preprocessed data.

---

## Requirements

See `requirements.txt`:

- numpy
- pandas
- tensorflow
- scikit-learn
- matplotlib

---

## License

This project is licensed under the GNU GPL v3.

---

## Contributing

Contributions are welcome!