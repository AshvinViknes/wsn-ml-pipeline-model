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

---

## Directory Structure

```
wsn_ml_pipeline_model/
├── config/                # Configuration, constants, and logging setup
│   ├── constants.py
│   ├── logger.py
├── data/                  # Data storage
│   ├── raw/               # Raw sensor data
│   ├── cleaned/           # Cleaned data output
│   └── preprocessed_data/ # Preprocessed/segmented data
├── logs/                  # Log files
│   └── app.log
├── preprocess/            # Data cleaning and preprocessing scripts
│   ├── clean_data.py
│   ├── preprocessing.py
│   └── preprocessing_workflow.py
├── utils/                 # Utility functions (e.g., saving data)
│   └── save_utils.py
├── requirements.txt       # Python dependencies
├── LICENSE                # License (GPL v3)
└── README.md              # This file
```

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd wsn_ml_pipeline_model
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## Usage

1. **Prepare your raw sensor data** in `data/raw/` as CSV files.
2. **Configure pipeline parameters** in `config/constants.py`.
3. **Run the preprocessing workflow:**
   ```sh
   python -m preprocess.preprocessing_workflow
   ```
   - This will clean, normalize, and segment your data, saving results in `data/cleaned/` and `data/preprocessed_data/`.

4. **Check logs** in `logs/app.log` for pipeline status and errors.

---

## Modules Overview

- `config/constants.py`: Pipeline configuration and constants.
- `config/logger.py`: Logging setup.
- `preprocess/clean_data.py`: Data cleaning logic.
- `preprocess/preprocessing.py`: Preprocessing functions (normalization, segmentation).
- `preprocess/preprocessing_workflow.py`: Orchestrates the full preprocessing pipeline.
- `utils/save_utils.py`: Utility functions for saving data.

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